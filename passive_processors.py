"""
Custom processors for passive listening bot.

This module contains processors for batching transcripts, detecting speakers,
and writing summary files for the passive monitoring bot.
"""

import datetime
import json
import re
from pathlib import Path
from typing import Any

import aiofiles
from loguru import logger
from pydantic import BaseModel, Field
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import (
    TranscriptionFrame,
    TextFrame,
    Frame,
    LLMMessagesFrame,
    LLMFullResponseEndFrame
)


class DigitalSignal(BaseModel):
    """Detected digital signal that couldn't be decoded."""
    timestamp: str = Field(description="ISO timestamp when signal was detected")
    duration_seconds: float = Field(description="Duration of the signal in seconds")
    signal_type: str | None = Field(default=None, description="Type of signal if identifiable (e.g., FT8, RTTY, PSK31)")


class RadioSummary(BaseModel):
    """Structured summary of radio communications."""
    topics: list[str] = Field(description="Main topics discussed in the conversation")
    callsigns: list[str] = Field(description="Amateur radio call signs mentioned (format: W1ABC, K2XYZ, etc.)")
    summary: str = Field(description="2-3 sentence summary of the radio communication")
    digital_signals: list[DigitalSignal] = Field(default_factory=list, description="Any digital signals detected but not decoded")


class TranscriptBatchProcessor:
    """Batches transcripts into time windows for summarization (event-driven, not a pipeline processor)."""

    def __init__(self, batch_window_seconds: int = 300, task=None, summary_writer=None):
        self._batch_window = batch_window_seconds
        self._current_batch = []
        self._batch_start_time = None
        self._last_activity = None
        self._task = task
        self._summary_writer = summary_writer
        self._speaker_tracker = {}
        self._current_speaker = 1
        self._last_speech_time = None
        self._pause_threshold = 2.0  # seconds for speaker change

    def set_task(self, task):
        """Set the pipeline task for queueing frames."""
        self._task = task

    def set_summary_writer(self, summary_writer):
        """Set reference to summary writer for passing metadata."""
        self._summary_writer = summary_writer

    async def add_transcript(self, msg):
        """Add a transcript message to the current batch."""
        now = datetime.datetime.now()

        # Detect speaker changes based on pauses
        speaker_id = self._detect_speaker(now)

        # Start new batch if first transcript
        if self._batch_start_time is None:
            self._batch_start_time = now
            logger.info("Starting new batch")

        # Add to current batch
        self._current_batch.append({
            "timestamp": msg.timestamp if msg.timestamp else now,
            "text": msg.content,
            "role": msg.role,
            "speaker_id": speaker_id
        })

        self._last_activity = now

        # Check if batch window elapsed
        elapsed = (now - self._batch_start_time).total_seconds()

        if elapsed >= self._batch_window:
            logger.info(f"Batch window elapsed ({elapsed:.1f}s), processing batch")
            await self._process_batch()

    def _detect_speaker(self, now):
        """Simple speaker detection based on pauses."""
        if self._last_speech_time:
            pause = (now - self._last_speech_time).total_seconds()
            if pause > self._pause_threshold:
                self._current_speaker += 1
                logger.debug(f"Speaker change detected (pause: {pause:.1f}s)")

        self._last_speech_time = now
        return self._current_speaker

    async def _process_batch(self):
        """Send accumulated batch to LLM for summarization."""
        if not self._current_batch:
            logger.warning("No transcripts in batch, skipping")
            return

        # Create summary prompt with JSON schema
        transcript_text = self._format_batch_for_llm()

        # Get JSON schema from Pydantic model
        schema = RadioSummary.model_json_schema()

        summary_prompt = f"""Analyze this radio communication transcript and extract structured information.

Transcript:
{transcript_text}

Please analyze and return a JSON object with:
- topics: Array of main topics discussed (strings)
- callsigns: Array of amateur radio call signs mentioned (format: W1ABC, K2XYZ, etc.)
- summary: A 2-3 sentence summary of the communication
- digital_signals: Array of any digital signals detected (empty array if none)

Respond with valid JSON matching this schema:
{json.dumps(schema, indent=2)}
"""

        logger.info(f"Sending batch to LLM ({len(self._current_batch)} transcripts)")

        # Store batch metadata in summary writer for access when LLM responds
        if self._summary_writer:
            self._summary_writer._pending_batch_metadata = {
                "batch_start": self._batch_start_time,
                "batch_end": self._last_activity,
                "transcript": self._current_batch.copy(),
                "speaker_count": len(set(t["speaker_id"] for t in self._current_batch))
            }

        # Queue LLM prompt to task
        if self._task:
            await self._task.queue_frames([LLMMessagesFrame(messages=[{
                "role": "user",
                "content": summary_prompt
            }])])

        # Reset batch
        self._current_batch = []
        self._batch_start_time = None

    def _format_batch_for_llm(self) -> str:
        """Format batch as readable transcript."""
        lines = []
        for entry in self._current_batch:
            time_str = entry["timestamp"].strftime("%H:%M:%S") if hasattr(entry["timestamp"], "strftime") else str(entry["timestamp"])
            speaker = entry.get("speaker_id", "?")
            role = entry.get("role", "user")
            lines.append(f"[{time_str}] Speaker {speaker} ({role}): {entry['text']}")
        return "\n".join(lines)


# SpeakerDetector removed - speaker detection is now handled by TranscriptBatchProcessor


class SummaryWriter(FrameProcessor):
    """Writes formatted summary files."""

    def __init__(self, output_dir: str = "summaries/"):
        super().__init__()
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(exist_ok=True)
        self._pending_batch_metadata = None
        self._llm_response = ""

    async def process_frame(self, frame: Frame, direction):
        """Receive LLM summary and write file."""

        # CRITICAL: Must call super().process_frame() FIRST to handle StartFrame and lifecycle frames
        await super().process_frame(frame, direction)

        # Collect LLM response text frames
        if isinstance(frame, TextFrame) and frame.text and frame.text.strip():
            self._llm_response += frame.text

        # Write summary when LLM finishes generating response
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._llm_response and self._pending_batch_metadata:
                logger.info("LLM response complete, writing summary")
                await self._write_summary_file(
                    self._llm_response,
                    self._pending_batch_metadata
                )
                self._pending_batch_metadata = None
                self._llm_response = ""
            elif self._llm_response:
                logger.warning("LLM response complete but no batch metadata available")
                self._llm_response = ""

        # Pass all frames through
        await self.push_frame(frame, direction)

    async def _write_summary_file(self, llm_response: str, metadata: dict):
        """Format and write summary file."""

        # Parse LLM response into structured object
        radio_summary = self._parse_llm_response(llm_response)

        # Generate filename
        start_time = metadata["batch_start"]
        timestamp = start_time.strftime("%Y%m%d-%H%M%S")
        topic_slug = self._slugify_topic(radio_summary.topics[0] if radio_summary.topics else "conversation")
        filename = f"{timestamp}_{topic_slug}.txt"

        # Format content
        content = self._format_summary(
            start_time=metadata["batch_start"],
            end_time=metadata["batch_end"],
            radio_summary=radio_summary,
            transcript=metadata["transcript"],
            speaker_count=metadata.get("speaker_count", None)
        )

        # Write file
        filepath = self._output_dir / filename
        async with aiofiles.open(filepath, "w") as f:
            await f.write(content)

        logger.info(f"✅ Summary saved: {filepath}")

    def _parse_llm_response(self, response: str) -> RadioSummary:
        """Parse JSON response from LLM into RadioSummary object.

        Args:
            response: JSON string from LLM (may be wrapped in markdown code fences)

        Returns:
            RadioSummary object with parsed data, or empty/default values if parsing fails
        """
        logger.debug(f"Parsing LLM JSON response (length: {len(response)})")

        try:
            # Strip markdown code fences if present (common with LLMs)
            json_str = response.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]  # Remove ```json
            elif json_str.startswith("```"):
                json_str = json_str[3:]  # Remove ```

            if json_str.endswith("```"):
                json_str = json_str[:-3]  # Remove trailing ```

            json_str = json_str.strip()

            # Parse JSON
            data = json.loads(json_str)
            summary = RadioSummary(**data)
            logger.info(f"Successfully parsed summary: {len(summary.topics)} topics, {len(summary.callsigns)} callsigns, {len(summary.digital_signals)} digital signals")
            return summary

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response that failed to parse: {response[:500]}")

            # Return default empty summary
            return RadioSummary(
                topics=["Unable to parse summary"],
                callsigns=[],
                summary="Error: LLM response was not valid JSON",
                digital_signals=[]
            )

        except Exception as e:
            logger.error(f"Unexpected error parsing LLM response: {e}")
            return RadioSummary(
                topics=["Error parsing response"],
                callsigns=[],
                summary=f"Error: {str(e)}",
                digital_signals=[]
            )

    def _format_summary(self, start_time, end_time, radio_summary: RadioSummary,
                       transcript, speaker_count=None) -> str:
        """Create formatted summary file content."""

        duration_seconds = (end_time - start_time).total_seconds()
        duration_str = f"{int(duration_seconds // 60)}m {int(duration_seconds % 60)}s"

        # Use provided speaker count or calculate from transcript
        if speaker_count is None:
            speaker_ids = set(t.get("speaker_id", 1) for t in transcript)
            speaker_count = len(speaker_ids)

        content = f"""=================================================
Auto Dispatch - Passive Monitoring Summary
=================================================

Session Start: {start_time.strftime("%Y-%m-%d %H:%M:%S")}
Session End:   {end_time.strftime("%Y-%m-%d %H:%M:%S")}
Duration:      {duration_str}
Speakers:      {speaker_count} detected
Call Signs:    {", ".join(radio_summary.callsigns) if radio_summary.callsigns else "None detected"}

Topics:
{chr(10).join("- " + t for t in radio_summary.topics) if radio_summary.topics else "- General conversation"}

=================================================
Summary
=================================================

{radio_summary.summary if radio_summary.summary else "No summary available"}
"""

        # Add digital signals section if any were detected
        if radio_summary.digital_signals:
            content += f"""
=================================================
Digital Signals Detected (Not Decoded)
=================================================

"""
            for sig in radio_summary.digital_signals:
                sig_type = f" ({sig.signal_type})" if sig.signal_type else ""
                content += f"[{sig.timestamp}] Duration: {sig.duration_seconds:.1f}s{sig_type}\n"

        content += """
=================================================
Full Transcript
=================================================

"""

        # Add full transcript
        for entry in transcript:
            time_str = entry["timestamp"].strftime("%H:%M:%S") if hasattr(entry["timestamp"], "strftime") else str(entry["timestamp"])
            speaker = entry.get("speaker_id", "?")
            role = entry.get("role", "user")
            content += f"[{time_str}] Speaker {speaker} ({role}): \"{entry['text']}\"\n"

        return content

    @staticmethod
    def _slugify_topic(topic: str) -> str:
        """Convert topic to filename-safe slug."""
        slug = topic.lower()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[\s_-]+', '_', slug)
        slug = slug[:50]  # Limit length
        return slug if slug else "conversation"
