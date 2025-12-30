#!/usr/bin/env python3
"""Test script for passive bot - processes audio file without WebRTC."""

import asyncio
import os
import sys
import wave
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from openai import NOT_GIVEN
from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    LLMMessagesFrame,
    StartFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import WhisperSTTService

from passive_processors import SummaryWriter, TranscriptBatchProcessor

load_dotenv(override=True)


async def test_passive_bot(audio_file: str):
    """Test passive bot with an audio file."""

    logger.info(f"Testing passive bot with audio file: {audio_file}")

    # Check file exists
    if not Path(audio_file).exists():
        logger.error(f"Audio file not found: {audio_file}")
        return

    # Load audio file using wave module
    logger.info("Loading audio file...")
    with wave.open(audio_file, 'rb') as wf:
        sample_rate = wf.getframerate()
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        num_frames = wf.getnframes()

        logger.info(f"Audio: {num_frames} frames, {sample_rate}Hz, {num_channels} channels, {sample_width} bytes/sample")

        # Read all frames
        audio_bytes = wf.readframes(num_frames)

        # Convert to numpy array
        if sample_width == 2:  # 16-bit
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        elif sample_width == 4:  # 32-bit
            audio_data = np.frombuffer(audio_bytes, dtype=np.int32)
            # Convert to 16-bit
            audio_data = (audio_data / 65536).astype(np.int16)
        else:
            logger.error(f"Unsupported sample width: {sample_width}")
            return

    # Convert to mono if stereo
    if num_channels == 2:
        audio_data = audio_data.reshape(-1, 2).mean(axis=1).astype(np.int16)
        logger.info("Converted stereo to mono")

    # Check sample rate (Whisper expects 16kHz)
    # For now, just warn if it's not 16kHz - we'll skip resampling to avoid scipy dependency
    target_rate = 16000
    if sample_rate != target_rate:
        logger.warning(f"Sample rate is {sample_rate}Hz, but Whisper expects {target_rate}Hz")
        logger.warning("Proceeding anyway - transcription may be affected")

    logger.info(f"Audio loaded: {len(audio_data)} samples")

    # Create services
    logger.info("Initializing services...")

    stt = WhisperSTTService(
        model=os.getenv("WHISPER_MODEL", "base"),
        device="cpu",
        compute_type="int8"
    )

    llm = OLLamaLLMService(model=os.getenv("OLLAMA_MODEL", "llama3.3"))

    # System prompt
    SUMMARY_SYSTEM_PROMPT = """You are a radio communication analyst. Your task is to analyze transcripts of radio conversations and extract:

1. Main topics discussed
2. Call signs mentioned (ham radio format: e.g., W1ABC, K2XYZ, N4DEF)
3. Technical details (frequencies, signal reports, equipment)
4. A concise summary

Be factual and technical. Focus on what was actually said.

Always respond in this exact format:
TOPICS: [comma-separated list]
CALLSIGNS: [comma-separated list]
SUMMARY: [2-3 sentences]
"""

    messages = [{"role": "system", "content": SUMMARY_SYSTEM_PROMPT}]
    context = LLMContext(messages, tools=NOT_GIVEN)
    context_aggregator = LLMContextAggregatorPair(context)

    # Create processors
    # Use a shorter batch window for testing (30 seconds instead of 5 minutes)
    batch_processor = TranscriptBatchProcessor(
        batch_window_seconds=int(os.getenv("BATCH_WINDOW_SECONDS", "30")),
        task=None
    )

    summary_writer = SummaryWriter(output_dir="summaries/")

    transcript_processor = TranscriptProcessor()

    # Create pipeline (no transport needed)
    pipeline = Pipeline([
        stt,
        transcript_processor.user(),
        context_aggregator.user(),
        llm,
        summary_writer,
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=False,
            audio_in_sample_rate=sample_rate,
        ),
    )

    # Set task reference for batch processor
    batch_processor.set_task(task)

    # Event handler to collect transcripts and batch them
    @transcript_processor.event_handler("on_transcript_update")
    async def on_transcript_update(processor, frame):
        logger.info(f"Transcript update: {len(frame.messages)} messages")
        from pipecat.frames.frames import TranscriptionMessage
        for msg in frame.messages:
            if isinstance(msg, TranscriptionMessage):
                logger.info(f"  - {msg.role}: {msg.content}")
                await batch_processor.add_transcript(msg)

    # Start the pipeline
    logger.info("Starting pipeline...")
    runner = PipelineRunner()

    async def feed_audio():
        """Feed audio data to the pipeline."""
        logger.info("Feeding audio to pipeline...")

        # Send StartFrame
        await task.queue_frames([StartFrame()])

        # Split audio into chunks (20ms frames = 320 samples at 16kHz)
        chunk_size = 320
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]

            # Pad last chunk if needed
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            # Create audio frame
            audio_frame = AudioRawFrame(
                audio=chunk.tobytes(),
                sample_rate=sample_rate,
                num_channels=1
            )

            await task.queue_frames([audio_frame])

            # Small delay to simulate real-time (optional, can remove for faster processing)
            # await asyncio.sleep(0.02)

        logger.info("Audio feeding complete, waiting for final batch...")

        # Process any remaining batch
        if batch_processor._current_batch:
            logger.info("Processing final batch...")
            await batch_processor._process_batch()

        # Small delay to let LLM finish
        await asyncio.sleep(5)

        # Send EndFrame
        await task.queue_frames([EndFrame()])

    # Run both the pipeline and audio feeding concurrently
    try:
        await asyncio.gather(
            runner.run(task),
            feed_audio()
        )
    except asyncio.CancelledError:
        logger.info("Pipeline cancelled")

    logger.info("Test complete!")
    logger.info("Check the summaries/ directory for output")


if __name__ == "__main__":
    # Use first recording by default, or accept argument
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "recordings/merged_20251229_010801.wav"

    asyncio.run(test_passive_bot(audio_file))
