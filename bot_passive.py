"""Auto Dispatch - Passive Listening Bot

This bot listens continuously without speaking, creating summarized transcripts
with timestamps, speaker detection, and topic identification.

Pipeline: Speech-to-Text → Batch Processor → LLM Summarizer → File Writer

Run the bot using::

    uv run bot_passive.py
"""

import datetime
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from openai import NOT_GIVEN
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat_tail.observer import TailObserver
from pipecat_whisker import WhiskerObserver

from passive_processors import (
    TranscriptBatchProcessor,
    SummaryWriter
)

load_dotenv(override=True)


async def run_passive_bot(transport: BaseTransport):
    """Passive monitoring bot - listens without responding."""
    logger.info("Starting passive listening bot")

    # Speech-to-Text only (no TTS)
    stt = WhisperSTTService(
        model=os.getenv("WHISPER_MODEL", "base"),
        device="cpu",
        compute_type="int8"
    )

    # LLM for summarization (not conversation)
    llm = OLLamaLLMService(model=os.getenv("OLLAMA_MODEL", "llama3.3"))

    # System prompt for summarization mode
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

    messages = [
        {
            "role": "system",
            "content": SUMMARY_SYSTEM_PROMPT
        }
    ]

    # Create context for LLM (no tools needed for summarization)
    context = LLMContext(messages, tools=NOT_GIVEN)
    context_aggregator = LLMContextAggregatorPair(context)

    # Custom processors for passive listening
    batch_window = int(os.getenv("BATCH_WINDOW_SECONDS", "300"))  # 5 minutes default
    output_dir = os.getenv("SUMMARY_OUTPUT_DIR", "summaries/")
    summary_writer = SummaryWriter(output_dir=output_dir)

    batch_processor = TranscriptBatchProcessor(
        batch_window_seconds=batch_window,
        task=None,  # Will be set after task is created
        summary_writer=summary_writer  # Pass reference for metadata sharing
    )

    # Use built-in transcript processor for observation
    from pipecat.processors.transcript_processor import TranscriptProcessor
    transcript_processor = TranscriptProcessor()

    # Pipeline: STT → Transcript → Context → LLM (for summarization only)
    # Note: No TTS or transport output - we're not speaking back
    pipeline = Pipeline([
        transport.input(),
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
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_in_sample_rate=16000,  # Whisper expects 16kHz
        ),
        observers=[
            WhiskerObserver(pipeline),
            TailObserver(),
        ],
    )

    # Set task reference for batch processor
    batch_processor.set_task(task)

    # Event handler to collect transcripts and batch them
    @transcript_processor.event_handler("on_transcript_update")
    async def on_transcript_update(processor, frame):
        from pipecat.frames.frames import TranscriptionMessage
        for msg in frame.messages:
            if isinstance(msg, TranscriptionMessage):
                await batch_processor.add_transcript(msg)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected - passive listening started")
        logger.info(f"Batch window: {batch_window} seconds")
        logger.info(f"Summary output directory: {output_dir}")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected - passive listening stopped")
        # Final batch processing if needed
        if batch_processor._current_batch:
            logger.info("Processing final batch before disconnect")
            await batch_processor._process_batch()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""

    transport = None

    match runner_args:
        case SmallWebRTCRunnerArguments():
            webrtc_connection: SmallWebRTCConnection = runner_args.webrtc_connection

            transport = SmallWebRTCTransport(
                webrtc_connection=webrtc_connection,
                params=TransportParams(
                    audio_in_enabled=True,
                    audio_out_enabled=False,  # No audio output - passive listening only
                    vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
                ),
            )
        case _:
            logger.error(f"Unsupported runner arguments type: {type(runner_args)}")
            return

    await run_passive_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
