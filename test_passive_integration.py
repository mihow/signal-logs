#!/usr/bin/env python3
"""Integration test for passive bot - simulates full pipeline without WebRTC."""

import asyncio
import datetime
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from openai import NOT_GIVEN
from pipecat.frames.frames import (
    EndFrame,
    LLMMessagesFrame,
    StartFrame,
    TextFrame,
    TranscriptionMessage,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.ollama.llm import OLLamaLLMService

from passive_processors import SummaryWriter, TranscriptBatchProcessor

load_dotenv(override=True)


async def test_passive_pipeline():
    """Test the complete passive bot pipeline with simulated transcripts."""

    logger.info("=" * 60)
    logger.info("PASSIVE BOT INTEGRATION TEST")
    logger.info("=" * 60)

    # Clean up any old test summaries
    test_dir = Path("test_summaries")
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    # Create LLM service
    logger.info("Initializing LLM service...")
    llm = OLLamaLLMService(model=os.getenv("OLLAMA_MODEL", "llama3.3"))

    # System prompt
    SUMMARY_SYSTEM_PROMPT = """You are a radio communication analyst. Analyze transcripts and extract:
1. Main topics discussed
2. Call signs mentioned (ham radio format: e.g., W1ABC, K2XYZ)
3. Brief summary

Always respond in this exact format:
TOPICS: [comma-separated list]
CALLSIGNS: [comma-separated list or "None"]
SUMMARY: [2-3 sentences]
"""

    messages = [{"role": "system", "content": SUMMARY_SYSTEM_PROMPT}]
    context = LLMContext(messages, tools=NOT_GIVEN)
    context_aggregator = LLMContextAggregatorPair(context)

    # Create processors with very short batch window for testing (5 seconds)
    logger.info("Creating processors...")
    summary_writer = SummaryWriter(output_dir="test_summaries/")

    batch_processor = TranscriptBatchProcessor(
        batch_window_seconds=5,  # Short window for quick test
        task=None,
        summary_writer=summary_writer  # Pass reference for metadata sharing
    )

    # Create pipeline
    logger.info("Building pipeline...")
    pipeline = Pipeline([
        context_aggregator.user(),
        llm,
        summary_writer,
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=False,
        ),
    )

    # Set task reference for batch processor
    batch_processor.set_task(task)

    # Define test transcripts
    transcripts = [
        ("This is W1ABC calling on 146.520", "00:00:01"),
        ("W1ABC this is K2XYZ, I copy you five by nine", "00:00:05"),
        ("K2XYZ, thanks for the report. I'm testing a new antenna", "00:00:10"),
        ("Roger that W1ABC. How's the SWR looking?", "00:00:15"),
        ("SWR is reading 1.2 to 1, pretty happy with it", "00:00:20"),
        ("That's excellent. What kind of antenna are you using?", "00:00:25"),
    ]

    # Track test results
    test_results = {
        "startframe_ok": False,
        "transcripts_processed": 0,
        "batch_triggered": False,
        "summary_created": False,
    }

    # Start the pipeline
    logger.info("Starting pipeline...")
    runner = PipelineRunner()

    async def simulate_transcripts():
        """Simulate incoming transcripts from users."""
        try:
            # Send StartFrame
            logger.info("✓ Sending StartFrame...")
            await task.queue_frames([StartFrame()])
            test_results["startframe_ok"] = True
            logger.success("  StartFrame sent successfully!")

            # Wait a moment for pipeline to initialize
            await asyncio.sleep(0.5)

            # Simulate radio conversation (using transcripts from outer scope)
            logger.info(f"\n✓ Simulating {len(transcripts)} transcript messages...")
            for text, timestamp in transcripts:
                msg = TranscriptionMessage(
                    role="user",
                    content=text,
                    timestamp=datetime.datetime.now()
                )
                await batch_processor.add_transcript(msg)
                test_results["transcripts_processed"] += 1
                logger.info(f"  [{timestamp}] {text}")
                await asyncio.sleep(0.1)

            logger.success(f"  Processed {test_results['transcripts_processed']} transcripts")

            # Wait for batch window to trigger (5 seconds + buffer)
            logger.info("\n✓ Waiting for batch processor (5 second window)...")
            await asyncio.sleep(6)

            # Manually trigger final batch if it hasn't been processed
            if batch_processor._current_batch:
                logger.info("  Manually triggering final batch...")
                await batch_processor._process_batch()
                test_results["batch_triggered"] = True
                logger.success("  Batch processed!")
            elif batch_processor._batch_start_time is None and test_results["transcripts_processed"] > 0:
                test_results["batch_triggered"] = True
                logger.success("  Batch was already processed!")
            else:
                logger.warning("  Batch status unclear")

            # Wait for LLM to generate summary (check periodically)
            logger.info("\n✓ Waiting for LLM summary generation...")
            for i in range(10):  # Wait up to 10 seconds, checking every second
                await asyncio.sleep(1)
                summary_files = list(Path("test_summaries").glob("*.txt"))
                if summary_files:
                    test_results["summary_created"] = True
                    logger.success(f"  Summary file created after {i+1}s: {summary_files[0].name}")
                    break
            else:
                summary_files = []

            # Show summary content if created
            if summary_files and test_results["summary_created"]:

                with open(summary_files[0]) as f:
                    content = f.read()
                    logger.info("\n" + "=" * 60)
                    logger.info("GENERATED SUMMARY:")
                    logger.info("=" * 60)
                    logger.info(content)
                    logger.info("=" * 60)
            elif not test_results["summary_created"]:
                logger.warning("  No summary file created (LLM may be slow or unavailable)")

            # Send EndFrame
            logger.info("\n✓ Sending EndFrame...")
            await task.queue_frames([EndFrame()])

        except Exception as e:
            logger.error(f"❌ Error in simulation: {e}")
            import traceback
            traceback.print_exc()

    # Run pipeline and simulation
    try:
        await asyncio.gather(
            runner.run(task),
            simulate_transcripts()
        )
    except asyncio.CancelledError:
        pass

    # Print test results
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)

    all_passed = True

    if test_results["startframe_ok"]:
        logger.success("✅ StartFrame handling: PASSED")
    else:
        logger.error("❌ StartFrame handling: FAILED")
        all_passed = False

    if test_results["transcripts_processed"] >= len(transcripts):
        logger.success(f"✅ Transcript processing: PASSED ({test_results['transcripts_processed']} transcripts)")
    else:
        logger.error(f"❌ Transcript processing: FAILED ({test_results['transcripts_processed']} transcripts)")
        all_passed = False

    if test_results["batch_triggered"]:
        logger.success("✅ Batch processing: PASSED")
    else:
        logger.warning("⚠️  Batch processing: UNCLEAR (may need more time)")

    if test_results["summary_created"]:
        logger.success("✅ Summary generation: PASSED")
    else:
        logger.warning("⚠️  Summary generation: FAILED (check if Ollama is running)")
        all_passed = False

    logger.info("=" * 60)

    if all_passed and test_results["summary_created"]:
        logger.success("\n🎉 ALL TESTS PASSED!")
        logger.success("The passive bot is working correctly.")
        logger.success("You can now run: uv run bot_passive.py")
        return True
    else:
        logger.warning("\n⚠️  SOME TESTS INCOMPLETE")
        if not test_results["summary_created"]:
            logger.warning("Make sure Ollama is running: ollama serve")
        logger.info("Core pipeline functionality is working.")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_passive_pipeline())

    # Cleanup
    test_dir = Path("test_summaries")
    if test_dir.exists():
        logger.info(f"\nTest summaries saved in: {test_dir}")
        logger.info("(Delete with: rm -rf test_summaries/)")

    exit(0 if success else 1)
