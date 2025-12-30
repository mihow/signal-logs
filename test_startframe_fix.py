#!/usr/bin/env python3
"""Unit test to verify SummaryWriter handles StartFrame correctly."""

import asyncio
from pathlib import Path

from loguru import logger
from pipecat.frames.frames import (
    EndFrame,
    StartFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from passive_processors import SummaryWriter


async def test_startframe_handling():
    """Test that SummaryWriter properly handles StartFrame without errors."""

    logger.info("Testing StartFrame handling in SummaryWriter...")

    # Create summary writer
    summary_writer = SummaryWriter(output_dir="test_summaries/")

    # Track if we got the StartFrame error
    got_error = False
    original_error_level = False

    try:
        # Process StartFrame - this should NOT raise an error
        logger.info("Sending StartFrame...")
        await summary_writer.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
        logger.success("✅ StartFrame processed without error!")

        # Send a test text frame
        logger.info("Sending test TextFrame...")
        await summary_writer.process_frame(
            TextFrame(text="Test message"),
            FrameDirection.DOWNSTREAM
        )
        logger.success("✅ TextFrame processed without error!")

        # Send EndFrame
        logger.info("Sending EndFrame...")
        await summary_writer.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
        logger.success("✅ EndFrame processed without error!")

        logger.success("\n🎉 All frames processed successfully!")
        logger.success("The StartFrame fix is working correctly.")

    except Exception as e:
        logger.error(f"❌ Error processing frames: {e}")
        got_error = True
        import traceback
        traceback.print_exc()

    # Cleanup
    test_dir = Path("test_summaries")
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
        logger.info("Cleaned up test directory")

    if got_error:
        logger.error("\n❌ TEST FAILED - StartFrame handling is broken")
        return False
    else:
        logger.success("\n✅ TEST PASSED - StartFrame handling is correct")
        return True


if __name__ == "__main__":
    success = asyncio.run(test_startframe_handling())
    exit(0 if success else 1)
