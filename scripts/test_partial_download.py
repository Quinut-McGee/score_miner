#!/usr/bin/env python3
"""
Quick test script to demonstrate the partial download speed trick.
Compares full download vs partial download times.
"""

import asyncio
import time
import sys
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "miner"))

from miner.utils.video_downloader import download_video, download_video_partial
from loguru import logger


async def test_download_speeds(url: str):
    """Test and compare download speeds."""
    
    logger.info("="*80)
    logger.info("PARTIAL DOWNLOAD SPEED TEST")
    logger.info("="*80)
    logger.info(f"Test URL: {url}\n")
    
    # Test 1: Full download
    logger.info("Test 1: FULL DOWNLOAD (current method)")
    logger.info("-"*80)
    start = time.time()
    try:
        full_path = await download_video(url)
        full_time = time.time() - start
        full_size = full_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Downloaded {full_size:.1f} MB in {full_time:.2f}s ({full_size/full_time:.2f} MB/s)")
        full_path.unlink()  # Cleanup
    except Exception as e:
        logger.error(f"✗ Full download failed: {e}")
        full_time = None
        full_size = None
    
    logger.info("")
    
    # Test 2: Partial download (2 MB)
    logger.info("Test 2: PARTIAL DOWNLOAD - 2 MB")
    logger.info("-"*80)
    start = time.time()
    try:
        partial_path = await download_video_partial(url, max_bytes=2*1024*1024)
        partial_time_2mb = time.time() - start
        partial_size_2mb = partial_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Downloaded {partial_size_2mb:.1f} MB in {partial_time_2mb:.2f}s ({partial_size_2mb/partial_time_2mb:.2f} MB/s)")
        partial_path.unlink()
    except Exception as e:
        logger.error(f"✗ Partial download (2MB) failed: {e}")
        partial_time_2mb = None
    
    logger.info("")
    
    # Test 3: Partial download (4 MB)
    logger.info("Test 3: PARTIAL DOWNLOAD - 4 MB (recommended)")
    logger.info("-"*80)
    start = time.time()
    try:
        partial_path = await download_video_partial(url, max_bytes=4*1024*1024)
        partial_time_4mb = time.time() - start
        partial_size_4mb = partial_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Downloaded {partial_size_4mb:.1f} MB in {partial_time_4mb:.2f}s ({partial_size_4mb/partial_time_4mb:.2f} MB/s)")
        partial_path.unlink()
    except Exception as e:
        logger.error(f"✗ Partial download (4MB) failed: {e}")
        partial_time_4mb = None
    
    logger.info("")
    
    # Test 4: Partial download (6 MB)
    logger.info("Test 4: PARTIAL DOWNLOAD - 6 MB")
    logger.info("-"*80)
    start = time.time()
    try:
        partial_path = await download_video_partial(url, max_bytes=6*1024*1024)
        partial_time_6mb = time.time() - start
        partial_size_6mb = partial_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Downloaded {partial_size_6mb:.1f} MB in {partial_time_6mb:.2f}s ({partial_size_6mb/partial_time_6mb:.2f} MB/s)")
        partial_path.unlink()
    except Exception as e:
        logger.error(f"✗ Partial download (6MB) failed: {e}")
        partial_time_6mb = None
    
    # Summary
    logger.info("")
    logger.info("="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    if full_time:
        logger.info(f"Full download:        {full_time:.2f}s ({full_size:.1f} MB)")
    if partial_time_2mb:
        speedup = full_time / partial_time_2mb if full_time else 0
        logger.info(f"Partial (2 MB):       {partial_time_2mb:.2f}s → {speedup:.1f}x faster")
    if partial_time_4mb:
        speedup = full_time / partial_time_4mb if full_time else 0
        logger.info(f"Partial (4 MB):       {partial_time_4mb:.2f}s → {speedup:.1f}x faster ⭐ RECOMMENDED")
    if partial_time_6mb:
        speedup = full_time / partial_time_6mb if full_time else 0
        logger.info(f"Partial (6 MB):       {partial_time_6mb:.2f}s → {speedup:.1f}x faster")
    
    logger.info("")
    logger.info("💡 TIP: For FRAME_STRIDE=5 and TIME_BUDGET_S=2.0, use PARTIAL_DOWNLOAD_MB=4")
    logger.info("="*80)


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test partial download speeds")
    parser.add_argument(
        "--url",
        type=str,
        default="https://scoredata.me/chunks/e10ab6a14f43457995316c7c0358d9.mp4",
        help="Video URL to test"
    )
    args = parser.parse_args()
    
    await test_download_speeds(args.url)


if __name__ == "__main__":
    asyncio.run(main())

