#!/usr/bin/env python3
"""
Test connection pool by making multiple downloads in the SAME process.
This simulates production behavior where PM2 keeps the process alive.
"""

import asyncio
import time
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "miner"))

from loguru import logger
from miner.utils.video_downloader import download_video_partial


async def test_connection_reuse():
    """Test connection pool by downloading the same video 3 times."""
    
    url = "https://scoredata.me/chunks/e10ab6a14f43457995316c7c0358d9.mp4"
    
    logger.info("="*80)
    logger.info("CONNECTION POOL REUSE TEST")
    logger.info("="*80)
    logger.info(f"URL: {url}")
    logger.info("Making 3 downloads in the same process to test connection reuse\n")
    
    timings = []
    
    for i in range(3):
        logger.info(f"{'='*80}")
        logger.info(f"Download {i+1}/3")
        logger.info(f"{'-'*80}")
        
        start = time.time()
        try:
            video_path = await download_video_partial(url, max_bytes=4*1024*1024)
            elapsed = time.time() - start
            timings.append(elapsed)
            
            size_mb = video_path.stat().st_size / 1024 / 1024
            logger.info(f"✓ Download {i+1}: {elapsed:.3f}s ({size_mb:.1f} MB @ {size_mb/elapsed:.2f} MB/s)")
            
            # Cleanup
            video_path.unlink()
            
        except Exception as e:
            logger.error(f"✗ Download {i+1} failed: {e}")
            timings.append(None)
        
        logger.info("")
        
        # Small delay between requests
        if i < 2:
            await asyncio.sleep(0.5)
    
    # Analysis
    logger.info("="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    
    for i, t in enumerate(timings, 1):
        if t:
            if i == 1:
                logger.info(f"Download {i}: {t:.3f}s (first - establishes connection)")
            else:
                improvement = ((timings[0] - t) / timings[0] * 100) if timings[0] else 0
                logger.info(f"Download {i}: {t:.3f}s ({improvement:+.1f}% vs first)")
    
    logger.info("")
    
    if all(timings):
        avg_after_first = sum(timings[1:]) / len(timings[1:]) if len(timings) > 1 else 0
        speedup = timings[0] / avg_after_first if avg_after_first > 0 else 1
        
        logger.info("="*80)
        logger.info("ANALYSIS")
        logger.info("="*80)
        logger.info(f"First download:     {timings[0]:.3f}s (cold start)")
        logger.info(f"Subsequent average: {avg_after_first:.3f}s (warm pool)")
        logger.info(f"Speedup:            {speedup:.2f}x faster")
        logger.info("")
        
        if speedup > 2.0:
            logger.info("✅ CONNECTION POOL IS WORKING! (>2x speedup)")
        elif speedup > 1.3:
            logger.info("⚠️  Pool working but not optimal (1.3-2x speedup)")
        else:
            logger.info("❌ Pool may not be reusing connections (<1.3x speedup)")
        
        logger.info("")
        logger.info("Expected in production (PM2):")
        logger.info(f"  First challenge:  ~{timings[0]:.1f}s")
        logger.info(f"  Later challenges: ~{avg_after_first:.1f}s")
        logger.info("="*80)


async def main():
    await test_connection_reuse()


if __name__ == "__main__":
    asyncio.run(main())

