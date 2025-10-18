#!/usr/bin/env python3
"""
Diagnostic script to identify where time is being spent in the pipeline.
Breaks down: DNS, connect, download, decode, first frame, inference.
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
import cv2


async def diagnose_bottleneck(url: str):
    """Detailed timing breakdown of the entire pipeline."""
    
    logger.info("="*80)
    logger.info("BOTTLENECK DIAGNOSTIC")
    logger.info("="*80)
    logger.info(f"URL: {url}\n")
    
    timings = {}
    
    # === Stage 1: DNS + Connection ===
    logger.info("Stage 1: DNS Resolution + Connection")
    logger.info("-"*80)
    start = time.time()
    
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Just HEAD to measure connection time
            response = await client.head(url, follow_redirects=True)
            connect_time = time.time() - start
            timings['connect'] = connect_time
            logger.info(f"✓ DNS + Connect: {connect_time:.3f}s")
    except Exception as e:
        logger.error(f"✗ Connection failed: {e}")
        return
    
    logger.info("")
    
    # === Stage 2: Download (Partial) ===
    logger.info("Stage 2: Partial Download (4 MB)")
    logger.info("-"*80)
    start = time.time()
    
    from miner.utils.video_downloader import download_video_partial
    try:
        video_path = await download_video_partial(url, max_bytes=4*1024*1024)
        download_time = time.time() - start
        timings['download'] = download_time
        size_mb = video_path.stat().st_size / 1024 / 1024
        logger.info(f"✓ Download: {download_time:.3f}s ({size_mb:.1f} MB @ {size_mb/download_time:.2f} MB/s)")
    except Exception as e:
        logger.error(f"✗ Download failed: {e}")
        return
    
    logger.info("")
    
    # === Stage 3: Video Open (OpenCV Init) ===
    logger.info("Stage 3: OpenCV Video Open")
    logger.info("-"*80)
    start = time.time()
    
    try:
        cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error("✗ Failed to open video")
            return
        open_time = time.time() - start
        timings['open'] = open_time
        logger.info(f"✓ Video open: {open_time:.3f}s")
    except Exception as e:
        logger.error(f"✗ Video open failed: {e}")
        return
    
    logger.info("")
    
    # === Stage 4: First Frame Decode ===
    logger.info("Stage 4: First Frame Decode")
    logger.info("-"*80)
    start = time.time()
    
    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.error("✗ Failed to read first frame")
            cap.release()
            return
        decode_time = time.time() - start
        timings['first_frame'] = decode_time
        logger.info(f"✓ First frame decoded: {decode_time:.3f}s")
        logger.info(f"  Frame shape: {frame.shape}")
    except Exception as e:
        logger.error(f"✗ First frame decode failed: {e}")
        cap.release()
        return
    
    logger.info("")
    
    # === Stage 5: Additional Frames (10 frames) ===
    logger.info("Stage 5: Decode 10 More Frames")
    logger.info("-"*80)
    start = time.time()
    
    try:
        frames_decoded = 0
        for i in range(10):
            ret, _ = cap.read()
            if ret:
                frames_decoded += 1
        
        multi_decode_time = time.time() - start
        timings['10_frames'] = multi_decode_time
        logger.info(f"✓ 10 frames decoded: {multi_decode_time:.3f}s ({10/multi_decode_time:.1f} fps)")
    except Exception as e:
        logger.error(f"✗ Multi-frame decode failed: {e}")
    finally:
        cap.release()
    
    logger.info("")
    
    # === Stage 6: Model Loading (simulated) ===
    logger.info("Stage 6: Model Loading + Warmup")
    logger.info("-"*80)
    start = time.time()
    
    try:
        from miner.utils.model_manager import ModelManager
        model_manager = ModelManager(device="cuda", batch_size=32)
        model_manager.load_all_models()
        
        load_time = time.time() - start
        timings['model_load'] = load_time
        logger.info(f"✓ Models loaded: {load_time:.3f}s")
        
        # Warmup
        start_warmup = time.time()
        await model_manager.warmup_models()
        warmup_time = time.time() - start_warmup
        timings['warmup'] = warmup_time
        logger.info(f"✓ Models warmed up: {warmup_time:.3f}s")
    except Exception as e:
        logger.error(f"✗ Model operations failed: {e}")
    
    # === Summary ===
    logger.info("")
    logger.info("="*80)
    logger.info("TIMING BREAKDOWN")
    logger.info("="*80)
    
    total = sum(timings.values())
    
    for stage, duration in timings.items():
        pct = (duration / total * 100) if total > 0 else 0
        logger.info(f"{stage:20s}: {duration:6.3f}s ({pct:5.1f}%)")
    
    logger.info(f"{'='*20}")
    logger.info(f"{'TOTAL':20s}: {total:6.3f}s (100.0%)")
    
    logger.info("")
    logger.info("="*80)
    logger.info("ANALYSIS & RECOMMENDATIONS")
    logger.info("="*80)
    
    # Identify bottlenecks
    bottlenecks = []
    
    if timings.get('connect', 0) > 1.0:
        bottlenecks.append(("DNS/Connection", timings['connect'], "Use persistent connections, check DNS"))
    
    if timings.get('download', 0) > 1.0:
        bottlenecks.append(("Download", timings['download'], "Reduce PARTIAL_DOWNLOAD_MB to 2 MB"))
    
    if timings.get('open', 0) > 0.5:
        bottlenecks.append(("Video Open", timings['open'], "Use NVDEC or faster codec"))
    
    if timings.get('first_frame', 0) > 0.5:
        bottlenecks.append(("First Frame", timings['first_frame'], "Reduce STREAM_MIN_START_BYTES, use NVDEC"))
    
    if timings.get('model_load', 0) > 1.0:
        bottlenecks.append(("Model Load", timings['model_load'], "Keep models in memory, use model server"))
    
    if bottlenecks:
        logger.info("🔴 BOTTLENECKS DETECTED:")
        for name, duration, fix in bottlenecks:
            logger.info(f"  • {name}: {duration:.3f}s → {fix}")
    else:
        logger.info("✅ No major bottlenecks detected!")
    
    logger.info("")
    logger.info("Estimated Total Pipeline Time (Challenge):")
    # Exclude model load (cached) and warmup (async)
    pipeline_time = timings.get('connect', 0) + timings.get('download', 0) + \
                   timings.get('open', 0) + timings.get('first_frame', 0) + 0.5  # +0.5s inference
    logger.info(f"  {pipeline_time:.3f}s (excluding model load)")
    
    if pipeline_time < 2.0:
        logger.info("  ✅ COMPETITIVE! (<2s)")
    elif pipeline_time < 3.0:
        logger.info("  ⚠️  MARGINAL (2-3s)")
    else:
        logger.info("  ❌ TOO SLOW (>3s)")
    
    logger.info("="*80)
    
    # Cleanup
    try:
        video_path.unlink()
    except:
        pass


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose pipeline bottlenecks")
    parser.add_argument(
        "--url",
        type=str,
        default="https://scoredata.me/chunks/e10ab6a14f43457995316c7c0358d9.mp4",
        help="Video URL to test"
    )
    args = parser.parse_args()
    
    await diagnose_bottleneck(args.url)


if __name__ == "__main__":
    asyncio.run(main())

