#!/usr/bin/env python3
"""
Test different streaming methods to find the fastest approach.
Compare: Full download, Direct URL streaming, Chunked streaming
"""
import asyncio
import time
import sys
import os
import subprocess
import tempfile
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "miner"))

from loguru import logger
from miner.utils.video_downloader import download_video, download_video_streaming, download_video_chunked_streaming
import cv2


TEST_URL = "https://scoredata.me/chunks/e10ab6a14f43457995316c7c0358d9.mp4"


async def test_full_download():
    """Test 1: Full download then process"""
    logger.info("=" * 80)
    logger.info("TEST 1: Full Download (Current Method)")
    logger.info("=" * 80)
    
    start = time.time()
    video_path = await download_video(TEST_URL)
    download_time = time.time() - start
    logger.info(f"✓ Download complete: {download_time:.2f}s")
    
    # Try to open and get first frame
    start_decode = time.time()
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error("Failed to open video")
        return None
    
    ret, frame = cap.read()
    first_frame_time = time.time() - start_decode
    cap.release()
    
    if ret:
        logger.info(f"✓ First frame decoded: {first_frame_time:.2f}s after open")
    else:
        logger.error("Failed to decode first frame")
        return None
    
    total_time = time.time() - start
    
    # Cleanup
    try:
        Path(video_path).unlink()
    except:
        pass
    
    return {
        "method": "Full Download",
        "download_time": download_time,
        "first_frame_time": first_frame_time,
        "total_time": total_time,
        "ttff": total_time  # Time to first frame
    }


async def test_direct_url_nvdec():
    """Test 2: Direct URL streaming with NVDEC"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Direct URL Streaming (NVDEC)")
    logger.info("=" * 80)
    
    # Check if ffmpeg available
    if not subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0:
        logger.error("ffmpeg not found - skipping NVDEC test")
        return None
    
    start = time.time()
    
    # Use ffmpeg to decode directly from URL with NVDEC
    # Match your implementation's flags
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-rw_timeout", "2000000",
        "-reconnect", "1",
        "-reconnect_streamed", "1", 
        "-reconnect_on_network_error", "1",
        "-hwaccel", "cuda",
        "-c:v", "h264_cuvid",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-avioflags", "direct",
        "-analyzeduration", "100000",
        "-probesize", "32768",
        "-i", TEST_URL,
        "-vf", "scale=416:416,select='not(mod(n\\,5))'",
        "-vframes", "1",  # Just get first frame
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-"
    ]
    
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Read first frame (416*416*3 bytes)
        frame_size = 416 * 416 * 3
        frame_data = proc.stdout.read(frame_size)
        
        ttff = time.time() - start
        
        proc.terminate()
        proc.wait(timeout=1)
        
        if len(frame_data) == frame_size:
            logger.info(f"✓ First frame received: {ttff:.2f}s (NVDEC GPU decode)")
            return {
                "method": "Direct URL + NVDEC",
                "download_time": 0,  # Overlapped
                "first_frame_time": ttff,
                "total_time": ttff,
                "ttff": ttff
            }
        else:
            logger.error(f"Incomplete frame data: {len(frame_data)} bytes")
            return None
            
    except Exception as e:
        logger.error(f"NVDEC streaming failed: {e}")
        return None


async def test_direct_url_opencv():
    """Test 3: Direct URL with OpenCV (CPU decode)"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Direct URL with OpenCV (CPU decode)")
    logger.info("=" * 80)
    
    start = time.time()
    
    # OpenCV can open HTTP URLs directly
    cap = cv2.VideoCapture(TEST_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error("Failed to open URL with OpenCV")
        return None
    
    ret, frame = cap.read()
    ttff = time.time() - start
    cap.release()
    
    if ret:
        logger.info(f"✓ First frame decoded: {ttff:.2f}s (OpenCV + FFmpeg CPU)")
        return {
            "method": "Direct URL + OpenCV",
            "download_time": 0,
            "first_frame_time": ttff,
            "total_time": ttff,
            "ttff": ttff
        }
    else:
        logger.error("Failed to decode first frame")
        return None


async def test_chunked_streaming():
    """Test 4: Chunked streaming (write while download)"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Chunked Streaming")
    logger.info("=" * 80)
    
    start = time.time()
    video_path = await download_video_chunked_streaming(TEST_URL)
    download_start_time = time.time() - start
    logger.info(f"✓ Chunked download started: {download_start_time:.2f}s")
    
    # Try to open immediately
    start_decode = time.time()
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error("Failed to open video")
        return None
    
    ret, frame = cap.read()
    first_frame_time = time.time() - start_decode
    cap.release()
    
    if ret:
        logger.info(f"✓ First frame decoded: {first_frame_time:.2f}s after open")
    else:
        logger.error("Failed to decode first frame")
        return None
    
    total_time = time.time() - start
    
    # Cleanup
    try:
        Path(video_path).unlink()
    except:
        pass
    
    return {
        "method": "Chunked Streaming",
        "download_time": download_start_time,
        "first_frame_time": first_frame_time,
        "total_time": total_time,
        "ttff": total_time
    }


async def main():
    logger.info("🎥 STREAMING METHODS COMPARISON TEST")
    logger.info(f"Test URL: {TEST_URL}\n")
    
    results = []
    
    # Test all methods
    result = await test_full_download()
    if result:
        results.append(result)
    
    await asyncio.sleep(1)  # Brief pause
    
    result = await test_direct_url_nvdec()
    if result:
        results.append(result)
    
    await asyncio.sleep(1)
    
    result = await test_direct_url_opencv()
    if result:
        results.append(result)
    
    await asyncio.sleep(1)
    
    result = await test_chunked_streaming()
    if result:
        results.append(result)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    
    if not results:
        logger.error("No successful tests!")
        return
    
    # Sort by TTFF (Time To First Frame)
    results.sort(key=lambda x: x["ttff"])
    
    logger.info(f"\n{'Method':<25} {'TTFF':<12} {'Download':<12} {'Decode':<12}")
    logger.info("-" * 65)
    
    for r in results:
        logger.info(
            f"{r['method']:<25} "
            f"{r['ttff']:.2f}s{' '*6} "
            f"{r['download_time']:.2f}s{' '*6} "
            f"{r['first_frame_time']:.2f}s"
        )
    
    # Winner
    winner = results[0]
    logger.info("\n" + "=" * 80)
    logger.info(f"🏆 FASTEST METHOD: {winner['method']}")
    logger.info(f"   Time to First Frame: {winner['ttff']:.2f}s")
    
    if winner['ttff'] < 1.0:
        logger.info("\n✅ SUB-1S TTFF ACHIEVED! This is the competitive method.")
    elif winner['ttff'] < 2.0:
        logger.info("\n⚠️  Close to competitive, but needs optimization")
    else:
        logger.info("\n❌ Still too slow - need sub-1s TTFF to be competitive")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

