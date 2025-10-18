#!/usr/bin/env python3
"""
Performance benchmark script for the Score Vision miner.

Tests different configurations to find optimal settings for your hardware.
Measures FPS, GPU utilization, and total processing time.

Usage:
    python scripts/benchmark_miner.py --video-path /path/to/video.mp4
    python scripts/benchmark_miner.py --video-url https://example.com/video.mp4
"""

import argparse
import asyncio
import time
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Add both to path to ensure imports work
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "miner") not in sys.path:
    sys.path.insert(0, str(project_root / "miner"))

from miner.utils.model_manager import ModelManager
from miner.endpoints.soccer import process_soccer_video
from miner.utils.video_downloader import download_video, download_video_streaming, download_video_partial
from loguru import logger
import torch


async def benchmark_configuration(
    video_path: str,
    batch_size: int,
    frame_stride: int,
    img_size: int,
    prefetch_frames: int,
    device: str = "cuda",
    is_url: bool = False,
):
    """
    Benchmark a specific configuration.
    
    Args:
        video_path: Path to video file
        batch_size: Batch size for inference
        frame_stride: Frame sampling stride (1=all frames, 2=every 2nd frame, etc)
        img_size: Image resolution for inference
        prefetch_frames: Number of frames to prefetch
        device: Device to use (cuda/mps/cpu)
    
    Returns:
        dict: Benchmark results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Benchmarking Configuration:")
    logger.info(f"  Device: {device}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Frame Stride: {frame_stride} (process every {frame_stride} frame(s))")
    logger.info(f"  Image Size: {img_size}px")
    logger.info(f"  Prefetch Frames: {prefetch_frames}")
    logger.info(f"{'='*80}\n")
    
    # Override inference config
    from miner.core.models.inference_config import InferenceConfig
    InferenceConfig.cuda_config = lambda: InferenceConfig(
        player_imgsz=img_size,
        pitch_imgsz=img_size,
        ball_imgsz=img_size,
        conf_threshold=0.35,
        iou_threshold=0.5,
        max_detections=150,
        half_precision=True,
        agnostic_nms=False,
        batch_size=batch_size,
    )
    
    # Initialize model manager
    model_manager = ModelManager(device=device, batch_size=batch_size)
    model_manager.load_all_models()
    
    # Get GPU memory before
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1024**3
    
    # Run benchmark
    start_time = time.time()
    
    # Temporarily modify VideoProcessor defaults
    from miner.utils import video_processor
    original_init = video_processor.VideoProcessor.__init__
    
    def patched_init(self, device="cpu", **kwargs):
        kwargs['prefetch_frames'] = prefetch_frames
        kwargs['frame_stride'] = frame_stride
        original_init(self, device=device, **kwargs)
    
    video_processor.VideoProcessor.__init__ = patched_init
    
    try:
        result = await process_soccer_video(video_path, model_manager, is_url=is_url)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Get GPU memory after
        if device == "cuda" and torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated() / 1024**3
            mem_peak = torch.cuda.max_memory_allocated() / 1024**3
        else:
            mem_after = 0
            mem_peak = 0
        
        num_frames = len(result['frames'])
        fps = num_frames / total_time if total_time > 0 else 0
        
        benchmark_result = {
            'success': True,
            'total_time': total_time,
            'num_frames': num_frames,
            'fps': fps,
            'batch_size': batch_size,
            'frame_stride': frame_stride,
            'img_size': img_size,
            'prefetch_frames': prefetch_frames,
            'device': device,
            'gpu_memory_used_gb': mem_after,
            'gpu_memory_peak_gb': mem_peak,
        }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Results:")
        logger.info(f"  Total Time: {total_time:.2f}s")
        logger.info(f"  Frames Processed: {num_frames}")
        logger.info(f"  FPS: {fps:.2f}")
        if device == "cuda":
            logger.info(f"  GPU Memory Peak: {mem_peak:.2f} GB")
        logger.info(f"{'='*80}\n")
        
        return benchmark_result
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'batch_size': batch_size,
            'frame_stride': frame_stride,
            'img_size': img_size,
        }
    finally:
        # Restore original init
        video_processor.VideoProcessor.__init__ = original_init
        
        # Clear GPU cache
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


async def run_benchmarks(video_path: str, device: str = "cuda", quick: bool = False, is_url: bool = False):
    """
    Run multiple benchmark configurations to find optimal settings.
    
    Args:
        video_path: Path to test video
        device: Device to use
        quick: If True, run fewer tests
    """
    logger.info(f"Starting benchmark suite for device: {device}")
    logger.info(f"Test video: {video_path}\n")
    
    if quick:
        # Quick test - just test current optimal config
        configs = [
            {'batch_size': 32, 'frame_stride': 2, 'img_size': 640, 'prefetch_frames': 64},
        ]
    else:
        # Full test suite
        configs = [
            # Current optimized config
            {'batch_size': 32, 'frame_stride': 2, 'img_size': 640, 'prefetch_frames': 64},
            
            # Test higher batch sizes
            {'batch_size': 48, 'frame_stride': 2, 'img_size': 640, 'prefetch_frames': 96},
            {'batch_size': 64, 'frame_stride': 2, 'img_size': 640, 'prefetch_frames': 128},
            
            # Test different frame strides
            {'batch_size': 32, 'frame_stride': 1, 'img_size': 640, 'prefetch_frames': 64},  # All frames
            {'batch_size': 32, 'frame_stride': 3, 'img_size': 640, 'prefetch_frames': 64},  # Every 3rd
            
            # Test different image sizes
            {'batch_size': 32, 'frame_stride': 2, 'img_size': 512, 'prefetch_frames': 64},
            {'batch_size': 32, 'frame_stride': 2, 'img_size': 800, 'prefetch_frames': 64},
            
            # Baseline (old config)
            {'batch_size': 4, 'frame_stride': 1, 'img_size': 1024, 'prefetch_frames': 32},
        ]
    
    results = []
    for i, config in enumerate(configs, 1):
        logger.info(f"\n>>> Running benchmark {i}/{len(configs)}")
        result = await benchmark_configuration(
            video_path=video_path,
            device=device,
            is_url=is_url,
            **config
        )
        results.append(result)
        
        # Wait a bit between tests
        await asyncio.sleep(2)
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info(f"BENCHMARK SUMMARY")
    logger.info(f"{'='*80}\n")
    
    # Sort by FPS (successful runs only)
    successful_results = [r for r in results if r.get('success')]
    successful_results.sort(key=lambda x: x.get('fps', 0), reverse=True)
    
    logger.info(f"{'Rank':<6} {'FPS':<8} {'Time':<8} {'Batch':<8} {'Stride':<8} {'ImgSize':<10} {'Memory':<10}")
    logger.info(f"{'-'*80}")
    
    for i, result in enumerate(successful_results, 1):
        fps = result.get('fps', 0)
        total_time = result.get('total_time', 0)
        batch_size = result.get('batch_size', 0)
        frame_stride = result.get('frame_stride', 0)
        img_size = result.get('img_size', 0)
        mem_peak = result.get('gpu_memory_peak_gb', 0)
        
        logger.info(
            f"{i:<6} {fps:>6.2f}  {total_time:>6.2f}s  {batch_size:<8} {frame_stride:<8} "
            f"{img_size:<10} {mem_peak:>6.2f} GB"
        )
    
    # Print failed runs
    failed_results = [r for r in results if not r.get('success')]
    if failed_results:
        logger.info(f"\nFailed configurations:")
        for result in failed_results:
            logger.info(f"  Batch={result['batch_size']}, Stride={result['frame_stride']}, "
                       f"ImgSize={result['img_size']}: {result.get('error', 'Unknown error')}")
    
    logger.info(f"\n{'='*80}")
    
    # Print recommendation
    if successful_results:
        best = successful_results[0]
        logger.info(f"\nRECOMMENDED CONFIGURATION:")
        logger.info(f"  Batch Size: {best['batch_size']}")
        logger.info(f"  Frame Stride: {best['frame_stride']}")
        logger.info(f"  Image Size: {best['img_size']}px")
        logger.info(f"  Prefetch Frames: {best['prefetch_frames']}")
        logger.info(f"  Expected FPS: {best['fps']:.2f}")
        if best['fps'] > 0:
            logger.info(f"  Expected Time (750 frames): {750 / best['fps']:.2f}s")
        else:
            logger.info(f"  Expected Time (750 frames): N/A (zero FPS)")
        
        # Calculate speedup vs baseline
        baseline = next((r for r in results if r.get('batch_size') == 4 and r.get('frame_stride') == 1), None)
        if baseline and baseline.get('success'):
            speedup = best['fps'] / baseline['fps']
            logger.info(f"  Speedup vs baseline: {speedup:.1f}x")
    
    logger.info(f"\n{'='*80}\n")


async def main():
    parser = argparse.ArgumentParser(description='Benchmark miner performance')
    parser.add_argument('--video-path', type=str, help='Path to local video file')
    parser.add_argument('--video-url', type=str, help='URL of video to download and test')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'],
                       help='Device to use for inference')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick test (only current config)')
    parser.add_argument('--direct-url', action='store_true',
                       help='Use URL directly for decoding (no download)')
    parser.add_argument('--streaming-download', action='store_true',
                       help='Start processing while the file is still downloading')
    parser.add_argument('--partial-download', action='store_true',
                       help='Download only first N MB (SPEED TRICK for sub-1s download)')
    parser.add_argument('--partial-mb', type=int, default=4,
                       help='How many MB to download in partial mode (default: 4)')
    
    args = parser.parse_args()
    
    # Get video path
    if args.video_path:
        video_path = args.video_path
        is_url = False
    elif args.video_url:
        if args.direct_url:
            video_path = args.video_url
            is_url = True
            logger.info("Using direct URL for decoding (no download)")
        elif args.partial_download:
            logger.info(f"Downloading first {args.partial_mb} MB (partial download SPEED TRICK)...")
            video_path = await download_video_partial(args.video_url, max_bytes=args.partial_mb * 1024 * 1024)
            is_url = False
            logger.info(f"Partial video downloaded to {video_path}")
        elif args.streaming_download:
            logger.info(f"Downloading (streaming) video from {args.video_url}...")
            video_path, bg_task = await download_video_streaming(args.video_url)
            is_url = False
            logger.info(f"Streaming to {video_path} (processing starts now; download continues in background)")
        else:
            logger.info(f"Downloading video from {args.video_url}...")
            video_path = await download_video(args.video_url)
            is_url = False
            logger.info(f"Video downloaded to {video_path}")
    else:
        logger.error("Please provide either --video-path or --video-url")
        return 1
    
    # Verify video exists
    if not (isinstance(video_path, str) and (video_path.startswith('http://') or video_path.startswith('https://'))):
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return 1
    
    # Run benchmarks
    await run_benchmarks(video_path, device=args.device, quick=args.quick, is_url=is_url)
    
    # Cleanup if downloaded
    if args.video_url and not args.direct_url:
        try:
            Path(video_path).unlink()
        except Exception:
            pass
    # Try to let the streaming download finish briefly for cleanup, if used
    if 'bg_task' in locals():
        try:
            await asyncio.wait_for(bg_task, timeout=0.1)
        except Exception:
            pass
    
    return 0


if __name__ == '__main__':
    exit(asyncio.run(main()))

