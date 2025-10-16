#!/usr/bin/env python3
"""
Benchmark script to measure inference performance improvements.

This script tests the miner's inference speed with different configurations
to help optimize performance for your specific hardware.

Usage:
    python scripts/benchmark_inference.py [video_path]
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import Dict, Any
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from miner.utils.model_manager import ModelManager
from miner.utils.video_processor import VideoProcessor
from miner.utils.batch_processor import BatchFrameProcessor, NoBatchProcessor
from miner.core.models.inference_config import InferenceConfig
import supervision as sv
from loguru import logger


async def benchmark_configuration(
    video_path: str,
    device: str,
    config: InferenceConfig,
    description: str,
) -> Dict[str, Any]:
    """
    Benchmark a specific configuration.

    Args:
        video_path: Path to test video
        device: Device to use ('cuda', 'mps', 'cpu')
        config: Inference configuration to test
        description: Description of this test

    Returns:
        Dict with benchmark results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {description}")
    logger.info(f"Device: {device}")
    logger.info(f"Config: player_imgsz={config.player_imgsz}, "
               f"half={config.half_precision}, batch_size={config.batch_size}")
    logger.info(f"{'='*60}\n")

    # Initialize model manager with this device
    model_manager = ModelManager(device=device)
    model_manager.inference_config = config  # Override with test config
    model_manager.load_all_models()

    # Create video processor
    video_processor = VideoProcessor(device=device)

    # Get models
    player_model = model_manager.get_model("player")
    pitch_model = model_manager.get_model("pitch")
    tracker = sv.ByteTrack()

    # Get inference kwargs
    player_kwargs = config.get_player_inference_kwargs()
    pitch_kwargs = config.get_pitch_inference_kwargs()

    # Choose processor
    if config.batch_size > 1:
        processor = BatchFrameProcessor(batch_size=config.batch_size)
    else:
        processor = NoBatchProcessor()

    # Run benchmark
    start_time = time.time()
    frame_count = 0

    try:
        frame_generator = video_processor.stream_frames(video_path)
        async for frame_data in processor.process_batched_frames(
            frame_generator,
            player_model,
            pitch_model,
            tracker,
            player_kwargs,
            pitch_kwargs,
        ):
            frame_count += 1

            # Log progress every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {frame_count} frames, {elapsed:.1f}s, {fps:.2f} fps")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return {
            "description": description,
            "device": device,
            "error": str(e),
            "frames_processed": frame_count,
        }

    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    results = {
        "description": description,
        "device": device,
        "player_imgsz": config.player_imgsz,
        "pitch_imgsz": config.pitch_imgsz,
        "half_precision": config.half_precision,
        "batch_size": config.batch_size,
        "frames_processed": frame_count,
        "total_time_seconds": round(elapsed_time, 2),
        "fps": round(fps, 2),
        "ms_per_frame": round((elapsed_time / frame_count * 1000), 2) if frame_count > 0 else 0,
    }

    logger.info(f"\n{'-'*60}")
    logger.info(f"Results for: {description}")
    logger.info(f"  Frames: {frame_count}")
    logger.info(f"  Time: {elapsed_time:.2f}s")
    logger.info(f"  FPS: {fps:.2f}")
    logger.info(f"  ms/frame: {results['ms_per_frame']:.2f}")
    logger.info(f"{'-'*60}\n")

    return results


async def run_benchmarks(video_path: str, device: str):
    """
    Run a suite of benchmarks comparing different configurations.

    Args:
        video_path: Path to test video
        device: Device to use for testing
    """
    logger.info(f"Starting benchmark suite on device: {device}")
    logger.info(f"Test video: {video_path}")

    all_results = []

    # Baseline: Original configuration (no optimizations)
    baseline_config = InferenceConfig(
        player_imgsz=1280,
        pitch_imgsz=1280,
        ball_imgsz=1280,
        conf_threshold=0.25,
        iou_threshold=0.45,
        max_detections=300,
        half_precision=False,  # Original didn't use FP16
        agnostic_nms=False,
        batch_size=1,  # Original didn't use batching
    )
    baseline_result = await benchmark_configuration(
        video_path,
        device,
        baseline_config,
        "Baseline (Original Settings)",
    )
    all_results.append(baseline_result)

    # Test 1: FP16 only
    if device == "cuda":
        fp16_config = InferenceConfig(
            player_imgsz=1280,
            pitch_imgsz=1280,
            ball_imgsz=1280,
            conf_threshold=0.25,
            iou_threshold=0.45,
            max_detections=300,
            half_precision=True,
            agnostic_nms=False,
            batch_size=1,
        )
        fp16_result = await benchmark_configuration(
            video_path,
            device,
            fp16_config,
            "FP16 Enabled (Tensor Cores)",
        )
        all_results.append(fp16_result)

        # Test 2: FP16 + Higher Resolution
        fp16_highres_config = InferenceConfig(
            player_imgsz=1536,
            pitch_imgsz=1280,
            ball_imgsz=1280,
            conf_threshold=0.25,
            iou_threshold=0.45,
            max_detections=300,
            half_precision=True,
            agnostic_nms=False,
            batch_size=1,
        )
        fp16_highres_result = await benchmark_configuration(
            video_path,
            device,
            fp16_highres_config,
            "FP16 + Higher Res (1536px)",
        )
        all_results.append(fp16_highres_result)

        # Test 3: Full Optimization (FP16 + High Res + Batching)
        optimized_config = InferenceConfig.cuda_config()
        optimized_result = await benchmark_configuration(
            video_path,
            device,
            optimized_config,
            "Full Optimization (FP16 + 1536px + Batch=4)",
        )
        all_results.append(optimized_result)

        # Test 4: Aggressive (Higher res + larger batch)
        aggressive_config = InferenceConfig(
            player_imgsz=1920,
            pitch_imgsz=1536,
            ball_imgsz=1280,
            conf_threshold=0.25,
            iou_threshold=0.45,
            max_detections=300,
            half_precision=True,
            agnostic_nms=False,
            batch_size=8,
        )
        aggressive_result = await benchmark_configuration(
            video_path,
            device,
            aggressive_config,
            "Aggressive (1920px + Batch=8)",
        )
        all_results.append(aggressive_result)

    elif device == "mps":
        # MPS optimized configuration
        mps_config = InferenceConfig.mps_config()
        mps_result = await benchmark_configuration(
            video_path,
            device,
            mps_config,
            "MPS Optimized",
        )
        all_results.append(mps_result)

    else:  # CPU
        cpu_config = InferenceConfig.cpu_config()
        cpu_result = await benchmark_configuration(
            video_path,
            device,
            cpu_config,
            "CPU Optimized",
        )
        all_results.append(cpu_result)

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Configuration':<40} {'FPS':>10} {'ms/frame':>12} {'Speedup':>10}")
    print("-"*80)

    baseline_fps = baseline_result.get("fps", 0)
    for result in all_results:
        desc = result["description"]
        fps = result.get("fps", 0)
        ms_per_frame = result.get("ms_per_frame", 0)
        speedup = fps / baseline_fps if baseline_fps > 0 else 0

        print(f"{desc:<40} {fps:>10.2f} {ms_per_frame:>12.2f} {speedup:>9.2f}x")

    print("="*80)

    # Recommendations
    print("\nRECOMMENDATIONS:")
    if device == "cuda":
        best_result = max(all_results, key=lambda x: x.get("fps", 0))
        print(f"  Best configuration: {best_result['description']}")
        print(f"  FPS: {best_result['fps']:.2f}")
        print(f"  Speedup vs baseline: {best_result['fps'] / baseline_fps:.2f}x")
        print(f"\n  For production on RTX 5070 Ti, use:")
        print(f"    - player_imgsz: {best_result['player_imgsz']}")
        print(f"    - half_precision: {best_result['half_precision']}")
        print(f"    - batch_size: {best_result['batch_size']}")
    else:
        print(f"  Use the default {device.upper()} configuration")

    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Score Vision miner inference performance"
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        help="Path to test video (optional - downloads sample if not provided)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu", "auto"],
        default="auto",
        help="Device to use for benchmarking (default: auto-detect)",
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # Handle video path
    if args.video_path:
        video_path = args.video_path
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            sys.exit(1)
    else:
        logger.info("No video path provided. Please provide a test video.")
        logger.info("Usage: python scripts/benchmark_inference.py <video_path>")
        sys.exit(1)

    # Run benchmarks
    asyncio.run(run_benchmarks(video_path, device))


if __name__ == "__main__":
    main()
