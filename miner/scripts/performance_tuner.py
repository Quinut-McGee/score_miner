#!/usr/bin/env python3
"""
Performance tuning script for Score Vision Miner.

This script helps optimize the miner for maximum performance on RTX 5070 Ti.
Run this script to find optimal batch sizes and configurations.
"""

import os
import time
import torch
import numpy as np
from pathlib import Path
import argparse
from loguru import logger

# Add miner to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from miner.utils.device import get_optimal_device
from miner.utils.model_manager import ModelManager
from miner.core.models.inference_config import InferenceConfig


def benchmark_batch_sizes(device: str = "cuda", max_batch_size: int = 128):
    """Benchmark different batch sizes to find optimal performance."""
    logger.info(f"Benchmarking batch sizes on {device} device...")
    
    # FIXED: Test smaller batch sizes to avoid OOM
    batch_sizes = [1, 2, 4, 8, 12, 16, 20, 24, 32]
    batch_sizes = [bs for bs in batch_sizes if bs <= max_batch_size]
    
    results = {}
    
    for batch_size in batch_sizes:
        try:
            logger.info(f"Testing batch size: {batch_size}")
            
            # Create inference config
            config = InferenceConfig.for_device(device, batch_size_override=batch_size)
            
            # Load model manager
            model_manager = ModelManager(device=device, batch_size=batch_size)
            model_manager.load_all_models()
            
            # Get models
            player_model = model_manager.get_model("player")
            pitch_model = model_manager.get_model("pitch")
            
            # Create dummy frames with proper size
            dummy_frames = []
            for _ in range(batch_size):
                frame = np.random.randint(0, 255, (config.player_imgsz, config.player_imgsz, 3), dtype=np.uint8)
                dummy_frames.append(frame)
            
            # Warmup
            for _ in range(3):
                try:
                    player_model(dummy_frames, **config.get_player_inference_kwargs())
                    pitch_model(dummy_frames, **config.get_pitch_inference_kwargs())
                except Exception as e:
                    logger.warning(f"Warmup failed for batch_size {batch_size}: {e}")
                    break
            
            # Benchmark
            num_iterations = 10
            times = []
            
            for i in range(num_iterations):
                start_time = time.time()
                
                try:
                    # Run inference
                    player_results = player_model(dummy_frames, **config.get_player_inference_kwargs())
                    pitch_results = pitch_model(dummy_frames, **config.get_pitch_inference_kwargs())
                    
                    # Synchronize GPU
                    if device == "cuda":
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    times.append(end_time - start_time)
                    
                except Exception as e:
                    logger.warning(f"Inference failed for batch_size {batch_size}: {e}")
                    break
            
            if times:
                avg_time = np.mean(times)
                fps = batch_size / avg_time
                results[batch_size] = {
                    'avg_time': avg_time,
                    'fps': fps,
                    'throughput': batch_size * len(times) / sum(times)
                }
                logger.info(f"Batch size {batch_size}: {avg_time:.3f}s, {fps:.1f} fps")
            else:
                logger.warning(f"No successful runs for batch_size {batch_size}")
                
        except Exception as e:
            logger.error(f"Failed to test batch_size {batch_size}: {e}")
            # Clear GPU memory after failure
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    return results


def find_optimal_configuration(device: str = "cuda"):
    """Find optimal configuration for the current hardware."""
    logger.info("Finding optimal configuration...")
    
    # Get GPU info
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory:.1f} GB")
        
        # Estimate max batch size based on memory
        if gpu_memory >= 16:
            max_batch_size = 128
        elif gpu_memory >= 12:
            max_batch_size = 96
        elif gpu_memory >= 8:
            max_batch_size = 64
        else:
            max_batch_size = 32
    else:
        max_batch_size = 32
    
    # Benchmark batch sizes
    results = benchmark_batch_sizes(device, max_batch_size)
    
    if not results:
        logger.error("No successful benchmark results")
        return None
    
    # Find optimal batch size
    best_batch_size = max(results.keys(), key=lambda k: results[k]['fps'])
    best_result = results[best_batch_size]
    
    logger.info(f"Optimal batch size: {best_batch_size}")
    logger.info(f"Best performance: {best_result['fps']:.1f} fps")
    
    # Generate configuration
    config = InferenceConfig.for_device(device, batch_size_override=best_batch_size)
    
    recommendations = {
        'batch_size': best_batch_size,
        'player_imgsz': config.player_imgsz,
        'pitch_imgsz': config.pitch_imgsz,
        'ball_imgsz': config.ball_imgsz,
        'conf_threshold': config.conf_threshold,
        'iou_threshold': config.iou_threshold,
        'max_detections': config.max_detections,
        'half_precision': config.half_precision,
        'prefetch_frames': 128,
        'expected_fps': best_result['fps']
    }
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(description="Performance tuner for Score Vision Miner")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda, mps, cpu)")
    parser.add_argument("--max-batch-size", type=int, default=128, help="Maximum batch size to test")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    # Set device
    device = get_optimal_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Find optimal configuration
    recommendations = find_optimal_configuration(device)
    
    if recommendations:
        logger.info("=== OPTIMAL CONFIGURATION ===")
        for key, value in recommendations.items():
            logger.info(f"{key}: {value}")
        
        # Save to file if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(recommendations, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        
        # Generate environment variables
        logger.info("\n=== ENVIRONMENT VARIABLES ===")
        logger.info(f"export DEVICE={device}")
        logger.info(f"export BATCH_SIZE={recommendations['batch_size']}")
        logger.info(f"export PREFETCH_FRAMES={recommendations['prefetch_frames']}")
        
    else:
        logger.error("Failed to find optimal configuration")


if __name__ == "__main__":
    main()
