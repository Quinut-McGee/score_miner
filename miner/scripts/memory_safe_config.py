#!/usr/bin/env python3
"""
Memory-safe configuration for Score Vision Miner.

This script provides a conservative configuration that should work
on RTX 5070 Ti without memory issues.
"""

import os
import torch
from loguru import logger

def get_memory_safe_config():
    """Get memory-safe configuration for RTX 5070 Ti."""
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory >= 15:  # RTX 5070 Ti
            return {
                'batch_size': 8,      # Conservative batch size
                'player_imgsz': 896,  # Reduced resolution
                'pitch_imgsz': 768,   # Reduced resolution
                'ball_imgsz': 768,   # Reduced resolution
                'prefetch_frames': 32,  # Reduced prefetch
                'memory_fraction': 0.6,  # Use only 60% of GPU memory
            }
        elif gpu_memory >= 12:
            return {
                'batch_size': 4,
                'player_imgsz': 768,
                'pitch_imgsz': 640,
                'ball_imgsz': 640,
                'prefetch_frames': 16,
                'memory_fraction': 0.5,
            }
        else:
            return {
                'batch_size': 2,
                'player_imgsz': 640,
                'pitch_imgsz': 512,
                'ball_imgsz': 512,
                'prefetch_frames': 8,
                'memory_fraction': 0.4,
            }
    else:
        return {
            'batch_size': 1,
            'player_imgsz': 512,
            'pitch_imgsz': 512,
            'ball_imgsz': 512,
            'prefetch_frames': 4,
            'memory_fraction': 0.0,
        }

def apply_memory_safe_settings():
    """Apply memory-safe settings to environment."""
    config = get_memory_safe_config()
    
    # Set environment variables
    os.environ['BATCH_SIZE'] = str(config['batch_size'])
    os.environ['PREFETCH_FRAMES'] = str(config['prefetch_frames'])
    
    # Set PyTorch memory settings
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(config['memory_fraction'])
        torch.cuda.empty_cache()
    
    logger.info("Memory-safe settings applied:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    return config

if __name__ == "__main__":
    config = apply_memory_safe_settings()
    print("\n=== MEMORY-SAFE CONFIGURATION ===")
    for key, value in config.items():
        print(f"{key}: {value}")
    
    print("\n=== ENVIRONMENT VARIABLES ===")
    print(f"export BATCH_SIZE={config['batch_size']}")
    print(f"export PREFETCH_FRAMES={config['prefetch_frames']}")
