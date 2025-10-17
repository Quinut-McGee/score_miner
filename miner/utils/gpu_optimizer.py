"""
GPU optimization utilities for maximum performance on RTX 5070 Ti.

This module provides GPU-specific optimizations including memory management,
CUDA settings, and performance tuning for optimal inference speed.
"""

import torch
import os
import gc
from loguru import logger
from typing import Optional


class GPUOptimizer:
    """GPU optimization manager for RTX 5070 Ti and similar GPUs."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.initialized = False
        
    def optimize_gpu_settings(self) -> None:
        """Apply GPU-specific optimizations for maximum performance."""
        if self.device != "cuda" or not torch.cuda.is_available():
            return
            
        try:
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
            torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster matmul
            torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for faster convolutions
            
            # SPEED OPTIMIZED: Use 90% of GPU memory for maximum throughput
            torch.cuda.set_per_process_memory_fraction(0.9)  # RTX 5070 Ti has 16GB, use most of it
            
            # Enable memory pool for faster allocation
            torch.cuda.empty_cache()
            
            # SPEED OPTIMIZED: Use many CPU threads for video decoding (96 cores available!)
            # Split threads: 32 for PyTorch operations, rest available for OpenCV
            torch.set_num_threads(32)  # Use 32 threads for tensor operations
            
            # Set OpenCV thread count for video decoding parallelization
            import cv2
            cv2.setNumThreads(32)  # Use 32 threads for OpenCV operations
            
            logger.info("GPU optimizations applied successfully (GPU mem: 90%, CPU threads: 32)")
            self.initialized = True
            
        except Exception as e:
            logger.warning(f"Failed to apply GPU optimizations: {e}")
    
    def clear_gpu_cache(self) -> None:
        """Clear GPU cache to free memory."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def get_gpu_memory_info(self) -> dict:
        """Get current GPU memory usage information."""
        if self.device != "cuda" or not torch.cuda.is_available():
            return {}
            
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            return {
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2),
                "free_gb": round(total - reserved, 2),
                "utilization_percent": round((reserved / total) * 100, 1)
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return {}
    
    def optimize_model_for_inference(self, model):
        """Apply model-specific optimizations for inference and return the model (possibly compiled)."""
        if self.device != "cuda":
            return model
            
        try:
            # Enable inference mode for better performance
            model.eval()
            
            # FIXED: Don't convert to half precision here - let YOLO handle it
            # The dtype mismatch occurs when we force half precision on models
            # that expect float32 inputs
            
            # Optionally compile model for faster execution (PyTorch 2.0+)
            # Default disabled to avoid first-run compile latency; enable via ENABLE_TORCH_COMPILE=1
            enable_compile = os.environ.get("ENABLE_TORCH_COMPILE", "0") in ("1", "true", "True")
            if enable_compile and hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, mode="max-autotune")
                    logger.info("Model compiled with torch.compile (enabled via env)")
                except Exception as e:
                    logger.warning(f"Failed to compile model: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to optimize model: {e}")
        return model


# Global optimizer instance
gpu_optimizer = GPUOptimizer()


def initialize_gpu_optimizations(device: str = "cuda") -> GPUOptimizer:
    """Initialize GPU optimizations for the specified device."""
    global gpu_optimizer
    gpu_optimizer = GPUOptimizer(device)
    gpu_optimizer.optimize_gpu_settings()
    return gpu_optimizer


def get_gpu_optimizer() -> GPUOptimizer:
    """Get the global GPU optimizer instance."""
    return gpu_optimizer
