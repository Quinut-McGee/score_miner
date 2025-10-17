"""
Inference configuration for YOLO models optimized for different hardware.

This module provides device-specific optimization settings to maximize
performance on different hardware configurations (CUDA, MPS, CPU).
"""

from dataclasses import dataclass
from typing import Optional
import platform
import os


@dataclass
class InferenceConfig:
    """Configuration for YOLO model inference optimized per device type."""

    # Image resolution for inference
    player_imgsz: int
    pitch_imgsz: int
    ball_imgsz: int

    # Inference parameters
    conf_threshold: float  # Confidence threshold for detections
    iou_threshold: float   # IoU threshold for NMS
    max_detections: int    # Maximum detections per image

    # Performance parameters
    half_precision: bool   # Use FP16 inference (2x faster on Tensor Cores)
    agnostic_nms: bool    # Class-agnostic NMS

    # Batch processing
    batch_size: int       # Number of frames to process in a batch

    @classmethod
    def for_device(cls, device: str, batch_size_override: Optional[int] = None) -> "InferenceConfig":
        """
        Create optimal inference configuration for the specified device.

        Args:
            device: Device type ('cuda', 'mps', or 'cpu')
            batch_size_override: Optional batch size override from environment

        Returns:
            InferenceConfig: Optimized configuration for the device
        """
        if device == "cuda":
            config = cls.cuda_config()
        elif device == "mps":
            config = cls.mps_config()
        else:
            config = cls.cpu_config()

        # Apply batch size override if provided
        if batch_size_override is not None:
            config.batch_size = batch_size_override

        return config

    @classmethod
    def cuda_config(cls) -> "InferenceConfig":
        """
        Configuration optimized for NVIDIA CUDA GPUs.

        Optimized for RTX 5070 Ti (16GB VRAM) and similar high-end GPUs:
        - Reduced resolution for 2-3x faster inference (640px)
        - FP16 enabled to utilize Tensor Cores (2x speed boost)
        - AGGRESSIVE batch size for maximum GPU utilization (32 frames)
        - Optimized NMS for speed
        
        Can be overridden with environment variables:
        - IMG_SIZE: Image resolution (default 640)
        - BATCH_SIZE: Batch size (default 32)
        """
        # Allow environment variable overrides for easy tuning
        fast_mode = os.getenv('FAST_MODE', '0') in ('1', 'true', 'True')
        img_size = int(os.getenv('IMG_SIZE', os.getenv('FAST_IMG_SIZE', '512' if fast_mode else '640')))
        batch_size = int(os.getenv('BATCH_SIZE', '32'))
        
        return cls(
            player_imgsz=img_size,      # SPEED OPTIMIZED: 640px is 2-3x faster than 1024px
            pitch_imgsz=img_size,       # SPEED OPTIMIZED: Smaller for faster keypoint detection
            ball_imgsz=img_size,        # SPEED OPTIMIZED: Consistent size
            conf_threshold=float(os.getenv('CONF_THRESHOLD', '0.40' if fast_mode else '0.35')),
            iou_threshold=0.5,          # Higher IoU to reduce duplicate detections
            max_detections=int(os.getenv('MAX_DETECTIONS', '120' if fast_mode else '150')),
            half_precision=True,        # CRITICAL: Use FP16 Tensor Cores
            agnostic_nms=False,         # Keep class-specific NMS
            batch_size=batch_size,      # AGGRESSIVE: RTX 5070 Ti can handle 32 frames easily (8x speedup)
        )

    @classmethod
    def mps_config(cls) -> "InferenceConfig":
        """
        Configuration optimized for Apple Silicon (M1/M2/M3) MPS.

        MPS has limited support for some operations:
        - Moderate resolution (MPS has memory limitations)
        - FP16 NOT reliably supported on MPS yet
        - Smaller batch size due to memory constraints
        """
        return cls(
            player_imgsz=1280,      # Conservative for MPS
            pitch_imgsz=1024,       # Reduced for MPS
            ball_imgsz=1024,        # Reduced for MPS
            conf_threshold=0.25,
            iou_threshold=0.45,
            max_detections=300,
            half_precision=False,   # MPS FP16 support is unreliable
            agnostic_nms=False,
            batch_size=2,           # Smaller batches for MPS
        )

    @classmethod
    def cpu_config(cls) -> "InferenceConfig":
        """
        Configuration optimized for CPU-only inference.

        Conservative settings for CPU:
        - Lower resolution to maintain reasonable FPS
        - No FP16 (not beneficial on CPU)
        - Batch size of 1 (CPU can't benefit from batching)
        """
        return cls(
            player_imgsz=1024,      # Lower resolution for CPU
            pitch_imgsz=896,        # Lower resolution
            ball_imgsz=896,         # Lower resolution
            conf_threshold=0.25,
            iou_threshold=0.45,
            max_detections=300,
            half_precision=False,   # FP16 not beneficial on CPU
            agnostic_nms=False,
            batch_size=1,           # No batching on CPU
        )

    def get_player_inference_kwargs(self) -> dict:
        """Get inference kwargs for player detection model."""
        return {
            "imgsz": self.player_imgsz,
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "max_det": self.max_detections,
            "half": self.half_precision,
            "agnostic_nms": self.agnostic_nms,
            "verbose": False,
        }

    def get_pitch_inference_kwargs(self) -> dict:
        """Get inference kwargs for pitch keypoint detection model."""
        return {
            "imgsz": self.pitch_imgsz,
            "half": self.half_precision,
            "verbose": False,
        }

    def get_ball_inference_kwargs(self) -> dict:
        """Get inference kwargs for ball detection model."""
        return {
            "imgsz": self.ball_imgsz,
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "max_det": 10,  # Usually only 1 ball, but allow some margin
            "half": self.half_precision,
            "agnostic_nms": self.agnostic_nms,
            "verbose": False,
        }


def get_inference_config(device: Optional[str] = None, batch_size: Optional[int] = None) -> InferenceConfig:
    """
    Get optimal inference configuration for the current or specified device.

    Args:
        device: Optional device specification ('cuda', 'mps', 'cpu')
                If None, auto-detects based on availability
        batch_size: Optional batch size override

    Returns:
        InferenceConfig: Optimized configuration
    """
    if device is None:
        # Auto-detect device
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif platform.system() == "Darwin" and hasattr(torch.backends, "mps"):
            if torch.backends.mps.is_built() and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = "cpu"

    return InferenceConfig.for_device(device, batch_size_override=batch_size)
