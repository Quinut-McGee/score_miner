import os
import sys
from pathlib import Path
from typing import Dict, Optional
from ultralytics import YOLO
from loguru import logger

from miner.utils.device import get_optimal_device
from miner.core.models.inference_config import InferenceConfig
from miner.utils.gpu_optimizer import initialize_gpu_optimizations

# Handle import of download_models from different directory contexts
try:
    from scripts.download_models import download_models
except ModuleNotFoundError:
    try:
        from miner.scripts.download_models import download_models
    except ModuleNotFoundError:
        # If we still can't import, we'll define a fallback function
        def download_models():
            logger.error("download_models function not available. Please ensure models are downloaded manually.")
            raise ImportError("Cannot import download_models. Please run 'python miner/scripts/download_models.py' manually.")

class ModelManager:
    """Manages the loading and caching of YOLO models."""

    def __init__(self, device: Optional[str] = None, batch_size: Optional[int] = None):
        self.device = get_optimal_device(device)
        self.models: Dict[str, YOLO] = {}
        self.data_dir = Path(__file__).parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)

        # Define model paths
        self.model_paths = {
            "player": self.data_dir / "football-player-detection.pt",
            "pitch": self.data_dir / "football-pitch-detection.pt",
            "ball": self.data_dir / "football-ball-detection.pt"
        }

        # Read batch size from environment if not provided
        if batch_size is None:
            batch_size_env = os.getenv("BATCH_SIZE")
            if batch_size_env is not None:
                try:
                    batch_size = int(batch_size_env)
                    logger.info(f"Using BATCH_SIZE from environment: {batch_size}")
                except ValueError:
                    logger.warning(f"Invalid BATCH_SIZE in environment: {batch_size_env}, using default")

        # Load inference configuration optimized for this device
        self.inference_config = InferenceConfig.for_device(self.device, batch_size_override=batch_size)
        logger.info(f"Loaded inference config for {self.device}: "
                   f"player_imgsz={self.inference_config.player_imgsz}, "
                   f"half={self.inference_config.half_precision}, "
                   f"batch_size={self.inference_config.batch_size}")

        # Check if models exist, download if missing
        self._ensure_models_exist()
        
        # OPTIMIZATION: Initialize GPU optimizations
        if self.device == "cuda":
            self.gpu_optimizer = initialize_gpu_optimizations(self.device)
    
    def _ensure_models_exist(self) -> None:
        """Check if required models exist, download if missing."""
        missing_models = [
            name for name, path in self.model_paths.items() 
            if not path.exists()
        ]
        
        if missing_models:
            logger.info(f"Missing models: {', '.join(missing_models)}")
            logger.info("Downloading required models...")
            download_models()
    
    def load_model(self, model_name: str) -> YOLO:
        """
        Load a model by name, using cache if available.
        
        Args:
            model_name: Name of the model to load ('player', 'pitch', or 'ball')
            
        Returns:
            YOLO: The loaded model
        """
        if model_name in self.models:
            return self.models[model_name]
        
        if model_name not in self.model_paths:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_path = self.model_paths[model_name]
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Please ensure all required models are downloaded."
            )
        
        logger.info(f"Loading {model_name} model from {model_path} to {self.device}")
        model = YOLO(str(model_path)).to(device=self.device)

        # OPTIMIZATION: Apply GPU optimizations to the model
        if self.device == "cuda" and hasattr(self, 'gpu_optimizer'):
            model.model = self.gpu_optimizer.optimize_model_for_inference(model.model)

        self.models[model_name] = model
        return model
    
    def load_all_models(self) -> None:
        """Load all models into cache."""
        for model_name in self.model_paths.keys():
            self.load_model(model_name)

    async def warmup_models(self) -> None:
        """Run a lightweight warmup pass to eliminate first-inference latency."""
        import torch
        import numpy as np
        try:
            player = self.get_model("player")
            pitch = self.get_model("pitch")

            bs = max(1, min(4, self.inference_config.batch_size))
            h = w = int(self.inference_config.player_imgsz)
            dummy = np.zeros((bs, 3, h, w), dtype=np.float16 if self.inference_config.half_precision else np.float32)
            tensor = torch.from_numpy(dummy).to(self.device)

            with torch.inference_mode():
                # Ultralytics models accept numpy/images; call low-level model for pure warmup
                _ = player.model(tensor)
                _ = pitch.model(tensor)

            if self.device == "cuda":
                torch.cuda.synchronize()
            logger.info("Models warmup completed")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
    
    def get_model(self, model_name: str) -> YOLO:
        """
        Get a model by name, loading it if necessary.
        
        Args:
            model_name: Name of the model to get ('player', 'pitch', or 'ball')
            
        Returns:
            YOLO: The requested model
        """
        return self.load_model(model_name)
    
    def clear_cache(self) -> None:
        """Clear the model cache."""
        self.models.clear() 