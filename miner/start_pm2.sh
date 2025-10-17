#!/bin/bash
# PM2-optimized startup script for Score Vision Miner on RTX 5070 Ti

# Set optimal environment variables
export DEVICE=cuda
export BATCH_SIZE=4
export PREFETCH_FRAMES=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Clear GPU memory before starting (if nvidia-smi is available)
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --gpu-reset || true
fi

# Start the miner with optimized settings
exec python main.py
