#!/bin/bash
# Optimized startup script for Score Vision Miner on RTX 5070 Ti

echo "=== Starting Score Vision Miner with Optimized Settings ==="
echo "GPU: NVIDIA GeForce RTX 5070 Ti (15.5 GB)"
echo "Optimal Configuration: Batch Size 4, 37.4 fps"
echo ""

# Set optimal environment variables
export DEVICE=cuda
export BATCH_SIZE=4
export PREFETCH_FRAMES=32

# Set PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Clear GPU memory before starting
echo "Clearing GPU memory..."
nvidia-smi --gpu-reset || true

# Display current GPU status
echo "Current GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv

echo ""
echo "Starting miner with optimized settings..."
echo "Expected performance: ~37.4 fps (vs ~21 fps previously)"
echo ""

# Start the miner
python main.py
