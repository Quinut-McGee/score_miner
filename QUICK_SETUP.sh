#!/bin/bash
# Quick setup script for optimal SN44 miner performance
# RTX 5070 Ti 16GB + 96-core Xeon
# Target: <2s processing time

echo "=========================================="
echo "SN44 Miner - Optimal Settings (RTX 5070 Ti)"
echo "=========================================="
echo ""

# === THE SPEED TRICK: Partial Download ===
export PARTIAL_DOWNLOAD=1
export PARTIAL_DOWNLOAD_MB=4
echo "✓ Partial download enabled (4 MB)"

# === Device Configuration ===
export DEVICE=cuda
echo "✓ Using CUDA (RTX 5070 Ti)"

# === Inference Settings (Optimized for 16GB VRAM) ===
export IMG_SIZE=416           # Smaller = faster (try 384 for even more speed)
export BATCH_SIZE=48          # Can handle larger batches with 16GB VRAM
export FRAME_STRIDE=5         # Process every 5th frame
export PREFETCH_FRAMES=192    # Large buffer for 96-core CPU
export CONF_THRESHOLD=0.5
export MAX_DETECTIONS=80
echo "✓ Inference config: IMG_SIZE=416, BATCH_SIZE=48, STRIDE=5"

# === Speed Optimizations ===
export RAMP_UP=1
export RAMP_UP_FIRST_BATCH=2
export DISABLE_TRACKING=1     # Disable ByteTrack for speed
export SKIP_PITCH=1           # Skip pitch detection for speed
echo "✓ Speed toggles enabled"

# === Time Budget ===
export TIME_BUDGET_S=2.0
export START_BUDGET_AFTER_FIRST_FRAME=1
export EARLY_FLUSH_FIRST_FRAME=1
echo "✓ Time budget: 2.0s"

# === GPU Decode (NVDEC) ===
export USE_NVDEC=1
export NVDEC_FIXED_SCALE=1
export NVDEC_OUT_W=416
export NVDEC_OUT_H=416
echo "✓ NVDEC GPU decode enabled"

# === Disable Old Methods ===
export DIRECT_URL_STREAM=0
export STREAMING_DOWNLOAD=0
export ENABLE_PARALLEL_DOWNLOAD=0
echo "✓ Old download methods disabled"

echo ""
echo "=========================================="
echo "Settings applied! Now run:"
echo "  cd miner"
echo "  pm2 restart sn44-miner --update-env"
echo ""
echo "Or test with:"
echo "  python scripts/benchmark_miner.py \\"
echo "    --video-url 'https://scoredata.me/chunks/e10ab6a14f43457995316c7c0358d9.mp4' \\"
echo "    --partial-download \\"
echo "    --device cuda \\"
echo "    --quick"
echo "=========================================="

