#!/bin/bash
# ULTRA-FAST Configuration for Sub-2s Processing
# Target: Download <1s, First Frame <0.5s, Total <2s
# RTX 5070 Ti 16GB + 96-core Xeon

echo "=========================================="
echo "SN44 Miner - ULTRA-FAST Mode (Sub-2s Target)"
echo "=========================================="
echo ""

# === DOWNLOAD OPTIMIZATION (Critical!) ===
export CHUNKED_STREAMING=1       # NEW: Stream with immediate decode start
export PARTIAL_DOWNLOAD_MB=2     # REDUCED: Only 2 MB for ultra-fast
echo "✓ Chunked streaming enabled (2 MB, immediate decode)"

# === FIRST FRAME OPTIMIZATION (Critical!) ===
export STREAM_MIN_START_BYTES=$((256*1024))   # 256 KB minimum (was 2-3 MB)
export STREAM_BUFFER_TIMEOUT_S=0.1            # 100ms wait max (was 1-2s)
export FIRST_FRAME_TIMEOUT_S=0.5              # 500ms first frame target
echo "✓ First frame optimizations: 256 KB min, 100ms timeout"

# === DEVICE & INFERENCE (Extreme Speed) ===
export DEVICE=cuda
export IMG_SIZE=384              # SMALLER: 384px (was 416)
export BATCH_SIZE=64             # LARGER: 64 (was 48) - maximize GPU
export FRAME_STRIDE=6            # MORE AGGRESSIVE: every 6th frame (was 5)
export PREFETCH_FRAMES=256       # HUGE: 256 frames (was 192)
export CONF_THRESHOLD=0.6        # HIGHER: 0.6 (was 0.5) - fewer detections
export MAX_DETECTIONS=60         # LOWER: 60 (was 80)
echo "✓ Extreme inference: IMG=384, BATCH=64, STRIDE=6"

# === SPEED TOGGLES (All On) ===
export RAMP_UP=1
export RAMP_UP_FIRST_BATCH=1     # SMALLEST: 1 frame first batch
export DISABLE_TRACKING=1
export SKIP_PITCH=1
export EARLY_FLUSH_FIRST_FRAME=1
echo "✓ All speed toggles enabled"

# === TIME BUDGET (Aggressive) ===
export TIME_BUDGET_S=1.5         # TIGHTER: 1.5s (was 2.0s)
export START_BUDGET_AFTER_FIRST_FRAME=1
echo "✓ Time budget: 1.5s"

# === GPU DECODE (Maximum Speed) ===
export USE_NVDEC=1
export NVDEC_FIXED_SCALE=1
export NVDEC_OUT_W=384          # Match IMG_SIZE
export NVDEC_OUT_H=384
echo "✓ NVDEC GPU decode at 384x384"

# === DISABLE SLOWER METHODS ===
export DIRECT_URL_STREAM=0
export STREAMING_DOWNLOAD=0
export PARTIAL_DOWNLOAD=0        # Disabled in favor of CHUNKED_STREAMING
export ENABLE_PARALLEL_DOWNLOAD=0
echo "✓ Old methods disabled"

echo ""
echo "=========================================="
echo "ULTRA-FAST mode configured!"
echo ""
echo "Expected Performance:"
echo "  Download:    <0.8s (2 MB chunked)"
echo "  First Frame: <0.5s (immediate decode)"
echo "  Processing:  <0.5s (stride=6, img=384)"
echo "  Total:       <1.8s ✅"
echo ""
echo "To deploy:"
echo "  source ULTRA_FAST_CONFIG.sh"
echo "  pm2 restart sn44-miner --update-env"
echo ""
echo "To test:"
echo "  python scripts/benchmark_miner.py \\"
echo "    --video-url 'https://scoredata.me/chunks/e10ab6a14f43457995316c7c0358d9.mp4' \\"
echo "    --device cuda --quick"
echo "=========================================="

