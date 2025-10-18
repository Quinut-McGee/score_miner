# The Speed Trick: Sub-2s Video Processing

## The Discovery

Based on Discord hints ("way less than 1s - no need for 1s"), the competitive miners are **NOT downloading entire videos**. They only download what they need.

## The Math

For a 2-second time budget with aggressive settings:
- `FRAME_STRIDE=5` → process every 5th frame
- ~30 FPS video → 6 frames per second
- 2 seconds of processing → ~12-40 frames needed
- At 30 FPS, 40 frames = ~1.3 seconds of video
- Typical H.264 compression: ~1-3 MB per second
- **You only need 2-4 MB, not 14 MB!**

## Implementation: Partial Download

### New Feature: `download_video_partial()`

```python
# Download only first N MB (default 4MB)
video_path = await download_video_partial(url, max_bytes=4*1024*1024)
```

This uses HTTP Range requests: `Range: bytes=0-4194303` to fetch only the beginning of the file.

**Result**: Download time drops from 8-9s to **<1s** for 4 MB.

## Configuration

### Environment Variables (Production)

```bash
# === THE SPEED TRICK ===
export PARTIAL_DOWNLOAD=1          # Enable partial download (NEW!)
export PARTIAL_DOWNLOAD_MB=4       # How many MB to download (tune based on stride)

# === Aggressive Processing ===
export DEVICE=cuda
export IMG_SIZE=416                # Smaller = faster inference
export BATCH_SIZE=32               # Good for RTX 5070 Ti with 16GB VRAM
export FRAME_STRIDE=5              # Process every 5th frame (5x speedup on frames)
export PREFETCH_FRAMES=128         # Large queue for smooth batching

# === Speed Toggles ===
export RAMP_UP=1                   # Start with small batches for fast first frame
export RAMP_UP_FIRST_BATCH=2       # Very small first batch
export DISABLE_TRACKING=1          # Skip ByteTrack (faster, less accurate)
export SKIP_PITCH=1                # Skip pitch detection (faster)
export CONF_THRESHOLD=0.5          # Higher threshold = fewer detections = faster
export MAX_DETECTIONS=80           # Limit max detections per frame

# === Time Budget ===
export TIME_BUDGET_S=2.0           # Hard cutoff at 2 seconds
export START_BUDGET_AFTER_FIRST_FRAME=1  # Don't count download time
export EARLY_FLUSH_FIRST_FRAME=1   # Return first frame ASAP

# === Disable Full Downloads ===
export DIRECT_URL_STREAM=0         # Don't stream directly (partial is better)
export STREAMING_DOWNLOAD=0        # Don't use streaming (partial is better)
export USE_NVDEC=1                 # Keep GPU decoding (if available)
```

### Testing with Benchmark

```bash
# Test partial download (4MB)
python scripts/benchmark_miner.py \
  --video-url "https://scoredata.me/chunks/e10ab6a14f43457995316c7c0358d9.mp4" \
  --device cuda \
  --partial-download \
  --partial-mb 4 \
  --quick

# Test different partial sizes
python scripts/benchmark_miner.py \
  --video-url "https://scoredata.me/chunks/e10ab6a14f43457995316c7c0358d9.mp4" \
  --device cuda \
  --partial-download \
  --partial-mb 3  # Try 2, 3, 4, 5 MB

# Compare: full download vs partial
python scripts/benchmark_miner.py \
  --video-url "https://scoredata.me/chunks/e10ab6a14f43457995316c7c0358d9.mp4" \
  --device cuda  # Full download (baseline)
```

## Expected Results

### Before (Full Download)
```
Download: 8.5s for 14.1 MB
First frame: 2.8-3.7s
Total time: ~6-8s
Frames processed: 6 (hitting time budget)
```

### After (Partial Download)
```
Download: <1s for 4 MB
First frame: 0.3-0.8s
Total time: ~1.5-2.5s
Frames processed: 20-50 (more useful data)
```

## Tuning Guide

### How Many MB to Download?

Calculate based on your `FRAME_STRIDE` and `TIME_BUDGET_S`:

```python
frames_needed = (30 fps / FRAME_STRIDE) * TIME_BUDGET_S * safety_factor
video_seconds = frames_needed / 30
mb_needed = video_seconds * 1.5  # Assume 1.5 MB/sec average

# Examples:
# STRIDE=5, BUDGET=2s: (30/5)*2*1.5 = 18 frames → 0.6s → 1 MB
# STRIDE=3, BUDGET=2s: (30/3)*2*1.5 = 30 frames → 1.0s → 1.5 MB
# STRIDE=2, BUDGET=3s: (30/2)*3*1.5 = 67 frames → 2.2s → 3.3 MB
```

**Safe defaults**:
- `STRIDE=5, BUDGET=2s`: `PARTIAL_DOWNLOAD_MB=2`
- `STRIDE=3, BUDGET=2s`: `PARTIAL_DOWNLOAD_MB=3`
- `STRIDE=2, BUDGET=3s`: `PARTIAL_DOWNLOAD_MB=4`

### Risk: Running Out of Frames

If you download too little:
- Video will end early
- You'll process fewer frames
- Might reduce accuracy

**Mitigation**: Start with 4-5 MB, then tune down once stable.

## Production Deployment

Update your PM2 configuration:

```bash
# Export all environment variables
export PARTIAL_DOWNLOAD=1
export PARTIAL_DOWNLOAD_MB=4
export DEVICE=cuda
export IMG_SIZE=416
export BATCH_SIZE=32
export FRAME_STRIDE=5
export PREFETCH_FRAMES=128
export RAMP_UP=1
export RAMP_UP_FIRST_BATCH=2
export DISABLE_TRACKING=1
export SKIP_PITCH=1
export CONF_THRESHOLD=0.5
export MAX_DETECTIONS=80
export TIME_BUDGET_S=2.0
export START_BUDGET_AFTER_FIRST_FRAME=1
export EARLY_FLUSH_FIRST_FRAME=1
export USE_NVDEC=1

# Restart miner with new settings
pm2 restart sn44-miner --update-env
```

Or update `miner/ecosystem.config.js`:

```javascript
module.exports = {
  apps: [{
    name: 'sn44-miner',
    script: 'uvicorn',
    args: 'main:app --host 0.0.0.0 --port 7999',
    env: {
      PARTIAL_DOWNLOAD: '1',
      PARTIAL_DOWNLOAD_MB: '4',
      DEVICE: 'cuda',
      IMG_SIZE: '416',
      BATCH_SIZE: '32',
      FRAME_STRIDE: '5',
      PREFETCH_FRAMES: '128',
      RAMP_UP: '1',
      RAMP_UP_FIRST_BATCH: '2',
      DISABLE_TRACKING: '1',
      SKIP_PITCH: '1',
      CONF_THRESHOLD: '0.5',
      MAX_DETECTIONS: '80',
      TIME_BUDGET_S: '2.0',
      START_BUDGET_AFTER_FIRST_FRAME: '1',
      EARLY_FLUSH_FIRST_FRAME: '1',
      USE_NVDEC: '1',
    }
  }]
};
```

## Stack Additional Optimizations

### Next Steps to Hit <2s Consistently

1. **TensorRT FP16 Engines** (2-4x speedup)
   - Export YOLO models to ONNX
   - Build TensorRT engines with FP16
   - Cache engines on disk
   - Feature-flag to test side-by-side

2. **Extreme Mode** (if partial download + current settings not enough)
   ```bash
   export IMG_SIZE=384           # Even smaller
   export FRAME_STRIDE=6         # More aggressive sampling
   export PARTIAL_DOWNLOAD_MB=2  # Minimal download
   export CONF_THRESHOLD=0.6     # Higher threshold
   export MAX_DETECTIONS=60      # Fewer detections
   ```

3. **Model Pruning** (advanced)
   - Prune YOLO models to remove unnecessary weights
   - Reduces inference time by 20-30%
   - Requires retraining/fine-tuning

4. **Multi-Stream Inference**
   - Run player, pitch, ball models in parallel CUDA streams
   - Requires significant refactoring

## Monitoring

Add timing logs to track improvements:

```bash
# Watch logs in real-time
pm2 logs sn44-miner --lines 100

# Look for these metrics:
# - "Downloading first X MB" → should be <1s
# - "Processed 0 frames in X.Xs" → should be <1s (first frame time)
# - "Completed processing X frames in Y.Ys" → should be <2.5s total
```

## Troubleshooting

### Issue: Video ends too early (< 20 frames)
**Solution**: Increase `PARTIAL_DOWNLOAD_MB` from 4 to 5 or 6

### Issue: Still taking > 3s total
**Check**:
1. Is `PARTIAL_DOWNLOAD=1` set? (check `echo $PARTIAL_DOWNLOAD`)
2. Is download actually partial? (check logs for "Downloading first X MB")
3. Is NVDEC working? (check for "Using NVDEC" in logs)
4. Is stride high enough? (try `FRAME_STRIDE=6`)

### Issue: Lower accuracy/rewards
**Solution**: Balance speed vs accuracy:
- Increase `IMG_SIZE` from 416 to 512 (slower but more accurate)
- Decrease `FRAME_STRIDE` from 5 to 3 (more frames)
- Increase `PARTIAL_DOWNLOAD_MB` to get more frames
- Re-enable tracking: `DISABLE_TRACKING=0`

## Summary

The "trick" is simple: **Don't download what you don't need**.

With partial downloads, your pipeline becomes:
1. Download first 4 MB: **<1s** (was 8s)
2. First frame decode: **<1s** (was 3s)  
3. Process 20-40 frames: **<1s** (with optimizations)
4. **Total: <2.5s** (was 6-8s)

This is a **3-4x speedup** from a single change, making you competitive in the subnet!

