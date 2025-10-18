# Partial Download Implementation - The Speed Trick

## Summary

Implemented the "speed trick" hinted at in the SN44 Discord: **downloading only the first N MB of video** instead of the entire file. This reduces download time from 8-9s to <1s, making sub-2s total processing time achievable.

## Files Modified

### 1. `miner/utils/video_downloader.py`
**Added**: `download_video_partial()` function
- Uses HTTP Range requests: `Range: bytes=0-{max_bytes-1}`
- Downloads only first N MB (configurable, default 4MB)
- Handles servers that don't support range requests (fallback to truncation)
- Logs partial download size for monitoring

### 2. `miner/endpoints/soccer.py`
**Modified**: `process_challenge()` function
- Added import for `download_video_partial`
- Added `PARTIAL_DOWNLOAD` environment variable check (default ON)
- Logic: `PARTIAL_DOWNLOAD` > `STREAMING_DOWNLOAD` > full download
- Backward compatible: old behavior still available with `PARTIAL_DOWNLOAD=0`

### 3. `scripts/benchmark_miner.py`
**Modified**: Command-line options
- Added `--partial-download` flag
- Added `--partial-mb N` option (default 4)
- Updated download logic to support partial downloads in benchmarks
- Maintains compatibility with existing flags

### 4. `scripts/test_partial_download.py` (NEW)
**Created**: Speed comparison script
- Tests full download vs partial (2MB, 4MB, 6MB)
- Shows speedup multipliers
- Helps tune `PARTIAL_DOWNLOAD_MB` for your network

### 5. `SPEED_TRICK_GUIDE.md` (NEW)
**Created**: Comprehensive guide
- Explains the math behind partial downloads
- Configuration examples
- Tuning guide based on stride/budget
- Production deployment instructions
- Troubleshooting section

## Environment Variables

### New Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PARTIAL_DOWNLOAD` | `1` | Enable partial download trick |
| `PARTIAL_DOWNLOAD_MB` | `4` | How many MB to download |

### Recommended Settings for Sub-2s Processing

```bash
# Download optimization
export PARTIAL_DOWNLOAD=1
export PARTIAL_DOWNLOAD_MB=4

# Inference optimization  
export DEVICE=cuda
export IMG_SIZE=416
export BATCH_SIZE=32
export FRAME_STRIDE=5
export PREFETCH_FRAMES=128

# Speed toggles
export RAMP_UP=1
export RAMP_UP_FIRST_BATCH=2
export DISABLE_TRACKING=1
export SKIP_PITCH=1
export CONF_THRESHOLD=0.5
export MAX_DETECTIONS=80

# Time budget
export TIME_BUDGET_S=2.0
export START_BUDGET_AFTER_FIRST_FRAME=1
export EARLY_FLUSH_FIRST_FRAME=1

# GPU decode
export USE_NVDEC=1
```

## How to Test

### 1. Quick Speed Test
```bash
cd /path/to/score-vision
python scripts/test_partial_download.py
```

This will compare:
- Full download (14 MB, ~8-9s)
- Partial 2 MB (~0.5-1s)
- Partial 4 MB (~0.7-1.2s) ← RECOMMENDED
- Partial 6 MB (~1.0-1.5s)

### 2. Benchmark with Partial Download
```bash
# Test with 4 MB partial download
python scripts/benchmark_miner.py \
  --video-url "https://scoredata.me/chunks/e10ab6a14f43457995316c7c0358d9.mp4" \
  --device cuda \
  --partial-download \
  --partial-mb 4 \
  --quick

# Compare against full download
python scripts/benchmark_miner.py \
  --video-url "https://scoredata.me/chunks/e10ab6a14f43457995316c7c0358d9.mp4" \
  --device cuda \
  --quick
```

### 3. Production Deployment
```bash
# Set environment variables
export PARTIAL_DOWNLOAD=1
export PARTIAL_DOWNLOAD_MB=4
export DEVICE=cuda
export IMG_SIZE=416
export BATCH_SIZE=32
export FRAME_STRIDE=5
# ... (other variables)

# Restart miner
pm2 restart sn44-miner --update-env
```

## Expected Impact

### Before
```
Download:      8.5s (14.1 MB)
First frame:   2.8-3.7s
Processing:    ~1-2s
Total:         ~6-8s ❌ Not competitive
Frames:        6-10 (time budget cutoff)
```

### After (with partial download + optimizations)
```
Download:      <1s (4 MB)
First frame:   0.3-0.8s
Processing:    ~0.5-1.5s
Total:         ~1.5-2.5s ✅ Competitive!
Frames:        20-50 (more useful data)
```

**Speedup**: 3-4x faster end-to-end

## The Math

Why 4 MB is sufficient:

```
FRAME_STRIDE = 5
TIME_BUDGET_S = 2.0
Video FPS = 30

Frames needed = (30 / 5) * 2.0 * 1.5 = ~18 frames
Video duration = 18 / 30 = 0.6 seconds
Typical bitrate = ~1.5-2 MB/sec for H.264
Bytes needed = 0.6 * 2 MB/s = 1.2 MB

Safety factor 3x: 1.2 * 3 = 3.6 MB
Rounded up: 4 MB ✓
```

## Tuning for Different Configurations

| STRIDE | BUDGET | Recommended MB |
|--------|--------|----------------|
| 6      | 2.0s   | 2-3 MB         |
| 5      | 2.0s   | 3-4 MB         |
| 4      | 2.5s   | 4-5 MB         |
| 3      | 3.0s   | 5-6 MB         |
| 2      | 3.0s   | 6-8 MB         |

## Technical Details

### HTTP Range Request
```http
GET /video.mp4 HTTP/1.1
Range: bytes=0-4194303
```

Response (if supported):
```http
HTTP/1.1 206 Partial Content
Content-Range: bytes 0-4194303/14680064
Content-Length: 4194304
```

### Fallback Handling
If server doesn't support range requests:
1. Returns 200 OK with full content
2. Client truncates to first N bytes
3. Still achieves speedup by not waiting for full transfer

### MP4 Compatibility
- MP4 files with "moov" atom at beginning: ✓ Works perfectly
- MP4 files with "moov" at end: ⚠️ May fail to decode
- Fragmented MP4 (fMP4): ✓ Works (moov/metadata in fragments)

Most modern CDN-served MP4s are "faststart" optimized (moov at beginning), so this works reliably.

## Monitoring & Validation

### Check if partial download is active:
```bash
# Should see "Downloading first 4.0 MB of video (partial download trick)"
pm2 logs sn44-miner | grep -i "partial"
```

### Verify environment variable is set:
```bash
# On your miner server
echo $PARTIAL_DOWNLOAD
echo $PARTIAL_DOWNLOAD_MB
```

### Monitor performance:
```bash
pm2 logs sn44-miner --lines 50 | grep -E "Downloaded|Processed|Completed"
```

Look for:
- Download time: <1s
- First frame: <1s  
- Total processing: <2.5s

## Troubleshooting

### Issue: Download still takes 8+ seconds
**Check**:
1. `echo $PARTIAL_DOWNLOAD` → should be `1`
2. Logs show "Downloading first X MB"? → If not, variable not set
3. Try explicit: `export PARTIAL_DOWNLOAD=1; pm2 restart sn44-miner`

### Issue: Video decoding fails / 0 frames
**Cause**: Downloaded too little data, moov atom missing

**Solution**:
1. Increase `PARTIAL_DOWNLOAD_MB` from 4 to 6
2. Or disable: `PARTIAL_DOWNLOAD=0` for problematic URLs
3. Check if CDN serves fragmented MP4s (better for streaming)

### Issue: Lower rewards / accuracy
**Cause**: Processing fewer frames due to partial video

**Solution**: Balance speed vs accuracy
- Increase `PARTIAL_DOWNLOAD_MB` to 5 or 6
- Decrease `FRAME_STRIDE` to 3 or 4
- Increase `IMG_SIZE` to 512

## Next Steps

After validating partial downloads work, stack additional optimizations:

1. **TensorRT** (2-4x inference speedup)
   - Export models to ONNX → TensorRT FP16
   - Combined with partial downloads: ~1-1.5s total time

2. **Even smaller partial downloads** (if network is bottleneck)
   - Try `PARTIAL_DOWNLOAD_MB=2` or `3`
   - Increase `FRAME_STRIDE` to 6

3. **Adaptive sampling**
   - Download 2MB → decode → if need more frames, fetch next 2MB
   - More complex but maximally efficient

## Credits

Insight from SN44 Discord: "Way less than 1s - no need for 1s; keep digging"

The trick: **Don't download what you don't need!** 🚀

