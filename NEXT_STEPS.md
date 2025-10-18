# Next Steps: Breaking the 2s Barrier

## 📊 Current Status

Your results showed:
- **Partial download (4 MB)**: 2.77s ← Still too slow
- **First frame delay**: 3.5s ← CRITICAL BOTTLENECK
- **Total time**: 6.5s ← Not competitive

The partial download helped (8.7s → 2.8s) but **first frame delay (3.5s)** is now the main problem.

## ✅ What We Just Implemented

### 1. **HTTP/2 + Connection Pooling**
- Faster connection setup
- Persistent connections
- 2s connect timeout

### 2. **Chunked Streaming Download (NEW)**
- `download_video_chunked_streaming()`
- Writes file as chunks arrive
- Decoder can start immediately while download continues!

### 3. **Aggressive First Frame Settings**
- Wait for only 256 KB (was 2-3 MB)
- 100ms timeout (was 1-2s)
- Faster polling (10ms instead of 20ms)

### 4. **Ultra-Fast Config**
- `ULTRA_FAST_CONFIG.sh`
- IMG_SIZE=384 (smaller)
- BATCH_SIZE=64 (larger)
- FRAME_STRIDE=6 (more aggressive)
- TIME_BUDGET=1.5s (tighter)

## 🧪 Testing Plan (Run on Your Miner)

### Test 1: Diagnose Bottleneck
```bash
cd ~/score/miner/score-vision/score_miner
python scripts/diagnose_bottleneck.py
```

**This will show you**:
- Where exactly the 3.5s first frame delay comes from
- Is it DNS? Connection? OpenCV init? Decode?

### Test 2: Try Chunked Streaming
```bash
# Enable the new method
export CHUNKED_STREAMING=1
export PARTIAL_DOWNLOAD=0
export PARTIAL_DOWNLOAD_MB=2

# Test download speed
python scripts/test_partial_download.py
```

**Expected**: Download should overlap with decode!

### Test 3: Full Ultra-Fast Benchmark
```bash
# Apply all optimizations
source ULTRA_FAST_CONFIG.sh

# Run benchmark
python scripts/benchmark_miner.py \
  --video-url "https://scoredata.me/chunks/e10ab6a14f43457995316c7c0358d9.mp4" \
  --device cuda \
  --quick
```

**Target**:
- Download + First frame: <1.5s (overlapped)
- Processing: <0.5s
- Total: <2s ✅

## 📈 Expected Improvements

### Scenario A: Chunked Streaming Works
```
Before:
  Download:    2.8s
  First frame: 3.5s  (sequential)
  Total:       6.3s

After:
  Download+Decode: 1.5s  (overlapped!)
  Processing:      0.4s
  Total:           1.9s ✅
```

### Scenario B: Just Aggressive Settings
```
Before:
  Download:    2.8s
  First frame: 3.5s
  Total:       6.3s

After:
  Download:    2.5s  (HTTP/2)
  First frame: 1.0s  (aggressive buffers)
  Processing:  0.3s
  Total:       3.8s ⚠️  (better but still marginal)
```

## 🚀 Deployment

If tests look good, deploy to production:

```bash
# On your miner server
cd ~/score/miner/score-vision/score_miner
git pull origin main

# Apply ultra-fast config
source ULTRA_FAST_CONFIG.sh

# Update ecosystem.config.js with new variables
cd miner
pm2 restart sn44-miner --update-env

# Monitor
pm2 logs sn44-miner --lines 50
```

### Key Variables to Set
```bash
CHUNKED_STREAMING=1              # NEW: Enable overlapped download+decode
PARTIAL_DOWNLOAD_MB=2            # Reduced to 2 MB
STREAM_MIN_START_BYTES=262144    # 256 KB (not 2-3 MB)
STREAM_BUFFER_TIMEOUT_S=0.1      # 100ms (not 1-2s)
IMG_SIZE=384                     # Smaller images
BATCH_SIZE=64                    # Larger batches
FRAME_STRIDE=6                   # More aggressive
TIME_BUDGET_S=1.5                # Tighter budget
```

## 🔍 What to Look For

### Success Indicators
```bash
# Logs should show:
"Streaming first 2.0 MB (chunked streaming for immediate decode)"
"Processed 0 frames in 0.8s"  # <1s first frame!
"Completed processing 20 frames in 1.8s"  # <2s total!
```

### Failure Indicators
```bash
# If you see:
"Processed 0 frames in 3.Xs"  # First frame still slow
"[mov,mp4,m4a...] partial file"  # Downloaded too little
"HTTP/2 not available"  # h2 package missing
```

## 🐛 Troubleshooting

### If chunked streaming fails:
```bash
# Fall back to partial download with aggressive settings
export CHUNKED_STREAMING=0
export PARTIAL_DOWNLOAD=1
export PARTIAL_DOWNLOAD_MB=3
export STREAM_MIN_START_BYTES=524288  # 512 KB
export STREAM_BUFFER_TIMEOUT_S=0.2
```

### If first frame still slow:
```bash
# Force NVDEC GPU decode
export USE_NVDEC=1
export NVDEC_FIXED_SCALE=1
export NVDEC_OUT_W=384
export NVDEC_OUT_H=384
```

### If download still slow:
```bash
# Check your network
curl -w "@-" -o /dev/null -s "https://scoredata.me/chunks/e10ab6a14f43457995316c7c0358d9.mp4" <<'EOF'
    time_namelookup:  %{time_namelookup}\n
       time_connect:  %{time_connect}\n
    time_starttransfer: %{time_starttransfer}\n
                      ------\n
        time_total:  %{time_total}\n
EOF
```

If `time_connect` > 1s, network/DNS is your bottleneck.

## 💡 Key Realizations

1. **Partial download alone isn't enough** - we saved on download but killed it on first frame delay
2. **The trick is overlapping operations** - decode while downloading
3. **Aggressive buffering is critical** - 256 KB threshold, not 2-3 MB
4. **Both bottlenecks must be addressed** - download AND first frame

## 📞 If Still Not Under 2s

The bottleneck might be:
1. **Network latency** - CDN is just slow from your location
2. **Server processing** - Your GPU/CPU initialization is slow
3. **Video format** - The MP4 format requires too much metadata

In that case, next level optimizations:
- **Pre-warm everything**: Keep models loaded, connections open
- **Local CDN cache**: Proxy cache frequent videos
- **Faster video format**: Request videos in a streaming-friendly format
- **TensorRT**: 2-4x inference speedup

## 🎯 Success Criteria

You'll know you're competitive when:
```bash
pm2 logs sn44-miner | grep "Completed processing"
# Shows: "Completed processing X frames in Y.Ys" where Y < 2.5
```

And you start seeing:
- ✅ Rewards in your wallet
- ✅ Higher ranking on leaderboard
- ✅ Consistent <2.5s processing times

---

**Run the tests, share the results, and we'll iterate from there!** 🚀

