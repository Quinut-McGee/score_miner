# Bottleneck Solutions: Getting Under 2s

## 🔍 Current Analysis

From your test results, we identified **two major bottlenecks**:

### Bottleneck #1: Download Time (2.7-2.8s)
```
Partial (2 MB): 2.70s
Partial (4 MB): 2.77s
```
**Issue**: Even downloading 2 MB takes 2.7s @ 0.7 MB/s throughput

### Bottleneck #2: First Frame Delay (3.5s) 🔴 CRITICAL
```
Processed 0 frames in 3.5s (0.00 fps)
```
**Issue**: 3.5 seconds from "video downloaded" to "first frame decoded"

### Total Impact
```
Download:        2.8s
First frame:     3.5s
Processing:      0.2s
━━━━━━━━━━━━━━━━━━━
Total:           6.5s ❌
```

## 🚀 New Optimizations Implemented

### 1. HTTP/2 + Connection Pooling
**File**: `miner/utils/video_downloader.py`

```python
# Added to download_video_partial():
limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
timeout = httpx.Timeout(10.0, connect=2.0)  # 2s connect timeout
http2=True  # HTTP/2 for multiplexing
```

**Expected Impact**: Reduce connection latency from ~1-2s to ~0.2-0.5s

### 2. Chunked Streaming Download (NEW)
**Function**: `download_video_chunked_streaming()`

Writes file as chunks arrive → decoder can start immediately!

```python
# Stream in 256 KB chunks, no buffering
async for chunk in response.aiter_bytes(chunk_size=256*1024):
    f.write(chunk)  # Write immediately
```

**Expected Impact**: Download and decode can overlap!

### 3. Aggressive First Frame Settings
**File**: `miner/utils/video_processor.py`

```python
# BEFORE:
min_bytes = 2-3 MB
timeout = 1.0-2.0s

# AFTER:
min_bytes = 512 KB (256 KB minimum)
timeout = 0.2s (200ms)
```

**Expected Impact**: Start decoding after 256 KB instead of 2-3 MB

### 4. Ultra-Fast OpenCV Settings
```python
# Force FFMPEG backend with minimal buffer
cap = cv2.VideoCapture(str(video_source), cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
```

**Expected Impact**: Reduce OpenCV init time

## 📋 Testing Strategy

### Step 1: Run Diagnostic
```bash
cd /path/to/score-vision
python scripts/diagnose_bottleneck.py
```

This will show you exactly where time is spent:
- DNS + Connection
- Download
- Video Open
- First Frame Decode
- Model Loading

**Look for**: Which stage takes > 1s?

### Step 2: Test Chunked Streaming
```bash
# Try the new chunked streaming method
export CHUNKED_STREAMING=1
export PARTIAL_DOWNLOAD_MB=2

python scripts/test_partial_download.py
```

**Expected**: Download should be faster, and decoding can start immediately

### Step 3: Ultra-Fast Configuration
```bash
# Source the ultra-fast config
source ULTRA_FAST_CONFIG.sh

# Run benchmark
python scripts/benchmark_miner.py \
  --video-url "https://scoredata.me/chunks/e10ab6a14f43457995316c7c0358d9.mp4" \
  --device cuda \
  --quick
```

**Target**:
- Download: <1s
- First frame: <0.5s
- Total: <2s

## 🎯 Ultra-Fast Configuration Details

The `ULTRA_FAST_CONFIG.sh` sets:

### Download
```bash
CHUNKED_STREAMING=1           # Stream with immediate decode
PARTIAL_DOWNLOAD_MB=2         # Only 2 MB (not 4)
STREAM_MIN_START_BYTES=256KB  # Start after 256 KB (not 2-3 MB)
STREAM_BUFFER_TIMEOUT_S=0.1   # Wait max 100ms (not 1-2s)
```

### Inference
```bash
IMG_SIZE=384                  # Smaller (was 416)
BATCH_SIZE=64                 # Larger (was 48)
FRAME_STRIDE=6                # More aggressive (was 5)
CONF_THRESHOLD=0.6            # Higher (was 0.5)
RAMP_UP_FIRST_BATCH=1         # Just 1 frame first
TIME_BUDGET_S=1.5             # Tighter (was 2.0)
```

## 🔧 If Still Not Fast Enough

### Option A: Even Smaller Download
```bash
export PARTIAL_DOWNLOAD_MB=1      # Just 1 MB
export FRAME_STRIDE=8             # Every 8th frame
```

### Option B: Pre-Connection Pool
Keep a persistent HTTP connection pool warm:
```python
# In your miner main.py, create a persistent client
client = httpx.AsyncClient(http2=True, limits=...)
# Reuse for all downloads → no connection setup time
```

### Option C: Local CDN Cache
If CDN is the bottleneck, consider:
- Cloudflare in front of scoredata.me
- Local nginx proxy cache
- Pre-warm popular video chunks

### Option D: Network Optimization
On your miner server:
```bash
# Increase TCP window size
sudo sysctl -w net.ipv4.tcp_window_scaling=1
sudo sysctl -w net.ipv4.tcp_rmem='4096 87380 16777216'
sudo sysctl -w net.ipv4.tcp_wmem='4096 65536 16777216'

# Enable TCP fast open
sudo sysctl -w net.ipv4.tcp_fastopen=3

# BBR congestion control (if kernel supports)
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr
```

## 📊 Expected Results

### Current (Before Optimizations)
```
Download:       2.8s
First frame:    3.5s
Processing:     0.2s
Total:          6.5s ❌
```

### After HTTP/2 + Aggressive Settings
```
Download:       1.5s  (↓ 1.3s)
First frame:    1.0s  (↓ 2.5s)
Processing:     0.2s
Total:          2.7s  ⚠️  (still marginal)
```

### After Chunked Streaming
```
Download+Decode: 1.2s  (overlapped!)
Processing:      0.3s
Total:           1.5s  ✅ (competitive!)
```

## 🐛 Troubleshooting

### Issue: Chunked streaming fails
**Cause**: HTTP/2 not available or h2 package missing

**Fix**:
```bash
pip install h2
# Or fall back to partial download:
export CHUNKED_STREAMING=0
export PARTIAL_DOWNLOAD=1
```

### Issue: First frame still slow (>1s)
**Cause**: Video codec or format issue

**Fix**:
```bash
# Force NVDEC
export USE_NVDEC=1
export NVDEC_FIXED_SCALE=1

# Or check video with ffprobe
ffprobe -v error -show_entries stream=codec_name,width,height <video_url>
```

### Issue: Download actually slower with HTTP/2
**Cause**: Server doesn't support HTTP/2 or has poor implementation

**Fix**:
```bash
# Disable HTTP/2
# Edit video_downloader.py, set http2=False
```

### Issue: Partial file errors
```
[mov,mp4,m4a,3gp,3g2,mj2 @ ...] stream 0, offset 0x...: partial file
```

**Cause**: Downloaded too little (moov atom incomplete)

**Fix**:
```bash
# Increase download size
export PARTIAL_DOWNLOAD_MB=3  # or 4
```

## 📈 Monitoring & Validation

### Check Download Speed
```bash
# Should see faster download
pm2 logs sn44-miner | grep -i "downloaded\|streamed"
```

### Check First Frame Time
```bash
# Should see "Processed 0 frames in X.Xs" where X < 1.0
pm2 logs sn44-miner | grep "Processed 0 frames"
```

### Check Total Time
```bash
# Should see "Completed processing X frames in Y.Ys" where Y < 2.5
pm2 logs sn44-miner | grep "Completed processing"
```

## 🎬 Next Steps

1. **Run diagnostic**: `python scripts/diagnose_bottleneck.py`
2. **Identify bottleneck**: Which stage > 1s?
3. **Apply fixes**:
   - If download slow → Chunked streaming
   - If first frame slow → Aggressive buffer settings
   - If both slow → Ultra-fast config
4. **Test**: `source ULTRA_FAST_CONFIG.sh && benchmark`
5. **Deploy**: `pm2 restart sn44-miner --update-env`
6. **Monitor**: Watch logs for <2s total time

## 💡 Key Insight

**The "trick" might not just be partial download—it's also IMMEDIATE DECODE START.**

Competitors might be:
1. Starting decode while download is in progress (chunked streaming)
2. Using extremely small buffer thresholds (256 KB)
3. Pre-warming connections
4. Using faster video codecs or GPU decode from the start

Our new chunked streaming approach targets all of these!

---

**Bottom Line**: We need to attack BOTH bottlenecks simultaneously. The new `CHUNKED_STREAMING` + ultra-aggressive settings should get you under 2s! 🚀

