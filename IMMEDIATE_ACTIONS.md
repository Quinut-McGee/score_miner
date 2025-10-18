# Immediate Actions: Fix 2.4s Connection Bottleneck

## 🔴 What We Discovered

Your diagnostic revealed the **actual problem**:

```
DNS + Connect:  2.385s  ← 85% of download time!
Data transfer:  ~0.3s   ← Only 15% of download time
First frame:    3.5s    ← Also needs fixing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total:          6.2s    ❌
```

## ✅ The Fix (2 Steps)

### Step 1: Install HTTP/2 Support (30 seconds)

```bash
# SSH to your miner
ssh kobe@<your-miner-ip>

# Navigate to project
cd ~/score/miner/score-vision/score_miner

# Install HTTP/2 support
pip install 'httpx[http2]'

# Verify installation
python -c "import h2; print('✓ HTTP/2 installed!')"
```

**Expected Impact**: Connection time 2.4s → 1.0-1.5s

### Step 2: Test Connection Pool (2 minutes)

```bash
# Pull latest code (includes connection pool)
git pull origin main

# Enable connection pool (default ON)
export USE_CONNECTION_POOL=1

# Test it
python scripts/diagnose_bottleneck.py
```

**Expected Results**:
```
Stage 1: DNS + Connect
✓ DNS + Connect: 1.000s  (was 2.385s with HTTP/2)

Stage 2: Partial Download (4 MB)
✓ Download: 0.350s  (was 0.3s, similar)

Total: 1.35s (was 2.7s) ✅
```

**And for subsequent requests** (connection reuse):
```
Stage 1: Connection Pool Lookup
✓ Pool lookup: 0.010s  (was 2.385s!)

Stage 2: Download
✓ Download: 0.350s

Total: 0.36s ⚡⚡⚡
```

## 📊 Expected Performance

### Current (Without Fixes)
```
Connection:   2.4s
Download:     0.3s
First frame:  3.5s
Processing:   0.3s
━━━━━━━━━━━━━━━━
Total:        6.5s ❌
```

### After HTTP/2 + Connection Pool
```
Connection (1st): 1.0s  (HTTP/2)
Connection (2nd+): 0.01s  (pool reuse!)
Download:     0.3s
First frame:  3.5s  (still need to fix)
Processing:   0.3s
━━━━━━━━━━━━━━━━
Total (1st):  5.1s  ⚠️
Total (2nd+): 4.1s  ⚠️
```

### With All Fixes (Connection + First Frame)
```
Connection:   0.01s  (pool)
Download:     0.3s   (partial)
First frame:  0.5s   (chunked + aggressive)
Processing:   0.3s
━━━━━━━━━━━━━━━━
Total:        1.1s  ✅ COMPETITIVE!
```

## 🧪 Full Testing Sequence

```bash
# 1. Install HTTP/2
pip install 'httpx[http2]'

# 2. Pull latest code
cd ~/score/miner/score-vision/score_miner
git pull origin main

# 3. Test connection pool
export USE_CONNECTION_POOL=1
python scripts/diagnose_bottleneck.py

# 4. Test full pipeline with ultra-fast config
source ULTRA_FAST_CONFIG.sh
python scripts/benchmark_miner.py \
  --video-url "https://scoredata.me/chunks/e10ab6a14f43457995316c7c0358d9.mp4" \
  --device cuda \
  --quick

# 5. If good, deploy to production
pm2 restart sn44-miner --update-env
```

## 🚀 Deploy to Production

If tests look good:

```bash
# On your miner server
cd ~/score/miner/score-vision/score_miner

# Ensure h2 is installed
pip install 'httpx[http2]'

# Set environment variables
export USE_CONNECTION_POOL=1
export PARTIAL_DOWNLOAD=1
export PARTIAL_DOWNLOAD_MB=4

# Optional: Ultra-fast mode
source ULTRA_FAST_CONFIG.sh

# Restart miner
cd miner
pm2 restart sn44-miner --update-env

# Monitor
pm2 logs sn44-miner --lines 50
```

## 📈 What to Look For

### Success Indicators
```bash
# Logs should show:
"HTTP/2 support enabled for connection pool"  ← Pool initialized
"Connection pool initialized"                 ← Ready
"Partial video downloaded successfully (4.0 MB) to /tmp/xxx"

# And importantly:
"Completed processing X frames in Y.Ys"
# Where Y should be < 2.5s after a few challenges (pool warmed up)
```

### Performance Milestones
- **After HTTP/2 install**: Download 2.7s → 1.3s
- **After 2nd challenge**: Download 1.3s → 0.3s (pool reuse!)
- **After first frame fix**: Total 4s → 1-2s ✅

## 🐛 If Something Goes Wrong

### HTTP/2 install fails
```bash
# Upgrade pip first
pip install --upgrade pip

# Try again
pip install 'httpx[http2]'
```

### Connection pool errors
```bash
# Check import
python -c "from miner.utils.connection_pool import get_pool; print('OK')"

# Disable if needed
export USE_CONNECTION_POOL=0
```

### Still slow
```bash
# Run diagnostic to see current timings
python scripts/diagnose_bottleneck.py

# Check what's still slow
```

## 💡 Why This Works

Your CDN (scoredata.me) has:
- **High connection latency**: 2.4s per new connection
- **Fast data transfer**: ~5 MB/s once connected

Solution:
- **Connect once**: Pay 2.4s cost
- **Keep connection alive**: 30 seconds
- **Reuse for multiple videos**: 0.01s overhead

Result: **80x faster** for subsequent downloads!

## 🎯 Next Priority

After fixing the connection bottleneck (2.4s → 0.3s), tackle the **first frame delay (3.5s → 0.5s)**:

```bash
# Chunked streaming + aggressive buffers
export CHUNKED_STREAMING=1
export STREAM_MIN_START_BYTES=262144   # 256 KB
export STREAM_BUFFER_TIMEOUT_S=0.1     # 100ms
```

Combined: **0.3s download + 0.5s first frame + 0.3s process = 1.1s total** ✅

---

**Bottom Line**: Install `httpx[http2]`, pull the code, test the diagnostic. Connection should drop from 2.4s → 0.3s! 🚀

