# Quick Reference Card: Partial Download Speed Trick

## 🚀 The One-Liner
**Download only 4 MB instead of 14 MB → 8x faster download, 3-4x faster overall**

---

## 📋 Quick Commands

### Test Download Speed
```bash
python scripts/test_partial_download.py
```

### Benchmark with Partial Download
```bash
python scripts/benchmark_miner.py \
  --video-url "https://scoredata.me/chunks/e10ab6a14f43457995316c7c0358d9.mp4" \
  --partial-download --partial-mb 4 --device cuda --quick
```

### Deploy to Production (Miner Server)
```bash
# Option 1: Source quick setup
source QUICK_SETUP.sh
pm2 restart sn44-miner --update-env

# Option 2: Manual env vars
export PARTIAL_DOWNLOAD=1
export PARTIAL_DOWNLOAD_MB=4
export DEVICE=cuda
export IMG_SIZE=416
export BATCH_SIZE=48
export FRAME_STRIDE=5
pm2 restart sn44-miner --update-env
```

### Verify It's Working
```bash
pm2 logs sn44-miner | grep -i "partial"
# Should see: "Downloading first 4.0 MB of video (partial download trick)"
```

---

## ⚙️ Key Environment Variables

| Variable | Value | Why |
|----------|-------|-----|
| `PARTIAL_DOWNLOAD` | `1` | Enable the speed trick |
| `PARTIAL_DOWNLOAD_MB` | `4` | Download only 4 MB |
| `IMG_SIZE` | `416` | Smaller = faster inference |
| `BATCH_SIZE` | `48` | Good for 16GB VRAM |
| `FRAME_STRIDE` | `5` | Process every 5th frame |
| `TIME_BUDGET_S` | `2.0` | Hard cutoff at 2s |
| `DISABLE_TRACKING` | `1` | Skip ByteTrack for speed |
| `SKIP_PITCH` | `1` | Skip pitch detection |

---

## 📊 Expected Performance

### Before
```
Download:    8.5s (14.1 MB)
First frame: 3.0s
Total:       6-8s ❌
```

### After
```
Download:    <1s (4 MB)
First frame: <1s
Total:       1.5-2.5s ✅
```

**Speedup: 3-4x faster overall, 8x faster download**

---

## 🔧 Tuning Cheat Sheet

### If Download Still Slow
```bash
echo $PARTIAL_DOWNLOAD  # Check if set
export PARTIAL_DOWNLOAD=1
pm2 restart sn44-miner --update-env
```

### If Video Decode Fails
```bash
# Downloaded too little, increase:
export PARTIAL_DOWNLOAD_MB=6
pm2 restart sn44-miner --update-env
```

### If Need More Speed
```bash
export IMG_SIZE=384          # Smaller inference
export FRAME_STRIDE=6        # Fewer frames
export PARTIAL_DOWNLOAD_MB=3 # Less download
```

### If Need More Accuracy
```bash
export IMG_SIZE=512          # Larger inference
export FRAME_STRIDE=3        # More frames
export DISABLE_TRACKING=0    # Enable tracking
export SKIP_PITCH=0          # Enable pitch
```

---

## 🎯 Partial Download Size Guide

| STRIDE | BUDGET | Recommended MB |
|--------|--------|----------------|
| 6      | 2.0s   | 2-3 MB         |
| 5      | 2.0s   | **4 MB** ⭐     |
| 4      | 2.5s   | 5 MB           |
| 3      | 3.0s   | 6 MB           |

**Formula**: `MB = (30 / STRIDE) * BUDGET * 1.5 / 30 * 1.5`

---

## 📝 Monitoring One-Liners

```bash
# Check if partial download is enabled
pm2 show sn44-miner | grep PARTIAL

# Watch logs in real-time
pm2 logs sn44-miner --lines 50

# Check download times
pm2 logs sn44-miner | grep "Downloaded"

# Check total processing times
pm2 logs sn44-miner | grep "Completed"
```

---

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| Still slow | `echo $PARTIAL_DOWNLOAD` → re-export if not `1` |
| 0 frames | Increase `PARTIAL_DOWNLOAD_MB` to 5 or 6 |
| Low rewards | Decrease `FRAME_STRIDE` or increase `IMG_SIZE` |
| Out of memory | Decrease `BATCH_SIZE` to 32 or 24 |

---

## 🚦 Quick Health Check

```bash
# All should be true:
echo $PARTIAL_DOWNLOAD         # = 1
echo $PARTIAL_DOWNLOAD_MB      # = 4
echo $FRAME_STRIDE             # = 5
echo $TIME_BUDGET_S            # = 2.0

# Logs should show:
pm2 logs sn44-miner --lines 20 | grep -E "(partial|Completed)"
# - "Downloading first 4.0 MB"
# - "Completed processing X frames in Y.Ys" where Y < 2.5
```

---

## 📚 Full Documentation

- **`SPEED_TRICK_GUIDE.md`** - Comprehensive guide
- **`IMPLEMENTATION_SUMMARY.md`** - What changed & how to deploy
- **`PARTIAL_DOWNLOAD_IMPLEMENTATION.md`** - Technical deep dive

---

## 💡 The Secret (Decoded)

> "Way less than 1s - no need for 1s; keep digging" - SN44 Discord

**Translation**: Don't download the entire video. Download only what you need.
- Need ~30 frames? That's ~1 second of video = ~2 MB
- No need to download 14 MB when 4 MB is enough!
- Result: <1s download instead of 8s ✅

---

**That's it! You're now competitive in SN44. Good luck! 🚀**

