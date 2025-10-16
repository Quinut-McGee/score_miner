# Score Vision Miner - Optimization Guide

## Overview

This guide documents the performance optimizations implemented for the Score Vision miner, specifically designed to maximize throughput on high-end GPU hardware like the NVIDIA RTX 5070 Ti.

## Optimizations Implemented

### 1. Device-Specific Inference Configuration

**Location**: `miner/core/models/inference_config.py`

The inference configuration system automatically optimizes YOLO inference parameters based on the detected hardware:

#### CUDA (NVIDIA GPU) Configuration
- **Resolution**: 1536px (increased from 1280px)
- **FP16 Precision**: Enabled - Utilizes Tensor Cores for 2x speed boost
- **Batch Size**: 4 frames processed simultaneously
- **Max Detections**: 300 (handles crowded scenes)
- **Confidence Threshold**: 0.25
- **IoU Threshold**: 0.45

**Expected Performance**: 40-60 FPS on RTX 5070 Ti (vs ~22 FPS baseline)

#### MPS (Apple Silicon) Configuration
- **Resolution**: 1280px player, 1024px pitch
- **FP16 Precision**: Disabled (unreliable on MPS)
- **Batch Size**: 2 frames
- **Optimized for**: M1/M2/M3 development machines

**Expected Performance**: 3-5 FPS on M1 MacBook

#### CPU Configuration
- **Resolution**: 1024px player, 896px pitch
- **FP16 Precision**: Disabled
- **Batch Size**: 1 (no batching benefit on CPU)
- **Optimized for**: Minimal resource usage

### 2. FP16 Inference (Half Precision)

**Impact**: ~2x speed improvement on NVIDIA GPUs

Modern NVIDIA GPUs (RTX series) have dedicated Tensor Cores that can process FP16 operations at 2x the throughput of FP32. This optimization:

- Reduces memory bandwidth requirements
- Doubles inference throughput
- Maintains accuracy (minimal precision loss for object detection)

**Configuration**: Automatically enabled on CUDA devices via `half=True` parameter

### 3. Batch Processing

**Location**: `miner/utils/batch_processor.py`

Processes multiple video frames in a single inference call, improving GPU utilization:

```python
# Old approach (sequential)
for frame in frames:
    result = model(frame)  # Underutilizes GPU

# New approach (batched)
results = model(frames_batch)  # Better GPU utilization
```

**Features**:
- `BatchFrameProcessor`: Processes frames in batches (CUDA: 4, MPS: 2)
- `NoBatchProcessor`: Fallback for CPU or batch_size=1
- Automatic fallback to sequential processing if batch inference fails

**Impact**: 1.5-2x throughput improvement on GPUs

### 4. Optimized YOLO Parameters

Applied to all inference calls:

```python
player_model(
    frame,
    imgsz=1536,        # Higher resolution
    conf=0.25,         # Confidence threshold
    iou=0.45,          # IoU for NMS
    max_det=300,       # More detections
    half=True,         # FP16 precision
    agnostic_nms=False,
    verbose=False
)
```

## Architecture

### Component Overview

```
miner/
├── core/models/
│   └── inference_config.py       # Device-specific configurations
├── utils/
│   ├── model_manager.py          # Loads config, manages models
│   ├── batch_processor.py        # Batch inference logic
│   └── video_processor.py        # Frame streaming
└── endpoints/
    └── soccer.py                 # Main endpoint (uses all above)
```

### Data Flow

```
Video Frame Stream
    ↓
Batch Processor (collects frames)
    ↓
YOLO Inference (batched, FP16, optimized params)
    ↓
ByteTrack Tracking
    ↓
Results
```

## Performance Benchmarking

### Running Benchmarks

```bash
# On your M1 MacBook (for testing)
python scripts/benchmark_inference.py /path/to/test_video.mp4 --device mps

# On your Ubuntu server (for production testing)
python scripts/benchmark_inference.py /path/to/test_video.mp4 --device cuda
```

The benchmark script tests multiple configurations:
1. Baseline (original settings)
2. FP16 only
3. FP16 + higher resolution
4. FP16 + higher resolution + batching
5. Aggressive (1920px + batch=8)

### Expected Results (RTX 5070 Ti)

| Configuration | FPS | Speedup | Notes |
|--------------|-----|---------|-------|
| Baseline (1280px, FP32, batch=1) | ~22 | 1.0x | Original |
| FP16 only | ~40 | 1.8x | Tensor Cores |
| FP16 + 1536px | ~45 | 2.0x | Recommended |
| FP16 + 1536px + batch=4 | ~55 | 2.5x | **Best balance** |
| Aggressive (1920px + batch=8) | ~50 | 2.3x | May be memory-limited |

### Expected Results (M1 MacBook)

| Configuration | FPS | Notes |
|--------------|-----|-------|
| MPS Optimized | ~3-5 | Good for development testing |

## Deployment Guide

### On Your M1 MacBook (Development)

1. The code automatically detects MPS and uses conservative settings
2. Test your changes with the benchmark script
3. Verify correctness of detections/tracking
4. Once satisfied, deploy to Ubuntu server

### On Your Ubuntu Server (Production)

1. Ensure NVIDIA drivers and CUDA are installed:
```bash
nvidia-smi  # Should show RTX 5070 Ti
```

2. The miner will automatically:
   - Detect CUDA availability
   - Load CUDA-optimized configuration
   - Enable FP16 inference
   - Use batch processing (batch_size=4)

3. Monitor performance:
```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Check miner logs for FPS
tail -f miner.log | grep fps
```

### Configuration Override

If you want to test different settings, modify `inference_config.py`:

```python
@classmethod
def cuda_config(cls) -> "InferenceConfig":
    return cls(
        player_imgsz=1920,      # Try higher resolution
        batch_size=8,           # Try larger batches
        half_precision=True,
        # ... other params
    )
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch_size: Change from 4 to 2
2. Reduce resolution: Change player_imgsz from 1536 to 1280
3. Monitor VRAM usage: `nvidia-smi`

### Issue: Batch Inference Fails

**Symptoms**: "Batch inference failed, falling back to sequential"

**Explanation**: Some models may not support true batch inference. The code automatically falls back to sequential processing.

**Impact**: You still get FP16 benefits, just not batching benefits.

### Issue: Lower FPS than Expected

**Checklist**:
1. Verify CUDA is detected: Check logs for "Using CUDA device"
2. Verify FP16 is enabled: Check logs for "half=True"
3. Check GPU utilization: Run `nvidia-smi` - should be 90%+
4. Verify no throttling: Check GPU temperature and power limits

### Issue: MPS Not Detected on macOS

**Solution**:
```bash
# Check PyTorch MPS support
python -c "import torch; print(torch.backends.mps.is_available())"
```

If False, update PyTorch:
```bash
pip install --upgrade torch torchvision
```

## Hardware Utilization Analysis

### Your RTX 5070 Ti (16GB VRAM)

**Current Utilization**:
- VRAM Usage: ~4-6GB (with batch_size=4, 1536px)
- GPU Compute: Should be 90%+ during inference
- Tensor Cores: Utilized via FP16
- Memory Bandwidth: Well utilized via batching

**Headroom Available**:
- VRAM: 10GB+ free - can increase batch_size or resolution
- Compute: Limited by model complexity, not by GPU
- Potential for concurrent challenge processing (future optimization)

### Your 96-core Xeon CPU

**Current Utilization**:
- Video Download: 1-2 cores
- Video Decoding: 2-4 cores
- Frame Buffering: Minimal
- **Underutilized**: 90+ cores idle

**Future Optimization Potential**:
- Concurrent challenge processing (remove global lock)
- Parallel video download/decode
- Could process 3-5 challenges simultaneously

## Next Steps & Future Optimizations

### Phase 1: Currently Implemented ✅
- [x] FP16 inference
- [x] Higher resolution (1536px)
- [x] Optimized YOLO parameters
- [x] Batch processing
- [x] Device-specific configurations

### Phase 2: Model Upgrades (Recommended)
- [ ] Upgrade to YOLO11 models
  - Better accuracy (54.7% mAP for YOLO11x)
  - More efficient architecture (42% fewer params than YOLOv8)
  - Train/fine-tune on same datasets as current models
- [ ] Export models to TensorRT format
  - Additional 2-3x speed improvement
  - GPU-specific optimizations

### Phase 3: Advanced Optimizations
- [ ] Remove global `miner_lock` for concurrent processing
  - Process 2-3 challenges simultaneously
  - Utilize all 96 CPU cores for video decode
  - 2-3x more challenges completed
- [ ] CUDA streams for GPU parallelism
- [ ] Optimized video download pipeline
- [ ] Dynamic batch size based on frame resolution

## Expected Production Performance

### Conservative Estimate
- **Current**: ~22 FPS (baseline)
- **With Phase 1**: ~45-55 FPS (2.5x improvement)
- **Processing Time**: 140 seconds for 600 frame video → 63 seconds

### Optimistic Estimate (with all phases)
- **With Phase 2**: ~60-80 FPS (3-4x improvement)
- **With Phase 3**: ~60-80 FPS + 3x concurrent challenges
- **Effective Throughput**: 9-12x improvement over baseline

## Reward Implications

Higher performance = More rewards:

1. **More Challenges Completed**: Faster processing → more challenges per hour
2. **Better Detection Quality**: Higher resolution → better mAP scores
3. **Improved Validation Scores**: Better accuracy → higher TAO emissions

**Estimated Impact**: 2-3x increase in daily TAO rewards with Phase 1 optimizations alone.

## Monitoring & Validation

### Key Metrics to Track

1. **FPS**: Logged every 100 frames
2. **Processing Time**: Total time per video
3. **GPU Utilization**: Via `nvidia-smi`
4. **Validation Scores**: From validator feedback
5. **Challenge Success Rate**: Challenges completed vs failed

### Logging

Watch for these log messages:
```
INFO: Loaded inference config for cuda: player_imgsz=1536, half=True, batch_size=4
INFO: Using batch processing with batch_size=4
INFO: Processed 100 frames in 2.2s (45.45 fps)
```

## Support & Resources

- **Benchmark Script**: `scripts/benchmark_inference.py`
- **Config Module**: `miner/core/models/inference_config.py`
- **Batch Processor**: `miner/utils/batch_processor.py`
- **Main Endpoint**: `miner/endpoints/soccer.py`

## Questions?

For issues or questions about these optimizations, check:
1. This guide's Troubleshooting section
2. Code comments in the implementation files
3. Benchmark script output for performance validation

---

**Last Updated**: 2025-10-11
**Optimized for**: NVIDIA RTX 5070 Ti 16GB + 96-core Xeon
