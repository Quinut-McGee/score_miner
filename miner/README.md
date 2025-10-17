# Score Vision (SN44) - Miner

This is the miner component for Subnet 44 (Soccer Video Analysis). For full subnet documentation, please see the [main README](../README.md).

## System Requirements

Please see [REQUIREMENTS.md](REQUIREMENTS.md) for detailed system requirements.

## Setup Instructions

1. **Bootstrap System Dependencies**

```bash
# Clone repository
git clone https://github.com/score-technologies/score-vision.git
cd score-vision
chmod +x bootstrap.sh
./bootstrap.sh
```

2. Setup Bittensor Wallet:

```bash
# Create hotkey directory
mkdir -p ~/.bittensor/wallets/[walletname]/hotkeys/

# If copying from local machine:
scp ~/.bittensor/wallets/[walletname]/hotkeys/[hotkeyname] [user]@[SERVERIP]:~/.bittensor/wallets/[walletname]/hotkeys/[hotkeyname]
scp ~/.bittensor/wallets/[walletname]/coldkeypub.txt [user]@[SERVERIP]:~/.bittensor/wallets/[walletname]/coldkeypub.txt
```

## Installation

1. Create and activate virtual environment:

```bash
uv venv
source .venv/bin/activate  # On Unix-like systems
# or
.venv\Scripts\activate  # On Windows
```

2. Install dependencies:

```bash
uv pip install -e ".[miner]"
```

3. Setup environment:

```bash
cp miner/.env.example miner/.env
# Edit .env with your configuration
```

## Register IP on Chain

1. Get your server IP:

```bash
curl ifconfig.me
```

2. Register your IP:

```bash
fiber-post-ip --netuid 44 --subtensor.network finney --external_port [YOUR-PORT] --wallet.name [WALLET_NAME] --wallet.hotkey [HOTKEY_NAME] --external_ip [YOUR-IP]
```

## Running the Miner

### Test the Pipeline

```bash
cd miner
python scripts/test_pipeline.py
```

### Production Deployment (PM2)

```bash
cd miner
pm2 start \
  --name "sn44-miner" \
  --interpreter "../.venv/bin/python" \
  "../.venv/bin/uvicorn" \
  -- main:app --host 0.0.0.0 --port 7999
```

### Development Mode

```bash
cd miner
uvicorn main:app --reload --host 0.0.0.0 --port 7999
```

### Testing the Pipeline

To test the inference pipeline locally:

```bash
cd miner
python scripts/test_pipeline.py
```

## Operational Overview

The miner operates several key processes to handle soccer video analysis:

### 1. Challenge Reception

- Listens for incoming challenges from validators
- Validates challenge authenticity using cryptographic signatures
- Downloads video content from provided URLs
- Manages concurrent challenge processing
- Implements exponential backoff for failed downloads

### 2. Video Processing Pipeline

- Loads video frames efficiently using OpenCV
- Processes frames through multiple detection models:
  - Player detection and tracking
  - Goalkeeper identification
  - Referee detection
  - Ball tracking
- Manages GPU memory for optimal performance
- Implements frame batching for efficiency

### 3. Response Generation

- Generates standardized bounding box annotations
- Formats responses according to subnet protocol
- Includes confidence scores for detections
- Implements quality checks before submission
- Handles response encryption and signing

### 4. Health Management

- Maintains availability endpoint for validator checks
- Monitors system resources (GPU/CPU usage)
- Implements graceful challenge rejection when overloaded
- Tracks processing metrics and timings
- Manages concurrent request limits

## Configuration Reference

Key environment variables in `.env`:

```bash
# Network
NETUID=261                                    # Subnet ID (261 for testnet, 44 for mainnnet)
SUBTENSOR_NETWORK=test                        # Network type (test/local)
SUBTENSOR_ADDRESS=wss://test.finney.opentensor.ai:443  # Network address

# Miner
WALLET_NAME=default                           # Your wallet name
HOTKEY_NAME=default                           # Your hotkey name
MIN_STAKE_THRESHOLD=2                         # Minimum stake requirement

# Hardware
DEVICE=cuda                                   # Computing device (cuda/cpu/mps)
```

## Troubleshooting

### Common Issues

1. **Video Download Failures**

   - Check network connectivity
   - Verify URL accessibility
   - Monitor disk space
   - Check download timeouts

2. **Model Loading Issues**

   - Verify model files in `data/` directory
   - Check CUDA/GPU availability
   - Monitor GPU memory usage
   - Verify model compatibility

3. **Performance Issues**

   - Adjust batch size settings
   - Monitor system resources
   - Check for memory leaks
   - Optimize frame processing

4. **Network Connectivity**
   - Ensure port 7999 is exposed
   - Check firewall settings
   - Verify validator connectivity
   - Monitor network latency

For advanced configuration options and architecture details, see the [main README](../README.md).

## Credit

A big shout out to Skalskip and the work they're doing over at Roboflow. The base miner utilizes models and techniques from:

- [Roboflow Sports](https://github.com/roboflow/sports) - An open-source repository providing computer vision tools and models for sports analytics, particularly focused on soccer/football detection tasks.

## Performance Optimizations (Implemented)

The miner has been tuned for SN44 real-time competitiveness with the following improvements:

- Background frame decode and queue (utils/video_processor.py)
  - Decoding runs in a dedicated thread into a bounded queue to overlap CPU decode with GPU inference.
  - Immediate first-frame availability; removes ~10–12s initial stall in some inputs.
  - Tunables: `PREFETCH_FRAMES` (default 64), `FRAME_STRIDE` (default 2).

- Overlapped download + model warmup (endpoints/soccer.py)
  - Starts model warmup concurrently with video download; continues if warmup exceeds 1s.

- Async model warmup (utils/model_manager.py)
  - Lightweight dummy forward pass on player and pitch models to eliminate first-inference latency.

- Safe torch.compile gating (utils/gpu_optimizer.py)
  - `torch.compile` is disabled by default to avoid first-run compile latency.
  - Enable explicitly with `ENABLE_TORCH_COMPILE=1` once the pipeline is stable.

- CUDA and threading tuning (utils/gpu_optimizer.py)
  - cuDNN benchmark + TF32 enabled; process memory fraction set to 0.9.
  - PyTorch and OpenCV threads set to 32 (adjust per system).

- Device-optimized inference config (core/models/inference_config.py)
  - CUDA default `IMG_SIZE=640`, `BATCH_SIZE=32`, FP16 enabled.
  - Override via environment variables for fast experimentation.

### Tuning & Environment Variables

Set before starting the miner (examples):

```bash
export DEVICE=cuda
export IMG_SIZE=640          # try 512 for more speed
export BATCH_SIZE=32         # tune by VRAM, 16–48 typical on 16GB
export PREFETCH_FRAMES=64    # queue capacity for background decode
export FRAME_STRIDE=2        # try 3–4 for aggressive sampling
export ENABLE_TORCH_COMPILE=0  # set to 1 once warmed and stable
```

Quick tips for <10s responses:
- Increase `FRAME_STRIDE` to 3–4 during Score challenges to downsample frames.
- Consider `IMG_SIZE=512` if accuracy loss is acceptable for speed.
- Keep the process warm (avoid cold starts) and reuse loaded models.

## Proposed Next Steps (Impact-Ordered)

1. TensorRT deployment for YOLO models (2–4x inference speedup)
   - Export to ONNX and build TensorRT engines (FP16); cache engines on disk.
   - Integrate a TensorRT-backed inference path behind a feature flag.

2. GPU video decode (NVDEC) via FFmpeg/PyAV
   - Replace OpenCV decode with FFmpeg + NVDEC for higher throughput and lower CPU.
   - Target zero-copy or minimal-copy path into GPU tensors.

3. Full triple-model parallelism
   - Run player, pitch, and ball models concurrently using separate CUDA streams.
   - Overlap post-processing with next-batch inference.

4. Memory pipeline upgrades
   - Use pinned host memory and async H2D copies (`non_blocking=True`).
   - Pre-allocate GPU tensors/buffers for steady-state workloads.

5. Pipeline scheduling
   - Enforce 3-stage overlap: decode(N+1) ⇄ infer(N) ⇄ postprocess(N-1).
   - Early-flush small first batch to minimize time-to-first-result.

6. Adaptive frame sampling
   - Motion-based keyframe selection; increase stride when motion is low.

7. Warm start on boot
   - Trigger warmup and a tiny synthetic job at service start to keep kernels hot.

## Benchmarking

Use the included benchmark scripts to measure stage-by-stage improvements:

```bash
# Inference-only benchmarking
python scripts/benchmark_inference.py --device cuda --imgsz 640 --batch-size 32

# End-to-end miner benchmarking
python scripts/benchmark_miner.py --url <video_url> --repeats 3
```

Recommended metrics to log:
- First-frame latency (s), Decode FPS, Inference FPS, End-to-end time (s)
- GPU utilization (%), VRAM used (GB), CPU utilization (%)
