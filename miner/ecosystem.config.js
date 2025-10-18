module.exports = {
  apps: [
    {
      name: 'sn44-miner',
      script: '/home/kobe/score/miner/score-vision/score_miner/.venv/bin/uvicorn',
      args: 'main:app --host 0.0.0.0 --port 7999 --no-access-log',
      cwd: '/home/kobe/score/miner/score-vision/score_miner/miner',
      interpreter: '/home/kobe/score/miner/score-vision/score_miner/.venv/bin/python',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '12G',
      env: {
        // Network configuration
        NETUID: '44',
        SUBTENSOR_NETWORK: 'finney',
        SUBTENSOR_ADDRESS: 'wss://entrypoint-finney.opentensor.ai:443',
        WALLET_NAME: 'validator',
        HOTKEY_NAME: 'sn44miner',
        MIN_STAKE_THRESHOLD: '1000',
        REFRESH_NODES: 'true',
        LOAD_OLD_NODES: 'true',
        
        // === SPEED TRICK: Partial Download ===
        PARTIAL_DOWNLOAD: '1',           // Download only first N MB (sub-1s download)
        PARTIAL_DOWNLOAD_MB: '4',        // 4 MB is optimal for STRIDE=5, BUDGET=2s
        
        // Device configuration
        DEVICE: 'cuda',
        
        // Inference settings (optimized for RTX 5070 Ti 16GB)
        BATCH_SIZE: '48',                // Larger batch for 16GB VRAM (was 32)
        IMG_SIZE: '416',                 // Smaller for speed (was 640)
        FRAME_STRIDE: '5',               // More aggressive sampling (was 2)
        PREFETCH_FRAMES: '192',          // Large buffer for 96-core CPU (was 64)
        CONF_THRESHOLD: '0.5',
        MAX_DETECTIONS: '80',
        
        // Speed optimizations
        RAMP_UP: '1',
        RAMP_UP_FIRST_BATCH: '2',
        DISABLE_TRACKING: '1',           // Skip ByteTrack for speed
        SKIP_PITCH: '1',                 // Skip pitch detection for speed
        
        // Time budget
        TIME_BUDGET_S: '2.0',
        START_BUDGET_AFTER_FIRST_FRAME: '1',
        EARLY_FLUSH_FIRST_FRAME: '1',
        
        // GPU decode
        USE_NVDEC: '1',
        NVDEC_FIXED_SCALE: '1',
        NVDEC_OUT_W: '416',
        NVDEC_OUT_H: '416',
        
        // Disable old download methods
        DIRECT_URL_STREAM: '0',
        STREAMING_DOWNLOAD: '0',
        ENABLE_PARALLEL_DOWNLOAD: '0',
        
        PYTORCH_CUDA_ALLOC_CONF: 'expandable_segments:True'
      },
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
      time: true
    }
  ]
};
