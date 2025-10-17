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
        NETUID: '44',
        SUBTENSOR_NETWORK: 'finney',
        SUBTENSOR_ADDRESS: 'wss://entrypoint-finney.opentensor.ai:443',
        WALLET_NAME: 'validator',
        HOTKEY_NAME: 'sn44miner',
        MIN_STAKE_THRESHOLD: '1000',
        REFRESH_NODES: 'true',
        LOAD_OLD_NODES: 'true',
        DEVICE: 'cuda',
        BATCH_SIZE: '32',
        IMG_SIZE: '640',
        FRAME_STRIDE: '2',
        PREFETCH_FRAMES: '64',
        PYTORCH_CUDA_ALLOC_CONF: 'expandable_segments:True'
      },
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
      time: true
    }
  ]
};
