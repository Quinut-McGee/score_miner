#!/usr/bin/env python3
"""
Optimized startup script for Score Vision Miner.
Sets environment variables and starts the miner with optimal settings.
"""

import os
import sys
import subprocess
from pathlib import Path

def load_env_file():
    """Load environment variables from .env file."""
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

def main():
    # Load environment variables from .env file
    load_env_file()
    
    # Set optimal environment variables
    os.environ['DEVICE'] = 'cuda'
    os.environ['BATCH_SIZE'] = '4'
    os.environ['PREFETCH_FRAMES'] = '32'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print("=== Score Vision Miner Starting ===")
    print("GPU: NVIDIA GeForce RTX 5070 Ti (15.5 GB)")
    print("Optimal Configuration: Batch Size 4, Expected 37.4 fps")
    print("Environment Variables Set:")
    print(f"  DEVICE={os.environ['DEVICE']}")
    print(f"  BATCH_SIZE={os.environ['BATCH_SIZE']}")
    print(f"  PREFETCH_FRAMES={os.environ['PREFETCH_FRAMES']}")
    print(f"  PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")
    print(f"  NETUID={os.environ.get('NETUID', 'NOT SET')}")
    print(f"  WALLET_NAME={os.environ.get('WALLET_NAME', 'NOT SET')}")
    print(f"  HOTKEY_NAME={os.environ.get('HOTKEY_NAME', 'NOT SET')}")
    print(f"  SUBTENSOR_NETWORK={os.environ.get('SUBTENSOR_NETWORK', 'NOT SET')}")
    print("")
    
    # Import and run the main application
    try:
        from main import app
        import uvicorn
        
        print("Starting FastAPI server with optimized settings...")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=7999,
            log_level="info"
        )
    except ImportError as e:
        print(f"Error importing main application: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
