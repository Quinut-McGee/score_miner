import asyncio
import time
import os
from typing import AsyncGenerator, Optional, Tuple, List
from collections import deque
import threading
import queue
import cv2
import numpy as np
import supervision as sv
from loguru import logger

class VideoProcessor:
    """Handles video processing with frame streaming and timeout management."""
    
    def __init__(
        self,
        device: str = "cpu",
        cuda_timeout: float = 900.0,  # 15 minutes for CUDA
        mps_timeout: float = 1800.0,  # 30 minutes for MPS
        cpu_timeout: float = 10800.0,  # 3 hours for CPU
        prefetch_frames: int = 64,   # SPEED OPTIMIZED: Larger prefetch for faster batch processing
        frame_stride: int = 2,        # SPEED OPTIMIZED: Process every Nth frame (2 = 2x faster)
    ):
        self.device = device
        # Allow environment variable overrides
        self.prefetch_frames = int(os.getenv('PREFETCH_FRAMES', str(prefetch_frames)))
        self.frame_stride = int(os.getenv('FRAME_STRIDE', str(frame_stride)))
        # Set timeout based on device
        if device == "cuda":
            self.processing_timeout = cuda_timeout
        elif device == "mps":
            self.processing_timeout = mps_timeout
        else:  # cpu or any other device
            self.processing_timeout = cpu_timeout

        logger.info(f"Video processor initialized with {device} device, timeout: {self.processing_timeout:.1f}s, prefetch={prefetch_frames}, frame_stride={frame_stride}")
    
    async def stream_frames(
        self,
        video_path: str
    ) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        """
        Stream video frames asynchronously with continuous background decode.
        A background thread decodes frames into a bounded queue so CPU decode
        overlaps with GPU inference.

        Args:
            video_path: Path to the video file

        Yields:
            Tuple[int, np.ndarray]: Frame index (sequential) and frame data
        """
        start_time = time.time()
        cap = cv2.VideoCapture(str(video_path))

        # Best-effort low-latency settings
        if self.device == "cuda":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        q: "queue.Queue[Tuple[int, np.ndarray] | None]" = queue.Queue(maxsize=max(2, self.prefetch_frames))
        stop_flag = threading.Event()

        def reader() -> None:
            try:
                frames_read = 0
                # Consume frames, applying stride by skipping reads
                while not stop_flag.is_set():
                    # Read the next frame we intend to keep
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames_read += 1

                    # Apply stride by skipping additional frames between yields
                    # We already read one above to keep; now skip stride-1
                    for _ in range(max(0, self.frame_stride - 1)):
                        ret_skip, _ = cap.read()
                        if not ret_skip:
                            break
                        frames_read += 1

                    # Block if queue is full (backpressure while GPU runs)
                    try:
                        q.put((frames_read - 1, frame), timeout=0.5)
                    except queue.Full:
                        # If consumer is slower, try again unless stopping
                        while not stop_flag.is_set():
                            try:
                                q.put((frames_read - 1, frame), timeout=0.5)
                                break
                            except queue.Full:
                                continue
            finally:
                # Signal completion
                try:
                    q.put(None, timeout=0.1)
                except Exception:
                    pass

        t = threading.Thread(target=reader, name="VideoReader", daemon=True)
        t.start()

        yielded_frame_count = 0
        try:
            while True:
                elapsed_time = time.time() - start_time
                if elapsed_time > self.processing_timeout:
                    logger.warning(
                        f"Video processing timeout reached after {elapsed_time:.1f}s "
                        f"on {self.device} device ({yielded_frame_count} frames processed)"
                    )
                    break

                try:
                    item = await asyncio.get_event_loop().run_in_executor(None, q.get)
                except Exception:
                    item = None

                if item is None:
                    logger.info(
                        f"Completed processing {yielded_frame_count} frames (stride={self.frame_stride}) "
                        f"in {elapsed_time:.1f}s on {self.device} device"
                    )
                    break

                _frame_num, frame = item
                yield yielded_frame_count, frame
                yielded_frame_count += 1
        finally:
            stop_flag.set()
            try:
                cap.release()
            except Exception:
                pass
    
    @staticmethod
    def get_video_info(video_path: str) -> sv.VideoInfo:
        """Get video information using supervision."""
        return sv.VideoInfo.from_video_path(video_path)
    
    @staticmethod
    async def ensure_video_readable(video_path: str, timeout: float = 5.0) -> bool:
        """
        Check if video is readable within timeout period.
        
        Args:
            video_path: Path to video file
            timeout: Maximum time to wait for video check
            
        Returns:
            bool: True if video is readable
        """
        try:
            async def _check_video():
                # Poll a few times to allow partially-downloaded files to become readable
                start = time.time()
                while time.time() - start < timeout:
                    cap = cv2.VideoCapture(str(video_path))
                    if cap.isOpened():
                        ret, _ = cap.read()
                        cap.release()
                        if ret:
                            return True
                    await asyncio.sleep(0.1)
                return False

            return await _check_video()
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout while checking video readability: {video_path}")
            return False
        except Exception as e:
            logger.error(f"Error checking video readability: {str(e)}")
            return False 