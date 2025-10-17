import asyncio
import time
import os
from typing import AsyncGenerator, Optional, Tuple, List
from collections import deque
import threading
import queue
import subprocess
import shlex
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
        # FAST_MODE can push stride higher by default
        fast_mode = os.getenv('FAST_MODE', '0') in ('1', 'true', 'True')
        default_stride = 3 if fast_mode and frame_stride <= 2 else frame_stride
        self.frame_stride = int(os.getenv('FRAME_STRIDE', str(default_stride)))
        # Set timeout based on device
        if device == "cuda":
            self.processing_timeout = cuda_timeout
        elif device == "mps":
            self.processing_timeout = mps_timeout
        else:  # cpu or any other device
            self.processing_timeout = cpu_timeout

        logger.info(f"Video processor initialized with {device} device, timeout: {self.processing_timeout:.1f}s, prefetch={self.prefetch_frames}, frame_stride={self.frame_stride}")
    
    async def stream_frames(
        self,
        video_source: str
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

        use_nvdec = os.getenv('USE_NVDEC', '1') in ('1', 'true', 'True') and self.device == 'cuda'
        is_url = isinstance(video_source, str) and (video_source.startswith('http://') or video_source.startswith('https://'))
        q: "queue.Queue[Tuple[int, np.ndarray] | None]" = queue.Queue(maxsize=max(2, self.prefetch_frames))
        stop_flag = threading.Event()

        if use_nvdec:
            width = 0
            height = 0
            try:
                # Try ffprobe first for URLs (faster/more reliable metadata)
                if is_url:
                    probe_cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x {shlex.quote(str(video_source))}"
                    out = subprocess.check_output(shlex.split(probe_cmd), stderr=subprocess.DEVNULL, timeout=2)
                    parts = out.decode().strip().split('x')
                    if len(parts) == 2:
                        width = int(parts[0])
                        height = int(parts[1])
                if width <= 0 or height <= 0:
                    # Fallback to OpenCV probe for local files or if ffprobe failed
                    cap_probe = cv2.VideoCapture(str(video_source))
                    if cap_probe.isOpened():
                        width = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
                        height = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
                    cap_probe.release()
            except Exception:
                pass
            if width <= 0 or height <= 0:
                logger.warning("NVDEC probe failed to get dimensions; falling back to OpenCV decode")
                use_nvdec = False

        def reader_opencv() -> None:
            cap = cv2.VideoCapture(str(video_source))
            # Best-effort low-latency settings
            if self.device == "cuda":
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            try:
                frames_read = 0
                while not stop_flag.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames_read += 1

                    for _ in range(max(0, self.frame_stride - 1)):
                        ret_skip, _ = cap.read()
                        if not ret_skip:
                            break
                        frames_read += 1

                    try:
                        q.put((frames_read - 1, frame), timeout=0.5)
                    except queue.Full:
                        while not stop_flag.is_set():
                            try:
                                q.put((frames_read - 1, frame), timeout=0.5)
                                break
                            except queue.Full:
                                continue
            finally:
                try:
                    q.put(None, timeout=0.1)
                except Exception:
                    pass
                try:
                    cap.release()
                except Exception:
                    pass

        def reader_nvdec() -> None:
            # Use ffmpeg with NVDEC to decode and pipe BGR frames
            # Apply stride with select to reduce decode work
            input_arg = shlex.quote(str(video_source))
            # If URL, let ffmpeg stream directly (fast start)
            cmd = (
                f"ffmpeg -v error -hwaccel cuda -c:v h264_cuvid -i {input_arg} "
                f"-vf select='not(mod(n,{max(1, self.frame_stride)}))' "
                f"-f rawvideo -pix_fmt bgr24 -"
            )
            try:
                proc = subprocess.Popen(
                    shlex.split(cmd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=10**8
                )
            except Exception as e:
                logger.warning(f"Failed to start NVDEC ffmpeg pipeline, falling back to OpenCV: {e}")
                reader_opencv()
                return

            frame_size = width * height * 3
            frames_read = 0
            try:
                # Wait briefly until ffmpeg starts producing frames
                startup_deadline = time.time() + float(os.getenv('STREAM_BUFFER_TIMEOUT_S', '1.0'))
                while not stop_flag.is_set():
                    if proc.stdout is None:
                        break
                    # If no data yet, avoid busy-waiting on empty pipe
                    if hasattr(proc.stdout, 'peek'):
                        ready = len(proc.stdout.peek(1)) > 0
                    else:
                        ready = True
                    if ready:
                        buf = proc.stdout.read(frame_size)
                    else:
                        if time.time() > startup_deadline:
                            break
                        time.sleep(0.01)
                        continue
                    if not buf or len(buf) < frame_size:
                        break
                    frame = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3))
                    frames_read += self.frame_stride  # approximate index advance with stride
                    try:
                        q.put((frames_read - 1, frame), timeout=0.5)
                    except queue.Full:
                        while not stop_flag.is_set():
                            try:
                                q.put((frames_read - 1, frame), timeout=0.5)
                                break
                            except queue.Full:
                                continue
            finally:
                try:
                    q.put(None, timeout=0.1)
                except Exception:
                    pass
                try:
                    if proc.stdout:
                        proc.stdout.close()
                except Exception:
                    pass
                try:
                    if proc.stderr:
                        proc.stderr.close()
                except Exception:
                    pass
                try:
                    proc.terminate()
                except Exception:
                    pass

        t = threading.Thread(target=(reader_nvdec if use_nvdec else reader_opencv), name="VideoReader", daemon=True)
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