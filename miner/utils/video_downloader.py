import tempfile
from pathlib import Path
import httpx
from fastapi import HTTPException
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
import asyncio
import os
from typing import Optional, Tuple, List

async def _parallel_range_download(client: httpx.AsyncClient, url: str, content_length: int, workers: int) -> Path:
    """
    Download a file using parallel HTTP range requests.

    Args:
        client: Shared AsyncClient
        url: Source URL
        content_length: Total file size in bytes
        workers: Number of parallel ranges

    Returns:
        Path to the completed temp file
    """
    # Create temp file and pre-allocate size
    temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
    os.close(temp_fd)
    # Pre-allocate size to reduce fragmentation
    with open(temp_path, 'wb') as f:
        f.truncate(content_length)

    chunk_size = max(256 * 1024, content_length // max(1, workers))
    ranges: List[Tuple[int, int]] = []
    start = 0
    while start < content_length:
        end = min(content_length - 1, start + chunk_size - 1)
        ranges.append((start, end))
        start = end + 1

    async def fetch_range(rng: Tuple[int, int]):
        start_byte, end_byte = rng
        headers = {"Range": f"bytes={start_byte}-{end_byte}"}
        # Use a separate connection for each range for maximum throughput
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.content
        if len(data) != (end_byte - start_byte + 1):
            # Some servers ignore Range; fail fast to fallback
            raise HTTPException(status_code=500, detail="Range request returned unexpected size")
        # Write to file at correct offset
        with open(temp_path, 'r+b', buffering=0) as f:
            f.seek(start_byte)
            f.write(data)

    await asyncio.gather(*[fetch_range(r) for r in ranges])
    return Path(temp_path)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def download_video(url: str) -> Path:
    """
    Download video with streaming for faster initial access.
    Uses chunked download so video can be accessed while downloading.
    
    Args:
        url: URL of the video to download
        
    Returns:
        Path: Path to the downloaded video file
        
    Raises:
        HTTPException: If download fails
    """
    try:
        # SPEED OPTIMIZED: Use longer timeout and streaming
        enable_parallel = os.getenv("ENABLE_PARALLEL_DOWNLOAD", "1") in ("1", "true", "True")
        parallel_workers = int(os.getenv("PARALLEL_DOWNLOAD_WORKERS", "8"))
        # Enable HTTP/2 only if explicitly requested AND the 'h2' package is installed
        enable_http2 = os.getenv("ENABLE_HTTP2", "0") in ("1", "true", "True")
        http2_available = False
        if enable_http2:
            try:
                import h2  # noqa: F401
                http2_available = True
            except Exception:
                http2_available = False
                logger.warning("ENABLE_HTTP2=1 but 'h2' is not installed; falling back to HTTP/1.1")
        limits = httpx.Limits(
            max_connections=max(20, parallel_workers * 2),
            max_keepalive_connections=max(20, parallel_workers * 2)
        )
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True, http2=http2_available, limits=limits) as client:
            # Use HEAD to get headers quickly; fallback to GET if HEAD not supported
            try:
                response = await client.head(url, follow_redirects=True)
                # Some servers don't return headers on HEAD properly; fallback check
                if response.status_code >= 400 or not response.headers:
                    raise httpx.HTTPError("HEAD not supported or no headers")
            except Exception:
                response = await client.get(url, follow_redirects=True)
            
            if "drive.google.com" in url:
                # For Google Drive, we need to handle the download URL specially
                if "drive.usercontent.google.com" not in str(response.url):
                    # If we got redirected to the Google Drive UI, construct the direct download URL
                    try:
                        file_id = url.split("id=")[1].split("&")[0]
                    except:
                        # Try alternative parsing
                        file_id = url.split("/")[-2] if "/d/" in url else url.split("id=")[1].split("&")[0]
                    download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
                    response = await client.get(download_url, follow_redirects=True)
            
            response.raise_for_status()

            # Try parallel range download when supported
            if enable_parallel:
                try:
                    # Determine content length and range support
                    content_length: Optional[int] = None
                    if response.headers.get('content-length'):
                        try:
                            content_length = int(response.headers['content-length'])
                        except Exception:
                            content_length = None
                    # Probe range support
                    probe = await client.get(str(response.url), headers={"Range": "bytes=0-0", "Connection": "keep-alive"})
                    accept_ranges = response.headers.get('accept-ranges', '')
                    if (probe.status_code in (206, 200)) and ("bytes" in accept_ranges or probe.status_code == 206) and content_length is not None and content_length > 0:
                        logger.info(f"Using parallel range download: {parallel_workers} workers for {content_length/1024/1024:.1f} MB")
                        return await _parallel_range_download(client, str(response.url), content_length, max(2, parallel_workers))
                except Exception as e:
                    logger.warning(f"Parallel download not used, falling back to streaming: {e}")
            
            # SPEED OPTIMIZED: Stream to file in chunks for faster access, without buffering full content in memory
            temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
            chunk_size = 1024 * 1024  # 1MB chunks
            total_downloaded = 0

            # Re-issue as stream to avoid loading entire body
            async with client.stream('GET', str(response.url)) as stream_resp:
                stream_resp.raise_for_status()
                with os.fdopen(temp_fd, 'wb') as temp_file:
                    async for chunk in stream_resp.aiter_bytes(chunk_size=chunk_size):
                        temp_file.write(chunk)
                        total_downloaded += len(chunk)
            
            logger.info(f"Video downloaded successfully ({total_downloaded / 1024 / 1024:.1f} MB) to {temp_path}")
            return Path(temp_path)
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error downloading video: {str(e)}")
        logger.error(f"Response status code: {e.response.status_code}")
        logger.error(f"Response headers: {e.response.headers}")
        raise HTTPException(status_code=500, detail=f"Failed to download video: {str(e)}")
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download video: {str(e)}")


async def download_video_streaming(url: str):
    """
    ADVANCED: Download video with true streaming support.
    Allows video processing to start while download is in progress.
    
    NOTE: This requires careful coordination with video processor.
    Currently not used - kept for future optimization.
    
    Args:
        url: URL of the video to download
        
    Returns:
        Path: Path to the downloading video file
    """
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            # Start streaming request
            async with client.stream('GET', url) as response:
                response.raise_for_status()
                
                # Create temp file
                temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
                
                # Start writing in background
                async def write_stream():
                    with os.fdopen(temp_fd, 'wb') as temp_file:
                        async for chunk in response.aiter_bytes(chunk_size=1024*1024):
                            temp_file.write(chunk)
                
                # Start download task
                download_task = asyncio.create_task(write_stream())
                
                # Wait for initial buffering to avoid partial-file decoder errors
                # Require at least MIN_START_BYTES or 0.5s timeout
                min_bytes = int(os.getenv("STREAM_MIN_START_BYTES", str(3 * 1024 * 1024)))
                # Clamp to avoid misconfiguration (0.5MB..16MB)
                min_bytes = max(512 * 1024, min(min_bytes, 16 * 1024 * 1024))
                start_deadline = asyncio.get_event_loop().time() + float(os.getenv("STREAM_BUFFER_TIMEOUT_S", "1.0"))
                while True:
                    try:
                        size_now = Path(temp_path).stat().st_size
                        if size_now >= min_bytes:
                            break
                    except FileNotFoundError:
                        pass
                    if asyncio.get_event_loop().time() >= start_deadline:
                        break
                    await asyncio.sleep(0.05)
                
                # Return path and download task immediately so processing can start
                return Path(temp_path), download_task
                
    except Exception as e:
        logger.error(f"Error in streaming download: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download video: {str(e)}")


async def download_video_partial(url: str, max_bytes: Optional[int] = None) -> Path:
    """
    SPEED TRICK: Download only the first N MB of video for rapid processing.
    
    For competitive subnet operation with stride=5 and ~2s budget, you only need
    ~20-40 frames, which is typically 1-3 MB of video data.
    
    This is the "trick" to sub-1s download times: don't download the whole file!
    
    Args:
        url: URL of the video to download
        max_bytes: Maximum bytes to download (default from env or 4MB)
        
    Returns:
        Path: Path to the partial video file
    """
    if max_bytes is None:
        max_bytes = int(os.getenv("PARTIAL_DOWNLOAD_MB", "4")) * 1024 * 1024
    
    try:
        # AGGRESSIVE: Use shorter timeout and HTTP/2 with connection pooling
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
        timeout = httpx.Timeout(10.0, connect=2.0)  # 2s connect timeout
        
        async with httpx.AsyncClient(
            timeout=timeout, 
            follow_redirects=True,
            limits=limits,
            http2=True  # Try HTTP/2 for better performance
        ) as client:
            # Use range request to download only first N bytes
            headers = {
                "Range": f"bytes=0-{max_bytes-1}",
                "Connection": "keep-alive",
                "Accept-Encoding": "identity",  # No compression to save CPU
            }
            
            logger.info(f"Downloading first {max_bytes/1024/1024:.1f} MB of video (partial download trick)")
            
            response = await client.get(url, headers=headers)
            
            # Handle servers that don't support range requests (fallback to normal download)
            if response.status_code == 416 or response.status_code == 200:
                # Server doesn't support range or returned full file anyway
                data = response.content[:max_bytes]  # Truncate if needed
            elif response.status_code == 206:
                # Partial content - exactly what we want
                data = response.content
            else:
                response.raise_for_status()
                data = response.content[:max_bytes]
            
            # Write to temp file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
            with os.fdopen(temp_fd, 'wb') as temp_file:
                temp_file.write(data)
            
            actual_mb = len(data) / 1024 / 1024
            logger.info(f"Partial video downloaded successfully ({actual_mb:.1f} MB) to {temp_path}")
            return Path(temp_path)
            
    except Exception as e:
        logger.error(f"Error in partial download: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download video: {str(e)}")


async def download_video_chunked_streaming(url: str, max_bytes: Optional[int] = None) -> Path:
    """
    ULTRA-FAST: Stream download with immediate file availability.
    Start writing immediately so decoder can start reading while download continues.
    
    This allows processing to begin before download completes!
    
    Args:
        url: URL of the video to download
        max_bytes: Maximum bytes to download (default from env or 4MB)
        
    Returns:
        Path: Path to the downloading video file (processing can start immediately)
    """
    if max_bytes is None:
        max_bytes = int(os.getenv("PARTIAL_DOWNLOAD_MB", "4")) * 1024 * 1024
    
    try:
        # Create temp file immediately
        temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
        
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
        timeout = httpx.Timeout(15.0, connect=2.0)
        
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            limits=limits,
            http2=True
        ) as client:
            headers = {
                "Range": f"bytes=0-{max_bytes-1}",
                "Connection": "keep-alive",
                "Accept-Encoding": "identity",
            }
            
            logger.info(f"Streaming first {max_bytes/1024/1024:.1f} MB (chunked streaming for immediate decode)")
            
            async with client.stream('GET', url, headers=headers) as response:
                response.raise_for_status()
                
                bytes_written = 0
                chunk_size = 256 * 1024  # 256 KB chunks for fast initial availability
                
                with os.fdopen(temp_fd, 'wb', buffering=0) as f:  # No buffering for immediate writes
                    async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                        f.write(chunk)
                        bytes_written += len(chunk)
                        
                        # Stop when we have enough
                        if bytes_written >= max_bytes:
                            break
                
                actual_mb = bytes_written / 1024 / 1024
                logger.info(f"Streamed {actual_mb:.1f} MB to {temp_path}")
                return Path(temp_path)
                
    except Exception as e:
        logger.error(f"Error in chunked streaming download: {str(e)}")
        # Cleanup on error
        try:
            if 'temp_path' in locals():
                Path(temp_path).unlink()
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to download video: {str(e)}") 