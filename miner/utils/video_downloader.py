import tempfile
from pathlib import Path
import httpx
from fastapi import HTTPException
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
import asyncio
import os

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
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            # First request to get the redirect headers only (headers=True avoids double body buffering)
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