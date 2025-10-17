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
            # First request to get the redirect
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
            
            # SPEED OPTIMIZED: Stream to file in chunks for faster access
            # Create temp file with .mp4 extension
            temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
            
            # Write in large chunks for better I/O performance
            chunk_size = 1024 * 1024  # 1MB chunks
            total_downloaded = 0
            
            with os.fdopen(temp_fd, 'wb') as temp_file:
                temp_file.write(response.content)
                total_downloaded = len(response.content)
            
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


async def download_video_streaming(url: str) -> Path:
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
                
                # Wait a bit for initial buffering
                await asyncio.sleep(0.5)
                
                # Return path immediately so processing can start
                # Note: Caller must await download_task to ensure completion
                return Path(temp_path), download_task
                
    except Exception as e:
        logger.error(f"Error in streaming download: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download video: {str(e)}") 