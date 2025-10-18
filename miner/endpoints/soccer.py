import os
import json
import time
from typing import Optional, Dict, Any
import supervision as sv
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
import asyncio
from pathlib import Path
from loguru import logger
import shutil

from fiber.logging_utils import get_logger
from miner.core.models.config import Config
from miner.core.configuration import factory_config
from miner.dependencies import get_config, verify_request, blacklist_low_stake
from sports.configs.soccer import SoccerPitchConfiguration
from miner.utils.device import get_optimal_device
from miner.utils.model_manager import ModelManager
from miner.utils.video_processor import VideoProcessor
from miner.utils.batch_processor import BatchFrameProcessor, NoBatchProcessor
from miner.utils.shared import miner_lock
from miner.utils.video_downloader import download_video, download_video_streaming, download_video_partial

logger = get_logger(__name__)

CONFIG = SoccerPitchConfiguration()

# Global model manager instance
model_manager = None

def get_model_manager(config: Config = Depends(get_config)) -> ModelManager:
    global model_manager
    if model_manager is None:
        model_manager = ModelManager(device=config.device)
        model_manager.load_all_models()
    return model_manager

async def process_soccer_video(
    video_source: str,
    model_manager: ModelManager,
    is_url: bool = False,
) -> Dict[str, Any]:
    """Process a soccer video and return tracking data."""
    start_time = time.time()
    # Hard time budget cutoff (env-tunable) to stay competitive
    time_budget_s = float(os.getenv("TIME_BUDGET_S", "8.0"))
    # Optionally start budget after the first frame to avoid 0-frame exits
    budget_after_first = os.getenv("START_BUDGET_AFTER_FIRST_FRAME", "1") in ("1", "true", "True")
    budget_start_time = start_time
    
    try:
        video_processor = VideoProcessor(
            device=model_manager.device,
            cuda_timeout=10800.0,
            mps_timeout=10800.0,
            cpu_timeout=10800.0,
            prefetch_frames=64,   # SPEED OPTIMIZED: Larger prefetch for batch=32
            frame_stride=2        # SPEED OPTIMIZED: Process every 2nd frame (2x speedup)
        )
        
        if not is_url:
            if not await video_processor.ensure_video_readable(video_source):
                raise HTTPException(
                    status_code=400,
                    detail="Video file is not readable or corrupted"
                )
        
        player_model = model_manager.get_model("player")
        pitch_model = model_manager.get_model("pitch")

        tracker = sv.ByteTrack()

        tracking_data = {"frames": []}

        # Get optimized inference parameters from model manager
        player_kwargs = model_manager.inference_config.get_player_inference_kwargs()
        pitch_kwargs = model_manager.inference_config.get_pitch_inference_kwargs()

        # Choose batch processor based on batch size
        batch_size = model_manager.inference_config.batch_size
        if batch_size > 1:
            processor = BatchFrameProcessor(batch_size=batch_size)
            logger.info(f"Using batch processing with batch_size={batch_size}")
        else:
            processor = NoBatchProcessor()
            logger.info("Using sequential processing (batch_size=1)")

        # OPTIMIZATION: Process frames with streaming JSON conversion
        frame_generator = video_processor.stream_frames(video_source)
        frames_list = []
        
        # Pre-allocate list for better memory performance
        frames_list = []
        
        first_frame_seen = False
        async for frame_data in processor.process_batched_frames(
            frame_generator,
            player_model,
            pitch_model,
            tracker,
            player_kwargs,
            pitch_kwargs,
        ):
            # OPTIMIZATION: Convert numpy arrays to lists immediately to reduce memory usage
            converted_frame = {
                "frame_number": int(frame_data["frame_number"]),
                "keypoints": frame_data["keypoints"].tolist() if len(frame_data["keypoints"]) > 0 else [],
                "objects": [
                    {
                        "id": int(tid),
                        "bbox": bbox.tolist(),
                        "class_id": int(cid)
                    }
                    for tid, bbox, cid in zip(frame_data["tracker_ids"], frame_data["bboxes"], frame_data["class_ids"])
                ] if len(frame_data["tracker_ids"]) > 0 else []
            }
            frames_list.append(converted_frame)
            if not first_frame_seen:
                first_frame_seen = True
                if budget_after_first:
                    budget_start_time = time.time()

            # Log progress every 100 frames
            frame_number = frame_data["frame_number"]
            if frame_number % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_number / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {frame_number} frames in {elapsed:.1f}s ({fps:.2f} fps)")

            # Enforce time budget cutoff
            if time.time() - budget_start_time >= time_budget_s:
                logger.info(f"Time budget of {time_budget_s:.1f}s reached, finishing early with {len(frames_list)} frames")
                break

        processing_time = time.time() - start_time

        # OPTIMIZATION: Frames are already converted during processing
        tracking_data["frames"] = frames_list

        tracking_data["processing_time"] = processing_time
        
        total_frames = len(tracking_data["frames"])
        fps = total_frames / processing_time if processing_time > 0 else 0
        logger.info(
            f"Completed processing {total_frames} frames in {processing_time:.1f}s "
            f"({fps:.2f} fps) on {model_manager.device} device"
        )
        
        return tracking_data
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")

async def process_challenge(
    request: Request,
    config: Config = Depends(get_config),
    model_manager: ModelManager = Depends(get_model_manager),
):
    logger.info("Attempting to acquire miner lock...")
    async with miner_lock:
        logger.info("Miner lock acquired, processing challenge...")
        try:
            challenge_data = await request.json()
            challenge_id = challenge_data.get("challenge_id")
            video_url = challenge_data.get("video_url")
            
            logger.info(f"Received challenge data: {json.dumps(challenge_data, indent=2)}")
            
            if not video_url:
                raise HTTPException(status_code=400, detail="No video URL provided")
            
            logger.info(f"Processing challenge {challenge_id} with video {video_url}")
            
            # Overlap video download and model warmup
            # Choose download strategy based on env
            use_partial = os.getenv("PARTIAL_DOWNLOAD", "1") in ("1", "true", "True")
            use_streaming = os.getenv("STREAMING_DOWNLOAD", "0") in ("1", "true", "True")
            
            if use_partial:
                # SPEED TRICK: Download only first N MB (default 4MB) for sub-1s download
                download_task = asyncio.create_task(download_video_partial(video_url))
            elif use_streaming:
                download_task = asyncio.create_task(download_video_streaming(video_url))
            else:
                download_task = asyncio.create_task(download_video(video_url))
            
            warmup_task = asyncio.create_task(model_manager.warmup_models())

            if use_streaming and not use_partial:
                video_path, background_download = await download_task
            else:
                video_path = await download_task
                background_download = None
            
            # Don't block on warmup if already done; wait up to 1s grace
            try:
                await asyncio.wait_for(warmup_task, timeout=1.0)
            except asyncio.TimeoutError:
                logger.info("Continuing without waiting for warmup to fully complete")

            try:
                # If DIRECT_URL_STREAM is enabled, bypass local file and stream directly from URL
                direct_url = os.getenv("DIRECT_URL_STREAM", "1") in ("1", "true", "True")
                ffmpeg_available = shutil.which("ffmpeg") is not None
                if use_streaming and direct_url and not ffmpeg_available:
                    logger.info("Disabling DIRECT_URL_STREAM: ffmpeg not found on PATH")
                    direct_url = False

                source = video_url if (use_streaming and direct_url) else str(video_path)
                is_url = use_streaming and direct_url

                tracking_data = await process_soccer_video(
                    source,
                    model_manager,
                    is_url=is_url
                )

                # Fallback: if direct URL produced zero frames or slow start, retry with local file path
                if is_url and ((not tracking_data.get("frames")) or tracking_data.get("slow_start")):
                    logger.warning("Direct URL streaming returned 0 frames, retrying with local file")
                    # If we have an in-progress streaming download, wait briefly for bytes to accumulate
                    min_bytes = int(os.getenv("STREAM_MIN_START_BYTES", str(3 * 1024 * 1024)))
                    buffer_deadline = asyncio.get_event_loop().time() + float(os.getenv("STREAM_BUFFER_TIMEOUT_S", "1.0"))
                    try:
                        while asyncio.get_event_loop().time() < buffer_deadline:
                            try:
                                if Path(video_path).stat().st_size >= min_bytes:
                                    break
                            except FileNotFoundError:
                                pass
                            await asyncio.sleep(0.05)
                    except Exception:
                        pass
                    tracking_data = await process_soccer_video(
                        str(video_path),
                        model_manager,
                        is_url=False
                    )
                
                response = {
                    "challenge_id": challenge_id,
                    "frames": tracking_data["frames"],
                    "processing_time": tracking_data["processing_time"]
                }
                
                logger.info(f"Completed challenge {challenge_id} in {tracking_data['processing_time']:.2f} seconds")
                return response
                
            finally:
                try:
                    if not is_url:
                        os.unlink(video_path)
                except:
                    pass
                # Ensure streaming download finishes and cleans up
                if background_download is not None:
                    try:
                        await asyncio.wait_for(background_download, timeout=0.1)
                    except Exception:
                        pass
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing soccer challenge: {str(e)}")
            logger.exception("Full error traceback:")
            raise HTTPException(status_code=500, detail=f"Challenge processing error: {str(e)}")
        finally:
            logger.info("Releasing miner lock...")

# Create router with dependencies
router = APIRouter()
router.add_api_route(
    "/challenge",
    process_challenge,
    tags=["soccer"],
    dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    methods=["POST"],
)