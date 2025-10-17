"""
Batch processing utilities for efficient GPU utilization.

This module provides batching capabilities for YOLO inference to maximize
GPU throughput by processing multiple frames simultaneously.
"""

import numpy as np
from typing import List, Tuple, AsyncGenerator
from loguru import logger
import supervision as sv


class BatchFrameProcessor:
    """
    Processes video frames in batches for optimal GPU utilization.

    Batching multiple frames together improves GPU utilization significantly,
    especially on modern GPUs with high memory bandwidth like RTX 5070 Ti.
    """

    def __init__(self, batch_size: int = 4):
        """
        Initialize the batch processor.

        Args:
            batch_size: Number of frames to process in each batch
        """
        self.batch_size = batch_size
        logger.info(f"BatchFrameProcessor initialized with batch_size={batch_size}")

    async def process_batched_frames(
        self,
        frame_generator: AsyncGenerator[Tuple[int, np.ndarray], None],
        player_model,
        pitch_model,
        tracker: sv.ByteTrack,
        player_kwargs: dict,
        pitch_kwargs: dict,
    ) -> AsyncGenerator[dict, None]:
        """
        Process frames in batches through YOLO models.

        Args:
            frame_generator: Async generator yielding (frame_number, frame) tuples
            player_model: YOLO model for player detection
            pitch_model: YOLO model for pitch keypoint detection
            tracker: ByteTrack tracker instance
            player_kwargs: Inference kwargs for player model
            pitch_kwargs: Inference kwargs for pitch model

        Yields:
            dict: Frame data with detections and keypoints
        """
        batch_frames = []
        batch_numbers = []

        async for frame_number, frame in frame_generator:
            batch_frames.append(frame)
            batch_numbers.append(frame_number)

            # Process batch when full or if this is the last frame
            if len(batch_frames) >= self.batch_size:
                async for frame_data in self._process_batch(
                    batch_frames,
                    batch_numbers,
                    player_model,
                    pitch_model,
                    tracker,
                    player_kwargs,
                    pitch_kwargs,
                ):
                    yield frame_data

                # Reset batch
                batch_frames = []
                batch_numbers = []

        # Process remaining frames in partial batch
        if batch_frames:
            async for frame_data in self._process_batch(
                batch_frames,
                batch_numbers,
                player_model,
                pitch_model,
                tracker,
                player_kwargs,
                pitch_kwargs,
            ):
                yield frame_data

    async def _process_batch(
        self,
        frames: List[np.ndarray],
        frame_numbers: List[int],
        player_model,
        pitch_model,
        tracker: sv.ByteTrack,
        player_kwargs: dict,
        pitch_kwargs: dict,
    ) -> AsyncGenerator[dict, None]:
        """
        Process a single batch of frames with CUDA streams for true parallelism.

        Args:
            frames: List of frame arrays
            frame_numbers: Corresponding frame numbers
            player_model: YOLO model for player detection
            pitch_model: YOLO model for pitch keypoint detection
            tracker: ByteTrack tracker instance
            player_kwargs: Inference kwargs for player model
            pitch_kwargs: Inference kwargs for pitch model

        Yields:
            dict: Frame data for each frame in the batch
        """
        batch_size = len(frames)
        
        # SPEED OPTIMIZED: Use CUDA streams for truly parallel execution
        import torch
        use_cuda_streams = torch.cuda.is_available() and hasattr(torch.cuda, 'Stream')

        def run_pitch():
            try:
                # Enable stream context if CUDA is available
                if use_cuda_streams:
                    with torch.cuda.stream(torch.cuda.Stream()):
                        if batch_size > 1:
                            return pitch_model(frames, **pitch_kwargs)
                        else:
                            return [pitch_model(frames[0], **pitch_kwargs)[0]]
                else:
                    if batch_size > 1:
                        return pitch_model(frames, **pitch_kwargs)
                    else:
                        return [pitch_model(frames[0], **pitch_kwargs)[0]]
            except Exception as e:
                logger.warning(f"Batch pitch inference failed, falling back to sequential: {e}")
                return [pitch_model(frame, **pitch_kwargs)[0] for frame in frames]

        def run_player():
            try:
                # Enable stream context if CUDA is available
                if use_cuda_streams:
                    with torch.cuda.stream(torch.cuda.Stream()):
                        if batch_size > 1:
                            return player_model(frames, **player_kwargs)
                        else:
                            return [player_model(frames[0], **player_kwargs)[0]]
                else:
                    if batch_size > 1:
                        return player_model(frames, **player_kwargs)
                    else:
                        return [player_model(frames[0], **player_kwargs)[0]]
            except Exception as e:
                logger.warning(f"Batch player inference failed, falling back to sequential: {e}")
                return [player_model(frame, **player_kwargs)[0] for frame in frames]

        # SPEED OPTIMIZED: Execute both models concurrently with CUDA streams
        # ThreadPoolExecutor allows true parallelism with separate CUDA streams
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            pitch_future = executor.submit(run_pitch)
            player_future = executor.submit(run_player)
            
            # Wait for both to complete
            pitch_results = pitch_future.result()
            player_results = player_future.result()
        
        # Synchronize CUDA if streams were used
        if use_cuda_streams:
            torch.cuda.synchronize()

        # Process each frame in the batch
        for i, (frame_number, pitch_result, player_result) in enumerate(
            zip(frame_numbers, pitch_results, player_results)
        ):
            # Extract keypoints
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)

            # Extract detections and update tracker
            detections = sv.Detections.from_ultralytics(player_result)
            detections = tracker.update_with_detections(detections)

            # Build frame data (optimized - minimal conversion)
            # Store numpy arrays directly, convert to lists in bulk later
            frame_data = {
                "frame_number": frame_number,  # Keep as int, don't convert
                "keypoints": keypoints.xy[0] if keypoints and keypoints.xy is not None else np.array([]),
                "tracker_ids": detections.tracker_id if detections and detections.tracker_id is not None else np.array([]),
                "bboxes": detections.xyxy if detections and detections.xyxy is not None else np.array([]),
                "class_ids": detections.class_id if detections and detections.class_id is not None else np.array([])
            }

            yield frame_data


class NoBatchProcessor:
    """
    Fallback processor that doesn't use batching (for CPU or single-frame processing).
    """

    def __init__(self):
        logger.info("NoBatchProcessor initialized (sequential frame processing)")

    async def process_batched_frames(
        self,
        frame_generator: AsyncGenerator[Tuple[int, np.ndarray], None],
        player_model,
        pitch_model,
        tracker: sv.ByteTrack,
        player_kwargs: dict,
        pitch_kwargs: dict,
    ) -> AsyncGenerator[dict, None]:
        """
        Process frames sequentially without batching.

        This is the original implementation, kept for CPU or when batch_size=1.
        """
        async for frame_number, frame in frame_generator:
            # Pitch keypoint detection
            pitch_result = pitch_model(frame, **pitch_kwargs)[0]
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)

            # Player detection
            player_result = player_model(frame, **player_kwargs)[0]
            detections = sv.Detections.from_ultralytics(player_result)
            detections = tracker.update_with_detections(detections)

            # Build frame data (optimized - minimal conversion)
            # Store numpy arrays directly, convert to lists in bulk later
            frame_data = {
                "frame_number": frame_number,  # Keep as int, don't convert
                "keypoints": keypoints.xy[0] if keypoints and keypoints.xy is not None else np.array([]),
                "tracker_ids": detections.tracker_id if detections and detections.tracker_id is not None else np.array([]),
                "bboxes": detections.xyxy if detections and detections.xyxy is not None else np.array([]),
                "class_ids": detections.class_id if detections and detections.class_id is not None else np.array([])
            }

            yield frame_data
