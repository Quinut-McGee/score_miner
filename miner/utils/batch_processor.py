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
        Process a single batch of frames.

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

        # Run batch inference for pitch keypoints
        # Note: Some models may not support true batching, so we handle both cases
        try:
            if batch_size > 1:
                # Try batch inference
                pitch_results = pitch_model(frames, **pitch_kwargs)
            else:
                # Single frame
                pitch_results = [pitch_model(frames[0], **pitch_kwargs)[0]]
        except Exception as e:
            # Fallback to sequential if batch fails
            logger.warning(f"Batch pitch inference failed, falling back to sequential: {e}")
            pitch_results = [pitch_model(frame, **pitch_kwargs)[0] for frame in frames]

        # Run batch inference for player detection
        try:
            if batch_size > 1:
                player_results = player_model(frames, **player_kwargs)
            else:
                player_results = [player_model(frames[0], **player_kwargs)[0]]
        except Exception as e:
            # Fallback to sequential if batch fails
            logger.warning(f"Batch player inference failed, falling back to sequential: {e}")
            player_results = [player_model(frame, **player_kwargs)[0] for frame in frames]

        # Process each frame in the batch
        for i, (frame_number, pitch_result, player_result) in enumerate(
            zip(frame_numbers, pitch_results, player_results)
        ):
            # Extract keypoints
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)

            # Extract detections and update tracker
            detections = sv.Detections.from_ultralytics(player_result)
            detections = tracker.update_with_detections(detections)

            # Build frame data
            frame_data = {
                "frame_number": int(frame_number),
                "keypoints": keypoints.xy[0].tolist() if keypoints and keypoints.xy is not None else [],
                "objects": [
                    {
                        "id": int(tracker_id),
                        "bbox": [float(x) for x in bbox],
                        "class_id": int(class_id)
                    }
                    for tracker_id, bbox, class_id in zip(
                        detections.tracker_id,
                        detections.xyxy,
                        detections.class_id
                    )
                ] if detections and detections.tracker_id is not None else []
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

            # Build frame data
            frame_data = {
                "frame_number": int(frame_number),
                "keypoints": keypoints.xy[0].tolist() if keypoints and keypoints.xy is not None else [],
                "objects": [
                    {
                        "id": int(tracker_id),
                        "bbox": [float(x) for x in bbox],
                        "class_id": int(class_id)
                    }
                    for tracker_id, bbox, class_id in zip(
                        detections.tracker_id,
                        detections.xyxy,
                        detections.class_id
                    )
                ] if detections and detections.tracker_id is not None else []
            }

            yield frame_data
