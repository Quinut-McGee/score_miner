"""
Batch processing utilities for efficient GPU utilization.

This module provides batching capabilities for YOLO inference to maximize
GPU throughput by processing multiple frames simultaneously.
"""

import numpy as np
from typing import List, Tuple, AsyncGenerator, Optional, Any
from loguru import logger
import supervision as sv
import concurrent.futures


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
        # Ramp-up: start with a smaller first batch to reduce time-to-first-result
        # Controlled via env RAMP_UP and RAMP_UP_FIRST_BATCH
        import os
        self.enable_ramp_up = os.getenv("RAMP_UP", "1") in ("1", "true", "True")
        self.first_batch_target = int(os.getenv("RAMP_UP_FIRST_BATCH", str(max(1, min(8, batch_size // 4)))))
        # Executor for overlapping batch inference and CPU postprocessing
        self._batch_executor: Optional[concurrent.futures.ThreadPoolExecutor] = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        # Optional fast-mode toggles
        self.disable_tracking = os.getenv("DISABLE_TRACKING", "0") in ("1", "true", "True")
        self.skip_pitch = os.getenv("SKIP_PITCH", "0") in ("1", "true", "True")
        self.early_flush_first = os.getenv("EARLY_FLUSH_FIRST_FRAME", "1") in ("1", "true", "True")
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

        first_batch_pending = self.enable_ramp_up
        current_target = self.first_batch_target if first_batch_pending else self.batch_size
        # Initialize double-buffering state
        prev_future: Optional[concurrent.futures.Future] = None
        prev_frame_numbers: List[int] = []
        first_frame_emitted = False

        async for frame_number, frame in frame_generator:
            batch_frames.append(frame)
            batch_numbers.append(frame_number)

            # Process batch when full or if this is the last frame
            # Early flush: if enabled, emit the very first frame ASAP
            if self.early_flush_first and not first_frame_emitted and len(batch_frames) >= 1:
                # Run a minimal batch of size 1, then continue normal batching
                mini_frames = [batch_frames.pop(0)]
                mini_numbers = [batch_numbers.pop(0)]
                # Submit and wait immediately
                mini_future = self._submit_infer_batch(mini_frames, player_model, pitch_model, player_kwargs, pitch_kwargs)
                pitch_results, player_results = await self._wait_future(mini_future)
                async for frame_data in self._postprocess_batch(mini_numbers, pitch_results, player_results, tracker):
                    yield frame_data
                first_frame_emitted = True
                # Recompute target (first batch may still be pending)
                current_target = (self.first_batch_target if first_batch_pending else self.batch_size)

            if len(batch_frames) >= current_target:
                # Pipeline: submit inference for the next batch before postprocessing the previous
                if prev_future is None:
                    prev_future = self._submit_infer_batch(batch_frames, player_model, pitch_model, player_kwargs, pitch_kwargs)
                    prev_frame_numbers = list(batch_numbers)
                else:
                    next_future = self._submit_infer_batch(batch_frames, player_model, pitch_model, player_kwargs, pitch_kwargs)
                    # Wait for prev inference and postprocess while next runs
                    pitch_results, player_results = await self._wait_future(prev_future)
                    async for frame_data in self._postprocess_batch(prev_frame_numbers, pitch_results, player_results, tracker):
                        yield frame_data
                    prev_future = next_future
                    prev_frame_numbers = list(batch_numbers)

                # Reset batch
                batch_frames = []
                batch_numbers = []

                # After the first small batch, switch to full size
                if first_batch_pending:
                    first_batch_pending = False
                    current_target = self.batch_size

        # Process remaining frames in partial batch
        if batch_frames:
            if prev_future is None:
                prev_future = self._submit_infer_batch(batch_frames, player_model, pitch_model, player_kwargs, pitch_kwargs)
                prev_frame_numbers = list(batch_numbers)
            else:
                next_future = self._submit_infer_batch(batch_frames, player_model, pitch_model, player_kwargs, pitch_kwargs)
                pitch_results, player_results = await self._wait_future(prev_future)
                async for frame_data in self._postprocess_batch(prev_frame_numbers, pitch_results, player_results, tracker):
                    yield frame_data
                prev_future = next_future
                prev_frame_numbers = list(batch_numbers)

        # Drain final pending inference
        if prev_future is not None:
            pitch_results, player_results = await self._wait_future(prev_future)
            async for frame_data in self._postprocess_batch(prev_frame_numbers, pitch_results, player_results, tracker):
                yield frame_data

    def _submit_infer_batch(
        self,
        frames: List[np.ndarray],
        player_model,
        pitch_model,
        player_kwargs: dict,
        pitch_kwargs: dict,
    ) -> concurrent.futures.Future:
        """Submit GPU inference for a batch to the executor and return a future of (pitch_results, player_results)."""
        def _infer() -> Tuple[List[Any], List[Any]]:
            import torch
            batch_size = len(frames)
            use_cuda_streams = torch.cuda.is_available() and hasattr(torch.cuda, 'Stream')

            def run_pitch():
                try:
                    if use_cuda_streams:
                        with torch.cuda.stream(torch.cuda.Stream()):
                            return pitch_model(frames, **pitch_kwargs) if batch_size > 1 else [pitch_model(frames[0], **pitch_kwargs)[0]]
                    return pitch_model(frames, **pitch_kwargs) if batch_size > 1 else [pitch_model(frames[0], **pitch_kwargs)[0]]
                except Exception as e:
                    logger.warning(f"Batch pitch inference failed, falling back to sequential: {e}")
                    return [pitch_model(frame, **pitch_kwargs)[0] for frame in frames]

            def run_player():
                try:
                    if use_cuda_streams:
                        with torch.cuda.stream(torch.cuda.Stream()):
                            return player_model(frames, **player_kwargs) if batch_size > 1 else [player_model(frames[0], **player_kwargs)[0]]
                    return player_model(frames, **player_kwargs) if batch_size > 1 else [player_model(frames[0], **player_kwargs)[0]]
                except Exception as e:
                    logger.warning(f"Batch player inference failed, falling back to sequential: {e}")
                    return [player_model(frame, **player_kwargs)[0] for frame in frames]

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as exec2:
                pf = exec2.submit(run_pitch)
                plf = exec2.submit(run_player)
                pr = pf.result()
                plr = plf.result()
            return pr, plr

        assert self._batch_executor is not None
        return self._batch_executor.submit(_infer)

    async def _wait_future(self, fut: concurrent.futures.Future) -> Tuple[List[Any], List[Any]]:
        loop = __import__("asyncio").get_event_loop()
        return await loop.run_in_executor(None, fut.result)

    async def _postprocess_batch(
        self,
        frame_numbers: List[int],
        pitch_results: List[Any],
        player_results: List[Any],
        tracker: sv.ByteTrack,
    ) -> AsyncGenerator[dict, None]:
        for frame_number, pitch_result, player_result in zip(frame_numbers, pitch_results, player_results):
            keypoints = None
            if not self.skip_pitch:
                keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
            detections = sv.Detections.from_ultralytics(player_result)
            if not self.disable_tracking and tracker is not None:
                detections = tracker.update_with_detections(detections)
            frame_data = {
                "frame_number": frame_number,
                "keypoints": keypoints.xy[0] if (keypoints is not None and keypoints.xy is not None) else np.array([]),
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
