import os
import json
import logging
from pathlib import Path, PurePosixPath

import cv2

from .transnet_wrapper import TransNetPredictor
from .keyframe_extractor import KeyframeExtractor
from .metadata_schema import VideoMetadata
from .resume_validation import keyframe_artifacts_are_valid

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(
        self,
        output_dir: str,
        device: str = None,
        webp_quality: int = 90,
        threshold: float = 0.5,
        data_root: str | None = None,
    ):
        """
        Orchestrator for processing a single video.
        """
        self.output_dir = output_dir
        self.threshold = threshold
        self.data_root = Path(data_root).resolve() if data_root else None
        
        # Initialize sub-modules
        self.transnet = TransNetPredictor(device=device)
        self.extractor = KeyframeExtractor(webp_quality=webp_quality)
        
        self.keyframes_dir = os.path.join(output_dir, "keyframes")
        self.metadata_dir = os.path.join(output_dir, "metadata")
        
        os.makedirs(self.keyframes_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
    def process_video(self, video_path: str) -> bool:
        """
        Process a single video: shot boundary detection + keyframe extraction + save metadata.
        Returns True if successful, False if failed/skipped.
        """
        try:
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            metadata_path = os.path.join(self.metadata_dir, f"{video_id}.json")
            
            # Idempotency / Resume check
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        metadata = VideoMetadata.model_validate(data)
                    if not keyframe_artifacts_are_valid(
                        metadata,
                        self.output_dir,
                    ):
                        raise ValueError(
                            "One or more keyframe artifacts are missing or unreadable"
                        )
                    logger.info(f"Skipping {video_id} - already processed successfully.")
                    return True
                except Exception as e:
                    logger.warning(f"Metadata for {video_id} is corrupted or invalid, re-processing. Error: {e}")
            
            logger.info(f"Processing video: {video_id} ({video_path})")
            
            # Step 1: Detect Shots
            shots = self.transnet.predict_shots(video_path, threshold=self.threshold)
            
            # Step 2: Extract Keyframes
            shots_metadata, fps = self.extractor.extract_keyframes(
                video_path=video_path,
                video_id=video_id,
                shots=shots,
                output_dir=self.keyframes_dir
            )

            frame_count, width, height = self._probe_video(video_path)
            duration_sec = frame_count / fps
            source_video_rel_path = self._relative_source_path(video_path)
                
            # Step 3: Build and Validate Metadata
            video_meta = VideoMetadata(
                video_id=video_id,
                source_path=source_video_rel_path,
                source_video_rel_path=source_video_rel_path,
                fps=fps,
                duration_sec=duration_sec,
                frame_count=frame_count,
                width=width,
                height=height,
                num_shots=len(shots_metadata),
                shots=shots_metadata
            )
            
            # Step 4: Save Metadata
            temporary_path = f"{metadata_path}.tmp"
            try:
                with open(temporary_path, 'w', encoding='utf-8') as f:
                    f.write(video_meta.model_dump_json(indent=2))
                os.replace(temporary_path, metadata_path)
            finally:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)
                
            logger.info(f"Successfully processed video: {video_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process video {video_path}. Error: {e}", exc_info=True)
            return False

    def _relative_source_path(self, video_path: str) -> str:
        """Return the source path relative to the configured dataset root."""
        source = Path(video_path).resolve()
        if self.data_root is None:
            relative = Path(source.name)
        else:
            try:
                relative = source.relative_to(self.data_root)
            except ValueError as exc:
                raise ValueError(
                    f"Video path is outside data root: {video_path}"
                ) from exc
        return PurePosixPath(*relative.parts).as_posix()

    @staticmethod
    def _probe_video(video_path: str) -> tuple[int, int, int]:
        """Read stable source dimensions needed by the database contract."""
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Cannot probe video {video_path}")
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        if frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Invalid video properties for {video_path}")
        return frame_count, width, height
