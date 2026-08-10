import os
import json
import logging
import shutil
from pathlib import Path, PurePosixPath
from uuid import uuid4

import cv2

from .transnet_wrapper import TransNetPredictor
from .keyframe_extractor import KeyframeExtractor
from .metadata_schema import VideoMetadata
from .fingerprints import (
    build_processing_config_fingerprint,
    sha256_file,
)
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
            source_video_rel_path = self._relative_source_path(video_path)
            source_fingerprint = sha256_file(video_path)
            config_fingerprint = build_processing_config_fingerprint(
                threshold=self.threshold,
                positions=self.extractor.positions,
                webp_quality=self.extractor.webp_quality,
            )
            
            # Idempotency / Resume check
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        metadata = VideoMetadata.model_validate(data)
                    if not keyframe_artifacts_are_valid(
                        metadata,
                        self.output_dir,
                        expected_source_fingerprint=source_fingerprint,
                        expected_config_fingerprint=config_fingerprint,
                        expected_source_video_rel_path=source_video_rel_path,
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

            staging_root = Path(self.keyframes_dir) / (
                f".{video_id}.staging-{uuid4().hex}"
            )
            staging_root.mkdir(parents=True)
            try:
                # Step 2: Extract into an unpublished run-scoped directory.
                shots_metadata, fps = self.extractor.extract_keyframes(
                    video_path=video_path,
                    video_id=video_id,
                    shots=shots,
                    output_dir=str(staging_root),
                )

                frame_count, width, height = self._probe_video(video_path)
                duration_sec = frame_count / fps
                if sha256_file(video_path) != source_fingerprint:
                    raise RuntimeError(
                        f"Source video changed during processing: {video_path}"
                    )

                # Step 3: Build and validate metadata before publication.
                video_meta = VideoMetadata(
                    video_id=video_id,
                    source_path=source_video_rel_path,
                    source_video_rel_path=source_video_rel_path,
                    source_fingerprint=source_fingerprint,
                    producer_config_fingerprint=config_fingerprint,
                    fps=fps,
                    duration_sec=duration_sec,
                    frame_count=frame_count,
                    width=width,
                    height=height,
                    num_shots=len(shots_metadata),
                    shots=shots_metadata,
                )

                # Step 4: Publish keyframes and metadata as one transaction.
                self._publish_video_artifacts(
                    video_id=video_id,
                    staged_video_dir=staging_root / video_id,
                    metadata_path=Path(metadata_path),
                    metadata=video_meta,
                )
            finally:
                shutil.rmtree(staging_root, ignore_errors=True)
                
            logger.info(f"Successfully processed video: {video_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process video {video_path}. Error: {e}", exc_info=True)
            return False

    def _publish_video_artifacts(
        self,
        *,
        video_id: str,
        staged_video_dir: Path,
        metadata_path: Path,
        metadata: VideoMetadata,
    ) -> None:
        """Atomically replace one video's last-known-good Module 1 output."""
        if not staged_video_dir.is_dir():
            raise ValueError(
                f"Missing staged keyframe directory for {video_id}"
            )

        final_video_dir = Path(self.keyframes_dir) / video_id
        backup_dir = Path(self.keyframes_dir) / (
            f".{video_id}.backup-{uuid4().hex}"
        )
        temporary_metadata = metadata_path.with_name(
            f".{metadata_path.name}.tmp-{uuid4().hex}"
        )
        had_previous_keyframes = final_video_dir.exists()
        previous_moved = False
        staged_published = False
        try:
            temporary_metadata.write_text(
                metadata.model_dump_json(indent=2),
                encoding="utf-8",
            )
            if had_previous_keyframes and not final_video_dir.is_dir():
                raise ValueError(
                    "Expected keyframe directory, found file: "
                    f"{final_video_dir}"
                )
            if had_previous_keyframes:
                final_video_dir.replace(backup_dir)
                previous_moved = True
            staged_video_dir.replace(final_video_dir)
            staged_published = True
            os.replace(temporary_metadata, metadata_path)
        except Exception:
            if staged_published and final_video_dir.exists():
                shutil.rmtree(final_video_dir)
            if previous_moved and backup_dir.exists():
                backup_dir.replace(final_video_dir)
            raise
        else:
            if backup_dir.exists():
                try:
                    shutil.rmtree(backup_dir)
                except OSError as exc:
                    logger.warning(
                        "Published %s but could not remove backup %s: %s",
                        video_id,
                        backup_dir,
                        exc,
                    )
        finally:
            temporary_metadata.unlink(missing_ok=True)

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
