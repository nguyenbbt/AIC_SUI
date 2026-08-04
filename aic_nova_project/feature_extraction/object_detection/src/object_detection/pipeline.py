import os
import json
import logging
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional

from .metadata_reader import read_metadata
from .box_fusion import apply_nms
from .detectors import YOLOWorldDetector, CoDETRDetector
from .resume_validation import (
    OBJECT_SCHEMA_VERSION,
    build_object_provenance,
    is_valid_object_artifact,
)

logger = logging.getLogger(__name__)


def _resolve_image_path(
    keyframe_dir: Path,
    frame_metadata: Dict[str, Any],
) -> Path:
    """Resolve the image path without treating the global frame ID as a filename."""
    file_path = frame_metadata.get("file_path")
    if not file_path:
        return keyframe_dir / f"{frame_metadata['frame_id']}.webp"

    metadata_path = Path(str(file_path))
    if metadata_path.is_absolute():
        return metadata_path

    return keyframe_dir / metadata_path.name


class ObjectDetectionPipeline:
    def __init__(
        self,
        yolo_world_model: Optional[str] = None,
        custom_vocab_file: Optional[str] = None,
        co_detr_backbone: Optional[str] = None,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.5,
        device: str = "cuda"
    ):
        self.detectors = []
        self.nms_threshold = nms_threshold
        self.provenance = build_object_provenance(
            yolo_world_model=yolo_world_model,
            custom_vocab_file=custom_vocab_file,
            co_detr_backbone=co_detr_backbone,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
        )
        
        if yolo_world_model:
            logger.info("Initializing YOLO-World detector...")
            self.detectors.append(
                YOLOWorldDetector(
                    model_path=yolo_world_model,
                    custom_vocab_file=custom_vocab_file,
                    confidence_threshold=confidence_threshold,
                    device=device
                )
            )
            
        if co_detr_backbone:
            logger.info(f"Initializing Co-DETR detector (backbone: {co_detr_backbone})...")
            self.detectors.append(
                CoDETRDetector(
                    backbone=co_detr_backbone,
                    confidence_threshold=confidence_threshold,
                    device=device
                )
            )
            
        if not self.detectors:
            raise ValueError("No detectors enabled. Please enable at least one detector.")

    def _process_batch_safe(self, images: List[Image.Image], batch_size: int) -> List[List[Dict[str, Any]]]:
        """Process batch with automatic OOM handling and batch size reduction."""
        if not images:
            return []
            
        try:
            batch_results = [[] for _ in range(len(images))]
            
            for detector in self.detectors:
                # Process with the current detector
                detector_results = detector.detect_batch(images)
                if len(detector_results) != len(images):
                    raise RuntimeError(
                        f"{detector.__class__.__name__} returned "
                        f"{len(detector_results)} results for "
                        f"{len(images)} images."
                    )
                
                # Merge results per image
                for i, res in enumerate(detector_results):
                    batch_results[i].extend(res)
                    
            return batch_results
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch_size > 1:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                new_batch_size = max(1, batch_size // 2)
                logger.warning(f"CUDA OOM detected. Reducing batch size from {batch_size} to {new_batch_size}")
                
                # Split current batch into two smaller batches and recurse
                mid = len(images) // 2
                res1 = self._process_batch_safe(images[:mid], new_batch_size)
                res2 = self._process_batch_safe(images[mid:], new_batch_size)
                return res1 + res2
            else:
                raise e

    def process_video(
        self,
        video_id: str,
        metadata_path: Path,
        keyframe_dir: Path,
        output_path: Path,
        batch_size: int = 16,
        force: bool = False
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        frames = read_metadata(metadata_path)
        expected_frame_ids = [
            str(frame.get("frame_id", ""))
            for frame in frames
        ]
        if output_path.exists() and not force:
            if is_valid_object_artifact(
                output_path,
                video_id,
                expected_frame_ids,
                self.provenance,
            ):
                logger.info(
                    "Valid output for %s already exists. Skipping.",
                    video_id,
                )
                return
            logger.warning(
                "Existing output for %s is stale, corrupt, or partial; "
                "regenerating it.",
                video_id,
            )

        output_data = {
            "schema_version": OBJECT_SCHEMA_VERSION,
            "video_id": video_id,
            "provenance": self.provenance,
            "frames": []
        }
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            images = []
            valid_indices = []
            try:
                for idx, frame_meta in enumerate(batch_frames):
                    image_path = _resolve_image_path(
                        keyframe_dir,
                        frame_meta,
                    )

                    if not image_path.exists():
                        raise FileNotFoundError(
                            f"Image not found: {image_path}"
                        )
                    try:
                        with Image.open(image_path) as source_image:
                            image = source_image.convert("RGB")
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to load image {image_path}: {exc}"
                        ) from exc
                    images.append(image)
                    valid_indices.append(idx)

                batch_results = self._process_batch_safe(
                    images,
                    batch_size,
                )
                if len(batch_results) != len(images):
                    raise RuntimeError(
                        f"Detector batch returned {len(batch_results)} "
                        f"results for {len(images)} images."
                    )

                for res_idx, img_results in enumerate(batch_results):
                    orig_idx = valid_indices[res_idx]
                    frame_meta = batch_frames[orig_idx]

                    # Apply Box Fusion (NMS)
                    final_objects = apply_nms(
                        img_results,
                        nms_threshold=self.nms_threshold,
                    )

                    output_data["frames"].append({
                        "frame_id": frame_meta["frame_id"],
                        "shot_id": frame_meta["shot_id"],
                        "position": frame_meta["position"],
                        "objects": final_objects
                    })
            finally:
                for image in images:
                    image.close()

        if len(output_data["frames"]) != len(frames):
            raise RuntimeError(
                f"Object completeness check failed for {video_id}: "
                f"processed {len(output_data['frames'])}/{len(frames)} frames."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            temp_path.replace(output_path)
        finally:
            temp_path.unlink(missing_ok=True)
        
        logger.info(f"Processed {video_id}. Saved to {output_path}")
