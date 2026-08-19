import json
import logging
import cv2
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from .detector import TextDetector
from .recognizer import TextRecognizer
from .region_cropper import crop_polygon
from .text_ordering import group_and_order_regions, concat_text
from .metadata_reader import get_keyframes_from_metadata
from .resume_validation import (
    OCR_SCHEMA_VERSION,
    build_ocr_provenance,
    is_valid_ocr_artifact,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _resolve_image_path(
    video_keyframes_dir: Path,
    frame_metadata: Dict[str, Any],
) -> Path:
    """Resolve the local keyframe path separately from its global frame ID."""
    file_path = frame_metadata.get("file_path")
    if not file_path:
        return video_keyframes_dir / f"{frame_metadata['frame_id']}.webp"

    metadata_path = Path(str(file_path))
    if metadata_path.is_absolute():
        return metadata_path

    return video_keyframes_dir / metadata_path.name


class OCRPipeline:
    def __init__(self, use_gpu: bool = True, backbone: str = 'vgg_transformer', 
                 confidence_threshold: float = 0.4):
        self.detector = TextDetector(use_gpu=use_gpu)
        self.recognizer = TextRecognizer(backbone=backbone, use_gpu=use_gpu)
        self.backbone = backbone
        self.confidence_threshold = confidence_threshold

    def process_video(self, video_id: str, keyframe_dir: Path, metadata_dir: Path, 
                      output_dir: Path, width_ths: float = 0.7, mag_ratio: float = 1.5,
                      force: bool = False, batch_size: int = 1):
        """Processes all keyframes for a single video."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        output_path = output_dir / f"{video_id}.json"

        metadata_path = metadata_dir / f"{video_id}.json"
        frames = get_keyframes_from_metadata(metadata_path)
        
        if not frames:
            logger.warning(f"No metadata frames found for {video_id}.")
            return

        provenance = build_ocr_provenance(
            backbone=self.backbone,
            confidence_threshold=self.confidence_threshold,
            width_ths=width_ths,
            mag_ratio=mag_ratio,
            batch_size=batch_size,
        )
        expected_frame_ids = [
            str(frame.get("frame_id", ""))
            for frame in frames
        ]
        if output_path.exists() and not force:
            if is_valid_ocr_artifact(
                output_path,
                video_id,
                expected_frame_ids,
                provenance,
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

        video_keyframes_dir = keyframe_dir / video_id
        if not video_keyframes_dir.exists():
            raise FileNotFoundError(
                f"Keyframe directory not found: {video_keyframes_dir}"
            )
            
        result_frames = []
        
        # We can implement batching later if needed, currently iterating through frames
        for frame in tqdm(frames, desc=f"Processing {video_id}"):
            frame_id = frame.get("frame_id")
            if not frame_id:
                raise ValueError(
                    f"Metadata contains a frame without frame_id for {video_id}."
                )
                
            image_path = _resolve_image_path(video_keyframes_dir, frame)
            if not image_path.exists():
                message = f"Image not found: {image_path}"
                logger.error(message)
                raise FileNotFoundError(message)
                
            # Read image using OpenCV
            image = cv2.imread(str(image_path))
            if image is None:
                message = f"Failed to load image: {image_path}"
                logger.error(message)
                raise RuntimeError(message)
                
            # 1. Detection
            polygons = self.detector.detect(image, width_ths=width_ths, mag_ratio=mag_ratio)
            
            region_inputs = []
            for poly in polygons:
                # 2. Crop
                cropped_img = crop_polygon(image, poly)
                
                # Check if cropped image is valid (not 1x1 fallback)
                if cropped_img.width <= 1 or cropped_img.height <= 1:
                    continue

                region_inputs.append((poly, cropped_img))

            ocr_regions = []
            for start in range(0, len(region_inputs), batch_size):
                region_batch = region_inputs[start:start + batch_size]
                predictions = self.recognizer.recognize_batch(
                    [cropped for _, cropped in region_batch]
                )
                if len(predictions) != len(region_batch):
                    raise RuntimeError(
                        "OCR recognizer returned an incomplete batch for "
                        f"{frame_id}."
                    )

                # 4. Filtering
                for (poly, _), (text, conf) in zip(
                    region_batch,
                    predictions,
                ):
                    if conf >= self.confidence_threshold and text.strip():
                        ocr_regions.append({
                            "bbox": poly,
                            "text": text.strip(),
                            "confidence": round(conf, 4)
                        })
            
            # 5. Ordering
            ordered_regions = group_and_order_regions(ocr_regions)
            concat_txt = concat_text(ordered_regions)
            
            # Append result
            frame_result = {
                "frame_id": frame_id,
                "shot_id": frame.get("shot_id", 0),
                "position": frame.get("position", 0.0),
                "ocr_regions": ordered_regions,
                "ocr_text_concat": concat_txt
            }
            result_frames.append(frame_result)
            
        if len(result_frames) != len(frames):
            raise RuntimeError(
                f"OCR completeness check failed for {video_id}: "
                f"processed {len(result_frames)}/{len(frames)} frames."
            )

        # Commit output only after every expected frame has been processed.
        output_dir.mkdir(parents=True, exist_ok=True)
        result_json = {
            "schema_version": OCR_SCHEMA_VERSION,
            "video_id": video_id,
            "provenance": provenance,
            "frames": result_frames
        }

        temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(result_json, f, ensure_ascii=False, indent=2)
            temp_path.replace(output_path)
        finally:
            temp_path.unlink(missing_ok=True)
            
        logger.info(f"Successfully processed {video_id}. Saved to {output_path}")

def run_pipeline(keyframe_dir: str, metadata_dir: str, output_dir: str, 
                 width_ths: float = 0.7, mag_ratio: float = 1.5,
                 confidence_threshold: float = 0.4, backbone: str = 'vgg_transformer',
                 use_gpu: bool = True, force: bool = False, workers: int = 1,
                 batch_size: int = 1, shard_index: int = 0,
                 shard_count: int = 1):
    """Entry point for CLI or multiprocessing wrapper."""
    if workers <= 0:
        raise ValueError("workers must be greater than zero.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if use_gpu and workers > 1:
        raise ValueError(
            "GPU OCR supports only one video worker; use --workers 1 "
            "and increase --batch-size instead."
        )
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1.")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            "shard_index must be between 0 and shard_count - 1."
        )

    k_dir = Path(keyframe_dir)
    m_dir = Path(metadata_dir)
    o_dir = Path(output_dir)

    # Identify videos to process based on metadata files
    all_video_ids = sorted(p.stem for p in m_dir.glob("*.json"))
    video_ids = all_video_ids[shard_index::shard_count]
    logger.info(
        "Selected %d/%d videos for OCR shard %d/%d.",
        len(video_ids),
        len(all_video_ids),
        shard_index,
        shard_count,
    )

    def process_with(pipeline: OCRPipeline, video_id: str) -> None:
        pipeline.process_video(
            video_id=video_id,
            keyframe_dir=k_dir,
            metadata_dir=m_dir,
            output_dir=o_dir,
            width_ths=width_ths,
            mag_ratio=mag_ratio,
            force=force,
            batch_size=batch_size,
        )

    if workers == 1:
        pipeline = OCRPipeline(
            use_gpu=use_gpu,
            backbone=backbone,
            confidence_threshold=confidence_threshold,
        )
        for video_id in video_ids:
            process_with(pipeline, video_id)
        return

    worker_state = threading.local()

    def process_in_worker(video_id: str) -> None:
        pipeline = getattr(worker_state, "pipeline", None)
        if pipeline is None:
            pipeline = OCRPipeline(
                use_gpu=False,
                backbone=backbone,
                confidence_threshold=confidence_threshold,
            )
            worker_state.pipeline = pipeline
        process_with(pipeline, video_id)

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_in_worker, video_id): video_id
            for video_id in video_ids
        }
        for future in as_completed(futures):
            video_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                logger.exception("OCR failed for %s", video_id)
                failures.append(f"{video_id}: {type(exc).__name__}: {exc}")

    if failures:
        raise RuntimeError(
            "OCR pipeline failed for one or more videos: "
            + "; ".join(failures)
        )
