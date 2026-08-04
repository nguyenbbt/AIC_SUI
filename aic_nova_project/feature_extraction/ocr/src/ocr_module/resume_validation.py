import json
from pathlib import Path
from typing import Any, Dict, Sequence


OCR_SCHEMA_VERSION = 1


def build_ocr_provenance(
    *,
    backbone: str,
    confidence_threshold: float,
    width_ths: float,
    mag_ratio: float,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Build the deterministic model/config contract stored with OCR output."""
    return {
        "detector_model": "easyocr/CRAFT-vi",
        "recognizer_backbone": backbone,
        "confidence_threshold": float(confidence_threshold),
        "width_ths": float(width_ths),
        "mag_ratio": float(mag_ratio),
        "recognition_batch_size": int(batch_size),
    }


def is_valid_ocr_artifact(
    output_path: Path,
    video_id: str,
    expected_frame_ids: Sequence[str],
    expected_provenance: Dict[str, Any],
) -> bool:
    """Validate that an OCR artifact is readable, current, and complete."""
    try:
        with output_path.open("r", encoding="utf-8") as output_file:
            data = json.load(output_file)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False

    if not isinstance(data, dict):
        return False
    if data.get("schema_version") != OCR_SCHEMA_VERSION:
        return False
    if data.get("video_id") != video_id:
        return False
    if data.get("provenance") != expected_provenance:
        return False

    frames = data.get("frames")
    if not isinstance(frames, list):
        return False

    actual_frame_ids = [
        frame.get("frame_id")
        for frame in frames
        if isinstance(frame, dict)
    ]
    expected_ids = list(expected_frame_ids)
    if len(actual_frame_ids) != len(frames):
        return False
    if any(not isinstance(frame_id, str) or not frame_id for frame_id in expected_ids):
        return False
    if len(set(expected_ids)) != len(expected_ids):
        return False

    return actual_frame_ids == expected_ids
