import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


OBJECT_SCHEMA_VERSION = 1


def build_object_provenance(
    *,
    yolo_world_model: Optional[str],
    custom_vocab_file: Optional[str],
    co_detr_backbone: Optional[str],
    confidence_threshold: float,
    nms_threshold: float,
) -> Dict[str, Any]:
    """Build the detector/config contract persisted with object output."""
    return {
        "yolo_world_model": yolo_world_model,
        "custom_vocab_file": custom_vocab_file,
        "co_detr_backbone": co_detr_backbone,
        "confidence_threshold": float(confidence_threshold),
        "nms_threshold": float(nms_threshold),
    }


def is_valid_object_artifact(
    output_path: Path,
    video_id: str,
    expected_frame_ids: Sequence[str],
    expected_provenance: Dict[str, Any],
) -> bool:
    """Validate object output before allowing resume to skip inference."""
    try:
        with output_path.open("r", encoding="utf-8") as output_file:
            data = json.load(output_file)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False

    if not isinstance(data, dict):
        return False
    if data.get("schema_version") != OBJECT_SCHEMA_VERSION:
        return False
    if data.get("video_id") != video_id:
        return False
    if data.get("provenance") != expected_provenance:
        return False

    frames = data.get("frames")
    if not isinstance(frames, list):
        return False

    actual_frame_ids = []
    for frame in frames:
        if not isinstance(frame, dict):
            return False
        frame_id = frame.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            return False
        if not isinstance(frame.get("objects"), list):
            return False
        actual_frame_ids.append(frame_id)

    expected_ids = list(expected_frame_ids)
    if any(not isinstance(frame_id, str) or not frame_id for frame_id in expected_ids):
        return False
    if len(expected_ids) != len(set(expected_ids)):
        return False

    return actual_frame_ids == expected_ids
