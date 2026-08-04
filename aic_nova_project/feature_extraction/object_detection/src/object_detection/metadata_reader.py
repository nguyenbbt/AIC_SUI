import json
from pathlib import Path
from typing import Any, Dict, List


def _generate_frame_id(video_id: str, shot_id: int, position: float) -> str:
    """Generate the canonical global frame ID used across offline artifacts."""
    position_code = f"{int(position * 100):03d}"
    return f"{video_id}_{shot_id:05d}_{position_code}"


def _flatten_module1_keyframes(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert Module 1 shot metadata into Module 5 frame records."""
    frames: List[Dict[str, Any]] = []
    shots = data.get("shots", [])
    if not shots:
        return frames

    video_id = str(data["video_id"])

    for shot in shots:
        shot_id = int(shot["shot_id"])
        for keyframe in shot.get("keyframes", []):
            position = float(keyframe["position"])
            file_path = str(keyframe["file_path"])
            frames.append(
                {
                    "frame_id": _generate_frame_id(video_id, shot_id, position),
                    "shot_id": shot_id,
                    "position": position,
                    "file_path": file_path,
                }
            )

    return frames


def read_metadata(metadata_path: Path) -> List[Dict[str, Any]]:
    """
    Read Module 1 metadata JSON and extract keyframe information.
    
    Args:
        metadata_path: Path to the JSON metadata file.
        
    Returns:
        A list of dictionaries, each containing:
        - frame_id: str
        - shot_id: int
        - position: float
        - file_path: str
        
    Raises:
        FileNotFoundError: If the metadata file does not exist.
        ValueError: If the metadata is invalid.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        frames = _flatten_module1_keyframes(data)
        if not frames:
            frames = data.get("frames", [])
        if not frames:
            raise ValueError("No frames found in metadata")
            
        return frames
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}") from e
