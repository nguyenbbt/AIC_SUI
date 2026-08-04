import json
from pathlib import Path
from typing import Any, Dict, List


def _generate_frame_id(video_id: str, shot_id: int, position: float) -> str:
    """Generate the canonical global frame ID used across offline artifacts."""
    position_code = f"{int(position * 100):03d}"
    return f"{video_id}_{shot_id:05d}_{position_code}"


def get_keyframes_from_metadata(metadata_path: Path) -> List[Dict[str, Any]]:
    """
    Reads the metadata JSON and returns the list of keyframes.
    
    Args:
        metadata_path (Path): Path to the metadata json file.
        
    Returns:
        List[Dict[str, Any]]: List of frame dictionaries containing at least frame_id and position.
    """
    if not metadata_path.exists():
        return []
        
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    frames: List[Dict[str, Any]] = []
    shots = data.get("shots", [])
    if shots:
        video_id = str(data["video_id"])
        for shot in shots:
            shot_id = int(shot.get("shot_id", 0))
            for kf in shot.get("keyframes", []):
                file_path = str(kf.get("file_path", ""))
                if not file_path:
                    continue
                position = float(kf.get("position", 0.0))
                frames.append(
                    {
                        "frame_id": _generate_frame_id(
                            video_id,
                            shot_id,
                            position,
                        ),
                        "shot_id": shot_id,
                        "position": position,
                        "file_path": file_path,
                    }
                )
        return frames

    if "frames" in data:
        return data["frames"]
        
    return []
