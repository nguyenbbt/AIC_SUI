import json
from pathlib import Path
from typing import Dict, Any, List

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
        
    frames = []
    if 'shots' in data:
        for shot in data['shots']:
            shot_id = shot.get('shot_id', 0)
            for kf in shot.get('keyframes', []):
                file_path = kf.get('file_path', '')
                if not file_path:
                    continue
                frame_id = Path(file_path).stem
                frames.append({
                    "frame_id": frame_id,
                    "shot_id": shot_id,
                    "position": kf.get('position', 0.0)
                })
        return frames

    if 'frames' in data:
        return data['frames']
        
    return []
