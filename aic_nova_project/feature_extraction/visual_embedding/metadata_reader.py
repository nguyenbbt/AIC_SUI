import json
import os
from pathlib import Path
from typing import List, Dict, Any

def generate_frame_id(video_id: str, shot_id: int, position: float) -> str:
    """
    Generate a unique frame ID.
    Example: V001_00000_015 for position 0.15
    """
    # Use 3 digits for position after multiplying by 100.
    pos_str = f"{int(position * 100):03d}"
    return f"{video_id}_{shot_id:05d}_{pos_str}"

def read_metadata(metadata_dir: str, keyframe_base_dir: str) -> List[Dict[str, Any]]:
    """
    Reads all metadata JSON files in the metadata directory.
    
    Args:
        metadata_dir: Path to directory containing .json files.
        keyframe_base_dir: Base path to resolve relative file_paths.
        
    Returns:
        List of dictionaries, each representing a keyframe to encode.
    """
    metadata_path = Path(metadata_dir)
    keyframe_base = Path(keyframe_base_dir)
    
    records = []
    
    if not metadata_path.exists() or not metadata_path.is_dir():
        return records
        
    for json_file in metadata_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            video_id = data.get("video_id")
            if not video_id:
                continue
                
            shots = data.get("shots", [])
            for shot in shots:
                shot_id = shot.get("shot_id")
                keyframes = shot.get("keyframes", [])
                
                for kf in keyframes:
                    position = kf.get("position")
                    rel_path = kf.get("file_path")
                    
                    if position is None or not rel_path:
                        continue
                        
                    frame_id = generate_frame_id(video_id, shot_id, position)
                    
                    # Resolve absolute/full path
                    # Some paths might start with "keyframes/", so we might need to handle it.
                    # Since the prompt says "file_path": "keyframes/V001/..." we'll just join.
                    # Usually keyframe_base_dir is just "data/".
                    # Let's ensure it exists.
                    abs_path = str(keyframe_base / rel_path)
                    
                    records.append({
                        "frame_id": frame_id,
                        "video_id": video_id,
                        "shot_id": shot_id,
                        "position": position,
                        "file_path": abs_path
                    })
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            
    return records
