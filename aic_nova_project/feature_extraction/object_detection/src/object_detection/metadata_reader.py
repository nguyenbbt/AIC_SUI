import json
from pathlib import Path
from typing import List, Dict, Any

def read_metadata(metadata_path: Path) -> List[Dict[str, Any]]:
    """
    Read metadata JSON and extract keyframe information.
    
    Args:
        metadata_path: Path to the JSON metadata file.
        
    Returns:
        A list of dictionaries, each containing:
        - frame_id: str
        - shot_id: int
        - position: float
        
    Raises:
        FileNotFoundError: If the metadata file does not exist.
        ValueError: If the metadata is invalid.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        frames = data.get("frames", [])
        if not frames:
            raise ValueError("No frames found in metadata")
            
        return frames
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
