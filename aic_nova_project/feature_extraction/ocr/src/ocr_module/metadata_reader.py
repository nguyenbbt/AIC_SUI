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
        
    # According to Module 1 schema, metadata usually contains 'frames' or similar list
    # Assuming standard schema based on AI Challenge metadata
    if 'frames' in data:
        return data['frames']
        
    return []
