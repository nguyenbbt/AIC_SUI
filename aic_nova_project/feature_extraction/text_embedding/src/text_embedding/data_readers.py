import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def read_json_safe(file_path: Path) -> Optional[Any]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read JSON from {file_path}: {e}")
        return None

def parse_asr_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Parses ASR cleaned JSON.
    Expected schema: list of objects with 'interval_id', 'start_time', 'end_time', 'cleaned_text'
    """
    data = read_json_safe(file_path)
    if not isinstance(data, list):
        if data is not None:
            logger.warning(f"ASR file {file_path} is not a list.")
        return []
        
    records = []
    video_id = file_path.stem.replace("_cleaned", "") # handle _cleaned suffix
    for item in data:
        text = item.get("cleaned_text", "").strip()
        if not text:
            continue
            
        records.append({
            "video_id": video_id,
            "interval_id": item.get("interval_id", ""),
            "start_time_sec": float(item.get("start_time", 0.0)),
            "end_time_sec": float(item.get("end_time", 0.0)),
            "text": text
        })
    return records

def parse_summary_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Parses Summary JSON.
    Expected schema: object with 'summary' field.
    """
    data = read_json_safe(file_path)
    if not isinstance(data, dict):
        if data is not None:
            logger.warning(f"Summary file {file_path} is not a dict.")
        return []
        
    video_id = file_path.stem
    text = data.get("summary", "").strip()
    if not text:
        return []
        
    return [{
        "video_id": video_id,
        "text": text
    }]

def parse_ocr_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Parses OCR JSON.
    Expected schema: object with 'frames' list containing 'frame_id', 'shot_id', 'ocr_text_concat'
    """
    data = read_json_safe(file_path)
    if not isinstance(data, dict):
        if data is not None:
            logger.warning(f"OCR file {file_path} is not a dict.")
        return []
        
    video_id = file_path.stem
    frames = data.get("frames", [])
    
    records = []
    for frame in frames:
        text = frame.get("ocr_text_concat", "").strip()
        if not text:
            continue
            
        records.append({
            "video_id": video_id,
            "frame_id": frame.get("frame_id", ""),
            "shot_id": frame.get("shot_id", ""),
            "text": text
        })
    return records
