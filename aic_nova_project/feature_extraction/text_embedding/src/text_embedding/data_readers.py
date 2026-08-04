import json
import logging
import math
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

    The canonical Module 3 artifact is an object envelope containing ``video_id``
    and an ``intervals`` list. Interval timestamps use ``start_time_sec`` and
    ``end_time_sec``. ``interval_id`` is normalized to a decimal string so the
    same identifier can be stored in Parquet, Milvus, and Elasticsearch.
    """
    data = read_json_safe(file_path)
    if not isinstance(data, dict):
        if data is not None:
            logger.warning(f"ASR file {file_path} must contain an object envelope.")
        return []

    intervals = data.get("intervals")
    if not isinstance(intervals, list):
        logger.warning(f"ASR file {file_path} has no valid 'intervals' list.")
        return []

    video_id = data.get("video_id")
    if not isinstance(video_id, str) or not video_id.strip():
        logger.warning(f"ASR file {file_path} has no valid 'video_id'.")
        return []
    video_id = video_id.strip()

    records = []
    for index, item in enumerate(intervals):
        if not isinstance(item, dict):
            logger.warning(
                f"Skipping non-object ASR interval at index {index} in {file_path}."
            )
            continue

        text_value = item.get("cleaned_text", "")
        if not isinstance(text_value, str):
            logger.warning(
                f"Skipping ASR interval {index} with invalid cleaned_text in {file_path}."
            )
            continue
        text = text_value.strip()
        if not text:
            continue

        raw_interval_id = item.get("interval_id")
        if isinstance(raw_interval_id, bool):
            interval_id = None
        elif isinstance(raw_interval_id, int) and raw_interval_id >= 0:
            interval_id = str(raw_interval_id)
        elif (
            isinstance(raw_interval_id, str)
            and raw_interval_id.isascii()
            and raw_interval_id.isdigit()
        ):
            interval_id = str(int(raw_interval_id))
        else:
            interval_id = None

        if interval_id is None:
            logger.warning(
                f"Skipping ASR interval {index} with invalid interval_id in {file_path}."
            )
            continue

        start_value = item.get("start_time_sec")
        end_value = item.get("end_time_sec")
        if (
            isinstance(start_value, bool)
            or not isinstance(start_value, (int, float))
            or isinstance(end_value, bool)
            or not isinstance(end_value, (int, float))
        ):
            logger.warning(
                f"Skipping ASR interval {interval_id} with invalid timestamps in {file_path}."
            )
            continue

        start_time_sec = float(start_value)
        end_time_sec = float(end_value)
        if (
            not math.isfinite(start_time_sec)
            or not math.isfinite(end_time_sec)
            or start_time_sec < 0.0
            or end_time_sec < start_time_sec
        ):
            logger.warning(
                f"Skipping ASR interval {interval_id} with invalid time range in {file_path}."
            )
            continue

        records.append({
            "video_id": video_id,
            "interval_id": interval_id,
            "start_time_sec": start_time_sec,
            "end_time_sec": end_time_sec,
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
        
    video_id_value = data.get("video_id")
    if not isinstance(video_id_value, str) or not video_id_value.strip():
        logger.warning(f"Summary file {file_path} has no valid 'video_id'.")
        return []
    video_id = video_id_value.strip()

    summary_value = data.get("summary")
    if not isinstance(summary_value, str):
        logger.warning(f"Summary file {file_path} has no valid 'summary'.")
        return []
    text = summary_value.strip()
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
        
    video_id_value = data.get("video_id")
    if not isinstance(video_id_value, str) or not video_id_value.strip():
        logger.warning(f"OCR file {file_path} has no valid 'video_id'.")
        return []
    video_id = video_id_value.strip()

    frames = data.get("frames", [])
    if not isinstance(frames, list):
        logger.warning(f"OCR file {file_path} has no valid 'frames' list.")
        return []
    
    records = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        text_value = frame.get("ocr_text_concat")
        if not isinstance(text_value, str):
            continue
        text = text_value.strip()
        if not text:
            continue
            
        records.append({
            "video_id": video_id,
            "frame_id": frame.get("frame_id", ""),
            "shot_id": frame.get("shot_id", ""),
            "text": text
        })
    return records
