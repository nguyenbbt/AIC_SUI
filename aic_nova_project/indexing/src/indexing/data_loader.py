"""
Data loader for reading and parsing all output artifacts from Module 1-6.

Reads:
- Visual embeddings (Parquet) from Module 2
- Text ASR/Summary/OCR embeddings (Parquet) from Module 6
- Metadata (JSON) from Module 1
- OCR raw text (JSON) from Module 4
- Object Detection (JSON) from Module 5

Detects vector dimensions dynamically from the first Parquet file found.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def read_json_safe(file_path: Path) -> Optional[Any]:
    """Safely read and parse a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read JSON from {file_path}: {e}")
        return None


def detect_embedding_dim(parquet_dir: Path) -> Optional[int]:
    """
    Detect embedding dimension dynamically by reading the first Parquet file
    in the given directory and inspecting the shape of the 'embedding' column.

    Returns None if no Parquet files are found or if the column is missing.
    """
    parquet_files = list(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning(f"No Parquet files found in {parquet_dir}")
        return None

    df = pd.read_parquet(parquet_files[0])
    if "embedding" not in df.columns or df.empty:
        logger.warning(f"No 'embedding' column or empty data in {parquet_files[0]}")
        return None

    first_embedding = df["embedding"].iloc[0]
    dim = len(first_embedding)
    logger.info(f"Detected embedding dim={dim} from {parquet_files[0].name}")
    return dim


def discover_video_ids(data_dir: Path) -> List[str]:
    """
    Discover all unique video_ids from metadata JSON files.
    Falls back to scanning other directories if metadata is empty.
    """
    metadata_dir = data_dir / "metadata"
    video_ids = set()

    if metadata_dir.exists():
        for f in metadata_dir.glob("*.json"):
            video_ids.add(f.stem)

    # Fallback: also scan OCR dir
    ocr_dir = data_dir / "ocr"
    if ocr_dir.exists():
        for f in ocr_dir.glob("*.json"):
            video_ids.add(f.stem)

    result = sorted(video_ids)
    logger.info(f"Discovered {len(result)} video(s) to process.")
    return result


def load_visual_embeddings(
    data_dir: Path, video_id: str
) -> List[Dict[str, Any]]:
    """Load visual embedding records for a video from Parquet."""
    # Visual embeddings are written by Module 2 into embeddings/visual/ or similar
    # Try multiple possible paths
    possible_paths = [
        data_dir / "embeddings" / "visual" / f"{video_id}.parquet",
        data_dir / "embeddings" / f"{video_id}.parquet",
    ]

    for path in possible_paths:
        if path.exists():
            df = pd.read_parquet(path)
            records = []
            for _, row in df.iterrows():
                embedding = row.get("embedding")
                if embedding is None:
                    continue
                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()

                records.append({
                    "frame_id": str(row.get("frame_id", "")),
                    "video_id": str(row.get("video_id", video_id)),
                    "shot_id": int(row.get("shot_id", 0)),
                    "embedding": embedding,
                })
            return records

    return []


def load_text_asr_embeddings(
    data_dir: Path, video_id: str
) -> List[Dict[str, Any]]:
    """Load ASR text embedding records for a video from Parquet."""
    path = data_dir / "embeddings" / "text_asr" / f"{video_id}.parquet"
    if not path.exists():
        return []

    df = pd.read_parquet(path)
    records = []
    for _, row in df.iterrows():
        embedding = row.get("embedding")
        if embedding is None:
            continue
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        records.append({
            "video_id": str(row.get("video_id", video_id)),
            "interval_id": str(row.get("interval_id", "")),
            "start_time_sec": float(row.get("start_time_sec", 0.0)),
            "end_time_sec": float(row.get("end_time_sec", 0.0)),
            "embedding": embedding,
        })
    return records


def load_text_summary_embeddings(
    data_dir: Path, video_id: str
) -> List[Dict[str, Any]]:
    """Load Summary text embedding records for a video from Parquet."""
    path = data_dir / "embeddings" / "text_summary" / f"{video_id}.parquet"
    if not path.exists():
        return []

    df = pd.read_parquet(path)
    records = []
    for _, row in df.iterrows():
        embedding = row.get("embedding")
        if embedding is None:
            continue
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        records.append({
            "video_id": str(row.get("video_id", video_id)),
            "embedding": embedding,
        })
    return records


def load_ocr_texts(data_dir: Path, video_id: str) -> List[Dict[str, Any]]:
    """Load raw OCR text records for Elasticsearch indexing."""
    path = data_dir / "ocr" / f"{video_id}.json"
    data = read_json_safe(path) if path.exists() else None
    if not isinstance(data, dict):
        return []

    records = []
    for frame in data.get("frames", []):
        text = frame.get("ocr_text_concat", "").strip()
        if not text:
            continue  # Graceful degradation: skip frames without OCR text

        records.append({
            "frame_id": str(frame.get("frame_id", "")),
            "video_id": video_id,
            "shot_id": str(frame.get("shot_id", "")),
            "ocr_text_concat": text,
        })
    return records


def load_asr_transcripts(data_dir: Path, video_id: str) -> List[Dict[str, Any]]:
    """Load cleaned ASR transcript records for Elasticsearch indexing."""
    # Try both with and without _cleaned suffix
    possible_paths = [
        data_dir / "transcripts" / f"{video_id}_cleaned.json",
        data_dir / "transcripts" / f"{video_id}.json",
    ]

    data = None
    for path in possible_paths:
        if path.exists():
            data = read_json_safe(path)
            break

    if not isinstance(data, list):
        return []

    records = []
    for item in data:
        text = item.get("cleaned_text", "").strip()
        if not text:
            continue

        records.append({
            "interval_id": str(item.get("interval_id", "")),
            "video_id": video_id,
            "start_time": float(item.get("start_time", 0.0)),
            "end_time": float(item.get("end_time", 0.0)),
            "cleaned_text": text,
        })
    return records


def load_video_summary(data_dir: Path, video_id: str) -> List[Dict[str, Any]]:
    """Load video summary for Elasticsearch indexing."""
    path = data_dir / "summaries" / f"{video_id}.json"
    data = read_json_safe(path) if path.exists() else None
    if not isinstance(data, dict):
        return []

    text = data.get("summary", "").strip()
    if not text:
        return []

    return [{"video_id": video_id, "summary": text}]


def load_metadata_and_objects(
    data_dir: Path, video_id: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load metadata (frame info) and object detection results for SQLite.

    Returns:
        (metadata_records, object_records)
    """
    # --- Metadata from Module 1 ---
    meta_path = data_dir / "metadata" / f"{video_id}.json"
    meta_data = read_json_safe(meta_path) if meta_path.exists() else None

    metadata_records = []
    if isinstance(meta_data, dict):
        for shot in meta_data.get("shots", []):
            shot_id = shot.get("shot_id", 0)
            for kf in shot.get("keyframes", []):
                frame_id = kf.get("file_path", "")
                # Extract frame_id from file_path: "keyframes/VIDEO_ID/shot_00000_pos_015.webp"
                if "/" in frame_id:
                    frame_id = Path(frame_id).stem

                metadata_records.append({
                    "frame_id": frame_id,
                    "video_id": video_id,
                    "shot_id": int(shot_id),
                    "timestamp": float(kf.get("time_sec", 0.0)),
                })

    # --- Object Detection from Module 5 ---
    # Object detection JSON follows same schema as OCR: {video_id, frames: [{frame_id, objects: [...]}]}
    obj_dir = data_dir / "object_detection"
    obj_path = obj_dir / f"{video_id}.json"
    obj_data = read_json_safe(obj_path) if obj_path.exists() else None

    object_records = []
    if isinstance(obj_data, dict):
        for frame in obj_data.get("frames", []):
            frame_id = str(frame.get("frame_id", ""))
            for obj in frame.get("objects", []):
                bbox = obj.get("bbox", obj.get("box", [0, 0, 0, 0]))
                object_records.append({
                    "frame_id": frame_id,
                    "label": str(obj.get("label", "unknown")).lower(),
                    "confidence": float(obj.get("confidence", obj.get("score", 0.0))),
                    "x_min": float(bbox[0]) if len(bbox) > 0 else 0.0,
                    "y_min": float(bbox[1]) if len(bbox) > 1 else 0.0,
                    "x_max": float(bbox[2]) if len(bbox) > 2 else 0.0,
                    "y_max": float(bbox[3]) if len(bbox) > 3 else 0.0,
                    "model_source": str(obj.get("model_source", "")),
                })

    return metadata_records, object_records
