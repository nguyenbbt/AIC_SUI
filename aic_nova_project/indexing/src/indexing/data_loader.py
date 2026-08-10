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
import math
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _relative_posix_path(value: Any, field_name: str) -> str:
    """Validate one artifact path relative to the configured dataset root."""
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{field_name} must be a normalized POSIX relative path")
    parsed = urlsplit(value)
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        parsed.scheme
        or parsed.netloc
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
    ):
        raise ValueError(f"{field_name} must be a safe relative path")
    return value


def _finite_float(value: Any, field_name: str) -> float:
    """Return a finite numeric value without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _positive_int(value: Any, field_name: str) -> int:
    """Return a strictly positive JSON integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def normalize_frame_id(raw_frame_id: str, video_id: str) -> str:
    """
    Normalize frame_id to the canonical Global ID format: {video_id}_{shot_id}_{position}.

    Handles multiple input formats:
      - 'shot_00000_pos_015' → '{video_id}_00000_015'
      - Already normalized (contains video_id) → returned as-is

    This ensures all 3 databases (Milvus, ES, SQLite) store exactly the same
    frame_id format, enabling JOIN operations across DBs.
    """
    if not isinstance(raw_frame_id, str) or not raw_frame_id:
        raise ValueError("Invalid frame_id: expected a non-empty string.")
    if not isinstance(video_id, str) or not video_id:
        raise ValueError("Invalid video_id: expected a non-empty string.")

    global_pattern = re.compile(
        rf"{re.escape(video_id)}_[0-9]{{5}}_[0-9]{{3}}"
    )
    if global_pattern.fullmatch(raw_frame_id):
        return raw_frame_id

    match = re.fullmatch(
        r"shot_([0-9]{5})_pos_([0-9]{3})",
        raw_frame_id,
    )
    if match:
        shot_str = match.group(1)   # e.g. '00000'
        pos_str = match.group(2)    # e.g. '015'
        return f"{video_id}_{shot_str}_{pos_str}"

    raise ValueError(
        f"Invalid frame_id '{raw_frame_id}' for video_id '{video_id}'. "
        "Expected '<video_id>_<5-digit-shot>_<3-digit-position>' or "
        "'shot_<5-digit-shot>_pos_<3-digit-position>'."
    )


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
    Detect and audit one embedding dimension across every Parquet row.

    Returns None only when no Parquet files exist. Invalid, non-finite,
    non-normalized, or dimensionally inconsistent artifacts fail closed.
    """
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning(f"No Parquet files found in {parquet_dir}")
        return None

    expected_dimension: Optional[int] = None
    audited_vectors = 0
    for parquet_path in parquet_files:
        try:
            dataframe = pd.read_parquet(parquet_path)
        except Exception as exc:
            raise ValueError(
                f"Failed to read embedding artifact {parquet_path}: {exc}"
            ) from exc

        if "embedding" not in dataframe.columns or dataframe.empty:
            raise ValueError(
                f"Invalid embedding artifact {parquet_path}: "
                "missing embedding column or rows."
            )

        for row_index, embedding in enumerate(
            dataframe["embedding"].tolist()
        ):
            try:
                vector = np.asarray(embedding, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid embedding at {parquet_path}:{row_index}."
                ) from exc

            if vector.ndim != 1 or vector.size == 0:
                raise ValueError(
                    f"Invalid embedding shape at "
                    f"{parquet_path}:{row_index}: {vector.shape}."
                )
            if not np.isfinite(vector).all():
                raise ValueError(
                    f"Invalid embedding values at "
                    f"{parquet_path}:{row_index}."
                )

            dimension = int(vector.size)
            if expected_dimension is None:
                expected_dimension = dimension
            elif dimension != expected_dimension:
                raise ValueError(
                    f"Embedding dimension mismatch at "
                    f"{parquet_path}:{row_index}: expected "
                    f"{expected_dimension}, got {dimension}."
                )

            norm = float(np.linalg.norm(vector))
            if not np.isclose(norm, 1.0, atol=1e-3, rtol=1e-3):
                raise ValueError(
                    f"Invalid embedding norm at "
                    f"{parquet_path}:{row_index}: {norm}."
                )
            audited_vectors += 1

    logger.info(
        "Audited %s vectors across %s Parquet file(s); dimension=%s.",
        audited_vectors,
        len(parquet_files),
        expected_dimension,
    )
    return expected_dimension


def discover_video_ids(data_dir: Path) -> List[str]:
    """
    Discover unique video IDs from every Module 1-6 artifact family.

    JSON payload IDs are authoritative. Parquet artifacts use the
    file-per-video stem contract.
    """
    video_ids = set()

    def add_json_video_id(
        path: Path,
        *,
        cleaned_suffix: bool = False,
    ) -> None:
        data = read_json_safe(path)
        payload_video_id = (
            data.get("video_id")
            if isinstance(data, dict)
            else None
        )
        if (
            isinstance(payload_video_id, str)
            and payload_video_id.strip()
        ):
            video_ids.add(payload_video_id.strip())
            return

        fallback = path.stem
        if cleaned_suffix:
            fallback = fallback.removesuffix("_cleaned")
        if fallback:
            video_ids.add(fallback)

    for directory_name in (
        "metadata",
        "ocr",
        "summaries",
        "object_detection",
    ):
        directory = data_dir / directory_name
        if directory.exists():
            for path in directory.glob("*.json"):
                add_json_video_id(path)

    transcript_dir = data_dir / "transcripts"
    if transcript_dir.exists():
        for path in transcript_dir.glob("*.json"):
            if path.stem.endswith("_raw"):
                continue
            add_json_video_id(path, cleaned_suffix=True)

    for relative_directory in (
        ("embeddings", "visual"),
        ("embeddings", "text_asr"),
        ("embeddings", "text_summary"),
        ("embeddings", "text_ocr"),
    ):
        directory = data_dir.joinpath(*relative_directory)
        if directory.exists():
            video_ids.update(
                path.stem
                for path in directory.glob("*.parquet")
            )

    legacy_embedding_dir = data_dir / "embeddings"
    if legacy_embedding_dir.exists():
        video_ids.update(
            path.stem
            for path in legacy_embedding_dir.glob("*.parquet")
        )

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

                raw_frame_id = str(row.get("frame_id", ""))
                record_video_id = str(row.get("video_id", video_id))
                records.append({
                    "frame_id": normalize_frame_id(
                        raw_frame_id,
                        record_video_id,
                    ),
                    "video_id": record_video_id,
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


def load_text_ocr_embeddings(
    data_dir: Path, video_id: str
) -> List[Dict[str, Any]]:
    """Load OCR text embedding records for a video from Parquet.

    These embeddings are produced by Module 6 and stored in
    data/embeddings/text_ocr/{video_id}.parquet.
    """
    path = data_dir / "embeddings" / "text_ocr" / f"{video_id}.parquet"
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

        raw_frame_id = str(row.get("frame_id", ""))
        vid = str(row.get("video_id", video_id))

        records.append({
            "frame_id": normalize_frame_id(raw_frame_id, vid),
            "video_id": vid,
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

        raw_fid = str(frame.get("frame_id", ""))
        records.append({
            "frame_id": normalize_frame_id(raw_fid, video_id),
            "video_id": video_id,
            "shot_id": str(frame.get("shot_id", "")),
            "ocr_text_concat": text,
        })
    return records


def load_asr_transcripts(data_dir: Path, video_id: str) -> List[Dict[str, Any]]:
    """Load canonical Module 3 ASR intervals for Elasticsearch indexing."""
    # Try both with and without _cleaned suffix
    possible_paths = [
        data_dir / "transcripts" / f"{video_id}_cleaned.json",
        data_dir / "transcripts" / f"{video_id}.json",
    ]

    data = None
    selected_path = None
    for path in possible_paths:
        if path.exists():
            selected_path = path
            data = read_json_safe(path)
            break

    if selected_path is None:
        return []

    if not isinstance(data, dict):
        if data is not None:
            logger.warning(
                f"ASR file {selected_path} must contain an object envelope."
            )
        return []

    intervals = data.get("intervals")
    if not isinstance(intervals, list):
        logger.warning(f"ASR file {selected_path} has no valid 'intervals' list.")
        return []

    payload_video_id = data.get("video_id")
    if not isinstance(payload_video_id, str) or not payload_video_id.strip():
        logger.warning(f"ASR file {selected_path} has no valid 'video_id'.")
        return []
    payload_video_id = payload_video_id.strip()

    if payload_video_id != video_id:
        logger.warning(
            f"ASR video_id mismatch in {selected_path}: "
            f"payload='{payload_video_id}', expected='{video_id}'."
        )
        return []

    records = []
    for index, item in enumerate(intervals):
        if not isinstance(item, dict):
            logger.warning(
                f"Skipping non-object ASR interval at index {index} in {selected_path}."
            )
            continue

        text_value = item.get("cleaned_text", "")
        if not isinstance(text_value, str):
            logger.warning(
                f"Skipping ASR interval {index} with invalid cleaned_text "
                f"in {selected_path}."
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
                f"Skipping ASR interval {index} with invalid interval_id "
                f"in {selected_path}."
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
                f"Skipping ASR interval {interval_id} with invalid timestamps "
                f"in {selected_path}."
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
                f"Skipping ASR interval {interval_id} with invalid time range "
                f"in {selected_path}."
            )
            continue

        records.append({
            "interval_id": interval_id,
            "video_id": payload_video_id,
            "start_time_sec": start_time_sec,
            "end_time_sec": end_time_sec,
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
    video_record = load_video_metadata(data_dir, video_id)

    # --- Metadata from Module 1 ---
    meta_path = data_dir / "metadata" / f"{video_id}.json"
    meta_data = read_json_safe(meta_path) if meta_path.exists() else None
    if not isinstance(meta_data, dict):
        raise ValueError(f"Missing or invalid metadata artifact for {video_id}")

    metadata_records = []
    for shot in meta_data.get("shots", []):
        if not isinstance(shot, dict):
            raise ValueError("metadata.shots must contain objects")
        shot_id = shot.get("shot_id")
        if isinstance(shot_id, bool) or not isinstance(shot_id, int) or shot_id < 0:
            raise ValueError("shot_id must be a non-negative integer")
        for keyframe in shot.get("keyframes", []):
            if not isinstance(keyframe, dict):
                raise ValueError("keyframes must contain objects")
            image_rel_path = _relative_posix_path(
                keyframe.get("image_rel_path"),
                "image_rel_path",
            )
            raw_frame_id = Path(image_rel_path).stem
            frame_id = normalize_frame_id(raw_frame_id, video_id)
            position_code = keyframe.get("position_code")
            if (
                isinstance(position_code, bool)
                or not isinstance(position_code, int)
                or not 0 <= position_code <= 100
                or not frame_id.endswith(
                    f"_{shot_id:05d}_{position_code:03d}"
                )
            ):
                raise ValueError("frame_id suffix does not match shot/position")
            source_frame_idx = keyframe.get("source_frame_idx")
            if (
                isinstance(source_frame_idx, bool)
                or not isinstance(source_frame_idx, int)
                or not 0 <= source_frame_idx < video_record["frame_count"]
            ):
                raise ValueError("source_frame_idx is outside source frame bounds")
            timestamp = _finite_float(keyframe.get("time_sec"), "time_sec")
            if timestamp < 0 or timestamp > video_record["duration_sec"]:
                raise ValueError("time_sec is outside source duration")

            metadata_records.append(
                {
                    "frame_id": frame_id,
                    "video_id": video_id,
                    "shot_id": shot_id,
                    "source_frame_idx": source_frame_idx,
                    "timestamp": timestamp,
                    "image_rel_path": image_rel_path,
                }
            )

    metadata_ids = {record["frame_id"] for record in metadata_records}
    if len(metadata_ids) != len(metadata_records):
        raise ValueError("metadata contains duplicate frame_id values")

    # --- Object Detection from Module 5 ---
    # Object detection JSON follows same schema as OCR: {video_id, frames: [{frame_id, objects: [...]}]}
    obj_dir = data_dir / "object_detection"
    obj_path = obj_dir / f"{video_id}.json"
    obj_data = read_json_safe(obj_path) if obj_path.exists() else None

    object_records = []
    if isinstance(obj_data, dict):
        for frame in obj_data.get("frames", []):
            raw_fid = str(frame.get("frame_id", ""))
            frame_id = normalize_frame_id(raw_fid, video_id)
            if frame_id not in metadata_ids:
                raise ValueError("object frame_id does not JOIN metadata")
            for obj in frame.get("objects", []):
                if not isinstance(obj, dict):
                    raise ValueError("objects must contain object records")
                bbox = obj.get("bbox", obj.get("box"))
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError("bbox must contain exactly four coordinates")
                x_min, y_min, x_max, y_max = (
                    _finite_float(value, "bbox") for value in bbox
                )
                if not (
                    0 <= x_min < x_max <= video_record["width"]
                    and 0 <= y_min < y_max <= video_record["height"]
                ):
                    raise ValueError("bbox is outside absolute pixel bounds")
                confidence = _finite_float(
                    obj.get("confidence", obj.get("score")),
                    "confidence",
                )
                if not 0 <= confidence <= 1:
                    raise ValueError("confidence must be within [0, 1]")
                label = obj.get("label")
                if not isinstance(label, str) or not label.strip():
                    raise ValueError("object label must not be empty")
                model_source = obj.get("model_source")
                if model_source is not None and (
                    not isinstance(model_source, str) or not model_source.strip()
                ):
                    raise ValueError("model_source must be non-empty text or null")
                object_records.append(
                    {
                        "frame_id": frame_id,
                        "label": label.strip().casefold(),
                        "confidence": confidence,
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                        "model_source": (
                            model_source.strip() if model_source is not None else None
                        ),
                    }
                )

    return metadata_records, object_records


def load_video_metadata(data_dir: Path, video_id: str) -> Dict[str, Any]:
    """Load and validate the self-indexed-v2 source-video SQLite row."""
    meta_path = data_dir / "metadata" / f"{video_id}.json"
    payload = read_json_safe(meta_path) if meta_path.exists() else None
    if not isinstance(payload, dict):
        raise ValueError(f"Missing or invalid metadata artifact for {video_id}")
    if payload.get("contract_version") != "self-indexed-v2":
        raise ValueError("metadata contract_version must be self-indexed-v2")
    if payload.get("video_id") != video_id:
        raise ValueError("metadata video_id does not match requested video")

    fps = _finite_float(payload.get("fps"), "fps")
    duration_sec = _finite_float(payload.get("duration_sec"), "duration_sec")
    if fps <= 0 or duration_sec < 0:
        raise ValueError("fps and duration_sec are outside valid bounds")

    return {
        "video_id": video_id,
        "source_video_rel_path": _relative_posix_path(
            payload.get("source_video_rel_path"),
            "source_video_rel_path",
        ),
        "fps": fps,
        "duration_sec": duration_sec,
        "frame_count": _positive_int(payload.get("frame_count"), "frame_count"),
        "width": _positive_int(payload.get("width"), "width"),
        "height": _positive_int(payload.get("height"), "height"),
    }
