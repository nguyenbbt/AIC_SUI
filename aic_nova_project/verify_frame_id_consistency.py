"""Verify join-key consistency across Milvus, Elasticsearch, and SQLite."""

import argparse
from datetime import datetime, timezone
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Iterable, List, Mapping, Set, Tuple

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import numpy as np
from PIL import Image
from pymilvus import Collection, connections, utility


CANONICAL_FRAME_ID = re.compile(r"^.+_[0-9]{5}_[0-9]{3}$")
DATASET_FINGERPRINT = re.compile(r"^sha256:[0-9a-f]{64}$")
TEXT_MODEL_REVISION = "4ab46e46ba5902328ba0742e489e75f787932f2b"


@dataclass(frozen=True)
class VerificationSnapshot:
    """Join keys read from every online storage branch."""

    visual_frame_ids: Set[str]
    ocr_vector_frame_ids: Set[str]
    ocr_text_frame_ids: Set[str]
    metadata_frame_ids: Set[str]
    object_frame_ids: Set[str]
    asr_vector_ids: Set[Tuple[str, str]]
    asr_text_ids: Set[Tuple[str, str]]
    summary_vector_ids: Set[str]
    summary_text_ids: Set[str]


@dataclass(frozen=True)
class FullVerificationSnapshot:
    """Complete records required by the self-indexed-v2 audit."""

    joins: VerificationSnapshot
    videos: Tuple[Dict[str, Any], ...]
    metadata: Tuple[Dict[str, Any], ...]
    objects: Tuple[Dict[str, Any], ...]
    milvus: Mapping[str, Tuple[Dict[str, Any], ...]]
    elasticsearch: Mapping[str, Tuple[Dict[str, Any], ...]]
    sqlite_schema_errors: Tuple[str, ...] = ()
    foreign_key_errors: Tuple[str, ...] = ()


def record_counts(snapshot: FullVerificationSnapshot) -> Dict[str, int]:
    """Return manifest record counts from the complete backend snapshot."""
    return {
        "videos": len(snapshot.videos),
        "metadata": len(snapshot.metadata),
        "objects": len(snapshot.objects),
        "visual_features": len(snapshot.milvus.get("visual_features", ())),
        "ocr_features": len(snapshot.milvus.get("ocr_features", ())),
        "asr_features": len(snapshot.milvus.get("asr_features", ())),
        "summary_features": len(snapshot.milvus.get("summary_features", ())),
        "ocr_texts": len(snapshot.elasticsearch.get("ocr_texts", ())),
        "asr_transcripts": len(
            snapshot.elasticsearch.get("asr_transcripts", ())
        ),
        "video_summaries": len(
            snapshot.elasticsearch.get("video_summaries", ())
        ),
    }


def _relative_data_path(
    data_root: Path,
    value: object,
) -> Path | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
    ):
        return None
    candidate = (data_root / Path(*posix_path.parts)).resolve()
    try:
        candidate.relative_to(data_root.resolve())
    except ValueError:
        return None
    return candidate


def _duplicate_keys(
    records: Tuple[Dict[str, Any], ...],
    fields: Tuple[str, ...],
) -> list[tuple[str, ...]]:
    seen = set()
    duplicates = set()
    for record in records:
        key = tuple(str(record.get(field, "")) for field in fields)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return sorted(duplicates)


def _validate_vectors(
    snapshot: FullVerificationSnapshot,
) -> List[str]:
    errors: List[str] = []
    specifications = (
        ("visual_features", 512, ("frame_id",)),
        ("ocr_features", 768, ("frame_id",)),
        ("asr_features", 768, ("video_id", "interval_id")),
        ("summary_features", 768, ("video_id",)),
    )
    for stream_name, expected_dimension, key_fields in specifications:
        records = snapshot.milvus.get(stream_name, ())
        duplicates = _duplicate_keys(records, key_fields)
        if duplicates:
            errors.append(
                f"{stream_name} contains duplicate domain keys: "
                f"{duplicates[:10]}"
            )
        for index, record in enumerate(records):
            try:
                vector = np.asarray(record["embedding"], dtype=np.float32)
            except (KeyError, TypeError, ValueError):
                errors.append(
                    f"{stream_name}[{index}] has an invalid embedding"
                )
                continue
            if vector.shape != (expected_dimension,):
                errors.append(
                    f"{stream_name}[{index}] dimension={vector.size}; "
                    f"expected={expected_dimension}"
                )
                continue
            if not np.isfinite(vector).all():
                errors.append(
                    f"{stream_name}[{index}] contains non-finite values"
                )
                continue
            if not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-3):
                errors.append(
                    f"{stream_name}[{index}] is not L2-normalized"
                )
    return errors


def _validate_manifest(
    snapshot: FullVerificationSnapshot,
    manifest: Mapping[str, Any],
) -> List[str]:
    errors: List[str] = []
    expected_fields = {
        "contract_version": "self-indexed-v2",
        "frame_index_base": 0,
        "bbox_space": "absolute_pixel_xyxy",
        "visual_model_id": "ViT-B-32::openai",
        "visual_dimension": 512,
        "visual_normalized": True,
        "text_model_name": "dangvantuan/vietnamese-embedding",
        "text_model_revision": TEXT_MODEL_REVISION,
        "text_dimension": 768,
        "text_max_length": 256,
    }
    for field, expected in expected_fields.items():
        if manifest.get(field) != expected:
            errors.append(
                f"manifest.{field}={manifest.get(field)!r}; "
                f"expected={expected!r}"
            )
    if manifest.get("status") not in {"BUILDING", "READY"}:
        errors.append("manifest.status must be BUILDING or READY")
    dataset_id = manifest.get("dataset_id")
    if (
        not isinstance(dataset_id, str)
        or not dataset_id
        or dataset_id != dataset_id.strip()
    ):
        errors.append("manifest.dataset_id must be a non-empty trimmed string")
    created_at_utc = manifest.get("created_at_utc")
    try:
        if not isinstance(created_at_utc, str) or not created_at_utc.endswith("Z"):
            raise ValueError
        created_at = datetime.fromisoformat(
            created_at_utc[:-1] + "+00:00"
        )
        if created_at.utcoffset() != timezone.utc.utcoffset(created_at):
            raise ValueError
    except ValueError:
        errors.append("manifest.created_at_utc must be an ISO-8601 UTC timestamp")
    fingerprint = manifest.get("dataset_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or DATASET_FINGERPRINT.fullmatch(fingerprint) is None
    ):
        errors.append("manifest.dataset_fingerprint is invalid")

    expected_counts = record_counts(snapshot)
    manifest_counts = manifest.get("record_counts")
    if not isinstance(manifest_counts, Mapping):
        errors.append("manifest.record_counts must be an object")
    else:
        for key, actual_count in expected_counts.items():
            if manifest_counts.get(key) != actual_count:
                errors.append(
                    f"manifest.record_counts.{key}="
                    f"{manifest_counts.get(key)!r}; actual={actual_count}"
                )
        extra = sorted(set(manifest_counts) - set(expected_counts))
        if extra:
            errors.append(
                f"manifest.record_counts has unexpected keys: {extra}"
            )
    return errors


def build_full_contract_report(
    snapshot: FullVerificationSnapshot,
    *,
    data_root: str | Path,
    manifest: Mapping[str, Any],
) -> List[str]:
    """Return every self-indexed-v2 violation found in a full snapshot."""
    errors = build_consistency_report(snapshot.joins)
    errors.extend(snapshot.sqlite_schema_errors)
    errors.extend(
        f"SQLite foreign key violation: {error}"
        for error in snapshot.foreign_key_errors
    )
    root = Path(data_root).resolve()

    videos_by_id: Dict[str, Dict[str, Any]] = {}
    for index, video in enumerate(snapshot.videos):
        video_id = video.get("video_id")
        if not isinstance(video_id, str) or not video_id.strip():
            errors.append(f"videos[{index}].video_id is invalid")
            continue
        if video_id in videos_by_id:
            errors.append(f"videos contains duplicate video_id {video_id!r}")
        videos_by_id[video_id] = video
        source_path = _relative_data_path(
            root,
            video.get("source_video_rel_path"),
        )
        if source_path is None:
            errors.append(
                f"videos[{index}].source_video_rel_path is unsafe"
            )
        for field in ("fps", "duration_sec"):
            value = video.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (field == "fps" and value <= 0)
                or (field == "duration_sec" and value < 0)
            ):
                errors.append(f"videos[{index}].{field} is invalid")
        for field in ("frame_count", "width", "height"):
            value = video.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                errors.append(f"videos[{index}].{field} is invalid")

    metadata_by_frame: Dict[str, Dict[str, Any]] = {}
    for index, record in enumerate(snapshot.metadata):
        frame_id = record.get("frame_id")
        video_id = record.get("video_id")
        if frame_id in metadata_by_frame:
            errors.append(f"metadata contains duplicate frame_id {frame_id!r}")
        if isinstance(frame_id, str):
            metadata_by_frame[frame_id] = record
        video = videos_by_id.get(video_id)
        if video is None:
            errors.append(
                f"metadata[{index}].video_id={video_id!r} has no videos row"
            )
            continue
        shot_id = record.get("shot_id")
        source_frame_idx = record.get("source_frame_idx")
        if (
            isinstance(shot_id, bool)
            or not isinstance(shot_id, int)
            or shot_id < 0
        ):
            errors.append(f"metadata[{index}].shot_id is invalid")
        frame_count = video.get("frame_count")
        frame_count_is_valid = (
            not isinstance(frame_count, bool)
            and isinstance(frame_count, int)
            and frame_count > 0
        )
        if (
            isinstance(source_frame_idx, bool)
            or not isinstance(source_frame_idx, int)
            or source_frame_idx < 0
            or not frame_count_is_valid
            or source_frame_idx >= frame_count
        ):
            errors.append(
                f"metadata[{index}].source_frame_idx is outside video bounds"
            )
        timestamp = record.get("timestamp")
        duration_sec = video.get("duration_sec")
        fps = video.get("fps")
        time_bounds_are_valid = (
            isinstance(duration_sec, (int, float))
            and not isinstance(duration_sec, bool)
            and math.isfinite(float(duration_sec))
            and duration_sec >= 0
            and isinstance(fps, (int, float))
            and not isinstance(fps, bool)
            and math.isfinite(float(fps))
            and fps > 0
        )
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(float(timestamp))
            or timestamp < 0
            or not time_bounds_are_valid
            or timestamp > duration_sec + (1.0 / fps)
        ):
            errors.append(f"metadata[{index}].timestamp is invalid")
        if isinstance(frame_id, str) and CANONICAL_FRAME_ID.fullmatch(frame_id):
            encoded_shot_id = int(frame_id.rsplit("_", 2)[1])
            if encoded_shot_id != shot_id:
                errors.append(
                    f"metadata[{index}].frame_id shot suffix does not match shot_id"
                )
        image_path = _relative_data_path(root, record.get("image_rel_path"))
        if image_path is None:
            errors.append(f"metadata[{index}].image_rel_path is unsafe")
        elif not image_path.is_file():
            errors.append(
                f"metadata[{index}].image_rel_path does not exist: {image_path}"
            )
        else:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, ValueError):
                errors.append(
                    f"metadata[{index}].image_rel_path is not decodable"
                )

    for index, record in enumerate(snapshot.objects):
        frame_id = record.get("frame_id")
        metadata = metadata_by_frame.get(frame_id)
        if metadata is None:
            errors.append(f"objects[{index}].frame_id is orphaned")
            continue
        video = videos_by_id.get(metadata.get("video_id"))
        label = record.get("label")
        if (
            not isinstance(label, str)
            or not label.strip()
            or label != label.casefold()
        ):
            errors.append(f"objects[{index}].label is not casefold-normalized")
        confidence = record.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0 <= confidence <= 1
        ):
            errors.append(f"objects[{index}].confidence is invalid")
        width = video.get("width") if video is not None else None
        height = video.get("height") if video is not None else None
        try:
            x_min = float(record["x_min"])
            y_min = float(record["y_min"])
            x_max = float(record["x_max"])
            y_max = float(record["y_max"])
            bbox_valid = (
                all(math.isfinite(value) for value in (x_min, y_min, x_max, y_max))
                and isinstance(width, int)
                and not isinstance(width, bool)
                and isinstance(height, int)
                and not isinstance(height, bool)
                and 0 <= x_min < x_max <= width
                and 0 <= y_min < y_max <= height
            )
        except (KeyError, TypeError, ValueError):
            bbox_valid = False
        if not bbox_valid:
            errors.append(f"objects[{index}].bbox is outside absolute pixel bounds")

    for stream_name, records in (
        ("visual_features", snapshot.milvus.get("visual_features", ())),
        ("ocr_features", snapshot.milvus.get("ocr_features", ())),
        ("ocr_texts", snapshot.elasticsearch.get("ocr_texts", ())),
    ):
        for index, record in enumerate(records):
            metadata = metadata_by_frame.get(record.get("frame_id"))
            if metadata is None:
                continue
            if record.get("video_id") != metadata.get("video_id"):
                errors.append(
                    f"{stream_name}[{index}].video_id does not match metadata"
                )
            if stream_name in {"visual_features", "ocr_texts"}:
                if str(record.get("shot_id")) != str(metadata.get("shot_id")):
                    errors.append(
                        f"{stream_name}[{index}].shot_id does not match metadata"
                    )

    lexical_specs = (
        ("ocr_texts", ("frame_id",)),
        ("asr_transcripts", ("video_id", "interval_id")),
        ("video_summaries", ("video_id",)),
    )
    for stream_name, fields in lexical_specs:
        duplicates = _duplicate_keys(
            snapshot.elasticsearch.get(stream_name, ()),
            fields,
        )
        if duplicates:
            errors.append(
                f"{stream_name} contains duplicate domain keys: {duplicates[:10]}"
            )

    for stream_name, records in (
        ("asr_features", snapshot.milvus.get("asr_features", ())),
        (
            "asr_transcripts",
            snapshot.elasticsearch.get("asr_transcripts", ()),
        ),
    ):
        for index, record in enumerate(records):
            video_id = record.get("video_id")
            if video_id not in videos_by_id:
                errors.append(
                    f"{stream_name}[{index}].video_id has no videos row"
                )
            interval_id = record.get("interval_id")
            if (
                not isinstance(interval_id, str)
                or not interval_id.isascii()
                or not interval_id.isdigit()
                or (len(interval_id) > 1 and interval_id.startswith("0"))
            ):
                errors.append(
                    f"{stream_name}[{index}].interval_id is invalid"
                )
            if stream_name == "asr_features":
                start = record.get("start_time_sec")
                end = record.get("end_time_sec")
                if (
                    isinstance(start, bool)
                    or not isinstance(start, (int, float))
                    or not math.isfinite(float(start))
                    or isinstance(end, bool)
                    or not isinstance(end, (int, float))
                    or not math.isfinite(float(end))
                    or start < 0
                    or end <= start
                ):
                    errors.append(
                        f"{stream_name}[{index}] has invalid time bounds"
                    )

    asr_vectors_by_key = {
        (str(record.get("video_id")), str(record.get("interval_id"))): record
        for record in snapshot.milvus.get("asr_features", ())
    }
    for index, transcript in enumerate(
        snapshot.elasticsearch.get("asr_transcripts", ())
    ):
        key = (
            str(transcript.get("video_id")),
            str(transcript.get("interval_id")),
        )
        vector_record = asr_vectors_by_key.get(key)
        if vector_record is None:
            continue
        for field in ("start_time_sec", "end_time_sec"):
            lexical_value = transcript.get(field)
            vector_value = vector_record.get(field)
            if (
                isinstance(lexical_value, bool)
                or not isinstance(lexical_value, (int, float))
                or not math.isfinite(float(lexical_value))
                or isinstance(vector_value, bool)
                or not isinstance(vector_value, (int, float))
                or not math.isfinite(float(vector_value))
                or not math.isclose(
                    float(lexical_value),
                    float(vector_value),
                    abs_tol=1e-6,
                )
            ):
                errors.append(
                    f"asr_transcripts[{index}].{field} does not match "
                    "asr_features"
                )

    for stream_name, records in (
        (
            "summary_features",
            snapshot.milvus.get("summary_features", ()),
        ),
        (
            "video_summaries",
            snapshot.elasticsearch.get("video_summaries", ()),
        ),
    ):
        for index, record in enumerate(records):
            if record.get("video_id") not in videos_by_id:
                errors.append(
                    f"{stream_name}[{index}].video_id has no videos row"
                )

    errors.extend(_validate_vectors(snapshot))
    errors.extend(_validate_manifest(snapshot, manifest))
    return errors


def _describe_difference(
    left_name: str,
    left: set,
    right_name: str,
    right: set,
) -> str:
    left_only = sorted(left - right)
    right_only = sorted(right - left)
    return (
        f"{left_name} and {right_name} do not JOIN: "
        f"{left_name}-only={left_only[:10]}, "
        f"{right_name}-only={right_only[:10]}"
    )


def build_consistency_report(
    snapshot: VerificationSnapshot,
) -> List[str]:
    """Return cross-database contract violations for matching record keys."""
    errors: List[str] = []
    frame_sets = {
        "Milvus visual": snapshot.visual_frame_ids,
        "Milvus OCR": snapshot.ocr_vector_frame_ids,
        "Elasticsearch OCR": snapshot.ocr_text_frame_ids,
        "SQLite metadata": snapshot.metadata_frame_ids,
        "SQLite objects": snapshot.object_frame_ids,
    }
    for source_name, frame_ids in frame_sets.items():
        invalid = sorted(
            frame_id
            for frame_id in frame_ids
            if not CANONICAL_FRAME_ID.fullmatch(frame_id)
        )
        if invalid:
            errors.append(
                f"{source_name} contains invalid frame IDs: {invalid[:10]}"
            )

    if snapshot.visual_frame_ids != snapshot.metadata_frame_ids:
        errors.append(
            _describe_difference(
                "Milvus visual",
                snapshot.visual_frame_ids,
                "SQLite metadata",
                snapshot.metadata_frame_ids,
            )
        )
    if snapshot.ocr_vector_frame_ids != snapshot.ocr_text_frame_ids:
        errors.append(
            _describe_difference(
                "Milvus OCR",
                snapshot.ocr_vector_frame_ids,
                "Elasticsearch OCR",
                snapshot.ocr_text_frame_ids,
            )
        )

    for source_name, frame_ids in (
        ("Milvus OCR", snapshot.ocr_vector_frame_ids),
        ("Elasticsearch OCR", snapshot.ocr_text_frame_ids),
        ("SQLite objects", snapshot.object_frame_ids),
    ):
        orphan_ids = sorted(frame_ids - snapshot.metadata_frame_ids)
        if orphan_ids:
            errors.append(
                f"{source_name} has frame IDs absent from SQLite metadata: "
                f"{orphan_ids[:10]}"
            )

    if snapshot.asr_vector_ids != snapshot.asr_text_ids:
        errors.append(
            _describe_difference(
                "Milvus ASR",
                snapshot.asr_vector_ids,
                "Elasticsearch ASR",
                snapshot.asr_text_ids,
            )
        )
    if snapshot.summary_vector_ids != snapshot.summary_text_ids:
        errors.append(
            _describe_difference(
                "Milvus summary",
                snapshot.summary_vector_ids,
                "Elasticsearch summary",
                snapshot.summary_text_ids,
            )
        )
    return errors


def _query_milvus(
    collection_name: str,
    output_fields: List[str],
) -> List[Dict]:
    if not utility.has_collection(collection_name, using="verify"):
        return []

    collection = Collection(collection_name, using="verify")
    collection.load()
    iterator = collection.query_iterator(
        batch_size=1_000,
        expr="pk >= 0",
        output_fields=output_fields,
    )
    records: List[Dict] = []
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            records.extend(batch)
    finally:
        iterator.close()
    return records


def collect_milvus_keys(uri: str) -> Dict[str, set]:
    """Read all relevant join keys from Milvus."""
    connections.connect(alias="verify", uri=uri)
    try:
        visual = _query_milvus(
            "visual_features",
            ["frame_id", "video_id"],
        )
        ocr = _query_milvus(
            "ocr_features",
            ["frame_id", "video_id"],
        )
        asr = _query_milvus(
            "asr_features",
            ["video_id", "interval_id"],
        )
        summaries = _query_milvus(
            "summary_features",
            ["video_id"],
        )
        return {
            "visual": {str(record["frame_id"]) for record in visual},
            "ocr": {str(record["frame_id"]) for record in ocr},
            "asr": {
                (str(record["video_id"]), str(record["interval_id"]))
                for record in asr
            },
            "summary": {
                str(record["video_id"])
                for record in summaries
            },
        }
    finally:
        connections.disconnect("verify")


def collect_milvus_records(
    uri: str,
) -> Dict[str, Tuple[Dict[str, Any], ...]]:
    """Read every vector and domain field needed by the full verifier."""
    connections.connect(alias="verify", uri=uri)
    try:
        specifications = {
            "visual_features": [
                "frame_id",
                "video_id",
                "shot_id",
                "embedding",
            ],
            "ocr_features": ["frame_id", "video_id", "embedding"],
            "asr_features": [
                "video_id",
                "interval_id",
                "start_time_sec",
                "end_time_sec",
                "embedding",
            ],
            "summary_features": ["video_id", "embedding"],
        }
        return {
            collection_name: tuple(
                _query_milvus(collection_name, output_fields)
            )
            for collection_name, output_fields in specifications.items()
        }
    finally:
        connections.disconnect("verify")


def _scan_es_index(
    client: Elasticsearch,
    index_name: str,
) -> Iterable[Dict]:
    if not client.indices.exists(index=index_name):
        return []
    return scan(
        client,
        index=index_name,
        query={"query": {"match_all": {}}},
    )


def collect_elasticsearch_keys(uri: str) -> Dict[str, set]:
    """Read all relevant join keys from Elasticsearch."""
    client = Elasticsearch(uri)
    try:
        ocr = list(_scan_es_index(client, "ocr_texts"))
        asr = list(_scan_es_index(client, "asr_transcripts"))
        summaries = list(_scan_es_index(client, "video_summaries"))
        return {
            "ocr": {
                str(hit["_source"]["frame_id"])
                for hit in ocr
            },
            "asr": {
                (
                    str(hit["_source"]["video_id"]),
                    str(hit["_source"]["interval_id"]),
                )
                for hit in asr
            },
            "summary": {
                str(hit["_source"]["video_id"])
                for hit in summaries
            },
        }
    finally:
        client.close()


def collect_elasticsearch_records(
    uri: str,
) -> Dict[str, Tuple[Dict[str, Any], ...]]:
    """Read every lexical source document needed by the full verifier."""
    client = Elasticsearch(uri)
    try:
        return {
            index_name: tuple(
                dict(hit.get("_source", {}))
                for hit in _scan_es_index(client, index_name)
            )
            for index_name in (
                "ocr_texts",
                "asr_transcripts",
                "video_summaries",
            )
        }
    finally:
        client.close()


def collect_sqlite_keys(db_uri: str) -> Dict[str, set]:
    """Read all relevant join keys from SQLite."""
    if db_uri.startswith("sqlite:///"):
        db_uri = db_uri[len("sqlite:///"):]
    db_path = Path(db_uri)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    connection = sqlite3.connect(str(db_path))
    try:
        metadata = {
            str(row[0])
            for row in connection.execute(
                "SELECT frame_id FROM metadata"
            ).fetchall()
        }
        objects = {
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT frame_id FROM objects"
            ).fetchall()
        }
        return {"metadata": metadata, "objects": objects}
    finally:
        connection.close()


def _sqlite_schema_report(connection: sqlite3.Connection) -> List[str]:
    expected_columns = {
        "videos": {
            "video_id": ("TEXT", 0, 1),
            "source_video_rel_path": ("TEXT", 1, 0),
            "fps": ("REAL", 1, 0),
            "duration_sec": ("REAL", 1, 0),
            "frame_count": ("INTEGER", 1, 0),
            "width": ("INTEGER", 1, 0),
            "height": ("INTEGER", 1, 0),
        },
        "metadata": {
            "frame_id": ("TEXT", 0, 1),
            "video_id": ("TEXT", 1, 0),
            "shot_id": ("INTEGER", 1, 0),
            "source_frame_idx": ("INTEGER", 1, 0),
            "timestamp": ("REAL", 1, 0),
            "image_rel_path": ("TEXT", 1, 0),
        },
        "objects": {
            "id": ("INTEGER", 0, 1),
            "frame_id": ("TEXT", 1, 0),
            "label": ("TEXT", 1, 0),
            "confidence": ("REAL", 1, 0),
            "x_min": ("REAL", 1, 0),
            "y_min": ("REAL", 1, 0),
            "x_max": ("REAL", 1, 0),
            "y_max": ("REAL", 1, 0),
            "model_source": ("TEXT", 0, 0),
        },
    }
    errors: List[str] = []
    for table_name, expected in expected_columns.items():
        rows = connection.execute(
            f"PRAGMA table_info({table_name})"
        ).fetchall()
        actual = {
            row[1]: (str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in rows
        }
        if actual != expected:
            errors.append(
                f"SQLite schema mismatch for {table_name}: "
                f"actual_columns={sorted(actual)}"
            )

    expected_indexes = {
        "idx_metadata_video_id": ("metadata", ("video_id",)),
        "idx_metadata_video_timeline": (
            "metadata",
            ("video_id", "timestamp", "frame_id"),
        ),
        "idx_metadata_video_source_frame": (
            "metadata",
            ("video_id", "source_frame_idx"),
        ),
        "idx_objects_frame_id": ("objects", ("frame_id",)),
        "idx_objects_label": ("objects", ("label",)),
    }
    for index_name, (table_name, expected_fields) in expected_indexes.items():
        index_names = {
            row[1]
            for row in connection.execute(
                f"PRAGMA index_list({table_name})"
            ).fetchall()
        }
        if index_name not in index_names:
            errors.append(f"SQLite schema missing index {index_name}")
            continue
        actual_fields = tuple(
            row[2]
            for row in connection.execute(
                f"PRAGMA index_info({index_name})"
            ).fetchall()
        )
        if actual_fields != expected_fields:
            errors.append(
                f"SQLite index {index_name} fields={actual_fields}; "
                f"expected={expected_fields}"
            )

    foreign_key_specs = (
        ("metadata", "videos", "video_id", "video_id"),
        ("objects", "metadata", "frame_id", "frame_id"),
    )
    for table_name, target_table, source_field, target_field in foreign_key_specs:
        foreign_keys = connection.execute(
            f"PRAGMA foreign_key_list({table_name})"
        ).fetchall()
        if not any(
            row[2] == target_table
            and row[3] == source_field
            and row[4] == target_field
            and str(row[6]).upper() == "CASCADE"
            for row in foreign_keys
        ):
            errors.append(
                f"SQLite schema missing FK {table_name}.{source_field} "
                f"-> {target_table}.{target_field} ON DELETE CASCADE"
            )
    return errors


def _sqlite_rows(
    connection: sqlite3.Connection,
    table_name: str,
) -> Tuple[Dict[str, Any], ...]:
    try:
        rows = connection.execute(f"SELECT * FROM {table_name}").fetchall()
    except sqlite3.OperationalError:
        return ()
    return tuple(dict(row) for row in rows)


def collect_sqlite_records(db_uri: str) -> Dict[str, tuple]:
    """Read and audit every self-indexed-v2 SQLite row without writes."""
    if db_uri.startswith("sqlite:///"):
        db_uri = db_uri[len("sqlite:///"):]
    db_path = Path(db_uri).resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    connection = sqlite3.connect(
        f"{db_path.as_uri()}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        schema_errors = tuple(_sqlite_schema_report(connection))
        foreign_key_errors = tuple(
            repr(tuple(row))
            for row in connection.execute("PRAGMA foreign_key_check")
        )
        return {
            "videos": _sqlite_rows(connection, "videos"),
            "metadata": _sqlite_rows(connection, "metadata"),
            "objects": _sqlite_rows(connection, "objects"),
            "schema_errors": schema_errors,
            "foreign_key_errors": foreign_key_errors,
        }
    finally:
        connection.close()


def build_full_snapshot(
    *,
    milvus: Mapping[str, Tuple[Dict[str, Any], ...]],
    elasticsearch: Mapping[str, Tuple[Dict[str, Any], ...]],
    sqlite: Mapping[str, tuple],
) -> FullVerificationSnapshot:
    """Build one immutable full snapshot and its backward-compatible joins."""
    visual_records = milvus.get("visual_features", ())
    ocr_vector_records = milvus.get("ocr_features", ())
    asr_vector_records = milvus.get("asr_features", ())
    summary_vector_records = milvus.get("summary_features", ())
    ocr_text_records = elasticsearch.get("ocr_texts", ())
    asr_text_records = elasticsearch.get("asr_transcripts", ())
    summary_text_records = elasticsearch.get("video_summaries", ())
    metadata_records = sqlite.get("metadata", ())
    object_records = sqlite.get("objects", ())
    joins = VerificationSnapshot(
        visual_frame_ids={
            str(record.get("frame_id", "")) for record in visual_records
        },
        ocr_vector_frame_ids={
            str(record.get("frame_id", ""))
            for record in ocr_vector_records
        },
        ocr_text_frame_ids={
            str(record.get("frame_id", "")) for record in ocr_text_records
        },
        metadata_frame_ids={
            str(record.get("frame_id", "")) for record in metadata_records
        },
        object_frame_ids={
            str(record.get("frame_id", "")) for record in object_records
        },
        asr_vector_ids={
            (
                str(record.get("video_id", "")),
                str(record.get("interval_id", "")),
            )
            for record in asr_vector_records
        },
        asr_text_ids={
            (
                str(record.get("video_id", "")),
                str(record.get("interval_id", "")),
            )
            for record in asr_text_records
        },
        summary_vector_ids={
            str(record.get("video_id", ""))
            for record in summary_vector_records
        },
        summary_text_ids={
            str(record.get("video_id", ""))
            for record in summary_text_records
        },
    )
    return FullVerificationSnapshot(
        joins=joins,
        videos=tuple(sqlite.get("videos", ())),
        metadata=tuple(metadata_records),
        objects=tuple(object_records),
        milvus=milvus,
        elasticsearch=elasticsearch,
        sqlite_schema_errors=tuple(sqlite.get("schema_errors", ())),
        foreign_key_errors=tuple(sqlite.get("foreign_key_errors", ())),
    )


def collect_full_snapshot(
    *,
    milvus_uri: str,
    es_uri: str,
    db_uri: str,
) -> FullVerificationSnapshot:
    """Read every backend fully before performing any contract checks."""
    return build_full_snapshot(
        milvus=collect_milvus_records(milvus_uri),
        elasticsearch=collect_elasticsearch_records(es_uri),
        sqlite=collect_sqlite_records(db_uri),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="JOIN all record IDs across Milvus, ES, and SQLite"
    )
    parser.add_argument("--milvus-uri", default="http://localhost:19530")
    parser.add_argument("--es-uri", default="http://localhost:9200")
    parser.add_argument(
        "--db-uri",
        default="sqlite:///data/metadata.db",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed"),
        help="Root used to resolve SQLite image_rel_path values",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/processed/dataset-manifest.building.json"),
        help="BUILDING or READY manifest to reconcile against live databases",
    )
    args = parser.parse_args()

    try:
        snapshot = collect_full_snapshot(
            milvus_uri=args.milvus_uri,
            es_uri=args.es_uri,
            db_uri=args.db_uri,
        )
        manifest = json.loads(
            args.manifest_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        print(f"Verification failed while reading resources: {exc}")
        return 1

    errors = build_full_contract_report(
        snapshot,
        data_root=args.data_root,
        manifest=manifest,
    )
    if errors:
        print("Full self-indexed-v2 verification FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Full self-indexed-v2 verification PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
