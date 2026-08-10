"""Milvus + SQLite full ordered visual corpus adapter for TRAKE/DANTE."""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

from online.config import MilvusResourceConfig
from online.domain.errors import ContractMismatchError, InvalidQueryError
from online.ports.metadata import MetadataReaderPort
from online.ports.visual_corpus import OrderedVisualBatch, OrderedVisualFrame


class MilvusRecordReader(Protocol):
    def iter_records(
        self,
        name: str,
        output_fields: Sequence[str],
        *,
        filter_expression: str,
        batch_size: int,
    ) -> Iterable[tuple[Mapping[str, Any], ...]]: ...


class MilvusSQLiteVisualCorpusAdapter:
    """Hydrate Milvus vectors with exact SQLite source-frame metadata."""

    def __init__(
        self,
        config: MilvusResourceConfig,
        *,
        milvus: MilvusRecordReader,
        metadata_reader: MetadataReaderPort,
        scan_batch_size: int = 500,
    ) -> None:
        if not isinstance(metadata_reader, MetadataReaderPort):
            raise TypeError("metadata_reader must implement MetadataReaderPort")
        if isinstance(scan_batch_size, bool) or not isinstance(scan_batch_size, int) or scan_batch_size < 1:
            raise ValueError("scan_batch_size must be a positive integer")
        self.config = config
        self._milvus = milvus
        self._metadata = metadata_reader
        self._scan_batch_size = scan_batch_size
        self._video_ids: tuple[str, ...] | None = None
        self._cache_lock = threading.RLock()

    def list_video_ids(self) -> Sequence[str]:
        with self._cache_lock:
            if self._video_ids is not None:
                return self._video_ids
        ids: set[str] = set()
        for batch in self._milvus.iter_records(
            self.config.visual_collection,
            ("video_id",),
            filter_expression='video_id != ""',
            batch_size=self._scan_batch_size,
        ):
            for record in batch:
                ids.add(_text(record, "video_id"))
        result = tuple(sorted(ids))
        with self._cache_lock:
            self._video_ids = result
        return result

    def iter_ordered_frame_embedding_batches(
        self,
        video_id: str,
        batch_size: int,
    ) -> Iterable[OrderedVisualBatch]:
        if not isinstance(video_id, str) or not video_id.strip() or video_id != video_id.strip():
            raise InvalidQueryError("video_id must be a canonical non-empty string")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise InvalidQueryError("batch_size must be a positive integer")

        records: dict[str, tuple[str, int, tuple[float, ...]]] = {}
        expression = f"video_id == {json.dumps(video_id, ensure_ascii=False)}"
        for batch in self._milvus.iter_records(
            self.config.visual_collection,
            ("frame_id", "video_id", "shot_id", "embedding"),
            filter_expression=expression,
            batch_size=self._scan_batch_size,
        ):
            for record in batch:
                frame_id = _text(record, "frame_id")
                if frame_id in records:
                    raise ContractMismatchError("Visual corpus contains a duplicate frame_id")
                record_video_id = _text(record, "video_id")
                if record_video_id != video_id:
                    raise ContractMismatchError("Milvus filter returned a different video_id")
                records[frame_id] = (
                    record_video_id,
                    _non_negative_int(record, "shot_id"),
                    _normalized_vector(record, "embedding", self.config.norm_tolerance),
                )

        if not records:
            return
        metadata = self._metadata.get_frames_by_ids(tuple(records))
        if not isinstance(metadata, Mapping):
            raise ContractMismatchError("Metadata reader returned a non-mapping value")
        missing = set(records) - set(metadata)
        extra = set(metadata) - set(records)
        if missing or extra:
            raise ContractMismatchError(
                "Visual corpus and SQLite metadata frame sets differ",
                details={"missing_count": len(missing), "extra_count": len(extra)},
            )

        ordered_metadata = sorted(
            metadata.values(), key=lambda item: (item.timestamp_sec, item.frame_id)
        )
        frames: list[OrderedVisualFrame] = []
        expected_dimension: int | None = None
        for local_index, frame in enumerate(ordered_metadata):
            record_video_id, shot_id, vector = records[frame.frame_id]
            if frame.video_id != record_video_id or frame.shot_id != shot_id:
                raise ContractMismatchError("Milvus visual identity disagrees with SQLite metadata")
            if expected_dimension is None:
                expected_dimension = len(vector)
            elif len(vector) != expected_dimension:
                raise ContractMismatchError("Visual corpus contains mixed vector dimensions")
            frames.append(
                OrderedVisualFrame(
                    frame_id=frame.frame_id,
                    video_id=frame.video_id,
                    shot_id=frame.shot_id,
                    local_index=local_index,
                    timestamp_sec=frame.timestamp_sec,
                    source_frame_idx=frame.source_frame_idx,
                    image_rel_path=frame.image_rel_path,
                    vector=vector,
                )
            )
        for start in range(0, len(frames), batch_size):
            yield tuple(frames[start : start + batch_size])


def _text(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ContractMismatchError(
            "Milvus corpus field must be canonical non-empty text",
            details={"field": field},
        )
    return value


def _non_negative_int(record: Mapping[str, Any], field: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractMismatchError(
            "Milvus corpus field must be a non-negative integer",
            details={"field": field},
        )
    return value


def _normalized_vector(
    record: Mapping[str, Any],
    field: str,
    tolerance: float,
) -> tuple[float, ...]:
    value = record.get(field)
    if isinstance(value, (str, bytes)):
        raise ContractMismatchError("Milvus corpus embedding must be a numeric sequence")
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise ContractMismatchError("Milvus corpus embedding must be a numeric sequence") from exc
    if not raw or any(isinstance(item, bool) for item in raw):
        raise ContractMismatchError("Milvus corpus embedding must be non-empty and numeric")
    try:
        vector = tuple(float(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise ContractMismatchError("Milvus corpus embedding must be numeric") from exc
    if not all(math.isfinite(item) for item in vector):
        raise ContractMismatchError("Milvus corpus embedding must be finite")
    norm = math.sqrt(sum(item * item for item in vector))
    if abs(norm - 1.0) > tolerance:
        raise ContractMismatchError("Milvus corpus embedding must be L2-normalized")
    return vector


__all__ = ["MilvusRecordReader", "MilvusSQLiteVisualCorpusAdapter"]
