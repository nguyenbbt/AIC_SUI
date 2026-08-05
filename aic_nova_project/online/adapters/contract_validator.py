"""Read-only audit of the Offline-to-Online database contract."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any

from pydantic import AfterValidator, Field, PlainSerializer

from online.config import OnlineDataConfig
from online.domain.base import (
    NonEmptyStr,
    StrictFrozenModel,
    StrictIntValue,
    freeze_mapping,
    serialize_mapping,
)
from online.domain.errors import DataInfrastructureError, ErrorCode
from online.domain.identifiers import validate_canonical_frame_id


VALIDATOR_VERSION = "4-self-indexed-v2"


class ValidationStatus(str, Enum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"


class CheckStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"


class ContractCheck(StrictFrozenModel):
    name: NonEmptyStr
    status: CheckStatus
    required: bool
    message: NonEmptyStr
    error_code: ErrorCode | None = None


class ContractValidationReport(StrictFrozenModel):
    status: ValidationStatus
    checks: tuple[ContractCheck, ...]
    dimensions: Annotated[
        Mapping[str, int],
        AfterValidator(freeze_mapping),
        PlainSerializer(serialize_mapping, return_type=dict),
    ] = Field(
        default_factory=dict
    )
    resources_checked: tuple[NonEmptyStr, ...] = ()
    checks_skipped: tuple[NonEmptyStr, ...] = ()
    sample_counts: Annotated[
        Mapping[str, Annotated[StrictIntValue, Field(ge=0)]],
        AfterValidator(freeze_mapping),
        PlainSerializer(serialize_mapping, return_type=dict),
    ] = Field(default_factory=dict)
    generated_at_utc: NonEmptyStr = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    validator_version: NonEmptyStr = VALIDATOR_VERSION

    @property
    def failed_checks(self) -> tuple[ContractCheck, ...]:
        return tuple(check for check in self.checks if check.status is CheckStatus.FAIL)

    @property
    def blocking_checks(self) -> tuple[ContractCheck, ...]:
        return tuple(
            check
            for check in self.checks
            if check.required and check.status in {CheckStatus.FAIL, CheckStatus.NOT_RUN}
        )


class OfflineContractValidator:
    """Validate schemas, canonical IDs, vector compatibility and logical JOINs.

    Every database call is read-only. Visual Milvus and SQLite metadata checks
    are required; OCR, ASR, summary, objects and Elasticsearch may degrade the
    report to ``PARTIAL``. A required encoder check that was not supplied is
    explicitly ``NOT_RUN`` and blocks a full integration ``PASS``.
    """

    MILVUS_REQUIREMENTS = {
        "visual": (
            {
                "frame_id": "VARCHAR",
                "video_id": "VARCHAR",
                "shot_id": "INT64",
                "embedding": "FLOAT_VECTOR",
            },
            True,
        ),
        "ocr": (
            {
                "frame_id": "VARCHAR",
                "video_id": "VARCHAR",
                "embedding": "FLOAT_VECTOR",
            },
            False,
        ),
        "asr": (
            {
                "video_id": "VARCHAR",
                "interval_id": "VARCHAR",
                "start_time_sec": "FLOAT",
                "end_time_sec": "FLOAT",
                "embedding": "FLOAT_VECTOR",
            },
            False,
        ),
        "summary": (
            {"video_id": "VARCHAR", "embedding": "FLOAT_VECTOR"},
            False,
        ),
    }
    ES_REQUIREMENTS = {
        "ocr": (
            {
                "frame_id": "keyword",
                "video_id": "keyword",
                "shot_id": "keyword",
                "ocr_text_concat": "text",
            },
            False,
        ),
        "asr": (
            {
                "video_id": "keyword",
                "interval_id": "keyword",
                "start_time_sec": "float",
                "end_time_sec": "float",
                "cleaned_text": "text",
            },
            False,
        ),
        "summary": (
            {"video_id": "keyword", "summary": "text"},
            False,
        ),
    }
    SQLITE_REQUIREMENTS = {
        "videos": (
            {
                "video_id": "TEXT",
                "source_video_rel_path": "TEXT",
                "fps": "REAL",
                "duration_sec": "REAL",
                "frame_count": "INTEGER",
                "width": "INTEGER",
                "height": "INTEGER",
            },
            True,
        ),
        "metadata": (
            {
                "frame_id": "TEXT",
                "video_id": "TEXT",
                "shot_id": "INTEGER",
                "source_frame_idx": "INTEGER",
                "timestamp": "REAL",
                "image_rel_path": "TEXT",
            },
            True,
        ),
        "objects": (
            {
                "id": "INTEGER",
                "frame_id": "TEXT",
                "label": "TEXT",
                "confidence": "REAL",
                "x_min": "REAL",
                "y_min": "REAL",
                "x_max": "REAL",
                "y_max": "REAL",
                "model_source": "TEXT",
            },
            False,
        ),
    }

    def __init__(
        self,
        config: OnlineDataConfig,
        *,
        milvus: Any,
        elasticsearch: Any,
        sqlite: Any,
        manifest_gate: Any | None = None,
        encoder_smoke_vectors: Mapping[str, Callable[[], Sequence[float]]] | None = None,
        sample_size: int = 5,
        norm_tolerance: float = 1e-2,
    ) -> None:
        if not isinstance(sample_size, int) or isinstance(sample_size, bool) or sample_size < 1:
            raise ValueError("sample_size must be >= 1")
        if not math.isfinite(norm_tolerance) or norm_tolerance <= 0:
            raise ValueError("norm_tolerance must be finite and > 0")
        self.config = config
        self.milvus = milvus
        self.elasticsearch = elasticsearch
        self.sqlite = sqlite
        self.manifest_gate = manifest_gate
        self.encoder_smoke_vectors = dict(encoder_smoke_vectors or {})
        self.sample_size = sample_size
        self.norm_tolerance = norm_tolerance
        self._checks: list[ContractCheck] = []
        self._dimensions: dict[str, int] = {}
        self._available: dict[str, bool] = {}
        self._samples: dict[str, tuple[Mapping[str, Any], ...]] = {}
        self._sample_counts: dict[str, int] = {}
        self._resources_checked: list[str] = []

    def _record(
        self,
        name: str,
        *,
        ok: bool,
        required: bool,
        success: str,
        failure: str,
        code: ErrorCode = ErrorCode.CONTRACT_MISMATCH,
    ) -> None:
        if ok:
            status = CheckStatus.PASS
            message = success
            error_code = None
        else:
            status = CheckStatus.FAIL if required else CheckStatus.WARNING
            message = failure
            error_code = code
        self._checks.append(
            ContractCheck(
                name=name,
                status=status,
                required=required,
                message=message,
                error_code=error_code,
            )
        )

    def _not_run(self, name: str, *, required: bool, reason: str) -> None:
        self._checks.append(
            ContractCheck(
                name=name,
                status=CheckStatus.NOT_RUN,
                required=required,
                message=reason,
                error_code=None,
            )
        )

    @staticmethod
    def _safe_exception(exc: Exception) -> str:
        if isinstance(exc, DataInfrastructureError):
            return f"{type(exc).__name__}: {exc.message}"
        return f"{type(exc).__name__}: check could not complete"

    def _guard(self, name: str, required: bool, function: Callable[[], None]) -> None:
        try:
            function()
        except Exception as exc:
            self._record(
                name,
                ok=False,
                required=required,
                success="check passed",
                failure=self._safe_exception(exc),
                code=getattr(exc, "code", ErrorCode.RESOURCE_UNAVAILABLE),
            )

    def _guard_resource(
        self,
        *,
        prefix: str,
        availability_key: str,
        required: bool,
        child_checks: Sequence[str],
        function: Callable[[], None],
    ) -> None:
        """Run one resource audit and explicitly account for unfinished checks."""

        try:
            function()
        except Exception as exc:
            self._available[availability_key] = False
            self._record(
                prefix,
                ok=False,
                required=required,
                success="resource validation passed",
                failure=self._safe_exception(exc),
                code=getattr(exc, "code", ErrorCode.RESOURCE_UNAVAILABLE),
            )
            self._skip_resource_checks(
                prefix,
                child_checks,
                required=required,
                reason="resource validation stopped before this check could run",
            )

    def _ensure_check_names(
        self,
        checks: Sequence[tuple[str, bool]],
        *,
        reason: str,
    ) -> None:
        existing = {check.name for check in self._checks}
        for name, required in checks:
            if name not in existing:
                self._not_run(name, required=required, reason=reason)
                existing.add(name)

    @staticmethod
    def _mapping_properties(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
        properties = mapping.get("properties", {})
        return properties if isinstance(properties, Mapping) else {}

    def _store_samples(
        self, key: str, records: Sequence[Mapping[str, Any]]
    ) -> tuple[Mapping[str, Any], ...]:
        samples = tuple(records)
        self._samples[key] = samples
        self._sample_counts[key] = len(samples)
        return samples

    def _skip_resource_checks(
        self,
        prefix: str,
        names: Sequence[str],
        *,
        required: bool,
        reason: str,
    ) -> None:
        existing = {check.name for check in self._checks}
        for name in names:
            full_name = f"{prefix}.{name}"
            if full_name not in existing:
                self._not_run(full_name, required=required, reason=reason)
                existing.add(full_name)

    def _validate_milvus_resource(
        self,
        logical_name: str,
        resource: str,
        fields: Mapping[str, str],
        required: bool,
    ) -> None:
        key = f"milvus:{logical_name}"
        prefix = f"milvus.{resource}"
        exists = bool(self.milvus.collection_exists(resource))
        self._available[key] = exists
        self._record(
            f"{prefix}.exists",
            ok=exists,
            required=required,
            success="collection exists",
            failure="collection is missing",
            code=ErrorCode.RESOURCE_UNAVAILABLE,
        )
        if not exists:
            self._skip_resource_checks(
                prefix,
                ("fields", "types", "dimension", "index", "non_empty", "vector_norm"),
                required=required,
                reason="collection is missing",
            )
            return

        description = self.milvus.describe_collection(resource)
        if not isinstance(description, Mapping):
            raise TypeError("Milvus collection description is not a mapping")
        field_types = description.get("fields", {})
        if not isinstance(field_types, Mapping):
            field_types = {}
        actual_fields = set(field_types)
        self._record(
            f"{prefix}.fields",
            ok=set(fields) <= actual_fields,
            required=required,
            success="required fields are present",
            failure=f"missing fields: {sorted(set(fields) - actual_fields)}",
        )
        types_ok = all(
            str(field_types.get(field, "")).upper() == expected
            for field, expected in fields.items()
        )
        self._record(
            f"{prefix}.types",
            ok=types_ok,
            required=required,
            success="field types match contract",
            failure="one or more field types do not match contract",
        )

        dimension = description.get("dimension")
        valid_dim = isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0
        self._record(
            f"{prefix}.dimension",
            ok=valid_dim,
            required=required,
            success=f"dimension={dimension}",
            failure="embedding dimension is missing or invalid",
            code=ErrorCode.DIMENSION_MISMATCH,
        )
        if valid_dim:
            self._dimensions[resource] = dimension

        metric = str(description.get("metric_type") or "").upper()
        index_type = str(description.get("index_type") or "").upper()
        self._record(
            f"{prefix}.index",
            ok=metric == "IP" and index_type == "HNSW",
            required=required,
            success="HNSW/IP index contract matches",
            failure=f"expected HNSW/IP, got {index_type or 'unknown'}/{metric or 'unknown'}",
        )

        records = self._store_samples(
            key,
            self.milvus.sample_records(resource, tuple(fields), self.sample_size),
        )
        self._record(
            f"{prefix}.non_empty",
            ok=bool(records),
            required=required,
            success=f"collection sample count={len(records)}",
            failure="collection is empty",
        )
        if not records:
            self._not_run(
                f"{prefix}.vector_norm",
                required=required,
                reason="collection has no sample vector",
            )
            return

        norms_ok = True
        for record in records:
            vector = record.get("embedding")
            if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
                norms_ok = False
                break
            try:
                values = tuple(float(value) for value in vector)
            except (TypeError, ValueError):
                norms_ok = False
                break
            norm = math.sqrt(sum(value * value for value in values))
            if (
                not values
                or not all(math.isfinite(value) for value in values)
                or (valid_dim and len(values) != dimension)
                or abs(norm - 1.0) > self.norm_tolerance
            ):
                norms_ok = False
                break
        self._record(
            f"{prefix}.vector_norm",
            ok=norms_ok,
            required=required,
            success="sample vectors are finite, dimensioned and L2-normalized",
            failure="sample vector is invalid, wrong-sized or not L2-normalized",
            code=ErrorCode.DIMENSION_MISMATCH,
        )

    def _validate_es_resource(
        self,
        logical_name: str,
        resource: str,
        fields: Mapping[str, str],
        required: bool,
    ) -> None:
        key = f"es:{logical_name}"
        prefix = f"elasticsearch.{resource}"
        exists = bool(self.elasticsearch.index_exists(resource))
        self._available[key] = exists
        self._record(
            f"{prefix}.exists",
            ok=exists,
            required=required,
            success="index exists",
            failure="index is missing",
            code=ErrorCode.RESOURCE_UNAVAILABLE,
        )
        if not exists:
            self._skip_resource_checks(
                prefix,
                ("fields", "types", "analyzer", "non_empty"),
                required=required,
                reason="index is missing",
            )
            return

        mapping = self.elasticsearch.get_mapping(resource)
        properties = self._mapping_properties(mapping)
        actual_fields = set(properties)
        self._record(
            f"{prefix}.fields",
            ok=set(fields) <= actual_fields,
            required=required,
            success="required fields are present",
            failure=f"missing fields: {sorted(set(fields) - actual_fields)}",
        )
        types_ok = all(
            isinstance(properties.get(field), Mapping)
            and str(properties[field].get("type", "")).lower() == expected
            for field, expected in fields.items()
        )
        self._record(
            f"{prefix}.types",
            ok=types_ok,
            required=required,
            success="field mappings match contract",
            failure="one or more field mappings do not match contract",
        )
        text_fields = [
            name
            for name, value in properties.items()
            if isinstance(value, Mapping) and value.get("type") == "text"
        ]
        analyzer_ok = bool(text_fields) and all(
            properties[field].get("analyzer") == "vietnamese_analyzer"
            for field in text_fields
        )
        self._record(
            f"{prefix}.analyzer",
            ok=analyzer_ok,
            required=required,
            success="text fields use vietnamese_analyzer",
            failure="text field analyzer does not match vietnamese_analyzer",
        )
        documents = self._store_samples(
            key,
            self.elasticsearch.sample_documents(resource, tuple(fields), self.sample_size),
        )
        self._record(
            f"{prefix}.non_empty",
            ok=bool(documents),
            required=required,
            success=f"index sample count={len(documents)}",
            failure="index is empty",
        )

    def _validate_sqlite_resource(
        self,
        logical_name: str,
        table: str,
        fields: Mapping[str, str],
        required: bool,
    ) -> None:
        key = f"sqlite:{logical_name}"
        prefix = f"sqlite.{table}"
        columns = self.sqlite.table_columns(table)
        exists = bool(columns)
        self._available[key] = exists
        self._record(
            f"{prefix}.exists",
            ok=exists,
            required=required,
            success="table exists",
            failure="table is missing",
            code=ErrorCode.RESOURCE_UNAVAILABLE,
        )
        if not exists:
            self._skip_resource_checks(
                prefix,
                ("fields", "types", "non_empty"),
                required=required,
                reason="table is missing",
            )
            return

        actual_fields = set(columns)
        self._record(
            f"{prefix}.fields",
            ok=set(fields) <= actual_fields,
            required=required,
            success="required columns are present",
            failure=f"missing columns: {sorted(set(fields) - actual_fields)}",
        )
        types_ok = all(
            str(columns.get(field, "")).upper() == expected
            for field, expected in fields.items()
        )
        self._record(
            f"{prefix}.types",
            ok=types_ok,
            required=required,
            success="column types match contract",
            failure="one or more column types do not match contract",
        )
        records = self._store_samples(
            key,
            self.sqlite.sample_records(table, tuple(fields), self.sample_size),
        )
        self._record(
            f"{prefix}.non_empty",
            ok=bool(records),
            required=required,
            success=f"table sample count={len(records)}",
            failure="table is empty",
        )

    def _validate_canonical_ids(self) -> None:
        specifications = (
            ("milvus:visual", "canonical_id.milvus.visual", True, True, True),
            ("milvus:ocr", "canonical_id.milvus.ocr", False, True, False),
            ("es:ocr", "canonical_id.elasticsearch.ocr", False, True, True),
            ("sqlite:metadata", "canonical_id.sqlite.metadata", True, True, True),
            ("sqlite:objects", "canonical_id.sqlite.objects", False, False, False),
        )
        for key, name, required, has_video, has_shot in specifications:
            if not self._available.get(key):
                self._not_run(name, required=required, reason=f"{key} is unavailable")
                continue
            records = self._samples.get(key, ())
            valid = bool(records)
            for record in records:
                frame_id = record.get("frame_id")
                video_id = record.get("video_id") if has_video else None
                shot_id = record.get("shot_id") if has_shot else None
                if not isinstance(frame_id, str):
                    valid = False
                    break
                if video_id is not None and not isinstance(video_id, str):
                    valid = False
                    break
                try:
                    semantic_shot = int(shot_id) if has_shot else None
                    validate_canonical_frame_id(
                        frame_id,
                        video_id=video_id,
                        shot_id=semantic_shot,
                    )
                except Exception:
                    valid = False
                    break
            self._record(
                name,
                ok=valid,
                required=required,
                success=f"{len(records)} sampled frame_id values are canonical",
                failure=f"{key} has malformed or semantically inconsistent frame_id samples",
            )

    def _validate_frame_joins(self) -> None:
        if not (
            self._available.get("sqlite:metadata")
            and self._available.get("sqlite:videos")
        ):
            self._not_run(
                "join.metadata_to_videos",
                required=True,
                reason="metadata or videos table is unavailable",
            )
        else:
            records = self._samples.get("sqlite:metadata", ())
            frame_ids = [
                record["frame_id"]
                for record in records
                if isinstance(record.get("frame_id"), str)
            ]
            video_ids = tuple(
                dict.fromkeys(
                    record["video_id"]
                    for record in records
                    if isinstance(record.get("video_id"), str)
                )
            )
            frames = self.sqlite.get_frames_by_ids(frame_ids)
            videos = self.sqlite.get_videos_by_ids(video_ids)
            matches = (
                bool(records)
                and len(frame_ids) == len(records)
                and len(frames) == len(set(frame_ids))
            )
            for record in records:
                frame_id = record.get("frame_id")
                video_id = record.get("video_id")
                frame = frames.get(frame_id) if isinstance(frame_id, str) else None
                video = videos.get(video_id) if isinstance(video_id, str) else None
                if (
                    frame is None
                    or video is None
                    or frame.video_id != video.video_id
                    or frame.source_frame_idx >= video.frame_count
                    or frame.timestamp_sec > video.duration_sec + max(1.0 / video.fps, 0.05)
                ):
                    matches = False
                    break
            self._record(
                "join.metadata_to_videos",
                ok=matches,
                required=True,
                success="metadata samples JOIN videos and stay within source bounds",
                failure="metadata samples do not JOIN videos or exceed source bounds",
            )

        if not (
            self._available.get("milvus:visual")
            and self._available.get("sqlite:metadata")
        ):
            self._not_run(
                "join.visual_to_metadata",
                required=True,
                reason="visual collection or metadata table is unavailable",
            )
        else:
            records = self._samples.get("milvus:visual", ())
            frame_ids = [
                record["frame_id"]
                for record in records
                if isinstance(record.get("frame_id"), str)
            ]
            metadata = self.sqlite.get_frames_by_ids(frame_ids)
            matches = bool(records) and len(frame_ids) == len(records)
            for record in records:
                frame_id = record.get("frame_id")
                row = metadata.get(frame_id) if isinstance(frame_id, str) else None
                try:
                    if (
                        row is None
                        or row.video_id != record.get("video_id")
                        or row.shot_id != int(record.get("shot_id"))
                    ):
                        matches = False
                        break
                except (TypeError, ValueError):
                    matches = False
                    break
            self._record(
                "join.visual_to_metadata",
                ok=matches,
                required=True,
                success="visual frame samples JOIN matching SQLite metadata",
                failure="visual samples do not JOIN canonical SQLite metadata",
            )

        if not (
            self._available.get("milvus:ocr")
            and self._available.get("sqlite:metadata")
        ):
            self._not_run(
                "join.ocr_dense_to_metadata",
                required=False,
                reason="OCR dense collection or metadata table is unavailable",
            )
        else:
            records = self._samples.get("milvus:ocr", ())
            ids = [
                record["frame_id"]
                for record in records
                if isinstance(record.get("frame_id"), str)
            ]
            metadata = self.sqlite.get_frames_by_ids(ids)
            matches = bool(records) and len(ids) == len(records) and all(
                frame_id in metadata
                and metadata[frame_id].video_id == record.get("video_id")
                for frame_id, record in zip(ids, records)
            )
            self._record(
                "join.ocr_dense_to_metadata",
                ok=matches,
                required=False,
                success="OCR dense samples JOIN SQLite metadata",
                failure="OCR dense samples do not JOIN SQLite metadata",
            )

        if not (
            self._available.get("es:ocr")
            and self._available.get("sqlite:metadata")
        ):
            self._not_run(
                "join.ocr_lexical_to_metadata",
                required=False,
                reason="OCR lexical index or metadata table is unavailable",
            )
        else:
            records = self._samples.get("es:ocr", ())
            ids = [
                record["frame_id"]
                for record in records
                if isinstance(record.get("frame_id"), str)
            ]
            metadata = self.sqlite.get_frames_by_ids(ids)
            matches = bool(records) and len(ids) == len(records)
            for frame_id, record in zip(ids, records):
                row = metadata.get(frame_id)
                try:
                    if (
                        row is None
                        or row.video_id != record.get("video_id")
                        or row.shot_id != int(record.get("shot_id"))
                    ):
                        matches = False
                        break
                except (TypeError, ValueError):
                    matches = False
                    break
            self._record(
                "join.ocr_lexical_to_metadata",
                ok=matches,
                required=False,
                success="OCR lexical samples JOIN SQLite metadata",
                failure="OCR lexical samples do not JOIN SQLite metadata",
            )

        if not (
            self._available.get("milvus:ocr")
            and self._available.get("es:ocr")
        ):
            self._not_run(
                "join.ocr_dense_to_lexical",
                required=False,
                reason="OCR dense collection or lexical index is unavailable",
            )
        else:
            records = self._samples.get("milvus:ocr", ())
            matches = bool(records)
            for record in records:
                documents = self.elasticsearch.find_documents(
                    self.config.elasticsearch.ocr_index,
                    {"frame_id": record.get("frame_id")},
                    ("frame_id", "video_id"),
                    limit=2,
                )
                if (
                    len(documents) != 1
                    or documents[0].get("frame_id") != record.get("frame_id")
                    or documents[0].get("video_id") != record.get("video_id")
                ):
                    matches = False
                    break
            self._record(
                "join.ocr_dense_to_lexical",
                ok=matches,
                required=False,
                success="OCR samples JOIN between Milvus and Elasticsearch",
                failure="OCR samples do not JOIN between Milvus and Elasticsearch",
            )

    def _validate_asr_join(self) -> None:
        if not (
            self._available.get("milvus:asr")
            and self._available.get("es:asr")
        ):
            self._not_run(
                "join.asr_interval",
                required=False,
                reason="ASR dense collection or lexical index is unavailable",
            )
            return
        records = self._samples.get("milvus:asr", ())
        matches = bool(records)
        for record in records:
            documents = self.elasticsearch.find_documents(
                self.config.elasticsearch.asr_index,
                {
                    "video_id": record.get("video_id"),
                    "interval_id": record.get("interval_id"),
                },
                ("video_id", "interval_id"),
                limit=2,
            )
            if (
                len(documents) != 1
                or documents[0].get("video_id") != record.get("video_id")
                or documents[0].get("interval_id") != record.get("interval_id")
            ):
                matches = False
                break
        self._record(
            "join.asr_interval",
            ok=matches,
            required=False,
            success="ASR (video_id, interval_id) samples JOIN",
            failure="ASR (video_id, interval_id) samples do not JOIN",
        )

    def _validate_summary_join(self) -> None:
        if not (
            self._available.get("milvus:summary")
            and self._available.get("es:summary")
        ):
            self._not_run(
                "join.summary_video",
                required=False,
                reason="summary dense collection or lexical index is unavailable",
            )
            return
        records = self._samples.get("milvus:summary", ())
        matches = bool(records)
        for record in records:
            documents = self.elasticsearch.find_documents(
                self.config.elasticsearch.summary_index,
                {"video_id": record.get("video_id")},
                ("video_id",),
                limit=2,
            )
            if (
                len(documents) != 1
                or documents[0].get("video_id") != record.get("video_id")
            ):
                matches = False
                break
        self._record(
            "join.summary_video",
            ok=matches,
            required=False,
            success="summary video_id samples JOIN",
            failure="summary video_id samples do not JOIN",
        )

    def _validate_object_join(self) -> None:
        if not (
            self._available.get("sqlite:objects")
            and self._available.get("sqlite:metadata")
        ):
            self._not_run(
                "join.objects_to_metadata",
                required=False,
                reason="objects or metadata table is unavailable",
            )
            return
        records = self._samples.get("sqlite:objects", ())
        ids = [
            record["frame_id"]
            for record in records
            if isinstance(record.get("frame_id"), str)
        ]
        metadata = self.sqlite.get_frames_by_ids(ids)
        video_ids = tuple(
            dict.fromkeys(
                metadata[frame_id].video_id
                for frame_id in ids
                if frame_id in metadata
            )
        )
        videos = self.sqlite.get_videos_by_ids(video_ids)
        matches = bool(records) and len(ids) == len(records)
        for record in records:
            frame_id = record.get("frame_id")
            frame = metadata.get(frame_id) if isinstance(frame_id, str) else None
            video = videos.get(frame.video_id) if frame is not None else None
            try:
                x_min = float(record.get("x_min"))
                y_min = float(record.get("y_min"))
                x_max = float(record.get("x_max"))
                y_max = float(record.get("y_max"))
                confidence = float(record.get("confidence"))
                label = record.get("label")
                if (
                    frame is None
                    or video is None
                    or not isinstance(label, str)
                    or not label
                    or label != label.casefold()
                    or not all(math.isfinite(value) for value in (x_min, y_min, x_max, y_max, confidence))
                    or not 0.0 <= confidence <= 1.0
                    or not 0.0 <= x_min < x_max <= video.width
                    or not 0.0 <= y_min < y_max <= video.height
                ):
                    matches = False
                    break
            except (TypeError, ValueError):
                matches = False
                break
        self._record(
            "join.objects_to_metadata",
            ok=matches,
            required=False,
            success="object frame samples JOIN metadata",
            failure="object frame samples do not JOIN metadata",
        )

    def _validate_encoders(self) -> None:
        resources = (
            ("visual", self.config.milvus.visual_collection, True),
            ("ocr", self.config.milvus.ocr_collection, False),
            ("asr", self.config.milvus.asr_collection, False),
            ("summary", self.config.milvus.summary_collection, False),
        )
        for logical_name, resource, required in resources:
            name = f"encoder.{resource}"
            if not self._available.get(f"milvus:{logical_name}"):
                self._not_run(
                    name,
                    required=required,
                    reason="collection is unavailable",
                )
                continue
            if resource not in self._dimensions:
                self._not_run(
                    name,
                    required=required,
                    reason="collection dimension is unavailable",
                )
                continue
            factory = self.encoder_smoke_vectors.get(resource)
            if factory is None:
                self._not_run(
                    name,
                    required=required,
                    reason="encoder smoke vector factory was not supplied",
                )
                continue
            try:
                vector = tuple(float(value) for value in factory())
                norm = math.sqrt(sum(value * value for value in vector))
                ok = (
                    len(vector) == self._dimensions[resource]
                    and all(math.isfinite(value) for value in vector)
                    and abs(norm - 1.0) <= self.norm_tolerance
                )
                failure = "encoder dimension or norm does not match collection"
            except Exception as exc:
                ok = False
                failure = f"encoder smoke factory failed: {type(exc).__name__}"
            self._record(
                name,
                ok=ok,
                required=required,
                success="encoder dimension and norm match collection",
                failure=failure,
                code=ErrorCode.DIMENSION_MISMATCH,
            )

    def _validate_icu_plugin(self) -> None:
        present = bool(self.elasticsearch.has_icu_plugin())
        self._record(
            "elasticsearch.analysis_icu",
            ok=present,
            required=False,
            success="analysis-icu plugin is installed",
            failure="analysis-icu plugin is missing",
        )

    def validate(self) -> ContractValidationReport:
        self._checks = []
        self._dimensions = {}
        self._available = {}
        self._samples = {}
        self._sample_counts = {}
        self._resources_checked = []

        if self.manifest_gate is not None:
            self._resources_checked.append("dataset_manifest")
            self._guard("dataset_manifest.ready_identity", True, self.manifest_gate.health_check)
        elif self.config.dataset.manifest_required:
            self._not_run(
                "dataset_manifest.ready_identity",
                required=True,
                reason="required dataset manifest gate was not supplied",
            )

        milvus_resources = {
            "visual": self.config.milvus.visual_collection,
            "ocr": self.config.milvus.ocr_collection,
            "asr": self.config.milvus.asr_collection,
            "summary": self.config.milvus.summary_collection,
        }
        for logical_name, (fields, required) in self.MILVUS_REQUIREMENTS.items():
            resource = milvus_resources[logical_name]
            self._resources_checked.append(f"milvus:{resource}")
            self._guard_resource(
                prefix=f"milvus.{resource}",
                availability_key=f"milvus:{logical_name}",
                required=required,
                child_checks=(
                    "exists",
                    "fields",
                    "types",
                    "dimension",
                    "index",
                    "non_empty",
                    "vector_norm",
                ),
                function=lambda logical_name=logical_name, resource=resource, fields=fields, required=required: self._validate_milvus_resource(
                    logical_name, resource, fields, required
                ),
            )

        es_resources = {
            "ocr": self.config.elasticsearch.ocr_index,
            "asr": self.config.elasticsearch.asr_index,
            "summary": self.config.elasticsearch.summary_index,
        }
        for logical_name, (fields, required) in self.ES_REQUIREMENTS.items():
            resource = es_resources[logical_name]
            self._resources_checked.append(f"elasticsearch:{resource}")
            self._guard_resource(
                prefix=f"elasticsearch.{resource}",
                availability_key=f"es:{logical_name}",
                required=required,
                child_checks=("exists", "fields", "types", "analyzer", "non_empty"),
                function=lambda logical_name=logical_name, resource=resource, fields=fields, required=required: self._validate_es_resource(
                    logical_name, resource, fields, required
                ),
            )
        self._guard("elasticsearch.analysis_icu", False, self._validate_icu_plugin)

        sqlite_resources = {
            "videos": self.config.sqlite.videos_table,
            "metadata": self.config.sqlite.metadata_table,
            "objects": self.config.sqlite.objects_table,
        }
        for logical_name, (fields, required) in self.SQLITE_REQUIREMENTS.items():
            table = sqlite_resources[logical_name]
            self._resources_checked.append(f"sqlite:{table}")
            self._guard_resource(
                prefix=f"sqlite.{table}",
                availability_key=f"sqlite:{logical_name}",
                required=required,
                child_checks=("exists", "fields", "types", "non_empty"),
                function=lambda logical_name=logical_name, table=table, fields=fields, required=required: self._validate_sqlite_resource(
                    logical_name, table, fields, required
                ),
            )

        self._guard("canonical_id", True, self._validate_canonical_ids)
        self._ensure_check_names(
            (
                ("canonical_id.milvus.visual", True),
                ("canonical_id.milvus.ocr", False),
                ("canonical_id.elasticsearch.ocr", False),
                ("canonical_id.sqlite.metadata", True),
                ("canonical_id.sqlite.objects", False),
            ),
            reason="canonical identifier validation did not complete",
        )
        self._guard("join.frame", True, self._validate_frame_joins)
        self._ensure_check_names(
            (
                ("join.metadata_to_videos", True),
                ("join.visual_to_metadata", True),
                ("join.ocr_dense_to_metadata", False),
                ("join.ocr_lexical_to_metadata", False),
                ("join.ocr_dense_to_lexical", False),
            ),
            reason="frame JOIN validation did not complete",
        )
        self._guard("join.asr", False, self._validate_asr_join)
        self._guard("join.summary", False, self._validate_summary_join)
        self._guard("join.objects", False, self._validate_object_join)
        self._ensure_check_names(
            (
                ("join.asr_interval", False),
                ("join.summary_video", False),
                ("join.objects_to_metadata", False),
            ),
            reason="JOIN validation did not complete",
        )
        self._validate_encoders()

        if any(
            check.required and check.status in {CheckStatus.FAIL, CheckStatus.NOT_RUN}
            for check in self._checks
        ):
            status = ValidationStatus.FAIL
        elif any(
            check.status in {CheckStatus.WARNING, CheckStatus.NOT_RUN}
            for check in self._checks
        ):
            status = ValidationStatus.PARTIAL
        else:
            status = ValidationStatus.PASS

        skipped = tuple(
            check.name for check in self._checks if check.status is CheckStatus.NOT_RUN
        )
        return ContractValidationReport(
            status=status,
            checks=tuple(self._checks),
            dimensions=dict(self._dimensions),
            resources_checked=tuple(self._resources_checked),
            checks_skipped=skipped,
            sample_counts=dict(self._sample_counts),
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
        )
