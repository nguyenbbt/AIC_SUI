"""Read-only audit of the Offline-to-Online database contract."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Any

from pydantic import Field

from online.config import OnlineDataConfig
from online.domain.base import NonEmptyStr, StrictFrozenModel
from online.domain.errors import ErrorCode


class ValidationStatus(str, Enum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"


class CheckStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class ContractCheck(StrictFrozenModel):
    name: NonEmptyStr
    status: CheckStatus
    required: bool
    message: NonEmptyStr
    error_code: ErrorCode | None = None


class ContractValidationReport(StrictFrozenModel):
    status: ValidationStatus
    checks: tuple[ContractCheck, ...]
    dimensions: dict[str, int] = Field(default_factory=dict)

    @property
    def failed_checks(self) -> tuple[ContractCheck, ...]:
        return tuple(check for check in self.checks if check.status is CheckStatus.FAIL)


class OfflineContractValidator:
    """Validates schemas, vector norms and cross-database logical keys.

    The validator only calls read methods on adapters. Missing visual/metadata
    resources are core failures. Missing OCR/ASR/summary/object resources are
    reported as PARTIAL because those branches may degrade by design.
    """

    MILVUS_REQUIREMENTS = {
        "visual": ({"frame_id": "VARCHAR", "video_id": "VARCHAR", "shot_id": "INT64", "embedding": "FLOAT_VECTOR"}, True),
        "ocr": ({"frame_id": "VARCHAR", "video_id": "VARCHAR", "embedding": "FLOAT_VECTOR"}, False),
        "asr": (
            {"video_id": "VARCHAR", "interval_id": "VARCHAR", "start_time_sec": "FLOAT", "end_time_sec": "FLOAT", "embedding": "FLOAT_VECTOR"},
            False,
        ),
        "summary": ({"video_id": "VARCHAR", "embedding": "FLOAT_VECTOR"}, False),
    }
    ES_REQUIREMENTS = {
        "ocr": ({"frame_id": "keyword", "video_id": "keyword", "shot_id": "keyword", "ocr_text_concat": "text"}, False),
        "asr": ({"video_id": "keyword", "interval_id": "keyword", "start_time": "float", "end_time": "float", "cleaned_text": "text"}, False),
        "summary": ({"video_id": "keyword", "summary": "text"}, False),
    }
    SQLITE_REQUIREMENTS = {
        "metadata": ({"frame_id": "TEXT", "video_id": "TEXT", "shot_id": "INTEGER", "timestamp": "REAL"}, True),
        "objects": (
            {"id": "INTEGER", "frame_id": "TEXT", "label": "TEXT", "confidence": "REAL", "x_min": "REAL", "y_min": "REAL", "x_max": "REAL", "y_max": "REAL"},
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
        encoder_smoke_vectors: Mapping[str, Callable[[], Sequence[float]]] | None = None,
        sample_size: int = 5,
        norm_tolerance: float = 1e-2,
    ) -> None:
        if sample_size < 1:
            raise ValueError("sample_size must be >= 1")
        self.config = config
        self.milvus = milvus
        self.elasticsearch = elasticsearch
        self.sqlite = sqlite
        self.encoder_smoke_vectors = dict(encoder_smoke_vectors or {})
        self.sample_size = sample_size
        self.norm_tolerance = norm_tolerance
        self._checks: list[ContractCheck] = []
        self._dimensions: dict[str, int] = {}
        self._available: dict[str, bool] = {}

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

    def _guard(self, name: str, required: bool, function: Callable[[], None]) -> None:
        try:
            function()
        except Exception as exc:
            self._record(
                name,
                ok=False,
                required=required,
                success="check passed",
                failure=f"{type(exc).__name__}: {exc}",
                code=getattr(exc, "code", ErrorCode.RESOURCE_UNAVAILABLE),
            )

    @staticmethod
    def _mapping_properties(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
        properties = mapping.get("properties", {})
        return properties if isinstance(properties, Mapping) else {}

    def _validate_milvus_resource(
        self, logical_name: str, resource: str, fields: Mapping[str, str], required: bool
    ) -> None:
        exists = bool(self.milvus.collection_exists(resource))
        self._available[f"milvus:{logical_name}"] = exists
        self._record(
            f"milvus.{resource}.exists",
            ok=exists,
            required=required,
            success="collection exists",
            failure="collection is missing",
            code=ErrorCode.RESOURCE_UNAVAILABLE,
        )
        if not exists:
            return
        description = self.milvus.describe_collection(resource)
        field_types = description.get("fields", {})
        actual_fields = set(field_types)
        self._record(
            f"milvus.{resource}.fields",
            ok=set(fields) <= actual_fields,
            required=required,
            success="required fields are present",
            failure=f"missing fields: {sorted(set(fields) - actual_fields)}",
        )
        types_ok = all(str(field_types.get(field, "")).upper() == expected for field, expected in fields.items())
        self._record(
            f"milvus.{resource}.types",
            ok=types_ok,
            required=required,
            success="field types match contract",
            failure="one or more field types do not match contract",
        )
        dimension = description.get("dimension")
        valid_dim = isinstance(dimension, int) and dimension > 0
        self._record(
            f"milvus.{resource}.dimension",
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
            f"milvus.{resource}.index",
            ok=metric == "IP" and index_type == "HNSW",
            required=required,
            success="HNSW/IP index contract matches",
            failure=f"expected HNSW/IP, got {index_type or 'unknown'}/{metric or 'unknown'}",
        )
        records = self.milvus.sample_records(resource, ("embedding",), self.sample_size)
        non_empty = bool(records)
        self._record(
            f"milvus.{resource}.non_empty",
            ok=non_empty,
            required=required,
            success="collection has sample records",
            failure="collection is empty",
        )
        if records:
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
                if not values or not all(math.isfinite(value) for value in values) or abs(norm - 1.0) > self.norm_tolerance:
                    norms_ok = False
                    break
            self._record(
                f"milvus.{resource}.vector_norm",
                ok=norms_ok,
                required=required,
                success="sample vectors are finite and L2-normalized",
                failure="sample vector is invalid or not L2-normalized",
            )

    def _validate_es_resource(
        self, logical_name: str, resource: str, fields: Mapping[str, str], required: bool
    ) -> None:
        exists = bool(self.elasticsearch.index_exists(resource))
        self._available[f"es:{logical_name}"] = exists
        self._record(
            f"elasticsearch.{resource}.exists",
            ok=exists,
            required=required,
            success="index exists",
            failure="index is missing",
            code=ErrorCode.RESOURCE_UNAVAILABLE,
        )
        if not exists:
            return
        mapping = self.elasticsearch.get_mapping(resource)
        properties = self._mapping_properties(mapping)
        actual_fields = set(properties)
        self._record(
            f"elasticsearch.{resource}.fields",
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
            f"elasticsearch.{resource}.types",
            ok=types_ok,
            required=required,
            success="field mappings match contract",
            failure="one or more field mappings do not match contract",
        )
        text_fields = [name for name, value in properties.items() if isinstance(value, Mapping) and value.get("type") == "text"]
        analyzer_ok = all(properties[field].get("analyzer") == "vietnamese_analyzer" for field in text_fields)
        self._record(
            f"elasticsearch.{resource}.analyzer",
            ok=analyzer_ok,
            required=required,
            success="text fields use vietnamese_analyzer",
            failure="text field analyzer does not match vietnamese_analyzer",
        )
        documents = self.elasticsearch.sample_documents(resource, tuple(fields), 1)
        self._record(
            f"elasticsearch.{resource}.non_empty",
            ok=bool(documents),
            required=required,
            success="index has sample documents",
            failure="index is empty",
        )

    def _validate_sqlite_resource(
        self, logical_name: str, table: str, fields: Mapping[str, str], required: bool
    ) -> None:
        columns = self.sqlite.table_columns(table)
        exists = bool(columns)
        self._available[f"sqlite:{logical_name}"] = exists
        self._record(
            f"sqlite.{table}.exists",
            ok=exists,
            required=required,
            success="table exists",
            failure="table is missing",
            code=ErrorCode.RESOURCE_UNAVAILABLE,
        )
        if exists:
            actual_fields = set(columns)
            self._record(
                f"sqlite.{table}.fields",
                ok=set(fields) <= actual_fields,
                required=required,
                success="required columns are present",
                failure=f"missing columns: {sorted(set(fields) - actual_fields)}",
            )
            types_ok = all(str(columns.get(field, "")).upper() == expected for field, expected in fields.items())
            self._record(
                f"sqlite.{table}.types",
                ok=types_ok,
                required=required,
                success="column types match contract",
                failure="one or more column types do not match contract",
            )
            records = self.sqlite.sample_records(table, tuple(fields), 1)
            self._record(
                f"sqlite.{table}.non_empty",
                ok=bool(records),
                required=required,
                success="table has sample records",
                failure="table is empty",
            )

    def _validate_frame_joins(self) -> None:
        if not (
            self._available.get("milvus:visual")
            and self._available.get("sqlite:metadata")
        ):
            return
        resource = self.config.milvus.visual_collection
        records = self.milvus.sample_records(
            resource, ("frame_id", "video_id", "shot_id"), self.sample_size
        )
        frame_ids = [str(record.get("frame_id", "")) for record in records]
        metadata = self.sqlite.get_frames_by_ids(frame_ids)
        matches = bool(records)
        for record in records:
            frame_id = str(record.get("frame_id", ""))
            row = metadata.get(frame_id)
            if row is None:
                matches = False
                break
            try:
                if row.video_id != str(record.get("video_id")) or row.shot_id != int(record.get("shot_id")):
                    matches = False
                    break
            except (TypeError, ValueError):
                matches = False
                break
        self._record(
            "join.visual_to_metadata",
            ok=matches,
            required=True,
            success="sample frame_id records JOIN with matching video_id/shot_id",
            failure="visual frame_id sample does not JOIN canonical SQLite metadata",
        )

        if self._available.get("milvus:ocr"):
            ocr_records = self.milvus.sample_records(
                self.config.milvus.ocr_collection,
                ("frame_id", "video_id"),
                self.sample_size,
            )
            ocr_ids = [str(record.get("frame_id", "")) for record in ocr_records]
            ocr_metadata = self.sqlite.get_frames_by_ids(ocr_ids)
            ocr_match = all(
                frame_id in ocr_metadata
                and ocr_metadata[frame_id].video_id == str(record.get("video_id"))
                for frame_id, record in zip(ocr_ids, ocr_records)
            ) and bool(ocr_records)
            self._record(
                "join.ocr_dense_to_metadata",
                ok=ocr_match,
                required=False,
                success="OCR dense frame IDs JOIN SQLite metadata",
                failure="OCR dense frame IDs do not JOIN SQLite metadata",
            )
            if self._available.get("es:ocr") and ocr_records:
                es_match = True
                for record in ocr_records:
                    documents = self.elasticsearch.find_documents(
                        self.config.elasticsearch.ocr_index,
                        {"frame_id": record.get("frame_id")},
                        ("frame_id", "video_id"),
                        limit=2,
                    )
                    if len(documents) != 1 or str(documents[0].get("video_id")) != str(record.get("video_id")):
                        es_match = False
                        break
                self._record(
                    "join.ocr_dense_to_lexical",
                    ok=es_match,
                    required=False,
                    success="OCR frame IDs JOIN between Milvus and Elasticsearch",
                    failure="OCR frame IDs do not JOIN between Milvus and Elasticsearch",
                )

    def _validate_asr_join(self) -> None:
        if not (self._available.get("milvus:asr") and self._available.get("es:asr")):
            return
        records = self.milvus.sample_records(
            self.config.milvus.asr_collection,
            ("video_id", "interval_id"),
            self.sample_size,
        )
        matches = bool(records)
        for record in records:
            docs = self.elasticsearch.find_documents(
                self.config.elasticsearch.asr_index,
                {"video_id": record.get("video_id"), "interval_id": record.get("interval_id")},
                ("video_id", "interval_id"),
                limit=2,
            )
            if len(docs) != 1:
                matches = False
                break
        self._record(
            "join.asr_interval",
            ok=matches,
            required=False,
            success="ASR (video_id, interval_id) sample JOINs",
            failure="ASR (video_id, interval_id) sample does not JOIN",
        )

    def _validate_summary_join(self) -> None:
        if not (
            self._available.get("milvus:summary")
            and self._available.get("es:summary")
        ):
            return
        records = self.milvus.sample_records(
            self.config.milvus.summary_collection, ("video_id",), self.sample_size
        )
        matches = bool(records)
        for record in records:
            docs = self.elasticsearch.find_documents(
                self.config.elasticsearch.summary_index,
                {"video_id": record.get("video_id")},
                ("video_id",),
                limit=2,
            )
            if len(docs) != 1:
                matches = False
                break
        self._record(
            "join.summary_video",
            ok=matches,
            required=False,
            success="summary video_id sample JOINs",
            failure="summary video_id sample does not JOIN",
        )

    def _validate_object_join(self) -> None:
        if not (
            self._available.get("sqlite:objects")
            and self._available.get("sqlite:metadata")
        ):
            return
        records = self.sqlite.sample_records(
            self.config.sqlite.objects_table, ("frame_id",), self.sample_size
        )
        ids = [str(record.get("frame_id", "")) for record in records]
        metadata = self.sqlite.get_frames_by_ids(ids)
        matches = bool(records) and all(frame_id in metadata for frame_id in ids)
        self._record(
            "join.objects_to_metadata",
            ok=matches,
            required=False,
            success="object frame IDs JOIN metadata",
            failure="object frame IDs do not JOIN metadata or objects table is empty",
        )

    def _validate_encoders(self) -> None:
        for resource, factory in self.encoder_smoke_vectors.items():
            required = resource == self.config.milvus.visual_collection
            if resource not in self._dimensions:
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
                failure = f"encoder smoke test failed: {type(exc).__name__}"
            self._record(
                f"encoder.{resource}",
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
            code=ErrorCode.CONTRACT_MISMATCH,
        )

    def validate(self) -> ContractValidationReport:
        self._checks = []
        self._dimensions = {}
        self._available = {}

        milvus_resources = {
            "visual": self.config.milvus.visual_collection,
            "ocr": self.config.milvus.ocr_collection,
            "asr": self.config.milvus.asr_collection,
            "summary": self.config.milvus.summary_collection,
        }
        for logical_name, (fields, required) in self.MILVUS_REQUIREMENTS.items():
            resource = milvus_resources[logical_name]
            self._guard(
                f"milvus.{resource}",
                required,
                lambda logical_name=logical_name, resource=resource, fields=fields, required=required: self._validate_milvus_resource(
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
            self._guard(
                f"elasticsearch.{resource}",
                required,
                lambda logical_name=logical_name, resource=resource, fields=fields, required=required: self._validate_es_resource(
                    logical_name, resource, fields, required
                ),
            )
        self._guard("elasticsearch.analysis_icu", False, self._validate_icu_plugin)

        sqlite_resources = {
            "metadata": self.config.sqlite.metadata_table,
            "objects": self.config.sqlite.objects_table,
        }
        for logical_name, (fields, required) in self.SQLITE_REQUIREMENTS.items():
            table = sqlite_resources[logical_name]
            self._guard(
                f"sqlite.{table}",
                required,
                lambda logical_name=logical_name, table=table, fields=fields, required=required: self._validate_sqlite_resource(
                    logical_name, table, fields, required
                ),
            )

        self._guard("join.frame", True, self._validate_frame_joins)
        self._guard("join.asr", False, self._validate_asr_join)
        self._guard("join.summary", False, self._validate_summary_join)
        self._guard("join.objects", False, self._validate_object_join)
        self._guard("encoder.smoke", False, self._validate_encoders)

        if any(check.status is CheckStatus.FAIL for check in self._checks):
            status = ValidationStatus.FAIL
        elif any(check.status is CheckStatus.WARNING for check in self._checks):
            status = ValidationStatus.PARTIAL
        else:
            status = ValidationStatus.PASS
        return ContractValidationReport(
            status=status,
            checks=tuple(self._checks),
            dimensions=dict(self._dimensions),
        )
