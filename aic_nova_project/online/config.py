"""Centralized, environment-loadable Data & Infrastructure configuration."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from .domain.base import FiniteFloat, NonEmptyStr, StrictFrozenModel, StrictIntValue


_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class MilvusResourceConfig(StrictFrozenModel):
    uri: NonEmptyStr = "http://localhost:19530"
    alias: NonEmptyStr = "online"
    visual_collection: NonEmptyStr = "visual_features"
    ocr_collection: NonEmptyStr = "ocr_features"
    asr_collection: NonEmptyStr = "asr_features"
    summary_collection: NonEmptyStr = "summary_features"
    search_ef: StrictIntValue = Field(default=128, ge=1)
    timeout_sec: FiniteFloat = Field(default=5.0, gt=0.0)
    norm_tolerance: FiniteFloat = Field(default=1e-3, gt=0.0)


class ElasticsearchResourceConfig(StrictFrozenModel):
    uri: NonEmptyStr = "http://localhost:9200"
    ocr_index: NonEmptyStr = "ocr_texts"
    asr_index: NonEmptyStr = "asr_transcripts"
    summary_index: NonEmptyStr = "video_summaries"
    timeout_sec: FiniteFloat = Field(default=5.0, gt=0.0)
    fuzzy_enabled: bool = False
    fuzziness: NonEmptyStr = "AUTO"
    evidence_lookup_limit: StrictIntValue = Field(default=1_000, ge=1, le=10_000)


class SQLiteResourceConfig(StrictFrozenModel):
    path: Path = Path("data/metadata.db")
    videos_table: NonEmptyStr = "videos"
    metadata_table: NonEmptyStr = "metadata"
    objects_table: NonEmptyStr = "objects"
    batch_size: StrictIntValue = Field(default=500, ge=1, le=900)
    timeout_sec: FiniteFloat = Field(default=5.0, gt=0.0)

    @field_validator("videos_table", "metadata_table", "objects_table")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not _SQL_IDENTIFIER.fullmatch(value):
            raise ValueError("SQLite table name must be a simple SQL identifier")
        return value


class DatasetResourceConfig(StrictFrozenModel):
    """Immutable identity and filesystem boundary for one published dataset."""

    manifest_path: Path = Path("data/dataset-manifest.json")
    data_root: Path = Path("data")
    expected_contract_version: NonEmptyStr = "self-indexed-v2"
    expected_fingerprint: str | None = None
    manifest_required: bool = True
    audit_batch_size: StrictIntValue = Field(default=500, ge=1, le=10_000)

    @field_validator("expected_fingerprint")
    @classmethod
    def validate_expected_fingerprint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
            raise ValueError("expected_fingerprint must be sha256:<64 lowercase hex chars>")
        return value


class OnlineDataConfig(StrictFrozenModel):
    milvus: MilvusResourceConfig = Field(default_factory=MilvusResourceConfig)
    elasticsearch: ElasticsearchResourceConfig = Field(
        default_factory=ElasticsearchResourceConfig
    )
    sqlite: SQLiteResourceConfig = Field(default_factory=SQLiteResourceConfig)
    dataset: DatasetResourceConfig = Field(default_factory=DatasetResourceConfig)

    @classmethod
    def from_env(cls, prefix: str = "AIC_ONLINE_") -> "OnlineDataConfig":
        """Load documented settings without requiring pydantic-settings."""

        def env(name: str, default: Any) -> Any:
            return os.getenv(f"{prefix}{name}", default)

        def as_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            normalized = str(value).strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            raise ValueError("boolean environment values must be true or false")

        return cls(
            milvus=MilvusResourceConfig(
                uri=env("MILVUS_URI", "http://localhost:19530"),
                alias=env("MILVUS_ALIAS", "online"),
                visual_collection=env("MILVUS_VISUAL_COLLECTION", "visual_features"),
                ocr_collection=env("MILVUS_OCR_COLLECTION", "ocr_features"),
                asr_collection=env("MILVUS_ASR_COLLECTION", "asr_features"),
                summary_collection=env("MILVUS_SUMMARY_COLLECTION", "summary_features"),
                search_ef=int(env("MILVUS_SEARCH_EF", 128)),
                timeout_sec=float(env("MILVUS_TIMEOUT_SEC", 5.0)),
                norm_tolerance=float(env("VECTOR_NORM_TOLERANCE", 1e-3)),
            ),
            elasticsearch=ElasticsearchResourceConfig(
                uri=env("ES_URI", "http://localhost:9200"),
                ocr_index=env("ES_OCR_INDEX", "ocr_texts"),
                asr_index=env("ES_ASR_INDEX", "asr_transcripts"),
                summary_index=env("ES_SUMMARY_INDEX", "video_summaries"),
                timeout_sec=float(env("ES_TIMEOUT_SEC", 5.0)),
                fuzzy_enabled=as_bool(env("ES_FUZZY_ENABLED", False)),
                fuzziness=env("ES_FUZZINESS", "AUTO"),
                evidence_lookup_limit=int(env("ES_EVIDENCE_LOOKUP_LIMIT", 1_000)),
            ),
            sqlite=SQLiteResourceConfig(
                path=Path(env("SQLITE_PATH", "data/metadata.db")),
                videos_table=env("SQLITE_VIDEOS_TABLE", "videos"),
                metadata_table=env("SQLITE_METADATA_TABLE", "metadata"),
                objects_table=env("SQLITE_OBJECTS_TABLE", "objects"),
                batch_size=int(env("SQLITE_BATCH_SIZE", 500)),
                timeout_sec=float(env("SQLITE_TIMEOUT_SEC", 5.0)),
            ),
            dataset=DatasetResourceConfig(
                manifest_path=Path(env("DATASET_MANIFEST_PATH", "data/dataset-manifest.json")),
                data_root=Path(env("DATA_ROOT", "data")),
                expected_contract_version=env("DATASET_CONTRACT_VERSION", "self-indexed-v2"),
                expected_fingerprint=(
                    str(env("DATASET_EXPECTED_FINGERPRINT", "")).strip() or None
                ),
                manifest_required=as_bool(env("DATASET_MANIFEST_REQUIRED", True)),
                audit_batch_size=int(env("DATASET_AUDIT_BATCH_SIZE", 500)),
            ),
        )
