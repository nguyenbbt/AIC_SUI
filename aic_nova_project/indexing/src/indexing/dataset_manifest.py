"""Deterministic BUILDING manifest for the self-indexed-v2 dataset."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CONTRACT_VERSION = "self-indexed-v2"
VISUAL_MODEL_ID = "ViT-B-32::openai"
TEXT_MODEL_NAME = "dangvantuan/vietnamese-embedding"
TEXT_MODEL_REVISION = "4ab46e46ba5902328ba0742e489e75f787932f2b"
RECORD_COUNT_KEYS = frozenset(
    {
        "videos",
        "metadata",
        "objects",
        "visual_features",
        "ocr_features",
        "asr_features",
        "summary_features",
        "ocr_texts",
        "asr_transcripts",
        "video_summaries",
    }
)
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class DatasetManifestDraft(BaseModel):
    """A validated manifest that cannot be consumed by Online yet."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["self-indexed-v2"] = CONTRACT_VERSION
    dataset_id: str = Field(min_length=1)
    dataset_fingerprint: str = Field(
        pattern=r"^sha256:[0-9a-f]{64}$"
    )
    status: Literal["BUILDING"] = "BUILDING"
    frame_index_base: Literal[0] = 0
    bbox_space: Literal["absolute_pixel_xyxy"] = "absolute_pixel_xyxy"
    visual_model_id: Literal["ViT-B-32::openai"] = VISUAL_MODEL_ID
    visual_dimension: Literal[512] = 512
    visual_normalized: Literal[True] = True
    text_model_name: Literal[
        "dangvantuan/vietnamese-embedding"
    ] = TEXT_MODEL_NAME
    text_model_revision: Literal[
        "4ab46e46ba5902328ba0742e489e75f787932f2b"
    ] = TEXT_MODEL_REVISION
    text_dimension: Literal[768] = 768
    text_max_length: Literal[256] = 256
    record_counts: Dict[str, int]
    created_at_utc: str

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("dataset_id cannot have surrounding whitespace")
        return value

    @field_validator("record_counts", mode="before")
    @classmethod
    def validate_record_counts(cls, value: object) -> Dict[str, int]:
        if not isinstance(value, Mapping):
            raise ValueError("record_counts must be an object")
        if set(value) != RECORD_COUNT_KEYS:
            missing = sorted(RECORD_COUNT_KEYS - set(value))
            extra = sorted(set(value) - RECORD_COUNT_KEYS)
            raise ValueError(
                "record_counts keys mismatch; "
                f"missing={missing}, extra={extra}"
            )
        counts: Dict[str, int] = {}
        for key, count in value.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(
                    f"record_counts.{key} must be a non-negative integer"
                )
            counts[str(key)] = count
        return counts

    @field_validator("created_at_utc")
    @classmethod
    def validate_created_at_utc(cls, value: str) -> str:
        if not value.endswith("Z"):
            raise ValueError("created_at_utc must use UTC Z notation")
        try:
            parsed = datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError as exc:
            raise ValueError(
                "created_at_utc must be an ISO-8601 timestamp"
            ) from exc
        if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ValueError("created_at_utc must be UTC")
        return value

    @model_validator(mode="after")
    def validate_cross_resource_counts(self) -> "DatasetManifestDraft":
        counts = self.record_counts
        pairs = (
            ("visual_features", "metadata"),
            ("ocr_features", "ocr_texts"),
            ("asr_features", "asr_transcripts"),
            ("summary_features", "video_summaries"),
        )
        for left, right in pairs:
            if counts[left] != counts[right]:
                raise ValueError(f"{left} count must equal {right} count")
        if counts["metadata"] and not counts["videos"]:
            raise ValueError("metadata cannot exist without videos")
        return self


class ReadyDatasetManifest(DatasetManifestDraft):
    """The exact manifest shape accepted by the Online startup gate."""

    status: Literal["READY"] = "READY"


def _created_at_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _load_source_identities(data_dir: Path) -> list[dict[str, str]]:
    metadata_dir = data_dir / "metadata"
    metadata_paths = sorted(metadata_dir.glob("*.json"))
    if not metadata_paths:
        raise ValueError(f"No Module 1 metadata found in {metadata_dir}")

    identities = []
    seen_video_ids = set()
    for metadata_path in metadata_paths:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Cannot read Module 1 metadata: {metadata_path}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Metadata must be an object: {metadata_path}")
        if payload.get("contract_version") != CONTRACT_VERSION:
            raise ValueError(
                f"Metadata contract mismatch: {metadata_path}"
            )

        video_id = payload.get("video_id")
        source_path = payload.get("source_video_rel_path")
        source_fingerprint = payload.get("source_fingerprint")
        config_fingerprint = payload.get("producer_config_fingerprint")
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError(f"Invalid video_id in {metadata_path}")
        if video_id in seen_video_ids:
            raise ValueError(f"Duplicate video_id in metadata: {video_id}")
        if metadata_path.stem != video_id:
            raise ValueError(
                f"Metadata filename does not match video_id: {metadata_path}"
            )
        if not isinstance(source_path, str) or not source_path:
            raise ValueError(
                f"Missing source_video_rel_path in {metadata_path}"
            )
        for field_name, fingerprint in (
            ("source_fingerprint", source_fingerprint),
            ("producer_config_fingerprint", config_fingerprint),
        ):
            if (
                not isinstance(fingerprint, str)
                or _SHA256_HEX.fullmatch(fingerprint) is None
            ):
                raise ValueError(
                    f"Invalid {field_name} in {metadata_path}"
                )
        seen_video_ids.add(video_id)
        identities.append(
            {
                "video_id": video_id,
                "source_video_rel_path": source_path,
                "source_sha256": source_fingerprint,
                "module1_config_sha256": config_fingerprint,
            }
        )
    return identities


def _dataset_fingerprint(payload: Mapping[str, Any]) -> str:
    try:
        canonical = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Producer config must be canonical JSON data") from exc
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def build_manifest_draft(
    *,
    data_dir: str | Path,
    dataset_id: str,
    record_counts: Mapping[str, int],
    created_at_utc: str | None = None,
    producer_config: Mapping[str, Any] | None = None,
) -> DatasetManifestDraft:
    """Build a deterministic, non-publishable manifest from verified inputs."""
    identities = _load_source_identities(Path(data_dir))
    counts = dict(record_counts)
    if counts.get("videos") != len(identities):
        raise ValueError(
            "record_counts.videos does not match Module 1 metadata count"
        )

    fingerprint_payload = {
        "contract_version": CONTRACT_VERSION,
        "models": {
            "visual_model_id": VISUAL_MODEL_ID,
            "visual_dimension": 512,
            "visual_normalized": True,
            "text_model_name": TEXT_MODEL_NAME,
            "text_model_revision": TEXT_MODEL_REVISION,
            "text_dimension": 768,
            "text_max_length": 256,
        },
        "producer_config": dict(producer_config or {}),
        "record_counts": counts,
        "sources": identities,
    }
    return DatasetManifestDraft(
        dataset_id=dataset_id,
        dataset_fingerprint=_dataset_fingerprint(fingerprint_payload),
        record_counts=counts,
        created_at_utc=created_at_utc or _created_at_now(),
    )


def _atomic_write_manifest(
    manifest: DatasetManifestDraft | ReadyDatasetManifest,
    output_path: str | Path,
) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(
        f".{destination.name}.tmp-{uuid4().hex}"
    )
    try:
        temporary_path.write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_manifest_draft(
    draft: DatasetManifestDraft,
    output_path: str | Path,
) -> None:
    """Atomically write a BUILDING manifest that Online must reject."""
    _atomic_write_manifest(draft, output_path)


def publish_ready_manifest(
    draft: DatasetManifestDraft,
    output_path: str | Path,
    *,
    verification_errors: Sequence[str],
) -> ReadyDatasetManifest:
    """Publish READY only after the caller supplies a clean verifier result."""
    errors = list(verification_errors)
    if errors:
        raise ValueError(
            "Dataset verification failed; READY was not published: "
            + "; ".join(errors[:5])
        )
    payload = draft.model_dump()
    payload["status"] = "READY"
    ready = ReadyDatasetManifest.model_validate(payload)
    _atomic_write_manifest(ready, output_path)
    return ready


__all__ = [
    "CONTRACT_VERSION",
    "DatasetManifestDraft",
    "ReadyDatasetManifest",
    "RECORD_COUNT_KEYS",
    "TEXT_MODEL_NAME",
    "TEXT_MODEL_REVISION",
    "VISUAL_MODEL_ID",
    "build_manifest_draft",
    "publish_ready_manifest",
    "write_manifest_draft",
]
