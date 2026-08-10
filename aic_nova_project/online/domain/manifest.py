"""Strict runtime contract for an Offline-published dataset manifest."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, field_serializer, field_validator, model_validator

from .base import FrozenDict, NonEmptyStr, StrictFrozenModel, StrictIntValue, freeze_mapping


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


def _fingerprint(value: str) -> str:
    if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError("dataset_fingerprint must be sha256:<64 lowercase hex chars>")
    return value


DatasetFingerprint = Annotated[str, AfterValidator(_fingerprint)]


class DatasetManifest(StrictFrozenModel):
    """Validated identity of one atomically published Offline dataset."""

    contract_version: Literal["self-indexed-v2"] = CONTRACT_VERSION
    dataset_id: NonEmptyStr
    dataset_fingerprint: DatasetFingerprint
    status: Literal["READY"]
    frame_index_base: Literal[0]
    bbox_space: Literal["absolute_pixel_xyxy"]
    visual_model_id: Literal["ViT-B-32::openai"] = VISUAL_MODEL_ID
    visual_dimension: Literal[512]
    visual_normalized: Literal[True]
    text_model_name: Literal["dangvantuan/vietnamese-embedding"] = TEXT_MODEL_NAME
    text_model_revision: Literal[
        "4ab46e46ba5902328ba0742e489e75f787932f2b"
    ] = TEXT_MODEL_REVISION
    text_dimension: Literal[768]
    text_max_length: Literal[256]
    record_counts: Mapping[str, StrictIntValue]
    created_at_utc: NonEmptyStr

    @field_validator("record_counts", mode="before")
    @classmethod
    def validate_record_counts(cls, value: object) -> FrozenDict[str, int]:
        if not isinstance(value, Mapping):
            raise ValueError("record_counts must be an object")
        if set(value) != RECORD_COUNT_KEYS:
            missing = sorted(RECORD_COUNT_KEYS - set(value))
            extra = sorted(set(value) - RECORD_COUNT_KEYS)
            raise ValueError(f"record_counts keys mismatch; missing={missing}, extra={extra}")
        output: dict[str, int] = {}
        for key, count in value.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"record_counts.{key} must be a non-negative integer")
            output[str(key)] = count
        return freeze_mapping(output)

    @field_serializer("record_counts")
    def serialize_record_counts(self, value: Mapping[str, int]) -> dict[str, int]:
        return dict(value)

    @field_validator("created_at_utc")
    @classmethod
    def validate_created_at_utc(cls, value: str) -> str:
        if not value.endswith("Z"):
            raise ValueError("created_at_utc must use UTC Z notation")
        try:
            parsed = datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError as exc:
            raise ValueError("created_at_utc must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ValueError("created_at_utc must be UTC")
        return value

    @model_validator(mode="after")
    def validate_cross_resource_counts(self) -> "DatasetManifest":
        counts = self.record_counts
        if counts["visual_features"] != counts["metadata"]:
            raise ValueError("visual_features count must equal metadata count")
        if counts["ocr_features"] != counts["ocr_texts"]:
            raise ValueError("ocr_features count must equal ocr_texts count")
        if counts["asr_features"] != counts["asr_transcripts"]:
            raise ValueError("asr_features count must equal asr_transcripts count")
        if counts["summary_features"] != counts["video_summaries"]:
            raise ValueError("summary_features count must equal video_summaries count")
        if counts["metadata"] and not counts["videos"]:
            raise ValueError("metadata cannot exist without videos")
        return self


__all__ = [
    "CONTRACT_VERSION",
    "DatasetFingerprint",
    "DatasetManifest",
    "RECORD_COUNT_KEYS",
    "TEXT_MODEL_NAME",
    "TEXT_MODEL_REVISION",
    "VISUAL_MODEL_ID",
]
