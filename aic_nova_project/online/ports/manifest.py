"""Read-only organizer ingestion manifest contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Protocol, runtime_checkable

from pydantic import AfterValidator, Field, PlainSerializer

from online.domain.base import (
    FiniteFloat,
    NonEmptyStr,
    StrictFrozenModel,
    StrictIntValue,
    freeze_mapping,
    serialize_mapping,
)


ORGANIZER_CONTRACT_VERSION = "organizer-v1"
ORGANIZER_FRAME_ID_CONTRACT_VERSION = "organizer-v1"
ORGANIZER_VISUAL_MODEL_ID = "ViT-B-32::openai"
ORGANIZER_VISUAL_DIMENSION = 512


class DatasetManifest(StrictFrozenModel):
    contract_version: NonEmptyStr
    visual_model_id: NonEmptyStr
    visual_dimension: StrictIntValue = Field(ge=1)
    visual_normalized: bool
    frame_id_contract_version: NonEmptyStr
    object_threshold: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
    object_nms_iou: Annotated[FiniteFloat, Field(gt=0.0, le=1.0)]
    record_counts: Annotated[
        Mapping[NonEmptyStr, Annotated[StrictIntValue, Field(ge=0)]],
        AfterValidator(freeze_mapping),
        PlainSerializer(serialize_mapping, return_type=dict),
    ]
    dataset_fingerprint: NonEmptyStr


@runtime_checkable
class ManifestReaderPort(Protocol):
    def read_manifest(self) -> DatasetManifest: ...


__all__ = [
    "DatasetManifest",
    "ManifestReaderPort",
    "ORGANIZER_CONTRACT_VERSION",
    "ORGANIZER_FRAME_ID_CONTRACT_VERSION",
    "ORGANIZER_VISUAL_DIMENSION",
    "ORGANIZER_VISUAL_MODEL_ID",
]
