"""Safe query-level diagnostics contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated

from pydantic import AfterValidator, Field, PlainSerializer

from .base import (
    FiniteFloat,
    NonEmptyStr,
    StrictIntValue,
    StrictFrozenModel,
    freeze_mapping,
    serialize_mapping,
)
from .enums import BranchStatus, RetrievalBranch


class BranchDiagnostics(StrictFrozenModel):
    status: BranchStatus
    latency_ms: Annotated[FiniteFloat, Field(ge=0.0)]
    requested_top_k: StrictIntValue = Field(ge=1)
    raw_result_count: StrictIntValue = Field(ge=0)
    output_candidate_count: StrictIntValue = Field(ge=0)
    mapping_loss_count: StrictIntValue = Field(default=0, ge=0)
    warnings: tuple[NonEmptyStr, ...] = ()


class QueryDiagnostics(StrictFrozenModel):
    query_id: NonEmptyStr
    total_latency_ms: Annotated[FiniteFloat, Field(ge=0.0)]
    stage_latencies_ms: Annotated[
        Mapping[NonEmptyStr, Annotated[FiniteFloat, Field(ge=0.0)]],
        AfterValidator(freeze_mapping),
        PlainSerializer(serialize_mapping, return_type=dict),
    ]
    branches: Annotated[
        Mapping[RetrievalBranch, BranchDiagnostics],
        AfterValidator(freeze_mapping),
        PlainSerializer(serialize_mapping, return_type=dict),
    ]
    missing_metadata_count: StrictIntValue = Field(default=0, ge=0)
    object_filter_removals: StrictIntValue = Field(default=0, ge=0)
    dedup_removals: StrictIntValue = Field(default=0, ge=0)
    normalization_method: NonEmptyStr
    fusion_method: NonEmptyStr
    fusion_weights: Annotated[
        Mapping[RetrievalBranch, Annotated[FiniteFloat, Field(ge=0.0)]],
        AfterValidator(freeze_mapping),
        PlainSerializer(serialize_mapping, return_type=dict),
    ]
    warnings: tuple[NonEmptyStr, ...] = ()
    errors: tuple[NonEmptyStr, ...] = ()
