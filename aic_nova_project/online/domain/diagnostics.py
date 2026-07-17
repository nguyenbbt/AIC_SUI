"""Safe query-level diagnostics contracts."""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, Field

from .base import FiniteFloat, NonEmptyStr, StrictFrozenModel, freeze_mapping
from .enums import BranchStatus, RetrievalBranch


class BranchDiagnostics(StrictFrozenModel):
    status: BranchStatus
    latency_ms: Annotated[FiniteFloat, Field(ge=0.0)]
    requested_top_k: int = Field(ge=1)
    raw_result_count: int = Field(ge=0)
    output_candidate_count: int = Field(ge=0)
    mapping_loss_count: int = Field(default=0, ge=0)
    warnings: tuple[NonEmptyStr, ...] = ()


class QueryDiagnostics(StrictFrozenModel):
    query_id: NonEmptyStr
    total_latency_ms: Annotated[FiniteFloat, Field(ge=0.0)]
    stage_latencies_ms: Annotated[
        dict[NonEmptyStr, Annotated[FiniteFloat, Field(ge=0.0)]],
        AfterValidator(freeze_mapping),
    ]
    branches: Annotated[
        dict[RetrievalBranch, BranchDiagnostics], AfterValidator(freeze_mapping)
    ]
    missing_metadata_count: int = Field(default=0, ge=0)
    object_filter_removals: int = Field(default=0, ge=0)
    dedup_removals: int = Field(default=0, ge=0)
    normalization_method: NonEmptyStr
    fusion_method: NonEmptyStr
    fusion_weights: Annotated[
        dict[RetrievalBranch, Annotated[FiniteFloat, Field(ge=0.0)]],
        AfterValidator(freeze_mapping),
    ]
    warnings: tuple[NonEmptyStr, ...] = ()
    errors: tuple[NonEmptyStr, ...] = ()
