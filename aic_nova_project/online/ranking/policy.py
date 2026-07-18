"""Explicit Person-C ranking policy configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, PlainSerializer, model_validator

from online.domain.base import (
    FiniteFloat,
    NonEmptyStr,
    StrictFrozenModel,
    StrictIntValue,
    freeze_mapping,
    serialize_mapping,
)
from online.domain.enums import RetrievalBranch


RankingPolicyStatus = Literal["experimental", "approved"]
CoreVisualPolicy = Literal["q0_required"]


class RankingPolicyConfig(StrictFrozenModel):
    """Frozen policy values used to build the C ranking pipeline."""

    policy_name: NonEmptyStr = "person_c_experimental_baseline_v1"
    policy_status: RankingPolicyStatus = "experimental"
    normalization_method: NonEmptyStr = "rrf"
    normalization_rrf_k: StrictIntValue = Field(default=60, ge=1)
    aggregation_method: NonEmptyStr = "weighted_sum_query_variant_v1"
    query_variant_weights: Annotated[
        Mapping[NonEmptyStr, Annotated[FiniteFloat, Field(ge=0.0)]],
        AfterValidator(freeze_mapping),
        PlainSerializer(serialize_mapping, return_type=dict),
    ] = Field(default_factory=lambda: {"q0": 1.0, "q1": 1.0, "q2": 1.0})
    fusion_method: NonEmptyStr = "experimental_weighted_sum_normalized_v1"
    fusion_weights: Annotated[
        Mapping[RetrievalBranch, Annotated[FiniteFloat, Field(ge=0.0)]],
        AfterValidator(freeze_mapping),
        PlainSerializer(serialize_mapping, return_type=dict),
    ] = Field(default_factory=dict)
    fusion_default_weight: Annotated[FiniteFloat, Field(ge=0.0)] = 1.0
    summary_method: NonEmptyStr = "summary_video_score_cap_v1"
    summary_weight: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)] = 0.1
    summary_max_boost: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)] = 0.2
    asr_mapping_method: NonEmptyStr = "timestamp_inclusive_distributed_v1"
    asr_max_frames_per_interval: StrictIntValue = Field(default=50, ge=1)
    asr_interval_rrf_k: StrictIntValue = Field(default=60, ge=1)
    object_soft_boost_per_constraint: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)] = 0.05
    object_max_total_boost: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)] = 0.2
    core_visual_policy: CoreVisualPolicy = "q0_required"

    @model_validator(mode="after")
    def validate_policy(self) -> "RankingPolicyConfig":
        if self.normalization_method != "rrf":
            raise ValueError("only rrf normalization is currently implemented")
        if self.aggregation_method != "weighted_sum_query_variant_v1":
            raise ValueError("only weighted_sum_query_variant_v1 aggregation is currently implemented")
        if self.asr_mapping_method != "timestamp_inclusive_distributed_v1":
            raise ValueError("only timestamp_inclusive_distributed_v1 ASR mapping is currently implemented")
        if self.fusion_default_weight == 0.0 and not self.fusion_weights:
            raise ValueError("fusion must have at least one positive weight")
        if self.summary_method != "summary_video_score_cap_v1":
            raise ValueError("only summary_video_score_cap_v1 summary propagation is currently implemented")
        _assert_finite_mapping(self.query_variant_weights, "query_variant_weights")
        _assert_finite_mapping(self.fusion_weights, "fusion_weights")
        return self

    @property
    def warning_tag(self) -> str:
        return f"ranking_policy={self.policy_name}:{self.policy_status}"


def _assert_finite_mapping(mapping: Mapping[object, float], name: str) -> None:
    for value in mapping.values():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must contain only finite values")
