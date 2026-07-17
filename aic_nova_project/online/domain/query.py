"""Strict internal query contracts shared by Online components."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, model_validator

from .base import FiniteFloat, NonEmptyStr, StrictFrozenModel, ensure_bbox_order
from .enums import CountOperator, FilterMode, QueryMode, RetrievalBranch


class NormalizedRegion(StrictFrozenModel):
    x_min: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
    y_min: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
    x_max: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
    y_max: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]

    _ordered = model_validator(mode="after")(ensure_bbox_order)


class ObjectConstraint(StrictFrozenModel):
    label: NonEmptyStr
    count_operator: CountOperator
    count: int = Field(ge=0)
    min_confidence: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)] = 0.5
    position: NormalizedRegion | None = None
    filter_mode: FilterMode = FilterMode.SOFT


class TextQueryVariant(StrictFrozenModel):
    """One independently retrieved text formulation of the same KIS request."""

    variant_id: NonEmptyStr
    text: NonEmptyStr
    # Reserved for an approved aggregation policy. Retrieval does not consume it.
    weight_hint: Annotated[FiniteFloat, Field(gt=0.0)] | None = None


class QueryOptions(StrictFrozenModel):
    """Reserved strict options object.

    Public API fields, top-k values and ranking behavior remain open questions.
    Fields are added here only after the corresponding decision is approved.
    """


class QueryBundle(StrictFrozenModel):
    """Validated internal KIS query passed to retrieval branches."""

    query_id: NonEmptyStr
    mode: QueryMode
    original_query: NonEmptyStr
    text_variants: tuple[TextQueryVariant, ...]
    object_constraints: tuple[ObjectConstraint, ...] = ()
    enabled_branches: tuple[RetrievalBranch, ...]
    options: QueryOptions = Field(default_factory=QueryOptions)

    @model_validator(mode="after")
    def validate_kis_contract(self) -> "QueryBundle":
        if self.mode not in {QueryMode.KIS_TEXT, QueryMode.KIS_VIDEO}:
            raise ValueError("QueryBundle currently supports only KIS text-query modes")

        if not 1 <= len(self.text_variants) <= 3:
            raise ValueError("KIS QueryBundle must contain q0 and at most q1/q2")

        variant_ids = tuple(variant.variant_id for variant in self.text_variants)
        expected_ids = tuple(f"q{index}" for index in range(len(self.text_variants)))
        if variant_ids != expected_ids:
            raise ValueError("text variant IDs must be ordered and contiguous: q0, q1, q2")

        if self.text_variants[0].text != self.original_query:
            raise ValueError("q0 text must equal original_query")

        if not self.enabled_branches:
            raise ValueError("at least one retrieval branch must be enabled")
        if len(self.enabled_branches) != len(set(self.enabled_branches)):
            raise ValueError("enabled_branches must not contain duplicates")
        return self
