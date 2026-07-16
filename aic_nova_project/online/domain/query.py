"""Structured object-query contract owned by Online."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, model_validator

from .base import FiniteFloat, NonEmptyStr, StrictFrozenModel, ensure_bbox_order
from .enums import CountOperator, FilterMode


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
