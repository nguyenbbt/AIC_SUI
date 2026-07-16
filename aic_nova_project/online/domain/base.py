"""Shared strict Pydantic model helpers."""

from __future__ import annotations

import math
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict


def _non_empty(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be empty or whitespace")
    return value


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    return value


NonEmptyStr = Annotated[str, AfterValidator(_non_empty)]
FiniteFloat = Annotated[float, AfterValidator(_finite)]


class StrictFrozenModel(BaseModel):
    """Immutable boundary model that rejects misspelled/unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def ensure_bbox_order(model: Any) -> Any:
    if model.x_max < model.x_min or model.y_max < model.y_min:
        raise ValueError("bbox max coordinates must be >= min coordinates")
    return model
