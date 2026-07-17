"""Shared strict Pydantic model helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated, Any, TypeVar

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
    """Deeply immutable boundary model that rejects unknown fields.

    Container fields still need to use :func:`freeze_mapping` (or tuples) because
    Pydantic's ``frozen=True`` only prevents attribute reassignment.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


_K = TypeVar("_K")
_V = TypeVar("_V")


class FrozenDict(dict[_K, _V]):
    """A JSON-serializable dictionary snapshot that cannot be mutated."""

    _IMMUTABLE_MESSAGE = "FrozenDict is immutable"

    def _immutable(self, *args: object, **kwargs: object) -> None:
        raise TypeError(self._IMMUTABLE_MESSAGE)

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def freeze_mapping(value: Mapping[_K, _V] | dict[_K, _V]) -> FrozenDict[_K, _V]:
    """Copy a mapping at the model boundary and freeze the resulting snapshot."""

    return FrozenDict(value)


def ensure_bbox_order(model: Any) -> Any:
    if model.x_max < model.x_min or model.y_max < model.y_min:
        raise ValueError("bbox max coordinates must be >= min coordinates")
    return model
