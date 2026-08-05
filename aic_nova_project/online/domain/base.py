"""Shared strict Pydantic model helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Any, Generic, Iterator, TypeVar

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict


def _non_empty(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be empty or whitespace")
    if value != value.strip():
        raise ValueError("value must not contain leading or trailing whitespace")
    return value


def _reject_bool(value: object) -> object:
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric value")
    return value


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    return value


NonEmptyStr = Annotated[str, AfterValidator(_non_empty)]
StrictIntValue = Annotated[int, BeforeValidator(_reject_bool)]
FiniteFloat = Annotated[
    float,
    BeforeValidator(_reject_bool),
    AfterValidator(_finite),
]


class StrictFrozenModel(BaseModel):
    """Deeply immutable boundary model that rejects unknown fields.

    Container fields still need to use :func:`freeze_mapping` (or tuples) because
    Pydantic's ``frozen=True`` only prevents attribute reassignment.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


_K = TypeVar("_K")
_V = TypeVar("_V")


class FrozenDict(Mapping[_K, _V], Generic[_K, _V]):
    """A read-only mapping snapshot with no mutable ``dict`` base to bypass.

    Subclassing ``dict`` and overriding ``__setitem__`` is insufficient because
    callers can still invoke ``dict.__setitem__`` directly.  Composition around
    ``MappingProxyType`` closes that escape hatch while remaining serializable
    through Pydantic's ``Mapping`` support.
    """

    __slots__ = ("_data",)

    def __init__(self, value: Mapping[_K, _V]) -> None:
        object.__setattr__(self, "_data", MappingProxyType(dict(value)))

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("FrozenDict is immutable")

    def __getitem__(self, key: _K) -> _V:
        return self._data[key]

    def __iter__(self) -> Iterator[_K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenDict({dict(self._data)!r})"


def freeze_mapping(value: Mapping[_K, _V] | dict[_K, _V]) -> FrozenDict[_K, _V]:
    """Copy a mapping at the model boundary and freeze the resulting snapshot."""

    return FrozenDict(value)


def serialize_mapping(value: Mapping[_K, _V]) -> dict[_K, _V]:
    """Convert an immutable mapping snapshot to a normal JSON object."""

    return dict(value)


def ensure_bbox_order(model: Any) -> Any:
    if model.x_max <= model.x_min or model.y_max <= model.y_min:
        raise ValueError("bbox max coordinates must be greater than min coordinates")
    return model
