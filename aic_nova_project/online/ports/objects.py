"""Read-only object detection port."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

from online.domain.candidates import ObjectDetection


@runtime_checkable
class ObjectReaderPort(Protocol):
    def get_objects_by_frame_ids(
        self,
        frame_ids: Sequence[str],
        *,
        label: str | None = None,
        min_confidence: float = 0.0,
    ) -> Mapping[str, Sequence[ObjectDetection]]: ...
