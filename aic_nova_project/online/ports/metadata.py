"""Read-only metadata hydration port."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

from .records import FrameMetadata


@runtime_checkable
class MetadataReaderPort(Protocol):
    def list_video_ids(self) -> Sequence[str]: ...

    def get_frames_by_ids(self, frame_ids: Sequence[str]) -> Mapping[str, FrameMetadata]: ...

    def get_ordered_frames_by_video(self, video_id: str) -> Sequence[FrameMetadata]: ...
