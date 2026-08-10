"""Read-only metadata hydration port."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

from .records import FrameMetadata, VideoMetadata


@runtime_checkable
class MetadataReaderPort(Protocol):
    def get_frames_by_ids(self, frame_ids: Sequence[str]) -> Mapping[str, FrameMetadata]: ...

    def get_ordered_frames_by_video(self, video_id: str) -> Sequence[FrameMetadata]: ...

    def get_videos_by_ids(self, video_ids: Sequence[str]) -> Mapping[str, VideoMetadata]: ...
