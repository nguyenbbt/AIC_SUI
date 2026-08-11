"""Read-only dataset resources used by the operator UI."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from pydantic import Field

from online.domain.base import NonEmptyStr, StrictFrozenModel, StrictIntValue
from online.domain.errors import MissingMetadataError, ResourceUnavailableError
from online.ports.metadata import MetadataReaderPort
from online.ports.objects import ObjectCatalogPort
from online.ports.records import FrameMetadata, ObjectLabelStat


COCO80_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)


class ObjectCatalogResponse(StrictFrozenModel):
    dataset_id: str | None = None
    dataset_fingerprint: str | None = None
    source: NonEmptyStr
    labels: tuple[ObjectLabelStat, ...]


class NeighborFrame(StrictFrozenModel):
    frame_id: NonEmptyStr
    video_id: NonEmptyStr
    source_frame_idx: StrictIntValue = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    image_url: NonEmptyStr


class NeighborFramesResponse(StrictFrozenModel):
    center_frame_id: NonEmptyStr
    frames: tuple[NeighborFrame, ...]


@dataclass(frozen=True)
class ResolvedMedia:
    path: Path
    media_type: str


class DatasetUIResources:
    """Serve only paths referenced by validated SQLite metadata under DATA_ROOT."""

    def __init__(
        self,
        *,
        data_root: Path,
        metadata_reader: MetadataReaderPort,
        object_catalog: ObjectCatalogPort | None = None,
        identity_provider: Callable[[], tuple[str, str]] | None = None,
    ) -> None:
        self._data_root = Path(data_root).expanduser().resolve()
        self._metadata = metadata_reader
        self._object_catalog = object_catalog
        self._identity_provider = identity_provider

    def object_labels(self) -> ObjectCatalogResponse:
        labels = (
            tuple(self._object_catalog.list_object_labels())
            if self._object_catalog is not None
            else ()
        )
        source = "sqlite"
        if not labels:
            labels = tuple(ObjectLabelStat(label=label, detection_count=0) for label in COCO80_LABELS)
            source = "coco80_fallback"
        dataset_id = dataset_fingerprint = None
        if self._identity_provider is not None:
            dataset_id, dataset_fingerprint = self._identity_provider()
        return ObjectCatalogResponse(
            dataset_id=dataset_id,
            dataset_fingerprint=dataset_fingerprint,
            source=source,
            labels=labels,
        )

    def resolve_keyframe(self, frame_id: str) -> ResolvedMedia:
        frames = self._metadata.get_frames_by_ids((frame_id,))
        frame = frames.get(frame_id)
        if frame is None:
            raise MissingMetadataError(
                "Keyframe metadata is missing", details={"frame_id": frame_id}
            )
        return self._resolve(frame.image_rel_path, default_type="image/jpeg")

    def resolve_video(self, video_id: str) -> ResolvedMedia:
        videos = self._metadata.get_videos_by_ids((video_id,))
        video = videos.get(video_id)
        if video is None:
            raise MissingMetadataError("Video metadata is missing")
        return self._resolve(video.source_video_rel_path, default_type="video/mp4")

    def neighbors(self, frame_id: str, *, before: int, after: int) -> NeighborFramesResponse:
        frames = self._metadata.get_frames_by_ids((frame_id,))
        center = frames.get(frame_id)
        if center is None:
            raise MissingMetadataError(
                "Keyframe metadata is missing", details={"frame_id": frame_id}
            )
        ordered = tuple(self._metadata.get_ordered_frames_by_video(center.video_id))
        position = next((i for i, item in enumerate(ordered) if item.frame_id == frame_id), None)
        if position is None:
            raise MissingMetadataError(
                "Keyframe is absent from its ordered video metadata",
                details={"frame_id": frame_id},
            )
        selected: Sequence[FrameMetadata] = ordered[
            max(0, position - before) : min(len(ordered), position + after + 1)
        ]
        return NeighborFramesResponse(
            center_frame_id=frame_id,
            frames=tuple(
                NeighborFrame(
                    frame_id=item.frame_id,
                    video_id=item.video_id,
                    source_frame_idx=item.source_frame_idx,
                    timestamp_sec=item.timestamp_sec,
                    image_url=f"/media/keyframes/{item.frame_id}",
                )
                for item in selected
            ),
        )

    def _resolve(self, relative_path: str, *, default_type: str) -> ResolvedMedia:
        candidate = (self._data_root / Path(relative_path)).resolve()
        if not candidate.is_relative_to(self._data_root):
            raise ResourceUnavailableError("Dataset media path escapes DATA_ROOT")
        if not candidate.is_file():
            raise ResourceUnavailableError("Dataset media file is unavailable")
        media_type = mimetypes.guess_type(candidate.name)[0] or default_type
        return ResolvedMedia(path=candidate, media_type=media_type)


__all__ = [
    "COCO80_LABELS",
    "DatasetUIResources",
    "NeighborFramesResponse",
    "ObjectCatalogResponse",
]
