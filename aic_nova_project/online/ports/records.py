"""SDK-neutral records returned by data adapters."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, model_validator

from online.domain.base import FiniteFloat, NonEmptyStr, StrictFrozenModel, StrictIntValue
from online.domain.candidates import ObjectDetection
from online.domain.errors import ContractMismatchError
from online.domain.identifiers import validate_canonical_frame_id


class FrameSearchHit(StrictFrozenModel):
    frame_id: NonEmptyStr
    video_id: NonEmptyStr
    raw_score: FiniteFloat

    @model_validator(mode="after")
    def validate_frame_identity(self) -> "FrameSearchHit":
        _validate_frame_identity(self.frame_id, self.video_id)
        return self


class ASRSearchHit(StrictFrozenModel):
    video_id: NonEmptyStr
    interval_id: NonEmptyStr
    start_time_sec: Annotated[FiniteFloat, Field(ge=0.0)]
    end_time_sec: Annotated[FiniteFloat, Field(ge=0.0)]
    raw_score: FiniteFloat
    text: str | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> "ASRSearchHit":
        if self.end_time_sec < self.start_time_sec:
            raise ValueError("end_time_sec must be >= start_time_sec")
        return self


class VideoSearchHit(StrictFrozenModel):
    video_id: NonEmptyStr
    raw_score: FiniteFloat
    summary: str | None = None


class FrameMetadata(StrictFrozenModel):
    frame_id: NonEmptyStr
    video_id: NonEmptyStr
    keyframe_no: StrictIntValue = Field(ge=1)
    local_index: StrictIntValue = Field(ge=0)
    timestamp_sec: Annotated[FiniteFloat, Field(ge=0.0)]
    fps: Annotated[FiniteFloat, Field(gt=0.0)]
    source_frame_idx: StrictIntValue = Field(ge=0)
    image_rel_path: NonEmptyStr

    @model_validator(mode="after")
    def validate_frame_identity(self) -> "FrameMetadata":
        _validate_frame_identity(
            self.frame_id,
            self.video_id,
            keyframe_no=self.keyframe_no,
        )
        if self.local_index != self.keyframe_no - 1:
            raise ValueError("local_index must equal keyframe_no - 1")
        return self


ObjectMap = dict[str, tuple[ObjectDetection, ...]]


def _validate_frame_identity(
    frame_id: str,
    video_id: str,
    *,
    keyframe_no: int | None = None,
) -> None:
    try:
        validate_canonical_frame_id(
            frame_id,
            video_id=video_id,
            keyframe_no=keyframe_no,
        )
    except ContractMismatchError as exc:
        raise ValueError(exc.message) from exc
