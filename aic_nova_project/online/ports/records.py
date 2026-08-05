"""SDK-neutral records returned by data adapters."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, field_validator, model_validator

from online.domain.base import FiniteFloat, NonEmptyStr, StrictFrozenModel, StrictIntValue
from online.domain.candidates import ObjectDetection
from online.domain.errors import ContractMismatchError
from online.domain.identifiers import (
    validate_canonical_frame_id,
    validate_interval_id,
    validate_relative_artifact_path,
)


class FrameSearchHit(StrictFrozenModel):
    frame_id: NonEmptyStr
    video_id: NonEmptyStr
    shot_id: StrictIntValue | None = Field(default=None, ge=0)
    raw_score: FiniteFloat

    @model_validator(mode="after")
    def validate_frame_identity(self) -> "FrameSearchHit":
        _validate_frame_identity(self.frame_id, self.video_id, shot_id=self.shot_id)
        return self


class ASRSearchHit(StrictFrozenModel):
    video_id: NonEmptyStr
    interval_id: NonEmptyStr
    start_time_sec: Annotated[FiniteFloat, Field(ge=0.0)]
    end_time_sec: Annotated[FiniteFloat, Field(ge=0.0)]
    raw_score: FiniteFloat
    text: str | None = None

    @field_validator("interval_id")
    @classmethod
    def validate_canonical_interval_id(cls, value: str) -> str:
        return validate_interval_id(value)

    @model_validator(mode="after")
    def validate_interval(self) -> "ASRSearchHit":
        if self.end_time_sec < self.start_time_sec:
            raise ValueError("end_time_sec must be >= start_time_sec")
        return self


class VideoSearchHit(StrictFrozenModel):
    video_id: NonEmptyStr
    raw_score: FiniteFloat
    summary: str | None = None


class VideoMetadata(StrictFrozenModel):
    video_id: NonEmptyStr
    source_video_rel_path: NonEmptyStr
    fps: Annotated[FiniteFloat, Field(gt=0.0)]
    duration_sec: Annotated[FiniteFloat, Field(ge=0.0)]
    frame_count: StrictIntValue = Field(gt=0)
    width: StrictIntValue = Field(gt=0)
    height: StrictIntValue = Field(gt=0)

    @field_validator("source_video_rel_path")
    @classmethod
    def validate_source_video_path(cls, value: str) -> str:
        return validate_relative_artifact_path(value)


class FrameMetadata(StrictFrozenModel):
    frame_id: NonEmptyStr
    video_id: NonEmptyStr
    shot_id: StrictIntValue = Field(ge=0)
    source_frame_idx: StrictIntValue = Field(ge=0)
    timestamp_sec: Annotated[FiniteFloat, Field(ge=0.0)]
    image_rel_path: NonEmptyStr

    @field_validator("image_rel_path")
    @classmethod
    def validate_image_path(cls, value: str) -> str:
        return validate_relative_artifact_path(value)

    @model_validator(mode="after")
    def validate_frame_identity(self) -> "FrameMetadata":
        _validate_frame_identity(self.frame_id, self.video_id, shot_id=self.shot_id)
        return self


ObjectMap = dict[str, tuple[ObjectDetection, ...]]


def _validate_frame_identity(
    frame_id: str,
    video_id: str,
    *,
    shot_id: int | None = None,
) -> None:
    try:
        validate_canonical_frame_id(frame_id, video_id=video_id, shot_id=shot_id)
    except ContractMismatchError as exc:
        raise ValueError(exc.message) from exc
