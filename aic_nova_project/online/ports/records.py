"""SDK-neutral records returned by data adapters."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, model_validator

from online.domain.base import FiniteFloat, NonEmptyStr, StrictFrozenModel
from online.domain.candidates import ObjectDetection


class FrameSearchHit(StrictFrozenModel):
    frame_id: NonEmptyStr
    video_id: NonEmptyStr
    shot_id: int | None = Field(default=None, ge=0)
    raw_score: FiniteFloat


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
    shot_id: int = Field(ge=0)
    timestamp_sec: Annotated[FiniteFloat, Field(ge=0.0)]


ObjectMap = dict[str, tuple[ObjectDetection, ...]]
