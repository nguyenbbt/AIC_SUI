"""Public, SDK-neutral TRAKE/DANTE domain contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .base import FiniteFloat, NonEmptyStr, StrictFrozenModel, StrictIntValue
from .errors import ContractMismatchError
from .identifiers import validate_canonical_frame_id


DANTE_POLICY_VERSION = "dante-index-gap-v1"
DANTE_DEFAULT_LAMBDA = 0.001
DANTE_MIN_LAMBDA = 0.001
DANTE_MAX_LAMBDA = 0.01


class TRAKEEvent(StrictFrozenModel):
    """One event in the narrative order supplied by a TRAKE query."""

    event_id: NonEmptyStr
    text: NonEmptyStr


class DANTEPolicy(StrictFrozenModel):
    """Versioned public DANTE policy selected for a TRAKE request."""

    policy_version: Literal["dante-index-gap-v1"] = DANTE_POLICY_VERSION
    lambda_penalty: Annotated[
        FiniteFloat,
        Field(ge=DANTE_MIN_LAMBDA, le=DANTE_MAX_LAMBDA),
    ] = DANTE_DEFAULT_LAMBDA


class TRAKEQuery(StrictFrozenModel):
    """Validated ordered events and result policy for one TRAKE request."""

    query_id: NonEmptyStr
    events: tuple[TRAKEEvent, ...] = Field(min_length=2)
    top_k_videos: StrictIntValue = Field(default=1, ge=1)
    policy: DANTEPolicy = Field(default_factory=DANTEPolicy)

    @model_validator(mode="after")
    def validate_event_ids(self) -> "TRAKEQuery":
        event_ids = tuple(event.event_id for event in self.events)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("TRAKE event IDs must be unique within a query")
        return self


class TRAKEFrameMatch(StrictFrozenModel):
    """One event-to-keyframe match in a DANTE sequence."""

    event_id: NonEmptyStr
    frame_id: NonEmptyStr
    video_id: NonEmptyStr
    shot_id: StrictIntValue = Field(ge=0)
    local_index: StrictIntValue = Field(ge=0)
    timestamp_sec: Annotated[FiniteFloat, Field(ge=0.0)]
    similarity_score: FiniteFloat

    @model_validator(mode="after")
    def validate_frame_identity(self) -> "TRAKEFrameMatch":
        try:
            validate_canonical_frame_id(
                self.frame_id,
                video_id=self.video_id,
                shot_id=self.shot_id,
            )
        except ContractMismatchError as exc:
            raise ValueError(exc.message) from exc
        return self


class TRAKEVideoResult(StrictFrozenModel):
    """The single best DANTE sequence for one video."""

    video_id: NonEmptyStr
    score: FiniteFloat
    event_ids: tuple[NonEmptyStr, ...] = Field(min_length=2)
    sequence: tuple[TRAKEFrameMatch, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_sequence(self) -> "TRAKEVideoResult":
        if len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("event_ids must be unique")
        sequence_event_ids = tuple(match.event_id for match in self.sequence)
        if sequence_event_ids != self.event_ids:
            raise ValueError(
                "sequence must contain exactly one match for every event in event order"
            )
        if any(match.video_id != self.video_id for match in self.sequence):
            raise ValueError("all sequence matches must belong to result video_id")
        local_indices = tuple(match.local_index for match in self.sequence)
        if any(
            current >= following
            for current, following in zip(local_indices, local_indices[1:])
        ):
            raise ValueError("sequence local indices must be strictly increasing")
        return self


class TRAKEDiagnostics(StrictFrozenModel):
    """Bounded diagnostics for one TRAKE service execution."""

    policy_version: Literal["dante-index-gap-v1"]
    lambda_penalty: Annotated[
        FiniteFloat,
        Field(ge=DANTE_MIN_LAMBDA, le=DANTE_MAX_LAMBDA),
    ]
    event_count: StrictIntValue = Field(ge=2)
    video_count: StrictIntValue = Field(ge=0)
    frame_count: StrictIntValue = Field(ge=0)
    similarity_latency_ms: Annotated[FiniteFloat, Field(ge=0.0)] = 0.0
    dp_latency_ms: Annotated[FiniteFloat, Field(ge=0.0)] = 0.0
    invalid_sequence_count: StrictIntValue = Field(default=0, ge=0)
    warnings: tuple[NonEmptyStr, ...] = ()
