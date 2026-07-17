"""Canonical candidate and branch-result contracts."""

from __future__ import annotations

from typing import Annotated, Generic, Literal, TypeVar

from pydantic import AfterValidator, Field, model_validator

from .base import (
    FiniteFloat,
    NonEmptyStr,
    StrictFrozenModel,
    ensure_bbox_order,
    freeze_mapping,
)
from .enums import BranchStatus, CandidateLevel, RetrievalBranch


NormalizedScore = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]


class CandidateProvenance(StrictFrozenModel):
    branch: RetrievalBranch
    backend: Literal["milvus", "elasticsearch", "derived"]
    source_resource: NonEmptyStr
    query_variant_id: NonEmptyStr
    query_text: NonEmptyStr


class FrameCandidate(StrictFrozenModel):
    frame_id: NonEmptyStr
    video_id: NonEmptyStr
    shot_id: int = Field(ge=0)
    timestamp_sec: Annotated[FiniteFloat, Field(ge=0.0)]
    rank: int = Field(ge=1)
    raw_score: FiniteFloat
    normalized_score: NormalizedScore | None = None
    provenance: CandidateProvenance


class ASRIntervalCandidate(StrictFrozenModel):
    video_id: NonEmptyStr
    interval_id: NonEmptyStr
    start_time_sec: Annotated[FiniteFloat, Field(ge=0.0)]
    end_time_sec: Annotated[FiniteFloat, Field(ge=0.0)]
    rank: int = Field(ge=1)
    raw_score: FiniteFloat
    normalized_score: NormalizedScore | None = None
    text: str | None = None
    provenance: CandidateProvenance

    @model_validator(mode="after")
    def validate_interval(self) -> "ASRIntervalCandidate":
        if self.end_time_sec < self.start_time_sec:
            raise ValueError("end_time_sec must be >= start_time_sec")
        return self


class VideoCandidate(StrictFrozenModel):
    video_id: NonEmptyStr
    rank: int = Field(ge=1)
    raw_score: FiniteFloat
    normalized_score: NormalizedScore | None = None
    summary: str | None = None
    provenance: CandidateProvenance


Candidate = FrameCandidate | ASRIntervalCandidate | VideoCandidate
CandidateT = TypeVar("CandidateT", FrameCandidate, ASRIntervalCandidate, VideoCandidate)


class BranchResult(StrictFrozenModel, Generic[CandidateT]):
    branch: RetrievalBranch
    candidate_level: CandidateLevel
    query_variant_id: NonEmptyStr
    candidates: tuple[CandidateT, ...]
    requested_top_k: int = Field(ge=1)
    latency_ms: Annotated[FiniteFloat, Field(ge=0.0)]
    status: BranchStatus
    warnings: tuple[NonEmptyStr, ...] = ()

    @property
    def returned_count(self) -> int:
        return len(self.candidates)

    @model_validator(mode="after")
    def validate_level_and_status(self) -> "BranchResult[CandidateT]":
        expected = {
            CandidateLevel.FRAME: FrameCandidate,
            CandidateLevel.ASR_INTERVAL: ASRIntervalCandidate,
            CandidateLevel.VIDEO: VideoCandidate,
        }[self.candidate_level]
        if any(not isinstance(candidate, expected) for candidate in self.candidates):
            raise ValueError(f"{self.candidate_level.value} result contains wrong candidate type")
        if self.status is not BranchStatus.SUCCESS and not self.warnings:
            raise ValueError("non-success BranchResult must contain a warning")
        if self.status in {BranchStatus.FAILED, BranchStatus.DISABLED} and self.candidates:
            raise ValueError("failed/disabled BranchResult must not contain candidates")
        if any(candidate.provenance.branch is not self.branch for candidate in self.candidates):
            raise ValueError("candidate provenance branch must match BranchResult branch")
        if any(
            candidate.provenance.query_variant_id != self.query_variant_id
            for candidate in self.candidates
        ):
            raise ValueError("candidate query_variant_id must match BranchResult")
        return self


class ObjectDetection(StrictFrozenModel):
    label: NonEmptyStr
    confidence: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
    x_min: Annotated[FiniteFloat, Field(ge=0.0)]
    y_min: Annotated[FiniteFloat, Field(ge=0.0)]
    x_max: Annotated[FiniteFloat, Field(ge=0.0)]
    y_max: Annotated[FiniteFloat, Field(ge=0.0)]
    model_source: NonEmptyStr | None = None

    _ordered = model_validator(mode="after")(ensure_bbox_order)


class NearFrameRef(StrictFrozenModel):
    frame_id: NonEmptyStr
    timestamp_sec: Annotated[FiniteFloat, Field(ge=0.0)]
    final_score: FiniteFloat


class CandidateEvidence(StrictFrozenModel):
    branch: RetrievalBranch
    query_variant_id: NonEmptyStr
    raw_score: FiniteFloat
    normalized_score: NormalizedScore


class CandidateDiagnostics(StrictFrozenModel):
    summary_boost: FiniteFloat = 0.0
    object_boost: FiniteFloat = 0.0
    object_constraints_satisfied: int = Field(default=0, ge=0)


class FusedFrameCandidate(StrictFrozenModel):
    frame_id: NonEmptyStr
    video_id: NonEmptyStr
    shot_id: int = Field(ge=0)
    timestamp_sec: Annotated[FiniteFloat, Field(ge=0.0)]
    final_score: FiniteFloat
    branch_scores: Annotated[
        dict[RetrievalBranch, NormalizedScore], AfterValidator(freeze_mapping)
    ]
    evidence: tuple[CandidateEvidence, ...]
    near_frames: tuple[NearFrameRef, ...] = ()
    objects: tuple[ObjectDetection, ...] = ()
    diagnostics: CandidateDiagnostics

    @model_validator(mode="after")
    def validate_near_frames(self) -> "FusedFrameCandidate":
        near_ids = [frame.frame_id for frame in self.near_frames]
        if self.frame_id in near_ids:
            raise ValueError("near_frames must not contain the representative frame")
        if len(near_ids) != len(set(near_ids)):
            raise ValueError("near_frames must not contain duplicates")
        return self
