"""Models for the explicitly unstable internal TRAKE and VQA API."""

from __future__ import annotations

from pydantic import Field, model_validator

from online.domain.base import NonEmptyStr, StrictFrozenModel, StrictIntValue
from online.domain.trake import DANTEPolicy, TRAKEDiagnostics, TRAKEEvent, TRAKEQuery, TRAKEVideoResult
from online.domain.vqa import VQAAnswerType, VQAEvidenceBudget, VQAQuestion, VQAResult


class InternalTRAKERequest(StrictFrozenModel):
    query_id: NonEmptyStr
    event_texts: tuple[NonEmptyStr, ...] = Field(min_length=2)
    event_ids: tuple[NonEmptyStr, ...] | None = None
    top_k_videos: StrictIntValue = Field(default=1, ge=1)
    policy: DANTEPolicy = Field(default_factory=DANTEPolicy)

    @model_validator(mode="after")
    def validate_event_ids(self) -> "InternalTRAKERequest":
        if self.event_ids is not None:
            if len(self.event_ids) != len(self.event_texts):
                raise ValueError("event_ids must match event_texts length")
            if len(set(self.event_ids)) != len(self.event_ids):
                raise ValueError("event_ids must be unique")
        return self

    def to_domain(self) -> TRAKEQuery:
        event_ids = self.event_ids or tuple(
            f"{self.query_id}:event:{index}" for index in range(len(self.event_texts))
        )
        return TRAKEQuery(
            query_id=self.query_id,
            events=tuple(
                TRAKEEvent(event_id=event_id, text=text)
                for event_id, text in zip(event_ids, self.event_texts, strict=True)
            ),
            top_k_videos=self.top_k_videos,
            policy=self.policy,
        )


class InternalTRAKEResponse(StrictFrozenModel):
    schema_version: str = "internal-unstable-v1"
    query_id: NonEmptyStr
    results: tuple[TRAKEVideoResult, ...]
    diagnostics: TRAKEDiagnostics


class InternalVQARequest(StrictFrozenModel):
    question_id: NonEmptyStr
    question: NonEmptyStr
    answer_type: VQAAnswerType
    evidence_budget: VQAEvidenceBudget = Field(default_factory=VQAEvidenceBudget)

    def to_domain(self) -> VQAQuestion:
        return VQAQuestion(
            question_id=self.question_id,
            question=self.question,
            answer_type=self.answer_type,
        )


class InternalVQAResponse(StrictFrozenModel):
    schema_version: str = "internal-unstable-v1"
    question_id: NonEmptyStr
    result: VQAResult


class TRAKEResponse(StrictFrozenModel):
    schema_version: str = "online-v1"
    query_id: NonEmptyStr
    results: tuple[TRAKEVideoResult, ...]
    diagnostics: TRAKEDiagnostics


class VQAResponse(StrictFrozenModel):
    schema_version: str = "online-v1"
    question_id: NonEmptyStr
    result: VQAResult


# Concise request aliases for callers that already scope imports to advanced_models.
TRAKERequest = InternalTRAKERequest
VQARequest = InternalVQARequest


__all__ = [
    "InternalTRAKERequest",
    "InternalTRAKEResponse",
    "InternalVQARequest",
    "InternalVQAResponse",
    "TRAKERequest",
    "TRAKEResponse",
    "VQARequest",
    "VQAResponse",
]
