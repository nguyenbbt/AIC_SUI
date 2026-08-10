"""Organizer-facing logical answer rows from the AIC 2026 preliminary rules.

The PDF fixes the ordered fields and the 100-answer cap, but does not define a
transport (HTTP payload, CSV headers, delimiter, or upload endpoint). These
models therefore represent logical answer rows only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from pydantic import Field

from online.domain.base import NonEmptyStr, StrictFrozenModel, StrictIntValue
from online.domain.candidates import FusedFrameCandidate
from online.domain.trake import TRAKEVideoResult
from online.domain.vqa import ImageEvidence, VLMResponse, VLMResponseStatus
from online.domain.errors import ContractMismatchError, InvalidQueryError


MAX_ANSWERS_PER_QUERY = 100
OrganizerFrameId = Annotated[StrictIntValue, Field(ge=0)]


class KISSubmissionRow(StrictFrozenModel):
    video_id: NonEmptyStr
    frame_id: OrganizerFrameId


class VQASubmissionRow(KISSubmissionRow):
    answer: NonEmptyStr


class TRAKESubmissionRow(StrictFrozenModel):
    video_id: NonEmptyStr
    frame_ids: tuple[OrganizerFrameId, ...] = Field(min_length=2)


def serialize_kis_submissions(
    candidates: Sequence[FusedFrameCandidate],
    *,
    limit: int = MAX_ANSWERS_PER_QUERY,
) -> tuple[KISSubmissionRow, ...]:
    """Map internal source-frame identity to BTC's external ``frame_id`` name."""

    values = _validated_sequence(candidates, FusedFrameCandidate, "candidates")
    bounded_limit = _validated_limit(limit)
    return tuple(
        KISSubmissionRow(video_id=item.video_id, frame_id=item.source_frame_idx)
        for item in values[:bounded_limit]
    )


def serialize_vqa_submission(
    *,
    image: ImageEvidence,
    response: VLMResponse,
) -> VQASubmissionRow:
    """Build one Q&A row from an explicitly selected source-frame evidence."""

    if not isinstance(image, ImageEvidence) or not isinstance(response, VLMResponse):
        raise ContractMismatchError("VQA submission requires validated image and response")
    if response.status is not VLMResponseStatus.ANSWERED or response.answer is None:
        raise ContractMismatchError("VQA submission requires an answered VLM response")
    return VQASubmissionRow(
        video_id=image.video_id,
        frame_id=image.source_frame_idx,
        answer=response.answer,
    )


def serialize_trake_submissions(
    results: Sequence[TRAKEVideoResult],
    *,
    limit: int = MAX_ANSWERS_PER_QUERY,
) -> tuple[TRAKESubmissionRow, ...]:
    """Preserve event order while mapping every match to its original frame index."""

    values = _validated_sequence(results, TRAKEVideoResult, "results")
    bounded_limit = _validated_limit(limit)
    return tuple(
        TRAKESubmissionRow(
            video_id=item.video_id,
            frame_ids=tuple(match.source_frame_idx for match in item.sequence),
        )
        for item in values[:bounded_limit]
    )


def _validated_limit(limit: int) -> int:
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_ANSWERS_PER_QUERY
    ):
        raise InvalidQueryError("submission limit must be within [1, 100]")
    return limit


def _validated_sequence(values: object, item_type: type, name: str) -> tuple:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise InvalidQueryError(f"{name} must be a sequence")
    result = tuple(values)
    if any(not isinstance(item, item_type) for item in result):
        raise ContractMismatchError(f"{name} contains an invalid submission item")
    return result


__all__ = [
    "KISSubmissionRow",
    "MAX_ANSWERS_PER_QUERY",
    "TRAKESubmissionRow",
    "VQASubmissionRow",
    "serialize_kis_submissions",
    "serialize_trake_submissions",
    "serialize_vqa_submission",
]
