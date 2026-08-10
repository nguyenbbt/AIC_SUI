"""Branch-local score normalization for Person-C ranking.

The papers used as implementation guidance combine heterogeneous retrieval
signals only after normalization.  These normalizers deliberately operate on one
branch/query-variant candidate list at a time; callers must not apply min-max
globally across Milvus similarity and Elasticsearch BM25 outputs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, TypeVar, runtime_checkable

from online.domain.candidates import (
    ASRIntervalCandidate,
    FrameCandidate,
    VideoCandidate,
)


NormalizableCandidate = FrameCandidate | ASRIntervalCandidate | VideoCandidate
CandidateT = TypeVar("CandidateT", FrameCandidate, ASRIntervalCandidate, VideoCandidate)


@runtime_checkable
class ScoreNormalizer(Protocol):
    name: str

    def normalize(self, candidates: Sequence[CandidateT]) -> tuple[CandidateT, ...]: ...


class RRFScoreNormalizer:
    """Rank-based reciprocal rank normalization.

    This is robust for branch/query-variant aggregation because it depends on
    rank order instead of raw backend scale.
    """

    name = "rrf"

    def __init__(self, *, k: int = 60) -> None:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        self.k = k

    def normalize(self, candidates: Sequence[CandidateT]) -> tuple[CandidateT, ...]:
        values = _as_tuple(candidates)
        return tuple(
            _with_normalized_score(candidate, 1.0 / (self.k + candidate.rank))
            for candidate in values
        )


class MinMaxScoreNormalizer:
    """Branch-local min-max normalization with equal-score handling."""

    name = "min_max"

    def normalize(self, candidates: Sequence[CandidateT]) -> tuple[CandidateT, ...]:
        values = _as_tuple(candidates)
        if not values:
            return ()
        scores = tuple(candidate.raw_score for candidate in values)
        minimum = min(scores)
        maximum = max(scores)
        if math.isclose(maximum, minimum):
            return tuple(_with_normalized_score(candidate, 1.0) for candidate in values)
        span = maximum - minimum
        return tuple(
            _with_normalized_score(candidate, (candidate.raw_score - minimum) / span)
            for candidate in values
        )


def _as_tuple(candidates: Sequence[CandidateT]) -> tuple[CandidateT, ...]:
    if isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be a sequence of domain candidates")
    values = tuple(candidates)
    if any(not isinstance(candidate, (FrameCandidate, ASRIntervalCandidate, VideoCandidate)) for candidate in values):
        raise TypeError("candidates must contain only domain candidates")
    return values


def _with_normalized_score(candidate: CandidateT, score: float) -> CandidateT:
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("normalized score must be finite and within [0, 1]")
    return candidate.model_copy(update={"normalized_score": score})  # type: ignore[return-value]
