"""Query-variant aggregation that preserves per-variant evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from online.domain.candidates import (
    ASRIntervalCandidate,
    BranchResult,
    FrameCandidate,
    VideoCandidate,
)
from online.domain.enums import BranchStatus


CandidateKey = tuple[str, ...]


class RRFQueryVariantAggregator:
    """Assign RRF contributions across q0/q1/q2 without breaking BranchResult.

    The shared ``BranchResult`` contract requires every result and candidate to
    keep one exact ``query_variant_id``.  For that reason this component does
    not collapse variants into a synthetic mixed-variant result.  It computes
    frame/interval/video totals across variants, stores each evidence item's RRF
    contribution in ``normalized_score``, and leaves fusion to aggregate those
    contributions by candidate ID and branch.
    """

    name = "rrf_query_variant_aggregation"

    def __init__(self, *, k: int = 60) -> None:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        self.k = k

    def aggregate(self, branch_results: Sequence[BranchResult[Any]]) -> tuple[BranchResult[Any], ...]:
        values = _as_branch_results(branch_results)
        totals: dict[tuple[object, CandidateKey], float] = defaultdict(float)
        for result in values:
            if result.status not in {BranchStatus.SUCCESS, BranchStatus.DEGRADED}:
                continue
            for candidate in result.candidates:
                totals[(result.branch, _candidate_key(candidate))] += self._rrf(candidate.rank)

        return tuple(
            result.model_copy(
                update={
                    "candidates": tuple(
                        candidate.model_copy(
                            update={
                                "normalized_score": self._variant_contribution(
                                    candidate,
                                    totals[(result.branch, _candidate_key(candidate))],
                                )
                            }
                        )
                        for candidate in result.candidates
                    )
                }
            )
            if result.status in {BranchStatus.SUCCESS, BranchStatus.DEGRADED}
            else result
            for result in values
        )

    def _rrf(self, rank: int) -> float:
        return 1.0 / (self.k + rank)

    def _variant_contribution(self, candidate: object, total: float) -> float:
        contribution = self._rrf(getattr(candidate, "rank"))
        # RRF values are naturally small, but keep the domain score invariant
        # intact if callers use an unusually small k or many query variants.
        if total <= 1.0:
            return contribution
        return contribution / total


def _candidate_key(candidate: object) -> CandidateKey:
    if isinstance(candidate, FrameCandidate):
        return ("frame", candidate.frame_id)
    if isinstance(candidate, ASRIntervalCandidate):
        return ("asr_interval", candidate.video_id, candidate.interval_id)
    if isinstance(candidate, VideoCandidate):
        return ("video", candidate.video_id)
    raise TypeError("BranchResult contains an unsupported candidate type")


def _as_branch_results(results: Sequence[BranchResult[Any]]) -> tuple[BranchResult[Any], ...]:
    if isinstance(results, (str, bytes)):
        raise TypeError("branch_results must be a sequence")
    values = tuple(results)
    if any(not isinstance(result, BranchResult) for result in values):
        raise TypeError("branch_results must contain BranchResult objects")
    return values
