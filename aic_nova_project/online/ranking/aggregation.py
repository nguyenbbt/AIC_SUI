"""Query-variant aggregation that preserves per-variant evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any

from online.domain.candidates import (
    ASRIntervalCandidate,
    BranchResult,
    FrameCandidate,
    VideoCandidate,
)
from online.domain.enums import BranchStatus
from online.domain.errors import ContractMismatchError


CandidateKey = tuple[str, ...]


@dataclass(frozen=True)
class QueryVariantAggregationConfig:
    """Policy for weighting already-normalized query variants."""

    query_variant_weights: Mapping[str, float]
    method_name: str = "weighted_sum_query_variant_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.method_name, str) or not self.method_name.strip():
            raise ValueError("method_name must be non-empty")
        weights: dict[str, float] = {}
        for raw_variant_id, raw_weight in self.query_variant_weights.items():
            if not isinstance(raw_variant_id, str) or not raw_variant_id.strip():
                raise ValueError("query variant IDs must be non-empty strings")
            if (
                isinstance(raw_weight, bool)
                or not isinstance(raw_weight, (int, float))
                or not math.isfinite(float(raw_weight))
                or float(raw_weight) < 0.0
            ):
                raise ValueError("query variant weights must be finite and >= 0")
            weights[raw_variant_id.strip()] = float(raw_weight)
        object.__setattr__(self, "query_variant_weights", MappingProxyType(weights))

    def weight_for(self, query_variant_id: str) -> float:
        return float(self.query_variant_weights.get(query_variant_id, 1.0))


class RRFQueryVariantAggregator:
    """Prepare q0/q1/q2 results and weight their normalized evidence.

    The class name is retained for import compatibility with earlier C code.
    Aggregation preparation deliberately runs before branch-local normalization
    in the KIS pipeline.  Variant weights are applied only after normalization
    so backend-specific raw-score scales are never multiplied together.
    """

    def __init__(self, config: QueryVariantAggregationConfig | None = None) -> None:
        self.config = config or QueryVariantAggregationConfig(
            query_variant_weights={"q0": 1.0, "q1": 1.0, "q2": 1.0}
        )

    @property
    def name(self) -> str:
        return self.config.method_name

    def aggregate(self, branch_results: Sequence[BranchResult[Any]]) -> tuple[BranchResult[Any], ...]:
        values = _as_branch_results(branch_results)
        seen: set[tuple[object, str]] = set()
        for result in values:
            key = (result.branch, result.query_variant_id)
            if key in seen:
                raise ContractMismatchError(
                    "query-variant aggregation received duplicate branch variants",
                    details={
                        "branch": result.branch.value,
                        "query_variant_id": result.query_variant_id,
                    },
                )
            seen.add(key)
        return values

    def apply_normalized_weights(
        self,
        branch_results: Sequence[BranchResult[Any]],
    ) -> tuple[BranchResult[Any], ...]:
        """Apply variant weights after each branch result has been normalized."""

        values = _as_branch_results(branch_results)
        for result in values:
            if result.status not in {BranchStatus.SUCCESS, BranchStatus.DEGRADED}:
                continue
            if any(candidate.normalized_score is None for candidate in result.candidates):
                raise ContractMismatchError(
                    "query-variant weighting requires normalized candidates",
                    details={
                        "branch": result.branch.value,
                        "query_variant_id": result.query_variant_id,
                    },
                )
        return tuple(
            result.model_copy(
                update={
                    "candidates": tuple(
                        self._with_weighted_score(candidate, result.query_variant_id)
                        for candidate in result.candidates
                    )
                }
            )
            if result.status in {BranchStatus.SUCCESS, BranchStatus.DEGRADED}
            else result
            for result in values
        )

    def _with_weighted_score(self, candidate: object, query_variant_id: str) -> object:
        score = getattr(candidate, "normalized_score", None)
        if score is None:
            raise ContractMismatchError("query-variant aggregation requires normalized candidates")
        weighted_score = min(1.0, score * self.config.weight_for(query_variant_id))
        return candidate.model_copy(update={"normalized_score": weighted_score})  # type: ignore[attr-defined]


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
