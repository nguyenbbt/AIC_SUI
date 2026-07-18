"""Frame-level fusion over normalized BranchResult handoffs."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from online.domain.candidates import (
    BranchResult,
    CandidateDiagnostics,
    CandidateEvidence,
    FrameCandidate,
    FusedFrameCandidate,
)
from online.domain.enums import BranchStatus, CandidateLevel, RetrievalBranch
from online.domain.errors import ContractMismatchError


@dataclass(frozen=True)
class FusionConfig:
    """Weights for normalized branch fusion.

    A missing branch is not inserted as fake evidence.  Branches absent from the
    mapping use ``default_weight`` so experiments can start with equal-weight
    fusion and later pin exact values for OQ-006/OQ-007.
    """

    weights: Mapping[RetrievalBranch, float]
    default_weight: float = 1.0
    method_name: str = "experimental_weighted_sum_normalized_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.method_name, str) or not self.method_name.strip():
            raise ValueError("method_name must be non-empty")
        if not _valid_weight(self.default_weight):
            raise ValueError("default_weight must be finite and >= 0")
        normalized: dict[RetrievalBranch, float] = {}
        for raw_branch, weight in self.weights.items():
            branch = RetrievalBranch(raw_branch)
            if not _valid_weight(weight):
                raise ValueError(f"invalid fusion weight for {branch.value}")
            normalized[branch] = float(weight)
        object.__setattr__(self, "weights", MappingProxyType(normalized))

    def weight_for(self, branch: RetrievalBranch) -> float:
        return float(self.weights.get(branch, self.default_weight))


class WeightedFrameFusion:
    """Merge normalized frame evidence into deterministic final candidates."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig(weights={})

    @property
    def name(self) -> str:
        return self.config.method_name

    def fuse(self, branch_results: Sequence[BranchResult[Any]]) -> tuple[FusedFrameCandidate, ...]:
        values = _as_branch_results(branch_results)
        grouped: dict[str, list[FrameCandidate]] = defaultdict(list)
        for result in values:
            if result.status not in {BranchStatus.SUCCESS, BranchStatus.DEGRADED}:
                continue
            if result.candidate_level is not CandidateLevel.FRAME:
                continue
            for candidate in result.candidates:
                if not isinstance(candidate, FrameCandidate):
                    raise TypeError("frame BranchResult contains a non-frame candidate")
                if candidate.normalized_score is None:
                    raise ValueError("frame fusion requires normalized scores")
                grouped[candidate.frame_id].append(candidate)

        fused = tuple(self._fuse_frame(candidates) for candidates in grouped.values())
        return tuple(sorted(fused, key=lambda item: (-item.final_score, item.frame_id)))

    def _fuse_frame(self, candidates: Sequence[FrameCandidate]) -> FusedFrameCandidate:
        ordered = tuple(sorted(candidates, key=_evidence_sort_key))
        representative = ordered[0]
        _validate_frame_contract(ordered)
        branch_scores: dict[RetrievalBranch, float] = {}
        evidence: list[CandidateEvidence] = []
        for candidate in ordered:
            assert candidate.normalized_score is not None
            branch = candidate.provenance.branch
            branch_scores[branch] = min(
                1.0,
                branch_scores.get(branch, 0.0) + candidate.normalized_score,
            )
            evidence.append(
                CandidateEvidence(
                    branch=branch,
                    query_variant_id=candidate.provenance.query_variant_id,
                    raw_score=candidate.raw_score,
                    normalized_score=candidate.normalized_score,
                )
            )

        weighted_scores = (
            self.config.weight_for(branch) * score
            for branch, score in branch_scores.items()
        )
        final_score = sum(weighted_scores)
        if not math.isfinite(final_score):
            raise ValueError("fusion produced a non-finite score")

        return FusedFrameCandidate(
            frame_id=representative.frame_id,
            video_id=representative.video_id,
            shot_id=representative.shot_id,
            timestamp_sec=representative.timestamp_sec,
            final_score=final_score,
            branch_scores=branch_scores,
            evidence=tuple(evidence),
            diagnostics=CandidateDiagnostics(),
        )


def _valid_weight(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _as_branch_results(results: Sequence[BranchResult[Any]]) -> tuple[BranchResult[Any], ...]:
    if isinstance(results, (str, bytes)):
        raise TypeError("branch_results must be a sequence")
    values = tuple(results)
    if any(not isinstance(result, BranchResult) for result in values):
        raise TypeError("branch_results must contain BranchResult objects")
    return values


def _evidence_sort_key(candidate: FrameCandidate) -> tuple[str, str, int]:
    return (
        candidate.provenance.branch.value,
        candidate.provenance.query_variant_id,
        candidate.rank,
    )


def _validate_frame_contract(candidates: Sequence[FrameCandidate]) -> None:
    first = candidates[0]
    for candidate in candidates[1:]:
        mismatches = {
            field
            for field in ("video_id", "shot_id", "timestamp_sec")
            if getattr(candidate, field) != getattr(first, field)
        }
        if mismatches:
            raise ContractMismatchError(
                "frame_id maps to conflicting metadata across branches",
                details={
                    "frame_id": first.frame_id,
                    "mismatched_fields": tuple(sorted(mismatches)),
                },
            )
