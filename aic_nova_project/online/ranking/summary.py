"""Video-summary propagation into existing frame evidence."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from online.domain.candidates import (
    BranchResult,
    CandidateEvidence,
    CandidateDiagnostics,
    FusedFrameCandidate,
    VideoCandidate,
)
from online.domain.enums import BranchStatus, CandidateLevel, RetrievalBranch


SUMMARY_BRANCHES = frozenset(
    {
        RetrievalBranch.SUMMARY_DENSE,
        RetrievalBranch.SUMMARY_BM25,
    }
)


@dataclass(frozen=True)
class SummaryPropagationConfig:
    """Controlled video-level boost for already-evidenced frames."""

    weight: float = 0.1
    max_boost: float = 0.2
    method_name: str = "summary_video_score_cap_v1"

    def __post_init__(self) -> None:
        if not _finite_non_negative(self.weight):
            raise ValueError("weight must be finite and >= 0")
        if not _finite_non_negative(self.max_boost):
            raise ValueError("max_boost must be finite and >= 0")
        if self.weight > 1.0:
            raise ValueError("weight must be <= 1 while stored in CandidateEvidence.normalized_score")
        if self.max_boost > 1.0:
            raise ValueError("max_boost must be <= 1 while stored in CandidateEvidence.normalized_score")
        if not isinstance(self.method_name, str) or not self.method_name.strip():
            raise ValueError("method_name must be non-empty")


@dataclass(frozen=True)
class SummaryContributionResult:
    total_boost: float
    evidence: tuple[CandidateEvidence, ...]


class SummaryScorePropagator:
    """Aggregate summary candidates by video and boost existing frames."""

    def __init__(self, config: SummaryPropagationConfig | None = None) -> None:
        self.config = config or SummaryPropagationConfig()

    @property
    def name(self) -> str:
        return self.config.method_name

    def propagate(
        self,
        frames: Sequence[FusedFrameCandidate],
        summary_results: Sequence[BranchResult[Any]],
    ) -> tuple[FusedFrameCandidate, ...]:
        frame_values = _as_frames(frames)
        if not frame_values:
            return ()
        contributions = self._contributions_by_video(summary_results)
        if not contributions:
            return frame_values

        boosted: list[FusedFrameCandidate] = []
        for frame in frame_values:
            contribution = contributions.get(frame.video_id)
            if contribution is None:
                boosted.append(frame)
                continue
            boost = contribution.total_boost
            base_score = frame.final_score - frame.diagnostics.summary_boost
            diagnostics = CandidateDiagnostics(
                summary_boost=boost,
                object_boost=frame.diagnostics.object_boost,
                object_constraints_satisfied=frame.diagnostics.object_constraints_satisfied,
            )
            base_evidence = tuple(
                evidence for evidence in frame.evidence if evidence.branch not in SUMMARY_BRANCHES
            )
            boosted.append(
                frame.model_copy(
                    update={
                        "final_score": base_score + boost,
                        "diagnostics": diagnostics,
                        "evidence": base_evidence + contribution.evidence,
                    }
                )
            )
        return tuple(sorted(boosted, key=lambda item: (-item.final_score, item.frame_id)))

    def aggregate_video_scores(
        self,
        summary_results: Sequence[BranchResult[Any]],
    ) -> dict[str, float]:
        return {
            video_id: min(1.0, contribution.total_boost / self.config.weight)
            if self.config.weight > 0.0
            else 0.0
            for video_id, contribution in self._contributions_by_video(summary_results).items()
        }

    def _contributions_by_video(
        self,
        summary_results: Sequence[BranchResult[Any]],
    ) -> dict[str, SummaryContributionResult]:
        values = _as_branch_results(summary_results)
        by_video: dict[str, list[CandidateEvidence]] = defaultdict(list)
        boost_by_video: dict[str, float] = defaultdict(float)
        for result in values:
            if result.status not in {BranchStatus.SUCCESS, BranchStatus.DEGRADED}:
                continue
            if result.candidate_level is not CandidateLevel.VIDEO or result.branch not in SUMMARY_BRANCHES:
                continue
            for candidate in result.candidates:
                if not isinstance(candidate, VideoCandidate):
                    raise TypeError("summary BranchResult contains a non-video candidate")
                score = candidate.normalized_score
                if score is None:
                    score = 1.0 / (60 + candidate.rank)
                remaining = max(0.0, self.config.max_boost - boost_by_video[candidate.video_id])
                contribution = min(remaining, score * self.config.weight)
                if contribution <= 0.0:
                    continue
                boost_by_video[candidate.video_id] += contribution
                by_video[candidate.video_id].append(
                    CandidateEvidence(
                        branch=candidate.provenance.branch,
                        query_variant_id=candidate.provenance.query_variant_id,
                        raw_score=candidate.raw_score,
                        normalized_score=contribution,
                    )
                )
        return {
            video_id: SummaryContributionResult(
                total_boost=boost_by_video[video_id],
                evidence=tuple(evidence),
            )
            for video_id, evidence in by_video.items()
        }


def _as_frames(frames: Sequence[FusedFrameCandidate]) -> tuple[FusedFrameCandidate, ...]:
    if isinstance(frames, (str, bytes)):
        raise TypeError("frames must be a sequence")
    values = tuple(frames)
    if any(not isinstance(frame, FusedFrameCandidate) for frame in values):
        raise TypeError("frames must contain FusedFrameCandidate objects")
    return values


def _as_branch_results(results: Sequence[BranchResult[Any]]) -> tuple[BranchResult[Any], ...]:
    if isinstance(results, (str, bytes)):
        raise TypeError("summary_results must be a sequence")
    values = tuple(results)
    if any(not isinstance(result, BranchResult) for result in values):
        raise TypeError("summary_results must contain BranchResult objects")
    return values


def _finite_non_negative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )
