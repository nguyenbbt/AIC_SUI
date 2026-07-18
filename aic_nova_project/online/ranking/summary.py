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
        if not isinstance(self.method_name, str) or not self.method_name.strip():
            raise ValueError("method_name must be non-empty")


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
        summary_scores = self.aggregate_video_scores(summary_results)
        if not summary_scores:
            return frame_values
        summary_evidence = self._summary_evidence_by_video(summary_results)

        boosted: list[FusedFrameCandidate] = []
        for frame in frame_values:
            video_score = summary_scores.get(frame.video_id, 0.0)
            boost = min(self.config.max_boost, video_score * self.config.weight)
            diagnostics = CandidateDiagnostics(
                summary_boost=frame.diagnostics.summary_boost + boost,
                object_boost=frame.diagnostics.object_boost,
                object_constraints_satisfied=frame.diagnostics.object_constraints_satisfied,
            )
            boosted.append(
                frame.model_copy(
                    update={
                        "final_score": frame.final_score + boost,
                        "diagnostics": diagnostics,
                        "evidence": frame.evidence + summary_evidence.get(frame.video_id, ()),
                    }
                )
            )
        return tuple(sorted(boosted, key=lambda item: (-item.final_score, item.frame_id)))

    def aggregate_video_scores(
        self,
        summary_results: Sequence[BranchResult[Any]],
    ) -> dict[str, float]:
        values = _as_branch_results(summary_results)
        scores: dict[str, float] = defaultdict(float)
        for result in values:
            if result.status not in {BranchStatus.SUCCESS, BranchStatus.DEGRADED}:
                continue
            if result.candidate_level is not CandidateLevel.VIDEO:
                continue
            if result.branch not in SUMMARY_BRANCHES:
                continue
            for candidate in result.candidates:
                if not isinstance(candidate, VideoCandidate):
                    raise TypeError("summary BranchResult contains a non-video candidate")
                score = candidate.normalized_score
                if score is None:
                    score = 1.0 / (60 + candidate.rank)
                scores[candidate.video_id] = min(1.0, scores[candidate.video_id] + score)
        return dict(scores)

    def _summary_evidence_by_video(
        self,
        summary_results: Sequence[BranchResult[Any]],
    ) -> dict[str, tuple[CandidateEvidence, ...]]:
        values = _as_branch_results(summary_results)
        by_video: dict[str, list[CandidateEvidence]] = defaultdict(list)
        used_boost_by_video: dict[str, float] = defaultdict(float)
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
                remaining = max(0.0, self.config.max_boost - used_boost_by_video[candidate.video_id])
                contribution = min(remaining, score * self.config.weight)
                if contribution <= 0.0:
                    continue
                used_boost_by_video[candidate.video_id] += contribution
                by_video[candidate.video_id].append(
                    CandidateEvidence(
                        branch=candidate.provenance.branch,
                        query_variant_id=candidate.provenance.query_variant_id,
                        raw_score=candidate.raw_score,
                        normalized_score=contribution,
                    )
                )
        return {
            video_id: tuple(evidence)
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
