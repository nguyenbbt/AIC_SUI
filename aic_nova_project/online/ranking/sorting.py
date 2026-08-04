"""Deterministic organizer-v1 ranking order shared by Person-C stages."""

from __future__ import annotations

from online.domain.candidates import FusedFrameCandidate


def fused_candidate_sort_key(
    candidate: FusedFrameCandidate,
) -> tuple[float, str, int, str]:
    """Sort by score DESC, then video/local identity ASC."""

    return (
        -candidate.final_score,
        candidate.video_id,
        candidate.local_index,
        candidate.frame_id,
    )


__all__ = ["fused_candidate_sort_key"]
