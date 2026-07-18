"""Shot-level deduplication and near-frame grouping."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from online.domain.candidates import FusedFrameCandidate, NearFrameRef


class ShotDeduplicator:
    """Keep one representative frame per ``(video_id, shot_id)`` group."""

    name = "shot_dedup"

    def deduplicate(self, candidates: Sequence[FusedFrameCandidate]) -> tuple[FusedFrameCandidate, ...]:
        if isinstance(candidates, (str, bytes)):
            raise TypeError("candidates must be a sequence")
        values = tuple(candidates)
        if any(not isinstance(candidate, FusedFrameCandidate) for candidate in values):
            raise TypeError("candidates must contain FusedFrameCandidate objects")

        groups: dict[tuple[str, int], list[FusedFrameCandidate]] = defaultdict(list)
        for candidate in values:
            groups[(candidate.video_id, candidate.shot_id)].append(candidate)

        representatives = tuple(
            self._represent_group(group)
            for group in groups.values()
        )
        return tuple(sorted(representatives, key=lambda item: (-item.final_score, item.frame_id)))

    @staticmethod
    def _represent_group(group: Sequence[FusedFrameCandidate]) -> FusedFrameCandidate:
        ordered = tuple(sorted(group, key=lambda item: (-item.final_score, item.frame_id)))
        representative = ordered[0]
        near_frames = tuple(
            NearFrameRef(
                frame_id=candidate.frame_id,
                timestamp_sec=candidate.timestamp_sec,
                final_score=candidate.final_score,
            )
            for candidate in ordered[1:]
        )
        return representative.model_copy(
            update={"near_frames": representative.near_frames + near_frames}
        )
