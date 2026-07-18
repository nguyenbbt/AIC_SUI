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
        near_by_id = {
            frame.frame_id: frame
            for frame in representative.near_frames
            if frame.frame_id != representative.frame_id
        }
        for candidate in ordered[1:]:
            if candidate.frame_id == representative.frame_id:
                continue
            near_by_id[candidate.frame_id] = NearFrameRef(
                frame_id=candidate.frame_id,
                timestamp_sec=candidate.timestamp_sec,
                final_score=candidate.final_score,
            )
        near_frames = tuple(
            sorted(
                near_by_id.values(),
                key=lambda item: (-item.final_score, item.frame_id),
            )
        )
        return FusedFrameCandidate(
            frame_id=representative.frame_id,
            video_id=representative.video_id,
            shot_id=representative.shot_id,
            timestamp_sec=representative.timestamp_sec,
            final_score=representative.final_score,
            branch_scores=representative.branch_scores,
            evidence=representative.evidence,
            near_frames=near_frames,
            objects=representative.objects,
            diagnostics=representative.diagnostics,
        )
