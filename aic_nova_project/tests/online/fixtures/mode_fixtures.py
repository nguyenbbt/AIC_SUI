"""C-owned consumer expectations for the self-indexed-v2 frame contract."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


Payload = Mapping[str, object]


def _payload(**values: object) -> Payload:
    return MappingProxyType(values)


def _provenance(frame_id: str) -> Payload:
    return _payload(
        branch="visual_dense",
        backend="milvus",
        source_resource="visual_features",
        query_variant_id="q0",
        query_text="a person lifts a cup",
        source_candidate_id=frame_id,
        source_start_time_sec=None,
        source_end_time_sec=None,
        source_normalized_score=None,
    )


def _candidate(metadata: Payload, *, rank: int, raw_score: float) -> Payload:
    return _payload(
        frame_id=metadata["frame_id"],
        video_id=metadata["video_id"],
        shot_id=metadata["shot_id"],
        timestamp_sec=metadata["timestamp_sec"],
        source_frame_idx=metadata["source_frame_idx"],
        image_rel_path=metadata["image_rel_path"],
        rank=rank,
        raw_score=raw_score,
        normalized_score=None,
        provenance=_provenance(str(metadata["frame_id"])),
    )


def _fused(candidate: Payload, *, final_score: float) -> Payload:
    return _payload(
        frame_id=candidate["frame_id"],
        video_id=candidate["video_id"],
        shot_id=candidate["shot_id"],
        timestamp_sec=candidate["timestamp_sec"],
        source_frame_idx=candidate["source_frame_idx"],
        image_rel_path=candidate["image_rel_path"],
        final_score=final_score,
        branch_scores=MappingProxyType({"visual_dense": final_score}),
        evidence=(
            _payload(
                branch="visual_dense",
                query_variant_id="q0",
                raw_score=candidate["raw_score"],
                normalized_score=final_score,
                backend="milvus",
                source_resource="visual_features",
                source_candidate_id=candidate["frame_id"],
                source_start_time_sec=None,
                source_end_time_sec=None,
                source_normalized_score=None,
            ),
        ),
        near_frames=(),
        objects=(),
        diagnostics=_payload(
            summary_boost=0.0,
            object_boost=0.0,
            object_constraints_satisfied=0,
        ),
    )


@dataclass(frozen=True)
class SelfIndexedModeExpectations:
    frame_metadata_payloads: tuple[Payload, ...]
    frame_candidate_payloads: tuple[Payload, ...]
    expected_fused_payloads: tuple[Payload, ...]
    asr_interval_payload: Payload
    expected_asr_mapped_frame_ids: tuple[str, ...]
    expected_kis_frame_ids_after_dedup: tuple[str, ...]
    expected_kis_competition_rows: tuple[tuple[str, int], ...]
    expected_trake_competition_row: tuple[str, tuple[int, ...]]
    expected_vqa_competition_row: tuple[str, int, str]


def self_indexed_mode_expectations() -> SelfIndexedModeExpectations:
    metadata = (
        _payload(frame_id="L21_V001_00000_015", video_id="L21_V001", shot_id=0, timestamp_sec=0.0, source_frame_idx=0, image_rel_path="keyframes/L21_V001/shot_00000_pos_015.webp"),
        _payload(frame_id="L21_V001_00000_050", video_id="L21_V001", shot_id=0, timestamp_sec=0.0333333, source_frame_idx=0, image_rel_path="keyframes/L21_V001/shot_00000_pos_050.webp"),
        _payload(frame_id="L21_V001_00001_085", video_id="L21_V001", shot_id=1, timestamp_sec=1.0, source_frame_idx=30, image_rel_path="keyframes/L21_V001/shot_00001_pos_085.webp"),
        _payload(frame_id="L21_V002_00000_050", video_id="L21_V002", shot_id=0, timestamp_sec=0.0, source_frame_idx=0, image_rel_path="keyframes/L21_V002/shot_00000_pos_050.webp"),
    )
    candidates = (
        _candidate(metadata[0], rank=2, raw_score=0.82),
        _candidate(metadata[1], rank=1, raw_score=0.92),
        _candidate(metadata[2], rank=3, raw_score=0.70),
        _candidate(metadata[3], rank=1, raw_score=0.82),
    )
    fused = (
        _fused(candidates[1], final_score=0.92),
        _fused(candidates[0], final_score=0.82),
        _fused(candidates[3], final_score=0.82),
        _fused(candidates[2], final_score=0.70),
    )
    return SelfIndexedModeExpectations(
        frame_metadata_payloads=metadata,
        frame_candidate_payloads=candidates,
        expected_fused_payloads=fused,
        asr_interval_payload=_payload(video_id="L21_V001", interval_id="0", start_time_sec=0.0, end_time_sec=0.05),
        expected_asr_mapped_frame_ids=("L21_V001_00000_015", "L21_V001_00000_050"),
        expected_kis_frame_ids_after_dedup=("L21_V001_00000_050", "L21_V002_00000_050", "L21_V001_00001_085"),
        expected_kis_competition_rows=(("L21_V001", 0), ("L21_V002", 0), ("L21_V001", 30)),
        expected_trake_competition_row=("L21_V001", (0, 30)),
        expected_vqa_competition_row=("L21_V001", 0, "The man lifts a cup."),
    )


# Compatibility aliases for downstream branches that still import the old names.
OrganizerWave0ModeExpectations = SelfIndexedModeExpectations
organizer_wave0_mode_expectations = self_indexed_mode_expectations


__all__ = [
    "SelfIndexedModeExpectations",
    "self_indexed_mode_expectations",
    "OrganizerWave0ModeExpectations",
    "organizer_wave0_mode_expectations",
]
