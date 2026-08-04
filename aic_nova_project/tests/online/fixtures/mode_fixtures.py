"""C-owned Wave 0 expectations for organizer-v1 mode behavior.

This module intentionally contains test vectors only.  It does not define a
replacement for A-owned domain models and it does not import database adapters.
Once A0 lands, C-side consumer tests materialize these payloads with the shared
types.
"""

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
        query_text="một người nâng chiếc cốc",
        source_candidate_id=frame_id,
        source_start_time_sec=None,
        source_end_time_sec=None,
        source_normalized_score=None,
    )


def _candidate(
    metadata: Payload,
    *,
    rank: int,
    raw_score: float,
) -> Payload:
    return _payload(
        frame_id=metadata["frame_id"],
        video_id=metadata["video_id"],
        keyframe_no=metadata["keyframe_no"],
        local_index=metadata["local_index"],
        timestamp_sec=metadata["timestamp_sec"],
        source_frame_idx=metadata["source_frame_idx"],
        rank=rank,
        raw_score=raw_score,
        normalized_score=None,
        provenance=_provenance(str(metadata["frame_id"])),
    )


def _fused(candidate: Payload, *, final_score: float) -> Payload:
    return _payload(
        frame_id=candidate["frame_id"],
        video_id=candidate["video_id"],
        keyframe_no=candidate["keyframe_no"],
        local_index=candidate["local_index"],
        timestamp_sec=candidate["timestamp_sec"],
        source_frame_idx=candidate["source_frame_idx"],
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
class OrganizerWave0ModeExpectations:
    """Inputs and expected C-layer outcomes before ranking implementation."""

    frame_metadata_payloads: tuple[Payload, ...]
    frame_candidate_payloads: tuple[Payload, ...]
    expected_fused_payloads: tuple[Payload, ...]
    asr_interval_payload: Payload
    expected_asr_mapped_frame_ids: tuple[str, ...]
    expected_kis_frame_ids_after_dedup: tuple[str, ...]
    expected_kis_competition_rows: tuple[tuple[str, int], ...]
    expected_trake_competition_row: tuple[str, tuple[int, ...]]
    expected_vqa_competition_row: tuple[str, int, str]


def organizer_wave0_mode_expectations() -> OrganizerWave0ModeExpectations:
    """Return deterministic C-side cases for A0, C0-C2, C5, C6 and C10."""

    metadata = (
        _payload(
            frame_id="L21_V001_001",
            video_id="L21_V001",
            keyframe_no=1,
            local_index=0,
            timestamp_sec=0.0,
            fps=30.0,
            source_frame_idx=0,
            image_rel_path="keyframes/L21_V001/001.jpg",
        ),
        _payload(
            frame_id="L21_V001_002",
            video_id="L21_V001",
            keyframe_no=2,
            local_index=1,
            timestamp_sec=0.0333333,
            fps=30.0,
            source_frame_idx=0,
            image_rel_path="keyframes/L21_V001/002.jpg",
        ),
        _payload(
            frame_id="L21_V001_003",
            video_id="L21_V001",
            keyframe_no=3,
            local_index=2,
            timestamp_sec=1.0,
            fps=30.0,
            source_frame_idx=30,
            image_rel_path="keyframes/L21_V001/003.jpg",
        ),
        _payload(
            frame_id="L21_V002_001",
            video_id="L21_V002",
            keyframe_no=1,
            local_index=0,
            timestamp_sec=0.0,
            fps=29.97,
            source_frame_idx=0,
            image_rel_path="keyframes/L21_V002/001.jpg",
        ),
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
    return OrganizerWave0ModeExpectations(
        frame_metadata_payloads=metadata,
        frame_candidate_payloads=candidates,
        expected_fused_payloads=fused,
        asr_interval_payload=_payload(
            video_id="L21_V001",
            interval_id="interval-0001",
            start_time_sec=0.0,
            end_time_sec=0.05,
        ),
        expected_asr_mapped_frame_ids=("L21_V001_001", "L21_V001_002"),
        expected_kis_frame_ids_after_dedup=(
            "L21_V001_002",
            "L21_V002_001",
            "L21_V001_003",
        ),
        expected_kis_competition_rows=(
            ("L21_V001", 0),
            ("L21_V002", 0),
            ("L21_V001", 30),
        ),
        expected_trake_competition_row=("L21_V001", (0, 30)),
        expected_vqa_competition_row=(
            "L21_V001",
            0,
            "Người đàn ông nâng một chiếc cốc.",
        ),
    )


__all__ = [
    "OrganizerWave0ModeExpectations",
    "organizer_wave0_mode_expectations",
]
