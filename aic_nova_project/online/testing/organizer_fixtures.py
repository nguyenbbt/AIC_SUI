"""Canonical synthetic organizer-v1 records shared by Online tests."""

from __future__ import annotations

from online.ports.records import FrameMetadata


ORGANIZER_FIXTURE_SCHEMA_VERSION = "organizer-v1"

ORGANIZER_FRAME_METADATA: tuple[FrameMetadata, ...] = (
    FrameMetadata(
        frame_id="L21_V001_001",
        video_id="L21_V001",
        keyframe_no=1,
        local_index=0,
        timestamp_sec=0.0,
        fps=30.0,
        source_frame_idx=0,
        image_rel_path="keyframes/L21_V001/001.jpg",
    ),
    FrameMetadata(
        frame_id="L21_V001_002",
        video_id="L21_V001",
        keyframe_no=2,
        local_index=1,
        timestamp_sec=1.0 / 30.0,
        fps=30.0,
        source_frame_idx=0,
        image_rel_path="keyframes/L21_V001/002.jpg",
    ),
    FrameMetadata(
        frame_id="L21_V001_003",
        video_id="L21_V001",
        keyframe_no=3,
        local_index=2,
        timestamp_sec=2.0,
        fps=30.0,
        source_frame_idx=60,
        image_rel_path="keyframes/L21_V001/003.jpg",
    ),
    FrameMetadata(
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


def build_organizer_frame_metadata() -> tuple[FrameMetadata, ...]:
    """Return the immutable canonical A0 fixture records."""

    return ORGANIZER_FRAME_METADATA
