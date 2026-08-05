"""Canonical synthetic self-indexed-v2 records shared by Online tests.

The module path and legacy aliases are retained so downstream imports do not
break during the contract migration.
"""

from __future__ import annotations

from online.ports.records import FrameMetadata


SELF_INDEXED_FIXTURE_SCHEMA_VERSION = "self-indexed-v2"

SELF_INDEXED_FRAME_METADATA: tuple[FrameMetadata, ...] = (
    FrameMetadata(
        frame_id="L21_V001_00000_015",
        video_id="L21_V001",
        shot_id=0,
        timestamp_sec=0.0,
        source_frame_idx=0,
        image_rel_path="keyframes/L21_V001/shot_00000_pos_015.webp",
    ),
    FrameMetadata(
        frame_id="L21_V001_00000_050",
        video_id="L21_V001",
        shot_id=0,
        timestamp_sec=1.0 / 30.0,
        source_frame_idx=0,
        image_rel_path="keyframes/L21_V001/shot_00000_pos_050.webp",
    ),
    FrameMetadata(
        frame_id="L21_V001_00001_085",
        video_id="L21_V001",
        shot_id=1,
        timestamp_sec=2.0,
        source_frame_idx=60,
        image_rel_path="keyframes/L21_V001/shot_00001_pos_085.webp",
    ),
    FrameMetadata(
        frame_id="L21_V002_00000_050",
        video_id="L21_V002",
        shot_id=0,
        timestamp_sec=0.0,
        source_frame_idx=0,
        image_rel_path="keyframes/L21_V002/shot_00000_pos_050.webp",
    ),
)

# Compatibility aliases; new code should use the SELF_INDEXED names.
ORGANIZER_FIXTURE_SCHEMA_VERSION = SELF_INDEXED_FIXTURE_SCHEMA_VERSION
ORGANIZER_FRAME_METADATA = SELF_INDEXED_FRAME_METADATA


def build_self_indexed_frame_metadata() -> tuple[FrameMetadata, ...]:
    return SELF_INDEXED_FRAME_METADATA


def build_organizer_frame_metadata() -> tuple[FrameMetadata, ...]:
    """Deprecated compatibility wrapper for pre-migration test imports."""

    return build_self_indexed_frame_metadata()
