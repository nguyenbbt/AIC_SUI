"""Canonical identifier parsing and validation shared by Online infrastructure."""

from __future__ import annotations

import re
from typing import NamedTuple

from .errors import ContractMismatchError


_FRAME_ID_PATTERN = re.compile(
    r"^(?P<video_id>L[0-9]+_V[0-9]+)_(?P<keyframe_no>[0-9]{3})$"
)


class CanonicalFrameId(NamedTuple):
    video_id: str
    keyframe_no: int


def parse_canonical_frame_id(frame_id: str) -> CanonicalFrameId:
    """Parse an organizer-v1 ``{video_id}_{keyframe_no:03d}`` identifier.

    Organizer video IDs contain an underscore (for example ``L21_V001``), so
    the three-digit keyframe suffix is parsed from the right. Legacy shot IDs
    and local image stems are deliberately rejected and never rewritten.
    """

    if not isinstance(frame_id, str) or not frame_id.strip():
        raise ContractMismatchError("frame_id must not be empty or whitespace")
    if frame_id != frame_id.strip():
        raise ContractMismatchError("frame_id must not contain surrounding whitespace")
    match = _FRAME_ID_PATTERN.fullmatch(frame_id)
    if (
        match is None
        or not match.group("video_id").strip()
        or match.group("video_id") != match.group("video_id").strip()
    ):
        raise ContractMismatchError(
            "frame_id is not canonical",
            details={"expected_format": "L<batch>_V<video-number>_<keyframe_no:03d>"},
        )
    keyframe_no = int(match.group("keyframe_no"))
    if keyframe_no < 1:
        raise ContractMismatchError(
            "frame_id keyframe suffix must be at least 001",
            details={"field": "keyframe_no"},
        )
    return CanonicalFrameId(
        video_id=match.group("video_id"),
        keyframe_no=keyframe_no,
    )


def validate_canonical_frame_id(
    frame_id: str,
    *,
    video_id: str | None = None,
    keyframe_no: int | None = None,
) -> CanonicalFrameId:
    """Validate canonical syntax and any semantic fields supplied by a backend."""

    parsed = parse_canonical_frame_id(frame_id)
    if video_id is not None and parsed.video_id != video_id:
        raise ContractMismatchError(
            "frame_id video suffix does not match video_id",
            details={"field": "video_id"},
        )
    if keyframe_no is not None and parsed.keyframe_no != keyframe_no:
        raise ContractMismatchError(
            "frame_id keyframe suffix does not match keyframe_no",
            details={"field": "keyframe_no"},
        )
    return parsed
