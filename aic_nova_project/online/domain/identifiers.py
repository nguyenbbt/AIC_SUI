"""Canonical identifier parsing and validation shared by Online infrastructure."""

from __future__ import annotations

import re
from typing import NamedTuple

from .errors import ContractMismatchError


_FRAME_ID_PATTERN = re.compile(
    r"^(?P<video_id>.+)_(?P<shot_id>[0-9]{5})_(?P<position>[0-9]{3})$"
)


class CanonicalFrameId(NamedTuple):
    video_id: str
    shot_id: int
    position: int


def parse_canonical_frame_id(frame_id: str) -> CanonicalFrameId:
    """Parse ``{video_id}_{shot_id:05d}_{position:03d}`` from the right.

    ``video_id`` may itself contain underscores. Local image stems such as
    ``shot_00000_pos_015`` are deliberately rejected and never rewritten.
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
            details={"expected_format": "{video_id}_{shot_id:05d}_{position:03d}"},
        )
    return CanonicalFrameId(
        video_id=match.group("video_id"),
        shot_id=int(match.group("shot_id")),
        position=int(match.group("position")),
    )


def validate_canonical_frame_id(
    frame_id: str,
    *,
    video_id: str | None = None,
    shot_id: int | None = None,
) -> CanonicalFrameId:
    """Validate canonical syntax and any semantic fields supplied by a backend."""

    parsed = parse_canonical_frame_id(frame_id)
    if video_id is not None and parsed.video_id != video_id:
        raise ContractMismatchError(
            "frame_id video suffix does not match video_id",
            details={"field": "video_id"},
        )
    if shot_id is not None and parsed.shot_id != shot_id:
        raise ContractMismatchError(
            "frame_id shot suffix does not match shot_id",
            details={"field": "shot_id"},
        )
    return parsed
