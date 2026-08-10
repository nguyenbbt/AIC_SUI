"""Canonical identifier parsing for the self-indexed Offline corpus."""

from __future__ import annotations

import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import NamedTuple
from urllib.parse import urlsplit

from .errors import ContractMismatchError


_FRAME_ID_PATTERN = re.compile(
    r"^(?P<video_id>.+)_(?P<shot_id>[0-9]{5})_(?P<position_code>[0-9]{3})$"
)
_INTERVAL_ID_PATTERN = re.compile(r"^(?:0|[1-9][0-9]*)$")


class CanonicalFrameId(NamedTuple):
    video_id: str
    shot_id: int
    position_code: int


def parse_canonical_frame_id(frame_id: str) -> CanonicalFrameId:
    """Parse ``{video_id}_{shot_id:05d}_{position_code:03d}`` from the right."""

    if not isinstance(frame_id, str) or not frame_id.strip():
        raise ContractMismatchError("frame_id must not be empty or whitespace")
    if frame_id != frame_id.strip():
        raise ContractMismatchError("frame_id must not contain surrounding whitespace")
    match = _FRAME_ID_PATTERN.fullmatch(frame_id)
    if match is None or not match.group("video_id").strip():
        raise ContractMismatchError(
            "frame_id is not canonical",
            details={
                "expected_format": "{video_id}_{shot_id:05d}_{position_code:03d}"
            },
        )
    position_code = int(match.group("position_code"))
    if position_code > 100:
        raise ContractMismatchError(
            "frame_id position code must be within 000..100",
            details={"field": "position_code"},
        )
    return CanonicalFrameId(
        video_id=match.group("video_id"),
        shot_id=int(match.group("shot_id")),
        position_code=position_code,
    )


def validate_canonical_frame_id(
    frame_id: str,
    *,
    video_id: str | None = None,
    shot_id: int | None = None,
    position_code: int | None = None,
) -> CanonicalFrameId:
    """Validate canonical syntax and semantic fields supplied by a backend."""

    parsed = parse_canonical_frame_id(frame_id)
    if video_id is not None and parsed.video_id != video_id:
        raise ContractMismatchError(
            "frame_id video prefix does not match video_id",
            details={"field": "video_id"},
        )
    if shot_id is not None and parsed.shot_id != shot_id:
        raise ContractMismatchError(
            "frame_id shot suffix does not match shot_id",
            details={"field": "shot_id"},
        )
    if position_code is not None and parsed.position_code != position_code:
        raise ContractMismatchError(
            "frame_id position suffix does not match position_code",
            details={"field": "position_code"},
        )
    return parsed


def validate_interval_id(interval_id: str) -> str:
    """Validate a zero-based decimal interval identifier without padding."""

    if not isinstance(interval_id, str) or _INTERVAL_ID_PATTERN.fullmatch(interval_id) is None:
        raise ValueError("interval_id must be an unpadded non-negative decimal string")
    return interval_id


def validate_relative_artifact_path(value: str) -> str:
    """Reject absolute, parent-traversing, URL and platform-specific paths."""

    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("artifact path must be normalized non-empty text")
    if "\\" in value or "\x00" in value:
        raise ValueError("artifact path must be a safe POSIX relative path")
    parsed = urlsplit(value)
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        parsed.scheme
        or parsed.netloc
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in posix.parts
    ):
        raise ValueError("artifact path must be a safe POSIX relative path")
    return value
