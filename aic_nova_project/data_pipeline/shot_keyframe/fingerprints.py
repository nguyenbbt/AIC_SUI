"""Deterministic fingerprints used by Module 1 resume validation."""

import hashlib
import json
from pathlib import Path
from typing import Sequence


PROCESSING_CONFIG_VERSION = 1
_HASH_CHUNK_SIZE = 4 * 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_processing_config_fingerprint(
    *,
    threshold: float,
    positions: Sequence[float],
    webp_quality: int,
) -> str:
    """Hash every Module 1 setting that can affect keyframe output."""
    payload = {
        "config_version": PROCESSING_CONFIG_VERSION,
        "keyframe_format": "webp",
        "positions": [float(position) for position in positions],
        "shot_detector": "TransNetV2",
        "threshold": float(threshold),
        "webp_quality": int(webp_quality),
    }
    canonical_payload = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical_payload).hexdigest()
