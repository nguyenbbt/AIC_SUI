"""Pure, deterministic evidence selection for VQA Wave 1."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from numbers import Real
from typing import Literal

from online.domain.candidates import FusedFrameCandidate
from online.ports.records import FrameMetadata

from .budget import EvidenceBudgetPolicy


TextEvidenceType = Literal["ocr", "asr", "summary"]


@dataclass(frozen=True, slots=True)
class _TextEvidenceChunk:
    stable_id: str
    evidence_type: TextEvidenceType
    source_rank: int
    source_order: int
    text: str
    video_id: str
    start_time_sec: float | None = None
    end_time_sec: float | None = None

    def __post_init__(self) -> None:
        for field_name in ("stable_id", "video_id", "text"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.evidence_type not in {"ocr", "asr", "summary"}:
            raise ValueError("unsupported evidence_type")
        if not isinstance(self.source_rank, int) or isinstance(self.source_rank, bool) or self.source_rank < 0:
            raise ValueError("source_rank must be a non-negative integer")
        if not isinstance(self.source_order, int) or isinstance(self.source_order, bool) or self.source_order < 0:
            raise ValueError("source_order must be a non-negative integer")
        if self.evidence_type == "asr":
            if self.start_time_sec is None or self.end_time_sec is None:
                raise ValueError("ASR chunks require start and end times")
            if any(
                isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0
                for value in (self.start_time_sec, self.end_time_sec)
            ):
                raise ValueError("ASR start and end times must be finite non-negative numbers")
            if self.end_time_sec < self.start_time_sec:
                raise ValueError("ASR end_time_sec must be >= start_time_sec")


def select_primary_frames(
    candidates: Sequence[FusedFrameCandidate],
    policy: EvidenceBudgetPolicy = EvidenceBudgetPolicy(),
) -> tuple[FusedFrameCandidate, ...]:
    """Select diverse primary frame evidence under DD-030 caps."""

    if any(not isinstance(candidate, FusedFrameCandidate) for candidate in candidates):
        raise TypeError("candidates must contain FusedFrameCandidate objects")

    deduplicated: dict[str, FusedFrameCandidate] = {}
    for candidate in candidates:
        current = deduplicated.get(candidate.frame_id)
        if current is None or _frame_sort_key(candidate) < _frame_sort_key(current):
            deduplicated[candidate.frame_id] = candidate
    ranked = sorted(deduplicated.values(), key=_frame_sort_key)

    selected_videos: list[str] = []
    for candidate in ranked:
        if candidate.video_id not in selected_videos:
            selected_videos.append(candidate.video_id)
            if len(selected_videos) == policy.max_videos:
                break

    selected: list[FusedFrameCandidate] = []
    selected_ids: set[str] = set()
    per_video: defaultdict[str, int] = defaultdict(int)

    for video_id in selected_videos:
        primary = next(candidate for candidate in ranked if candidate.video_id == video_id)
        selected.append(primary)
        selected_ids.add(primary.frame_id)
        per_video[video_id] += 1

    for candidate in ranked:
        if len(selected) >= policy.max_primary_total:
            break
        if candidate.frame_id in selected_ids or candidate.video_id not in selected_videos:
            continue
        if per_video[candidate.video_id] >= policy.max_primary_per_video:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.frame_id)
        per_video[candidate.video_id] += 1

    return tuple(selected)


def select_neighbor_frames(
    primary_frames: Sequence[FusedFrameCandidate],
    ordered_frames_by_video: Mapping[str, Sequence[FrameMetadata]],
    policy: EvidenceBudgetPolicy = EvidenceBudgetPolicy(),
) -> tuple[FrameMetadata, ...]:
    """Return local previous/next neighbors; the image cap includes primaries."""

    if len(primary_frames) > policy.max_images_total:
        raise ValueError("primary frame count exceeds max_images_total")
    remaining = policy.max_images_total - len({frame.frame_id for frame in primary_frames})
    primary_ids = {frame.frame_id for frame in primary_frames}
    selected_ids = set(primary_ids)
    neighbors: list[FrameMetadata] = []

    indexed: dict[str, tuple[tuple[FrameMetadata, ...], dict[str, int]]] = {}
    for video_id, sequence in ordered_frames_by_video.items():
        frames = tuple(sequence)
        if any(frame.video_id != video_id for frame in frames):
            raise ValueError("ordered frame sequence contains a different video_id")
        ids = [frame.frame_id for frame in frames]
        if len(ids) != len(set(ids)):
            raise ValueError("ordered frame sequence contains duplicate frame_id")
        indexed[video_id] = (frames, {frame_id: index for index, frame_id in enumerate(ids)})

    for primary in primary_frames:
        sequence_and_index = indexed.get(primary.video_id)
        if sequence_and_index is None:
            continue
        sequence, positions = sequence_and_index
        position = positions.get(primary.frame_id)
        if position is None:
            continue
        for neighbor_position in (position - 1, position + 1):
            if remaining == 0:
                return tuple(neighbors)
            if not 0 <= neighbor_position < len(sequence):
                continue
            neighbor = sequence[neighbor_position]
            if neighbor.frame_id in selected_ids:
                continue
            neighbors.append(neighbor)
            selected_ids.add(neighbor.frame_id)
            remaining -= 1
    return tuple(neighbors)


def filter_asr_chunks_for_windows(
    chunks: Sequence[_TextEvidenceChunk],
    primary_frames: Sequence[FusedFrameCandidate],
    policy: EvidenceBudgetPolicy = EvidenceBudgetPolicy(),
) -> tuple[_TextEvidenceChunk, ...]:
    """Keep ASR intervals overlapping any selected-frame ±window interval."""

    timestamps: defaultdict[str, list[float]] = defaultdict(list)
    for frame in primary_frames:
        timestamps[frame.video_id].append(frame.timestamp_sec)

    output: list[_TextEvidenceChunk] = []
    for chunk in _ordered_unique_chunks(chunks):
        if chunk.evidence_type != "asr":
            continue
        assert chunk.start_time_sec is not None and chunk.end_time_sec is not None
        if any(
            chunk.end_time_sec >= timestamp - policy.asr_window_seconds
            and chunk.start_time_sec <= timestamp + policy.asr_window_seconds
            for timestamp in timestamps.get(chunk.video_id, ())
        ):
            output.append(chunk)
    return tuple(output)


def apply_text_budget(
    chunks: Sequence[_TextEvidenceChunk],
    selected_video_ids: Sequence[str],
    policy: EvidenceBudgetPolicy = EvidenceBudgetPolicy(),
) -> tuple[_TextEvidenceChunk, ...]:
    """Deterministically truncate text evidence under individual and total caps."""

    allowed_videos = set(selected_video_ids)
    used_by_type = {"ocr": 0, "asr": 0, "summary": 0}
    summary_by_video: defaultdict[str, int] = defaultdict(int)
    used_total = 0
    output: list[_TextEvidenceChunk] = []
    type_caps = {
        "ocr": policy.ocr_chars,
        "asr": policy.asr_chars,
        "summary": policy.summary_chars_total,
    }

    for chunk in _ordered_unique_chunks(chunks):
        if chunk.video_id not in allowed_videos or not chunk.text:
            continue
        remaining = min(type_caps[chunk.evidence_type] - used_by_type[chunk.evidence_type], policy.text_chars_total - used_total)
        if chunk.evidence_type == "summary":
            remaining = min(remaining, policy.summary_chars_per_video - summary_by_video[chunk.video_id])
        if remaining <= 0:
            continue
        text = chunk.text[:remaining]
        if not text:
            continue
        output.append(chunk if text == chunk.text else replace(chunk, text=text))
        size = len(text)
        used_by_type[chunk.evidence_type] += size
        used_total += size
        if chunk.evidence_type == "summary":
            summary_by_video[chunk.video_id] += size
        if used_total == policy.text_chars_total:
            break
    return tuple(output)


def _frame_sort_key(candidate: FusedFrameCandidate) -> tuple[float, str]:
    return (-candidate.final_score, candidate.frame_id)


def _ordered_unique_chunks(chunks: Sequence[_TextEvidenceChunk]) -> tuple[_TextEvidenceChunk, ...]:
    if any(not isinstance(chunk, _TextEvidenceChunk) for chunk in chunks):
        raise TypeError("chunks must contain _TextEvidenceChunk objects")
    ordered = sorted(chunks, key=lambda chunk: (chunk.source_rank, chunk.source_order, chunk.stable_id))
    seen: set[str] = set()
    output: list[_TextEvidenceChunk] = []
    for chunk in ordered:
        if chunk.stable_id not in seen:
            output.append(chunk)
            seen.add(chunk.stable_id)
    return tuple(output)
