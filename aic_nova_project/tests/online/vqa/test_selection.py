from __future__ import annotations

import pytest

from online.domain.candidates import CandidateDiagnostics, FusedFrameCandidate
from online.domain.enums import RetrievalBranch
from online.ports.records import FrameMetadata
from online.vqa.budget import EvidenceBudgetPolicy
from online.vqa.selection import (
    _TextEvidenceChunk,
    apply_text_budget,
    filter_asr_chunks_for_windows,
    select_neighbor_frames,
    select_primary_frames,
)


def frame(frame_id: str, video_id: str, score: float, timestamp: float = 10.0) -> FusedFrameCandidate:
    return FusedFrameCandidate(
        frame_id=frame_id,
        video_id=video_id,
        shot_id=0,
        timestamp_sec=timestamp,
        final_score=score,
        branch_scores={RetrievalBranch.VISUAL_DENSE: min(score, 1.0)},
        evidence=(),
        diagnostics=CandidateDiagnostics(),
    )


def metadata(frame_id: str, video_id: str, timestamp: float) -> FrameMetadata:
    return FrameMetadata(frame_id=frame_id, video_id=video_id, shot_id=0, timestamp_sec=timestamp)


def chunk(stable_id: str, kind: str, text: str, *, video: str = "V1", rank: int = 0, order: int = 0, start: float | None = None, end: float | None = None) -> _TextEvidenceChunk:
    return _TextEvidenceChunk(stable_id, kind, rank, order, text, video, start, end)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("stable_id", ""),
        ("stable_id", "   "),
        ("stable_id", 123),
        ("video_id", ""),
        ("video_id", "   "),
        ("video_id", 123),
        ("text", ""),
        ("text", "   "),
        ("text", 123),
        ("source_rank", True),
        ("source_rank", 0.5),
        ("source_rank", float("nan")),
        ("source_rank", float("inf")),
        ("source_order", False),
        ("source_order", 0.5),
        ("source_order", float("nan")),
        ("source_order", float("inf")),
    ),
)
def test_text_chunk_rejects_invalid_identity_text_and_ordering_fields(field: str, value: object) -> None:
    values: dict[str, object] = {
        "stable_id": "chunk-1",
        "evidence_type": "ocr",
        "source_rank": 0,
        "source_order": 0,
        "text": "text",
        "video_id": "V1",
    }
    values[field] = value

    with pytest.raises(ValueError):
        _TextEvidenceChunk(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("start", "end"),
    (
        (True, 1),
        (0, False),
        ("0", 1),
        (0, "1"),
        (float("nan"), 1),
        (0, float("nan")),
        (float("inf"), 1),
        (0, float("inf")),
        (-0.1, 1),
        (0, -0.1),
        (2, 1),
    ),
)
def test_text_chunk_rejects_invalid_asr_times(start: object, end: object) -> None:
    with pytest.raises(ValueError):
        _TextEvidenceChunk("chunk-1", "asr", 0, 0, "text", "V1", start, end)  # type: ignore[arg-type]


def test_primary_selection_guarantees_diversity_then_fills_caps_and_deduplicates() -> None:
    candidates = [frame(f"V1_{i}", "V1", 20 - i) for i in range(5)]
    candidates += [frame(f"V2_{i}", "V2", 10 - i) for i in range(4)]
    candidates += [frame(f"V3_{i}", "V3", 5 - i) for i in range(3)]
    candidates += [frame("V4_0", "V4", 4), frame("V1_0", "V1", 1)]

    selected = select_primary_frames(tuple(reversed(candidates)))
    assert [item.video_id for item in selected[:3]] == ["V1", "V2", "V3"]
    assert len(selected) == 8
    assert len({item.frame_id for item in selected}) == 8
    assert all(sum(item.video_id == video for item in selected) <= 3 for video in {"V1", "V2", "V3"})
    assert "V4" not in {item.video_id for item in selected}


def test_primary_equal_score_uses_frame_id_and_output_is_deterministic() -> None:
    values = (frame("V2_B", "V2", 1), frame("V1_A", "V1", 1), frame("V1_C", "V1", 1))
    expected = ("V1_A", "V2_B", "V1_C")
    assert tuple(item.frame_id for item in select_primary_frames(values)) == expected
    assert tuple(item.frame_id for item in select_primary_frames(tuple(reversed(values)))) == expected


def test_neighbors_handle_boundaries_dedup_and_total_image_cap() -> None:
    sequence = tuple(metadata(f"V1_{i}", "V1", float(i)) for i in range(14))
    primaries = tuple(frame(f"V1_{i}", "V1", 20 - i, float(i)) for i in range(8))
    neighbors = select_neighbor_frames(primaries, {"V1": sequence})
    assert tuple(item.frame_id for item in neighbors) == ("V1_8",)
    assert len(primaries) + len(neighbors) <= 12

    boundary = select_neighbor_frames((frame("V1_0", "V1", 1, 0),), {"V1": sequence})
    assert tuple(item.frame_id for item in boundary) == ("V1_1",)


def test_asr_overlap_is_inclusive_at_plus_minus_five_seconds() -> None:
    primary = (frame("V1_F", "V1", 1, 10),)
    chunks = (
        chunk("before", "asr", "x", start=0, end=4.99),
        chunk("left", "asr", "x", start=0, end=5),
        chunk("right", "asr", "x", start=15, end=16),
        chunk("after", "asr", "x", start=15.01, end=16),
        chunk("other", "asr", "x", video="V2", start=8, end=12),
    )
    assert tuple(item.stable_id for item in filter_asr_chunks_for_windows(chunks, primary)) == ("left", "right")


def test_text_caps_summary_only_video_dedup_and_combined_cap() -> None:
    policy = EvidenceBudgetPolicy(ocr_chars=4, asr_chars=5, summary_chars_per_video=3, summary_chars_total=5, text_chars_total=10)
    chunks = (
        chunk("summary-v2", "summary", "ignored", video="V2", rank=0),
        chunk("ocr", "ocr", "abcdef", rank=1),
        chunk("ocr", "ocr", "duplicate", rank=2),
        chunk("asr", "asr", "123456", rank=3, start=0, end=1),
        chunk("summary", "summary", "WXYZ", rank=4),
    )
    selected = apply_text_budget(chunks, ("V1",), policy)
    assert [(item.stable_id, item.text) for item in selected] == [("ocr", "abcd"), ("asr", "12345"), ("summary", "W")]
    assert sum(len(item.text) for item in selected) == 10


def test_text_output_order_is_deterministic() -> None:
    chunks = (
        chunk("c", "ocr", "c", rank=1, order=1),
        chunk("a", "ocr", "a", rank=0, order=0),
        chunk("b", "summary", "b", rank=1, order=0),
    )
    expected = ("a", "b", "c")
    assert tuple(item.stable_id for item in apply_text_budget(chunks, ("V1",))) == expected
    assert tuple(item.stable_id for item in apply_text_budget(tuple(reversed(chunks)), ("V1",))) == expected
