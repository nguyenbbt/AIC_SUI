"""Build cleaning intervals from timestamped transcript segments."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Dict, List, Optional, Sequence, Tuple


IndexedSegment = Tuple[int, Dict[str, Any]]


class SegmentGrouper:
    """Group validated transcript segments without inventing timestamps."""

    @staticmethod
    def validate_segments(segments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return normalized segments or raise for an invalid timestamp contract."""
        return [
            segment
            for _, segment in SegmentGrouper._validate_segments(segments)
        ]

    @staticmethod
    def group_segments(
        segments: List[Dict[str, Any]],
        group_size: Optional[int] = None,
        *,
        min_duration_sec: float = 20.0,
        target_duration_sec: float = 40.0,
        max_duration_sec: float = 60.0,
    ) -> List[Dict[str, Any]]:
        """Return cleaning intervals while preserving the existing JSON schema.

        ``group_size`` remains available for callers that explicitly require the
        legacy count-based behavior. When omitted, boundaries are selected from
        the real segment timestamps and target intervals of 20--60 seconds.
        """
        indexed_segments = SegmentGrouper._validate_segments(segments)
        if not indexed_segments:
            return []

        if group_size is not None:
            if group_size <= 0:
                raise ValueError("group_size must be greater than zero")
            chunks = [
                indexed_segments[index:index + group_size]
                for index in range(0, len(indexed_segments), group_size)
            ]
        else:
            SegmentGrouper._validate_duration_settings(
                min_duration_sec,
                target_duration_sec,
                max_duration_sec,
            )
            chunks = SegmentGrouper._group_by_duration(
                indexed_segments,
                min_duration_sec=min_duration_sec,
                target_duration_sec=target_duration_sec,
                max_duration_sec=max_duration_sec,
            )

        return [
            SegmentGrouper._to_interval(interval_id, chunk)
            for interval_id, chunk in enumerate(chunks)
        ]

    @staticmethod
    def _validate_segments(segments: Sequence[Dict[str, Any]]) -> List[IndexedSegment]:
        validated: List[IndexedSegment] = []
        previous_start = -math.inf

        for segment_id, segment in enumerate(segments):
            timestamp = segment.get("timestamp")
            if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
                raise ValueError(f"segment {segment_id} has an invalid timestamp")

            start_time, end_time = timestamp
            if not SegmentGrouper._is_finite_number(start_time) or not SegmentGrouper._is_finite_number(end_time):
                raise ValueError(f"segment {segment_id} has an invalid timestamp")

            start = float(start_time)
            end = float(end_time)
            if start < 0.0 or end <= start or start < previous_start:
                raise ValueError(f"segment {segment_id} has an invalid timestamp range")

            normalized = dict(segment)
            normalized["timestamp"] = (start, end)
            validated.append((segment_id, normalized))
            previous_start = start

        return validated

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))

    @staticmethod
    def _validate_duration_settings(
        min_duration_sec: float,
        target_duration_sec: float,
        max_duration_sec: float,
    ) -> None:
        if not 0.0 < min_duration_sec <= target_duration_sec <= max_duration_sec:
            raise ValueError(
                "interval durations must satisfy 0 < min <= target <= max"
            )

    @staticmethod
    def _group_by_duration(
        segments: List[IndexedSegment],
        *,
        min_duration_sec: float,
        target_duration_sec: float,
        max_duration_sec: float,
    ) -> List[List[IndexedSegment]]:
        chunks: List[List[IndexedSegment]] = []
        current: List[IndexedSegment] = []

        for indexed_segment in segments:
            segment = indexed_segment[1]
            segment_start, segment_end = segment["timestamp"]
            if segment_end - segment_start > max_duration_sec:
                raise ValueError(
                    f"segment {indexed_segment[0]} exceeds maximum interval duration"
                )

            if current:
                proposed_duration = segment_end - current[0][1]["timestamp"][0]
                if proposed_duration > max_duration_sec:
                    chunks.append(current)
                    current = []

            current.append(indexed_segment)
            current_duration = (
                current[-1][1]["timestamp"][1]
                - current[0][1]["timestamp"][0]
            )
            if current_duration >= target_duration_sec:
                chunks.append(current)
                current = []

        if current:
            SegmentGrouper._append_tail(
                chunks,
                current,
                min_duration_sec=min_duration_sec,
                max_duration_sec=max_duration_sec,
            )

        return chunks

    @staticmethod
    def _append_tail(
        chunks: List[List[IndexedSegment]],
        tail: List[IndexedSegment],
        *,
        min_duration_sec: float,
        max_duration_sec: float,
    ) -> None:
        tail_duration = tail[-1][1]["timestamp"][1] - tail[0][1]["timestamp"][0]
        if chunks and tail_duration < min_duration_sec:
            merged_duration = (
                tail[-1][1]["timestamp"][1]
                - chunks[-1][0][1]["timestamp"][0]
            )
            if merged_duration <= max_duration_sec:
                chunks[-1].extend(tail)
                return

        chunks.append(tail)

    @staticmethod
    def _to_interval(interval_id: int, chunk: List[IndexedSegment]) -> Dict[str, Any]:
        raw_text = " ".join(
            segment.get("text", "").strip()
            for _, segment in chunk
            if isinstance(segment.get("text"), str) and segment["text"].strip()
        )
        return {
            "interval_id": str(interval_id),
            "start_time_sec": chunk[0][1]["timestamp"][0],
            "end_time_sec": chunk[-1][1]["timestamp"][1],
            "raw_text": raw_text,
            "segment_ids": [segment_id for segment_id, _ in chunk],
        }
