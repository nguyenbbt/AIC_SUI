"""Linear-time DANTE dynamic programming over one video's similarity matrix."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .config import DANTEConfig


SimilarityMatrix = Sequence[Sequence[float]]
_UNREACHABLE = -math.inf


@dataclass(frozen=True)
class DANTEPath:
    """Algorithm-level result; domain frame hydration belongs to the TRAKE service."""

    score: float
    positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("score must be numeric")
        if not math.isfinite(float(self.score)):
            raise ValueError("score must be finite")
        try:
            normalized_positions = tuple(self.positions)
        except TypeError as error:
            raise TypeError("positions must be an iterable of integers") from error
        if not normalized_positions:
            raise ValueError("positions must not be empty")
        if any(
            isinstance(position, bool) or not isinstance(position, int) or position < 0
            for position in normalized_positions
        ):
            raise ValueError("positions must contain non-negative integers")
        if any(
            current <= previous
            for previous, current in zip(
                normalized_positions, normalized_positions[1:]
            )
        ):
            raise ValueError("positions must be strictly increasing")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "positions", normalized_positions)


def solve_dante(
    similarity_matrix: SimilarityMatrix,
    config: DANTEConfig | None = None,
) -> DANTEPath | None:
    """Return the best strictly ordered event-to-frame path for one video.

    Rows are ordered events and columns are local ordered-keyframe positions.
    The running-maximum recurrence is equivalent to Equation (2) in the
    AIO_DANTE+QUEST paper while reducing time from ``O(N*T^2)`` to ``O(N*T)``.
    ``None`` means the video cannot provide one strictly increasing frame per
    event, for example when it has fewer frames than events.
    """

    if config is None:
        policy = DANTEConfig()
    elif isinstance(config, DANTEConfig):
        policy = config
    else:
        raise TypeError("config must be a DANTEConfig or None")
    similarities = _validated_matrix(similarity_matrix)
    event_count = len(similarities)
    frame_count = len(similarities[0])
    if frame_count < event_count:
        return None

    previous_scores = list(similarities[0])
    backpointers = [[-1] * frame_count for _ in range(event_count)]
    lambda_penalty = policy.lambda_penalty

    for event_index in range(1, event_count):
        current_scores = [_UNREACHABLE] * frame_count
        running_score = _UNREACHABLE
        running_position = -1

        for frame_position in range(frame_count):
            predecessor = frame_position - 1
            if predecessor >= 0 and math.isfinite(previous_scores[predecessor]):
                predecessor_score = (
                    previous_scores[predecessor] + lambda_penalty * predecessor
                )
                if _is_better_predecessor(
                    predecessor_score,
                    predecessor,
                    running_score,
                    running_position,
                ):
                    running_score = predecessor_score
                    running_position = predecessor

            if running_position < 0:
                continue
            score = (
                similarities[event_index][frame_position]
                + running_score
                - lambda_penalty * frame_position
            )
            if not math.isfinite(score):
                raise ValueError("DANTE recurrence produced a non-finite score")
            current_scores[frame_position] = score
            backpointers[event_index][frame_position] = running_position

        previous_scores = current_scores

    final_position = _best_final_position(previous_scores)
    if final_position is None:
        return None

    positions = [final_position]
    current_position = final_position
    for event_index in range(event_count - 1, 0, -1):
        current_position = backpointers[event_index][current_position]
        if current_position < 0:
            raise RuntimeError("DANTE backpointer chain is incomplete")
        positions.append(current_position)
    positions.reverse()

    return DANTEPath(
        score=previous_scores[final_position],
        positions=tuple(positions),
    )


def _validated_matrix(matrix: SimilarityMatrix) -> tuple[tuple[float, ...], ...]:
    if isinstance(matrix, (str, bytes)) or not isinstance(matrix, Sequence):
        raise TypeError("similarity_matrix must be a sequence of rows")
    rows = tuple(matrix)
    if not rows:
        raise ValueError("similarity_matrix must contain at least one event row")

    normalized: list[tuple[float, ...]] = []
    frame_count: int | None = None
    for row in rows:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise TypeError("each similarity row must be a sequence")
        values = tuple(row)
        if not values:
            raise ValueError("similarity rows must not be empty")
        if frame_count is None:
            frame_count = len(values)
        elif len(values) != frame_count:
            raise ValueError("similarity_matrix rows must have equal length")

        normalized_row: list[float] = []
        for value in values:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError("similarity values must be finite numbers")
            similarity = float(value)
            if not -1.0 <= similarity <= 1.0:
                raise ValueError("cosine similarity values must be in [-1, 1]")
            normalized_row.append(similarity)
        normalized.append(tuple(normalized_row))
    return tuple(normalized)


def _is_better_predecessor(
    candidate_score: float,
    candidate_position: int,
    best_score: float,
    best_position: int,
) -> bool:
    return candidate_score > best_score or (
        candidate_score == best_score
        and (best_position < 0 or candidate_position < best_position)
    )


def _best_final_position(scores: Sequence[float]) -> int | None:
    best_position: int | None = None
    best_score = _UNREACHABLE
    for position, score in enumerate(scores):
        if not math.isfinite(score):
            continue
        if score > best_score:
            best_score = score
            best_position = position
    return best_position


__all__ = ["DANTEPath", "SimilarityMatrix", "solve_dante"]
