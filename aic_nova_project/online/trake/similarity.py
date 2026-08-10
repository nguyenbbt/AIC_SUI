"""Validated event encoding and per-video OpenCLIP cosine similarity."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from online.domain.errors import (
    ContractMismatchError,
    DataInfrastructureError,
    DimensionMismatchError,
    ResourceUnavailableError,
)
from online.domain.identifiers import validate_canonical_frame_id
from online.domain.trake import TRAKEQuery
from online.ports.encoders import TextEncoderPort
from online.ports.visual_corpus import (
    OrderedVisualFrame,
    VisualCorpusPort,
    validate_ordered_visual_stream,
)


DEFAULT_NORM_TOLERANCE = 1e-4


@dataclass(frozen=True, slots=True)
class EncodedTRAKEEvents:
    """Ordered event IDs and validated unit vectors in OpenCLIP space."""

    event_ids: tuple[str, ...]
    vectors: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        event_ids = tuple(self.event_ids)
        vectors = tuple(tuple(row) for row in self.vectors)
        if len(event_ids) < 2 or len(vectors) != len(event_ids):
            raise ContractMismatchError(
                "Encoded TRAKE rows must match at least two ordered event IDs"
            )
        if any(not isinstance(event_id, str) or not event_id.strip() for event_id in event_ids):
            raise ContractMismatchError("Encoded TRAKE event IDs must be non-empty")
        if len(set(event_ids)) != len(event_ids):
            raise ContractMismatchError("Encoded TRAKE event IDs must be unique")
        dimension = len(vectors[0]) if vectors else 0
        if dimension < 1:
            raise DimensionMismatchError("Encoded TRAKE vectors must be non-empty")
        normalized = tuple(
            _validated_unit_vector(
                row,
                expected_dimension=dimension,
                label=f"event_vector[{index}]",
            )
            for index, row in enumerate(vectors)
        )
        object.__setattr__(self, "event_ids", event_ids)
        object.__setattr__(self, "vectors", normalized)

    @property
    def dimension(self) -> int:
        return len(self.vectors[0])


@dataclass(frozen=True, slots=True)
class VideoSimilarityMatrix:
    """One video's full ordered frames and event-by-frame cosine matrix."""

    video_id: str
    event_ids: tuple[str, ...]
    frames: tuple[OrderedVisualFrame, ...]
    similarities: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ContractMismatchError("video_id must be non-empty")
        event_ids = tuple(self.event_ids)
        frames = tuple(self.frames)
        similarities = tuple(tuple(row) for row in self.similarities)
        if len(event_ids) < 2 or len(similarities) != len(event_ids):
            raise ContractMismatchError(
                "Similarity rows must match at least two ordered event IDs"
            )
        if any(frame.video_id != self.video_id for frame in frames):
            raise ContractMismatchError("Similarity frames contain the wrong video_id")
        if any(frame.local_index != index for index, frame in enumerate(frames)):
            raise ContractMismatchError(
                "Similarity frames must use contiguous local indices from zero"
            )
        for row in similarities:
            if len(row) != len(frames):
                raise DimensionMismatchError(
                    "Similarity matrix column count must match frame count"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not -1.0 <= float(value) <= 1.0
                for value in row
            ):
                raise ContractMismatchError(
                    "Similarity matrix values must be finite cosine scores"
                )
        object.__setattr__(self, "event_ids", event_ids)
        object.__setattr__(self, "frames", frames)
        object.__setattr__(
            self,
            "similarities",
            tuple(tuple(float(value) for value in row) for row in similarities),
        )


def encode_trake_events(
    query: TRAKEQuery,
    encoder: TextEncoderPort,
) -> EncodedTRAKEEvents:
    """Encode all ordered events once and reject malformed encoder output."""

    if not isinstance(query, TRAKEQuery):
        raise ContractMismatchError("query must be a validated TRAKEQuery")
    if not isinstance(encoder, TextEncoderPort):
        raise TypeError("encoder must implement TextEncoderPort")
    dimension = encoder.dimension
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise DimensionMismatchError(
            "TRAKE encoder reported an invalid dimension",
            details={"actual": dimension},
        )
    raw_vectors = encoder.encode_texts(tuple(event.text for event in query.events))
    if isinstance(raw_vectors, (str, bytes)):
        raise ContractMismatchError("TRAKE encoder returned a non-matrix output")
    try:
        rows = tuple(raw_vectors)
    except TypeError as exc:
        raise ContractMismatchError("TRAKE encoder returned a non-matrix output") from exc
    if len(rows) != len(query.events):
        raise ContractMismatchError(
            "TRAKE encoder row count does not match event count",
            details={"expected": len(query.events), "actual": len(rows)},
        )
    normalized_rows = tuple(
        _validated_unit_vector(
            row,
            expected_dimension=dimension,
            label=f"event_vector[{index}]",
        )
        for index, row in enumerate(rows)
    )
    return EncodedTRAKEEvents(
        event_ids=tuple(event.event_id for event in query.events),
        vectors=normalized_rows,
    )


def load_video_similarity(
    corpus: VisualCorpusPort,
    video_id: str,
    encoded_events: EncodedTRAKEEvents,
    *,
    batch_size: int,
) -> VideoSimilarityMatrix:
    """Read and validate every ordered frame for one video, then score it."""

    if not isinstance(corpus, VisualCorpusPort):
        raise TypeError("corpus must implement VisualCorpusPort")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    try:
        batches = corpus.iter_ordered_frame_embedding_batches(video_id, batch_size)
        frames = validate_ordered_visual_stream(video_id, batches)
    except DataInfrastructureError:
        raise
    except (TypeError, ValueError) as exc:
        raise ContractMismatchError(
            "Ordered visual corpus stream is invalid",
            details={"video_id": video_id},
        ) from exc
    except Exception as exc:
        raise ResourceUnavailableError(
            "Ordered visual corpus could not be read",
            details={"video_id": video_id, "cause_type": type(exc).__name__},
        ) from exc
    return compute_video_similarity(encoded_events, video_id, frames)


def compute_video_similarity(
    encoded_events: EncodedTRAKEEvents,
    video_id: str,
    frames: Sequence[OrderedVisualFrame],
) -> VideoSimilarityMatrix:
    """Compute exact cosine rows without thresholding, prefiltering or clamping."""

    if not isinstance(encoded_events, EncodedTRAKEEvents):
        raise TypeError("encoded_events must be EncodedTRAKEEvents")
    if isinstance(frames, (str, bytes)):
        raise TypeError("frames must be a sequence of OrderedVisualFrame values")
    try:
        frame_values = tuple(frames)
    except TypeError as exc:
        raise TypeError("frames must be a sequence of OrderedVisualFrame values") from exc

    validated_frame_vectors: list[tuple[float, ...]] = []
    for index, frame in enumerate(frame_values):
        if not isinstance(frame, OrderedVisualFrame):
            raise ContractMismatchError(
                "Similarity input contains a non-OrderedVisualFrame value",
                details={"index": index},
            )
        if frame.video_id != video_id or frame.local_index != index:
            raise ContractMismatchError(
                "Similarity frames must match video_id and contiguous local order",
                details={"index": index, "video_id": video_id},
            )
        validate_canonical_frame_id(
            frame.frame_id,
            video_id=frame.video_id,
            shot_id=frame.shot_id,
        )
        validated_frame_vectors.append(
            _validated_unit_vector(
                frame.vector,
                expected_dimension=encoded_events.dimension,
                label=f"frame_vector[{index}]",
            )
        )

    similarities = tuple(
        tuple(_cosine(event_vector, frame_vector) for frame_vector in validated_frame_vectors)
        for event_vector in encoded_events.vectors
    )
    return VideoSimilarityMatrix(
        video_id=video_id,
        event_ids=encoded_events.event_ids,
        frames=frame_values,
        similarities=similarities,
    )


def _validated_unit_vector(
    vector: Sequence[float],
    *,
    expected_dimension: int,
    label: str,
) -> tuple[float, ...]:
    if isinstance(vector, (str, bytes)):
        raise ContractMismatchError(f"{label} must be a numeric sequence")
    try:
        raw_values = tuple(vector)
    except TypeError as exc:
        raise ContractMismatchError(f"{label} must be a numeric sequence") from exc
    if len(raw_values) != expected_dimension:
        raise DimensionMismatchError(
            f"{label} dimension does not match OpenCLIP space",
            details={"expected": expected_dimension, "actual": len(raw_values)},
        )
    if any(isinstance(value, bool) for value in raw_values):
        raise ContractMismatchError(f"{label} contains a boolean value")
    try:
        values = tuple(float(value) for value in raw_values)
    except (TypeError, ValueError) as exc:
        raise ContractMismatchError(f"{label} must contain numeric values") from exc
    if not all(math.isfinite(value) for value in values):
        raise ContractMismatchError(f"{label} must contain finite values")
    norm = math.sqrt(math.fsum(value * value for value in values))
    if not math.isfinite(norm) or not math.isclose(
        norm,
        1.0,
        rel_tol=DEFAULT_NORM_TOLERANCE,
        abs_tol=DEFAULT_NORM_TOLERANCE,
    ):
        raise ContractMismatchError(f"{label} must be L2-normalized")
    return values


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    score = math.fsum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
    if not math.isfinite(score):
        raise ContractMismatchError("Cosine similarity produced a non-finite score")
    if score < -1.0 or score > 1.0:
        if score < -1.0 - 1e-12 or score > 1.0 + 1e-12:
            raise ContractMismatchError("Cosine similarity is outside [-1, 1]")
        score = min(1.0, max(-1.0, score))
    return score


__all__ = [
    "DEFAULT_NORM_TOLERANCE",
    "EncodedTRAKEEvents",
    "VideoSimilarityMatrix",
    "compute_video_similarity",
    "encode_trake_events",
    "load_video_similarity",
]
