"""Read-only full ordered visual corpus boundary for TRAKE."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Annotated, Protocol, runtime_checkable

from pydantic import Field, model_validator

from online.domain.base import FiniteFloat, NonEmptyStr, StrictFrozenModel, StrictIntValue
from online.domain.errors import ContractMismatchError
from online.domain.identifiers import validate_canonical_frame_id


class OrderedVisualFrame(StrictFrozenModel):
    """One PE-Core visual vector at its local ordered position in a video."""

    frame_id: NonEmptyStr
    video_id: NonEmptyStr
    shot_id: StrictIntValue = Field(ge=0)
    local_index: StrictIntValue = Field(ge=0)
    timestamp_sec: Annotated[FiniteFloat, Field(ge=0.0)]
    vector: tuple[FiniteFloat, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_record(self) -> "OrderedVisualFrame":
        try:
            validate_canonical_frame_id(
                self.frame_id,
                video_id=self.video_id,
                shot_id=self.shot_id,
            )
        except ContractMismatchError as exc:
            raise ValueError(exc.message) from exc
        norm = math.sqrt(sum(value * value for value in self.vector))
        if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError("visual vector must be L2-normalized")
        return self


OrderedVisualBatch = Sequence[OrderedVisualFrame]


def validate_ordered_visual_batch(
    video_id: str,
    batch: OrderedVisualBatch,
    *,
    expected_dimension: int | None = None,
) -> tuple[OrderedVisualFrame, ...]:
    """Validate one ordered batch before it enters a corpus stream.

    A batch may start at any local index because the stream validator checks
    continuity across batch boundaries. Within a batch, records must already be
    strictly ordered and must belong to the requested video.
    """

    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError("video_id must not be empty or whitespace")
    if not isinstance(batch, Sequence) or not batch:
        raise ValueError("visual batches must be non-empty sequences")

    validated = tuple(batch)
    dimensions = expected_dimension
    seen_frame_ids: set[str] = set()
    previous_index: int | None = None
    for frame in validated:
        if not isinstance(frame, OrderedVisualFrame):
            raise ValueError("visual batches must contain OrderedVisualFrame values")
        if frame.video_id != video_id:
            raise ValueError("visual frame video_id does not match requested video_id")
        if frame.frame_id in seen_frame_ids:
            raise ValueError("duplicate frame_id in visual batch")
        if previous_index is not None and frame.local_index <= previous_index:
            raise ValueError("visual batch local_index values must be strictly increasing")
        dimension = len(frame.vector)
        if dimensions is None:
            dimensions = dimension
        elif dimension != dimensions:
            raise ValueError("visual vectors in one corpus must have one dimension")
        seen_frame_ids.add(frame.frame_id)
        previous_index = frame.local_index
    return validated


def validate_ordered_visual_stream(
    video_id: str,
    batches: Iterable[OrderedVisualBatch],
) -> tuple[OrderedVisualFrame, ...]:
    """Materialize and validate the complete ordered stream for one video.

    This is intentionally the shared boundary validator B can call after
    consuming adapter batches. It detects wrong-video records, duplicate IDs,
    missing/non-contiguous local positions, reordered batches, and dimension
    changes across batches.
    """

    frames: list[OrderedVisualFrame] = []
    expected_index = 0
    expected_dimension: int | None = None
    seen_frame_ids: set[str] = set()
    seen_local_indices: set[int] = set()
    for batch in batches:
        validated_batch = validate_ordered_visual_batch(
            video_id,
            batch,
            expected_dimension=expected_dimension,
        )
        if expected_dimension is None:
            expected_dimension = len(validated_batch[0].vector)
        for frame in validated_batch:
            if frame.frame_id in seen_frame_ids:
                raise ValueError("duplicate frame_id in visual corpus stream")
            if frame.local_index in seen_local_indices:
                raise ValueError("duplicate local_index in visual corpus stream")
            if frame.local_index != expected_index:
                raise ValueError(
                    "visual stream local_index values must be contiguous and ordered "
                    "from zero"
                )
            seen_frame_ids.add(frame.frame_id)
            seen_local_indices.add(frame.local_index)
            frames.append(frame)
            expected_index += 1
    return tuple(frames)


@runtime_checkable
class VisualCorpusPort(Protocol):
    def list_video_ids(self) -> Sequence[str]: ...

    def iter_ordered_frame_embedding_batches(
        self,
        video_id: str,
        batch_size: int,
    ) -> Iterable[OrderedVisualBatch]: ...
