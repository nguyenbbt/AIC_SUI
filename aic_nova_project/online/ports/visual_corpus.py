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


@runtime_checkable
class VisualCorpusPort(Protocol):
    def list_video_ids(self) -> Sequence[str]: ...

    def iter_ordered_frame_embedding_batches(
        self,
        video_id: str,
        batch_size: int,
    ) -> Iterable[OrderedVisualBatch]: ...
