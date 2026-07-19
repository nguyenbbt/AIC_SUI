from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import pytest

from online.domain.errors import ContractMismatchError, DimensionMismatchError
from online.domain.trake import TRAKEEvent, TRAKEQuery
from online.ports.visual_corpus import OrderedVisualBatch, OrderedVisualFrame
from online.trake.similarity import (
    EncodedTRAKEEvents,
    compute_video_similarity,
    encode_trake_events,
    load_video_similarity,
)


class StaticEncoder:
    def __init__(
        self,
        rows: object,
        *,
        dimension: object = 2,
    ) -> None:
        self.rows = rows
        self._dimension = dimension
        self.calls: list[tuple[str, ...]] = []

    @property
    def dimension(self) -> int:
        return self._dimension  # type: ignore[return-value]

    def encode_texts(self, texts: Sequence[str]) -> object:
        self.calls.append(tuple(texts))
        return self.rows


class StaticCorpus:
    def __init__(self, frames: Sequence[OrderedVisualFrame]) -> None:
        self.frames = tuple(frames)
        self.batch_sizes: list[int] = []

    def list_video_ids(self) -> Sequence[str]:
        return tuple(dict.fromkeys(frame.video_id for frame in self.frames))

    def iter_ordered_frame_embedding_batches(
        self,
        video_id: str,
        batch_size: int,
    ) -> Iterable[OrderedVisualBatch]:
        self.batch_sizes.append(batch_size)
        values = tuple(frame for frame in self.frames if frame.video_id == video_id)
        return tuple(
            values[index : index + batch_size]
            for index in range(0, len(values), batch_size)
        )


def query() -> TRAKEQuery:
    return TRAKEQuery(
        query_id="trake-sim",
        events=(
            TRAKEEvent(event_id="e1", text="first"),
            TRAKEEvent(event_id="e2", text="second"),
        ),
    )


def visual_frame(
    video_id: str,
    local_index: int,
    vector: tuple[float, ...],
) -> OrderedVisualFrame:
    return OrderedVisualFrame(
        frame_id=f"{video_id}_{local_index:05d}_015",
        video_id=video_id,
        shot_id=local_index,
        local_index=local_index,
        timestamp_sec=float(local_index),
        vector=vector,
    )


def test_event_encoding_preserves_order_and_requires_unit_vectors() -> None:
    encoder = StaticEncoder(((1.0, 0.0), (0.0, 1.0)))

    encoded = encode_trake_events(query(), encoder)  # type: ignore[arg-type]

    assert encoder.calls == [("first", "second")]
    assert encoded.event_ids == ("e1", "e2")
    assert encoded.vectors == ((1.0, 0.0), (0.0, 1.0))
    assert encoded.dimension == 2


@pytest.mark.parametrize(
    ("rows", "dimension", "error_type"),
    [
        (((1.0, 0.0),), 2, ContractMismatchError),
        (((1.0, 0.0), (0.0, 1.0, 0.0)), 2, DimensionMismatchError),
        (((0.0, 0.0), (0.0, 1.0)), 2, ContractMismatchError),
        (((math.nan, 0.0), (0.0, 1.0)), 2, ContractMismatchError),
        (((True, 0.0), (0.0, 1.0)), 2, ContractMismatchError),
        (((1.0, 0.0), (0.0, 1.0)), True, DimensionMismatchError),
    ],
)
def test_rejects_malformed_encoder_output(
    rows: object,
    dimension: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        encode_trake_events(
            query(),
            StaticEncoder(rows, dimension=dimension),  # type: ignore[arg-type]
        )


def test_similarity_shape_values_negative_scores_and_empty_video() -> None:
    root_half = math.sqrt(0.5)
    encoded = EncodedTRAKEEvents(
        event_ids=("e1", "e2"),
        vectors=((1.0, 0.0), (0.0, -1.0)),
    )
    frames = (
        visual_frame("V001", 0, (1.0, 0.0)),
        visual_frame("V001", 1, (root_half, root_half)),
        visual_frame("V001", 2, (0.0, 1.0)),
    )

    result = compute_video_similarity(encoded, "V001", frames)

    assert result.event_ids == ("e1", "e2")
    assert result.frames == frames
    assert result.similarities[0] == pytest.approx((1.0, root_half, 0.0))
    assert result.similarities[1] == pytest.approx((0.0, -root_half, -1.0))

    empty = compute_video_similarity(encoded, "V999", ())
    assert empty.frames == ()
    assert empty.similarities == ((), ())


def test_batch_size_does_not_change_similarity_output() -> None:
    encoded = EncodedTRAKEEvents(
        event_ids=("e1", "e2"),
        vectors=((1.0, 0.0), (0.0, 1.0)),
    )
    frames = tuple(
        visual_frame("V001", index, vector)
        for index, vector in enumerate(
            ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))
        )
    )
    corpus = StaticCorpus(frames)

    one = load_video_similarity(corpus, "V001", encoded, batch_size=1)
    four = load_video_similarity(corpus, "V001", encoded, batch_size=4)

    assert one == four
    assert corpus.batch_sizes == [1, 4]


def test_rejects_dimension_norm_video_and_order_mismatches() -> None:
    encoded = EncodedTRAKEEvents(
        event_ids=("e1", "e2"),
        vectors=((1.0, 0.0), (0.0, 1.0)),
    )
    dimension_mismatch = visual_frame("V001", 0, (1.0, 0.0, 0.0))
    with pytest.raises(DimensionMismatchError):
        compute_video_similarity(encoded, "V001", (dimension_mismatch,))

    valid = visual_frame("V001", 0, (1.0, 0.0))
    non_normalized = valid.model_copy(update={"vector": (2.0, 0.0)})
    with pytest.raises(ContractMismatchError, match="L2-normalized"):
        compute_video_similarity(encoded, "V001", (non_normalized,))

    wrong_video = valid.model_copy(
        update={"video_id": "V002", "frame_id": "V002_00000_015"}
    )
    with pytest.raises(ContractMismatchError, match="local order"):
        compute_video_similarity(encoded, "V001", (wrong_video,))

    gap = visual_frame("V001", 1, (0.0, 1.0))
    with pytest.raises(ContractMismatchError, match="local order"):
        compute_video_similarity(encoded, "V001", (valid, gap.model_copy(update={"local_index": 2})))
