from __future__ import annotations

import asyncio
import math
import threading
import time
from collections.abc import Iterable, Sequence

import pytest

from online.domain.errors import (
    BranchTimeoutError,
    ContractMismatchError,
    ResourceUnavailableError,
)
from online.domain.trake import DANTEPolicy, TRAKEEvent, TRAKEQuery
from online.ports.visual_corpus import OrderedVisualBatch, OrderedVisualFrame
from online.trake.service import TRAKEService, TRAKEServiceConfig
from query_understanding import parse_trake_query


class MappedEncoder:
    def __init__(self, mapping: dict[str, tuple[float, ...]]) -> None:
        self.mapping = mapping
        self.calls: list[tuple[str, ...]] = []

    @property
    def dimension(self) -> int:
        return len(next(iter(self.mapping.values())))

    def encode_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        values = tuple(texts)
        self.calls.append(values)
        return tuple(self.mapping[text] for text in values)


class FakeVisualCorpus:
    def __init__(
        self,
        videos: dict[str, tuple[OrderedVisualFrame, ...]],
        *,
        listed_video_ids: Sequence[str] | None = None,
        delay_sec: float = 0.0,
        entered: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.videos = videos
        self.listed_video_ids = (
            tuple(videos) if listed_video_ids is None else tuple(listed_video_ids)
        )
        self.delay_sec = delay_sec
        self.entered = entered
        self.release = release
        self.batch_calls: list[tuple[str, int]] = []
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def list_video_ids(self) -> Sequence[str]:
        return self.listed_video_ids

    def iter_ordered_frame_embedding_batches(
        self,
        video_id: str,
        batch_size: int,
    ) -> Iterable[OrderedVisualBatch]:
        with self._lock:
            self.batch_calls.append((video_id, batch_size))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.entered is not None and self.active >= 2:
                self.entered.set()
        try:
            if self.release is not None:
                self.release.wait(timeout=1.0)
            elif self.delay_sec:
                time.sleep(self.delay_sec)
            frames = self.videos[video_id]
            return tuple(
                frames[index : index + batch_size]
                for index in range(0, len(frames), batch_size)
            )
        finally:
            with self._lock:
                self.active -= 1


def frame(
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
        source_frame_idx=local_index * 30,
        image_rel_path=f"keyframes/{video_id}/{local_index:05d}.webp",
        vector=vector,
    )


def videos() -> dict[str, tuple[OrderedVisualFrame, ...]]:
    e1 = (1.0, 0.0, 0.0)
    e2 = (0.0, 1.0, 0.0)
    e3 = (0.0, 0.0, 1.0)
    return {
        "V001": tuple(frame("V001", index, vector) for index, vector in enumerate((e1, e2, e3))),
        "V002": tuple(frame("V002", index, vector) for index, vector in enumerate((e3, e2, e1))),
        "V003": tuple(frame("V003", index, vector) for index, vector in enumerate((e1, e2))),
        "V004": tuple(frame("V004", index, vector) for index, vector in enumerate((e1, e2, e3))),
    }


def query(*, top_k: int = 2) -> TRAKEQuery:
    return TRAKEQuery(
        query_id="trake-service",
        events=(
            TRAKEEvent(event_id="e1", text="first"),
            TRAKEEvent(event_id="e2", text="second"),
            TRAKEEvent(event_id="e3", text="third"),
        ),
        top_k_videos=top_k,
        policy=DANTEPolicy(lambda_penalty=0.001),
    )


def encoder() -> MappedEncoder:
    return MappedEncoder(
        {
            "first": (1.0, 0.0, 0.0),
            "second": (0.0, 1.0, 0.0),
            "third": (0.0, 0.0, 1.0),
        }
    )


def run(coroutine: object) -> object:
    return asyncio.run(coroutine)  # type: ignore[arg-type]


def test_full_service_returns_deterministic_top_k_hydration_and_diagnostics() -> None:
    text_encoder = encoder()
    corpus = FakeVisualCorpus(videos(), listed_video_ids=("V004", "V003", "V002", "V001"))
    service = TRAKEService(
        corpus=corpus,
        encoder=text_encoder,
        config=TRAKEServiceConfig(batch_size=2, max_workers=2, total_timeout_sec=1.0),
    )
    try:
        execution = run(service.execute(query()))
    finally:
        service.close()

    assert tuple(result.video_id for result in execution.results) == ("V001", "V004")
    assert execution.results[0].score == pytest.approx(2.998)
    assert tuple(match.event_id for match in execution.results[0].sequence) == (
        "e1",
        "e2",
        "e3",
    )
    assert tuple(match.local_index for match in execution.results[0].sequence) == (0, 1, 2)
    assert tuple(match.similarity_score for match in execution.results[0].sequence) == (
        1.0,
        1.0,
        1.0,
    )
    assert text_encoder.calls == [("first", "second", "third")]
    assert execution.diagnostics.video_count == 4
    assert execution.diagnostics.frame_count == 11
    assert execution.diagnostics.invalid_sequence_count == 1
    assert execution.diagnostics.warnings == ("unreachable_video_count=1",)
    assert all(batch_size == 2 for _, batch_size in corpus.batch_calls)


def test_local_fake_end_to_end_from_parser_through_dante_result() -> None:
    parsed_query = parse_trake_query(
        "trake-local-e2e",
        ("first", "second", "third"),
        event_ids=("e1", "e2", "e3"),
        top_k_videos=1,
    )
    service = TRAKEService(
        corpus=FakeVisualCorpus(videos()),
        encoder=encoder(),
        config=TRAKEServiceConfig(batch_size=1, max_workers=2),
    )
    try:
        execution = run(service.execute(parsed_query))
    finally:
        service.close()

    assert execution.query_id == "trake-local-e2e"
    assert tuple(result.video_id for result in execution.results) == ("V001",)
    assert tuple(match.frame_id for match in execution.results[0].sequence) == (
        "V001_00000_015",
        "V001_00001_015",
        "V001_00002_015",
    )


def test_search_returns_public_results_and_negative_scores_are_not_clamped() -> None:
    negative = {
        "V900": (
            frame("V900", 0, (-1.0, 0.0, 0.0)),
            frame("V900", 1, (0.0, -1.0, 0.0)),
            frame("V900", 2, (0.0, 0.0, -1.0)),
        )
    }
    service = TRAKEService(corpus=FakeVisualCorpus(negative), encoder=encoder())
    try:
        results = run(service.search(query(top_k=1)))
    finally:
        service.close()

    assert len(results) == 1
    assert results[0].score == pytest.approx(-3.002)
    assert all(match.similarity_score == -1.0 for match in results[0].sequence)


def test_repeated_execution_is_deterministic_and_empty_corpus_is_valid() -> None:
    service = TRAKEService(
        corpus=FakeVisualCorpus(videos()),
        encoder=encoder(),
        config=TRAKEServiceConfig(max_workers=3),
    )
    try:
        first = run(service.execute(query()))
        second = run(service.execute(query()))
    finally:
        service.close()

    assert first.results == second.results

    empty_service = TRAKEService(corpus=FakeVisualCorpus({}), encoder=encoder())
    try:
        empty = run(empty_service.execute(query()))
    finally:
        empty_service.close()

    assert empty.results == ()
    assert empty.diagnostics.video_count == 0
    assert empty.diagnostics.frame_count == 0
    assert empty.diagnostics.invalid_sequence_count == 0


def test_bounded_executor_runs_video_work_in_parallel() -> None:
    entered = threading.Event()
    release = threading.Event()
    corpus = FakeVisualCorpus(videos(), entered=entered, release=release)
    service = TRAKEService(
        corpus=corpus,
        encoder=encoder(),
        config=TRAKEServiceConfig(max_workers=2, total_timeout_sec=1.0),
    )

    async def scenario() -> None:
        task = asyncio.create_task(service.execute(query()))
        reached_two = await asyncio.to_thread(entered.wait, 0.5)
        assert reached_two
        assert corpus.max_active == 2
        with pytest.raises(ResourceUnavailableError, match="active executions"):
            service.close(wait=False)
        release.set()
        execution = await task
        assert len(execution.results) == 2

    try:
        run(scenario())
    finally:
        release.set()
        service.close()


def test_total_timeout_never_returns_partial_success() -> None:
    corpus = FakeVisualCorpus(videos(), delay_sec=0.08)
    service = TRAKEService(
        corpus=corpus,
        encoder=encoder(),
        config=TRAKEServiceConfig(max_workers=2, total_timeout_sec=0.02),
    )
    try:
        with pytest.raises(BranchTimeoutError) as raised:
            run(service.execute(query()))
        assert raised.value.details["stage"] == "per_video_dante"
    finally:
        service.close()


@pytest.mark.parametrize(
    "listed_video_ids",
    [
        ("V001", "V001"),
        ("V001", " "),
    ],
)
def test_invalid_video_enumeration_is_a_contract_failure(
    listed_video_ids: tuple[str, ...],
) -> None:
    service = TRAKEService(
        corpus=FakeVisualCorpus(videos(), listed_video_ids=listed_video_ids),
        encoder=encoder(),
    )
    try:
        with pytest.raises(ContractMismatchError):
            run(service.execute(query()))
    finally:
        service.close()


def test_wrong_video_stream_is_a_contract_failure_not_a_skipped_video() -> None:
    values = videos()
    values["V001"] = (
        values["V001"][0].model_copy(
            update={"video_id": "V999", "frame_id": "V999_00000_015"}
        ),
        *values["V001"][1:],
    )
    service = TRAKEService(corpus=FakeVisualCorpus(values), encoder=encoder())
    try:
        with pytest.raises(ContractMismatchError):
            run(service.execute(query()))
    finally:
        service.close()


def test_close_is_idempotent_and_rejects_new_work() -> None:
    service = TRAKEService(corpus=FakeVisualCorpus(videos()), encoder=encoder())
    service.close()
    service.close()

    with pytest.raises(ResourceUnavailableError):
        run(service.execute(query()))


@pytest.mark.parametrize(
    "config",
    [
        {"batch_size": True},
        {"batch_size": 0},
        {"max_workers": 0},
        {"total_timeout_sec": 0},
        {"total_timeout_sec": math.nan},
    ],
)
def test_service_config_rejects_invalid_bounds(config: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TRAKEServiceConfig(**config)  # type: ignore[arg-type]
