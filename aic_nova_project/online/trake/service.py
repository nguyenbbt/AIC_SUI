"""Bounded fake-ready TRAKE orchestration over a full ordered visual corpus."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from threading import Condition
from time import perf_counter
from typing import Callable, TypeVar

from pydantic import ValidationError

from online.domain.errors import (
    BranchTimeoutError,
    ContractMismatchError,
    DataInfrastructureError,
    ResourceUnavailableError,
)
from online.domain.trake import (
    TRAKEDiagnostics,
    TRAKEFrameMatch,
    TRAKEQuery,
    TRAKEVideoResult,
)
from online.ports.encoders import TextEncoderPort
from online.ports.visual_corpus import VisualCorpusPort

from .config import DANTEConfig
from .dante import solve_dante
from .similarity import (
    EncodedTRAKEEvents,
    VideoSimilarityMatrix,
    encode_trake_events,
    load_video_similarity,
)


_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class TRAKEServiceConfig:
    """Validated execution bounds; production tuning waits for real data."""

    batch_size: int = 256
    max_workers: int = 4
    total_timeout_sec: float = 10.0

    def __post_init__(self) -> None:
        for name in ("batch_size", "max_workers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.total_timeout_sec, bool)
            or not isinstance(self.total_timeout_sec, (int, float))
            or not math.isfinite(float(self.total_timeout_sec))
            or float(self.total_timeout_sec) <= 0.0
        ):
            raise ValueError("total_timeout_sec must be a positive finite number")
        object.__setattr__(self, "total_timeout_sec", float(self.total_timeout_sec))


@dataclass(frozen=True, slots=True)
class TRAKEExecution:
    """Internal service envelope; Wave 3 mode code consumes public child models."""

    query_id: str
    results: tuple[TRAKEVideoResult, ...]
    diagnostics: TRAKEDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id.strip():
            raise ValueError("query_id must be non-empty")
        if any(not isinstance(result, TRAKEVideoResult) for result in self.results):
            raise TypeError("results must contain TRAKEVideoResult values")
        if not isinstance(self.diagnostics, TRAKEDiagnostics):
            raise TypeError("diagnostics must be TRAKEDiagnostics")


@dataclass(frozen=True, slots=True)
class _VideoOutcome:
    video_id: str
    result: TRAKEVideoResult | None
    frame_count: int
    similarity_latency_ms: float
    dp_latency_ms: float

    @property
    def unreachable(self) -> bool:
        return self.result is None


class TRAKEService:
    """Encode once, run DANTE per video, then return deterministic top-k results."""

    def __init__(
        self,
        *,
        corpus: VisualCorpusPort,
        encoder: TextEncoderPort,
        config: TRAKEServiceConfig | None = None,
        executor: Executor | None = None,
    ) -> None:
        if not isinstance(corpus, VisualCorpusPort):
            raise TypeError("corpus must implement VisualCorpusPort")
        if not isinstance(encoder, TextEncoderPort):
            raise TypeError("encoder must implement TextEncoderPort")
        self.corpus = corpus
        self.encoder = encoder
        self.config = config or TRAKEServiceConfig()
        if not isinstance(self.config, TRAKEServiceConfig):
            raise TypeError("config must be TRAKEServiceConfig")
        self._executor = executor or ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix="aic-trake",
        )
        self._owns_executor = executor is None
        self._state = Condition()
        self._active_executions = 0
        self._closing = False
        self._closed = False

    async def execute(self, query: TRAKEQuery) -> TRAKEExecution:
        """Run one full-corpus TRAKE request under a single total deadline."""

        if not isinstance(query, TRAKEQuery):
            raise ContractMismatchError("query must be a validated TRAKEQuery")
        self._begin_execution()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.config.total_timeout_sec
        try:
            encoded = await self._run_executor_call(
                partial(encode_trake_events, query, self.encoder),
                deadline=deadline,
                stage="event_encoding",
            )
            video_ids = await self._run_executor_call(
                self._list_video_ids,
                deadline=deadline,
                stage="video_enumeration",
            )
            outcomes = await self._run_video_jobs(
                query,
                encoded,
                video_ids,
                deadline=deadline,
            )
            ordered_results = tuple(
                sorted(
                    (outcome.result for outcome in outcomes if outcome.result is not None),
                    key=lambda result: (-result.score, result.video_id),
                )[: query.top_k_videos]
            )
            unreachable_count = sum(outcome.unreachable for outcome in outcomes)
            warnings = (
                (f"unreachable_video_count={unreachable_count}",)
                if unreachable_count
                else ()
            )
            diagnostics = TRAKEDiagnostics(
                policy_version=query.policy.policy_version,
                lambda_penalty=query.policy.lambda_penalty,
                event_count=len(query.events),
                video_count=len(video_ids),
                frame_count=sum(outcome.frame_count for outcome in outcomes),
                similarity_latency_ms=sum(
                    outcome.similarity_latency_ms for outcome in outcomes
                ),
                dp_latency_ms=sum(outcome.dp_latency_ms for outcome in outcomes),
                invalid_sequence_count=unreachable_count,
                warnings=warnings,
            )
            return TRAKEExecution(
                query_id=query.query_id,
                results=ordered_results,
                diagnostics=diagnostics,
            )
        finally:
            self._end_execution()

    async def search(self, query: TRAKEQuery) -> tuple[TRAKEVideoResult, ...]:
        """Convenience handoff for Wave 3 mode routing."""

        return (await self.execute(query)).results

    def close(self, *, wait: bool = True) -> None:
        """Stop accepting work and release the owned executor."""

        with self._state:
            if self._closed:
                return
            self._closing = True
            if wait:
                while self._active_executions:
                    self._state.wait()
            elif self._active_executions:
                raise ResourceUnavailableError(
                    "TRAKE service has active executions",
                    details={"resource": "trake_service"},
                )
            self._closed = True
        if self._owns_executor:
            self._executor.shutdown(wait=wait, cancel_futures=True)

    async def __aenter__(self) -> "TRAKEService":
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _begin_execution(self) -> None:
        with self._state:
            if self._closing or self._closed:
                raise ResourceUnavailableError(
                    "TRAKE service is closing",
                    details={"resource": "trake_service"},
                )
            self._active_executions += 1

    def _end_execution(self) -> None:
        with self._state:
            self._active_executions -= 1
            self._state.notify_all()

    async def _run_executor_call(
        self,
        call: Callable[[], _T],
        *,
        deadline: float,
        stage: str,
    ) -> _T:
        loop = asyncio.get_running_loop()
        remaining = deadline - loop.time()
        if remaining <= 0.0:
            raise BranchTimeoutError(
                "TRAKE total deadline expired",
                details={"stage": stage},
            )
        future = loop.run_in_executor(self._executor, call)
        try:
            return await asyncio.wait_for(future, timeout=remaining)
        except TimeoutError as exc:
            raise BranchTimeoutError(
                "TRAKE total deadline expired",
                details={"stage": stage},
            ) from exc
        except DataInfrastructureError:
            raise
        except Exception as exc:
            raise ResourceUnavailableError(
                "TRAKE dependency call failed",
                details={"stage": stage, "cause_type": type(exc).__name__},
            ) from exc

    async def _run_video_jobs(
        self,
        query: TRAKEQuery,
        encoded: EncodedTRAKEEvents,
        video_ids: tuple[str, ...],
        *,
        deadline: float,
    ) -> tuple[_VideoOutcome, ...]:
        if not video_ids:
            return ()
        loop = asyncio.get_running_loop()
        video_iterator = iter(video_ids)
        pending: dict[asyncio.Future[_VideoOutcome], str] = {}
        outcomes: list[_VideoOutcome] = []

        def submit_next() -> bool:
            try:
                video_id = next(video_iterator)
            except StopIteration:
                return False
            future = loop.run_in_executor(
                self._executor,
                self._process_video,
                query,
                encoded,
                video_id,
            )
            pending[future] = video_id
            return True

        for _ in range(min(self.config.max_workers, len(video_ids))):
            submit_next()

        try:
            while pending:
                remaining = deadline - loop.time()
                if remaining <= 0.0:
                    raise BranchTimeoutError(
                        "TRAKE total deadline expired",
                        details={"stage": "per_video_dante"},
                    )
                done, _ = await asyncio.wait(
                    tuple(pending),
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    raise BranchTimeoutError(
                        "TRAKE total deadline expired",
                        details={"stage": "per_video_dante"},
                    )
                batch_error: BaseException | None = None
                for future in sorted(done, key=lambda item: pending[item]):
                    pending.pop(future)
                    try:
                        outcomes.append(future.result())
                    except DataInfrastructureError as exc:
                        if batch_error is None:
                            batch_error = exc
                    except Exception as exc:
                        if batch_error is None:
                            batch_error = ResourceUnavailableError(
                                "TRAKE per-video execution failed",
                                details={
                                    "stage": "per_video_dante",
                                    "cause_type": type(exc).__name__,
                                },
                            )
                            batch_error.__cause__ = exc
                if batch_error is not None:
                    raise batch_error
                for _ in done:
                    submit_next()
        finally:
            for future in pending:
                future.cancel()
        return tuple(sorted(outcomes, key=lambda outcome: outcome.video_id))

    def _process_video(
        self,
        query: TRAKEQuery,
        encoded: EncodedTRAKEEvents,
        video_id: str,
    ) -> _VideoOutcome:
        similarity_started = perf_counter()
        matrix = load_video_similarity(
            self.corpus,
            video_id,
            encoded,
            batch_size=self.config.batch_size,
        )
        similarity_latency_ms = _elapsed_ms(similarity_started)
        if len(matrix.frames) < len(query.events):
            return _VideoOutcome(
                video_id=video_id,
                result=None,
                frame_count=len(matrix.frames),
                similarity_latency_ms=similarity_latency_ms,
                dp_latency_ms=0.0,
            )

        dp_started = perf_counter()
        try:
            path = solve_dante(
                matrix.similarities,
                DANTEConfig(
                    lambda_penalty=query.policy.lambda_penalty,
                    policy_name=query.policy.policy_version,
                ),
            )
            result = None if path is None else _hydrate_result(query, matrix, path.positions, path.score)
        except DataInfrastructureError:
            raise
        except (TypeError, ValueError, RuntimeError, ValidationError) as exc:
            raise ContractMismatchError(
                "DANTE result could not be hydrated",
                details={"video_id": video_id, "cause_type": type(exc).__name__},
            ) from exc
        dp_latency_ms = _elapsed_ms(dp_started)
        return _VideoOutcome(
            video_id=video_id,
            result=result,
            frame_count=len(matrix.frames),
            similarity_latency_ms=similarity_latency_ms,
            dp_latency_ms=dp_latency_ms,
        )

    def _list_video_ids(self) -> tuple[str, ...]:
        raw_video_ids = self.corpus.list_video_ids()
        if isinstance(raw_video_ids, (str, bytes)):
            raise ContractMismatchError("Visual corpus returned invalid video IDs")
        try:
            video_ids = tuple(raw_video_ids)
        except TypeError as exc:
            raise ContractMismatchError("Visual corpus returned invalid video IDs") from exc
        if any(
            not isinstance(video_id, str)
            or not video_id.strip()
            or video_id != video_id.strip()
            for video_id in video_ids
        ):
            raise ContractMismatchError("Visual corpus returned an invalid video_id")
        if len(set(video_ids)) != len(video_ids):
            raise ContractMismatchError("Visual corpus returned duplicate video IDs")
        return tuple(sorted(video_ids))


def _hydrate_result(
    query: TRAKEQuery,
    matrix: VideoSimilarityMatrix,
    positions: Sequence[int],
    score: float,
) -> TRAKEVideoResult:
    if len(positions) != len(query.events):
        raise ContractMismatchError("DANTE path length does not match event count")
    matches = tuple(
        TRAKEFrameMatch(
            event_id=event.event_id,
            frame_id=matrix.frames[position].frame_id,
            video_id=matrix.video_id,
            shot_id=matrix.frames[position].shot_id,
            local_index=matrix.frames[position].local_index,
            timestamp_sec=matrix.frames[position].timestamp_sec,
            similarity_score=matrix.similarities[event_index][position],
        )
        for event_index, (event, position) in enumerate(
            zip(query.events, positions, strict=True)
        )
    )
    return TRAKEVideoResult(
        video_id=matrix.video_id,
        score=score,
        event_ids=tuple(event.event_id for event in query.events),
        sequence=matches,
    )


def _elapsed_ms(started_at: float) -> float:
    elapsed = (perf_counter() - started_at) * 1000.0
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise RuntimeError("monotonic clock returned an invalid duration")
    return elapsed


__all__ = ["TRAKEExecution", "TRAKEService", "TRAKEServiceConfig"]
