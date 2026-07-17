"""Async orchestration for synchronous retrieval branches."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from functools import partial
from threading import Lock
from time import perf_counter
from typing import Any, Protocol, runtime_checkable

from pydantic import Field, ValidationError

from online.domain.base import FiniteFloat, NonEmptyStr, StrictFrozenModel, StrictIntValue
from online.domain.candidates import BranchResult
from online.domain.diagnostics import BranchDiagnostics
from online.domain.enums import BranchStatus, CandidateLevel, RetrievalBranch
from online.domain.errors import BranchTimeoutError, ContractMismatchError, DataInfrastructureError
from online.domain.query import QueryBundle, TextQueryVariant

from .query_builder import BASELINE_KIS_BRANCHES


MULTI_VARIANT_BRANCHES = frozenset(
    {
        RetrievalBranch.VISUAL_DENSE,
        RetrievalBranch.OCR_DENSE,
        RetrievalBranch.ASR_DENSE,
        RetrievalBranch.SUMMARY_DENSE,
    }
)
CORE_BRANCHES = frozenset({RetrievalBranch.VISUAL_DENSE})

_CANDIDATE_LEVELS = {
    RetrievalBranch.VISUAL_DENSE: CandidateLevel.FRAME,
    RetrievalBranch.OCR_DENSE: CandidateLevel.FRAME,
    RetrievalBranch.OCR_BM25: CandidateLevel.FRAME,
    RetrievalBranch.ASR_DENSE: CandidateLevel.ASR_INTERVAL,
    RetrievalBranch.ASR_BM25: CandidateLevel.ASR_INTERVAL,
    RetrievalBranch.SUMMARY_DENSE: CandidateLevel.VIDEO,
    RetrievalBranch.SUMMARY_BM25: CandidateLevel.VIDEO,
}


class RetrievalBranchRunner(Protocol):
    branch: RetrievalBranch

    def retrieve_variant(
        self,
        variant: TextQueryVariant,
        *,
        top_k: int,
    ) -> BranchResult[Any]: ...


@runtime_checkable
class RetrievalServicePort(Protocol):
    """Minimal Person-B handoff consumed by Person C orchestration."""

    async def retrieve(self, bundle: QueryBundle) -> tuple[BranchResult[Any], ...]: ...


class RetrievalInvocationConfig(StrictFrozenModel):
    """Required tuning for one exact branch/query-variant invocation."""

    top_k: StrictIntValue = Field(ge=1)
    timeout_sec: FiniteFloat = Field(gt=0.0)


class BranchInvocationDiagnostics(StrictFrozenModel):
    branch: RetrievalBranch
    query_variant_id: NonEmptyStr
    metrics: BranchDiagnostics


class RetrievalExecution(StrictFrozenModel):
    query_id: NonEmptyStr
    total_latency_ms: FiniteFloat = Field(ge=0.0)
    results: tuple[BranchResult[Any], ...]
    invocations: tuple[BranchInvocationDiagnostics, ...]


class _Invocation:
    def __init__(
        self,
        *,
        branch: RetrievalBranch,
        runner: RetrievalBranchRunner,
        variant: TextQueryVariant,
        config: RetrievalInvocationConfig,
    ) -> None:
        self.branch = branch
        self.runner = runner
        self.variant = variant
        self.config = config


class RetrievalService:
    """Run enabled branch invocations concurrently in deterministic order.

    Branch implementations and database ports are synchronous. Every invocation
    is therefore submitted to a bounded executor instead of blocking the event
    loop. ``asyncio.wait_for`` bounds how long the caller waits; as with any
    Python thread timeout, an already-running SDK call cannot be force-killed and
    must also have its own adapter-level timeout.
    """

    def __init__(
        self,
        *,
        branches: Mapping[RetrievalBranch | str, RetrievalBranchRunner],
        invocation_configs: Mapping[
            tuple[RetrievalBranch | str, str],
            RetrievalInvocationConfig | Mapping[str, object],
        ],
        max_workers: int,
        executor: Executor | None = None,
    ) -> None:
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        self._branches = self._normalize_branches(branches)
        self._configs = self._normalize_configs(invocation_configs)
        self._executor = executor or ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="aic-retrieval",
        )
        self._max_workers = max_workers
        self._owns_executor = executor is None
        self._state_lock = Lock()
        self._active_executions = 0
        self._closed = False

    async def retrieve(self, bundle: QueryBundle) -> tuple[BranchResult[Any], ...]:
        """Person-B handoff contract consumed by Person C."""

        return (await self.execute(bundle)).results

    async def execute(self, bundle: QueryBundle) -> RetrievalExecution:
        """Retrieve plus per-invocation diagnostics for integration/debugging."""

        with self._state_lock:
            if self._closed:
                raise RuntimeError("RetrievalService is closed")
            self._active_executions += 1
        try:
            if not isinstance(bundle, QueryBundle):
                raise ContractMismatchError("bundle must be a validated QueryBundle")
            invocations = self._plan(bundle)
            started_at = perf_counter()
            # asyncio synchronization primitives are event-loop scoped. Keep
            # the limiter local so one service can safely be called from a new
            # loop in a later CLI/test invocation.
            concurrency_limit = asyncio.Semaphore(self._max_workers)
            outcomes = await asyncio.gather(
                *(
                    self._run_invocation(invocation, concurrency_limit)
                    for invocation in invocations
                )
            )
            total_latency_ms = self._elapsed_ms(started_at)
            return RetrievalExecution(
                query_id=bundle.query_id,
                total_latency_ms=total_latency_ms,
                results=tuple(result for result, _ in outcomes),
                invocations=tuple(diagnostics for _, diagnostics in outcomes),
            )
        finally:
            with self._state_lock:
                self._active_executions -= 1

    def close(self, *, wait: bool = True) -> None:
        """Release an owned executor; running SDK calls cannot be force-killed."""

        with self._state_lock:
            if self._closed:
                return
            if self._active_executions:
                raise RuntimeError("RetrievalService cannot close during active execution")
            self._closed = True
        if self._owns_executor:
            self._executor.shutdown(wait=wait, cancel_futures=True)

    async def __aenter__(self) -> "RetrievalService":
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _plan(self, bundle: QueryBundle) -> tuple[_Invocation, ...]:
        enabled = set(bundle.enabled_branches)
        planned: list[_Invocation] = []
        for branch in BASELINE_KIS_BRANCHES:
            if branch not in enabled:
                continue
            runner = self._branches.get(branch)
            if runner is None:
                raise ContractMismatchError(
                    "Enabled retrieval branch is not registered",
                    details={"branch": branch.value},
                )
            variants: Sequence[TextQueryVariant]
            if branch in MULTI_VARIANT_BRANCHES:
                variants = bundle.text_variants
            else:
                variants = bundle.text_variants[:1]
            for variant in variants:
                key = (branch, variant.variant_id)
                config = self._configs.get(key)
                if config is None:
                    raise ContractMismatchError(
                        "Retrieval invocation config is missing",
                        details={
                            "branch": branch.value,
                            "query_variant_id": variant.variant_id,
                        },
                    )
                planned.append(
                    _Invocation(
                        branch=branch,
                        runner=runner,
                        variant=variant,
                        config=config,
                    )
                )
        return tuple(planned)

    async def _run_invocation(
        self,
        invocation: _Invocation,
        concurrency_limit: asyncio.Semaphore,
    ) -> tuple[BranchResult[Any], BranchInvocationDiagnostics]:
        started_at = perf_counter()
        loop = asyncio.get_running_loop()
        call = partial(
            invocation.runner.retrieve_variant,
            invocation.variant,
            top_k=invocation.config.top_k,
        )
        try:
            async with concurrency_limit:
                future = loop.run_in_executor(self._executor, call)
                result = await asyncio.wait_for(
                    future,
                    timeout=invocation.config.timeout_sec,
                )
            self._validate_result(invocation, result)
        except TimeoutError:
            result = self._failure_result(
                invocation,
                warning="BRANCH_TIMEOUT",
                latency_ms=self._elapsed_ms(started_at),
            )
        except BranchTimeoutError:
            result = self._failure_result(
                invocation,
                warning="BRANCH_TIMEOUT",
                latency_ms=self._elapsed_ms(started_at),
            )
        except DataInfrastructureError as exc:
            result = self._failure_result(
                invocation,
                warning=exc.code.value,
                latency_ms=self._elapsed_ms(started_at),
            )
        except Exception:
            result = self._failure_result(
                invocation,
                warning="UNEXPECTED_ERROR",
                latency_ms=self._elapsed_ms(started_at),
            )
        return result, self._diagnostics(result)

    @staticmethod
    def _validate_result(invocation: _Invocation, result: object) -> None:
        if not isinstance(result, BranchResult):
            raise ContractMismatchError("Retrieval branch returned an invalid result")
        if result.branch is not invocation.branch:
            raise ContractMismatchError("Retrieval branch result has the wrong branch")
        if result.query_variant_id != invocation.variant.variant_id:
            raise ContractMismatchError("Retrieval branch result has the wrong query variant")
        if result.requested_top_k != invocation.config.top_k:
            raise ContractMismatchError("Retrieval branch result has the wrong top_k")
        if result.candidate_level is not _CANDIDATE_LEVELS[invocation.branch]:
            raise ContractMismatchError("Retrieval branch result has the wrong candidate level")
        if result.returned_count > invocation.config.top_k:
            raise ContractMismatchError("Retrieval branch returned more candidates than requested")
        if any(
            candidate.provenance.query_text != invocation.variant.text
            for candidate in result.candidates
        ):
            raise ContractMismatchError("Retrieval branch result has the wrong query text")

    def _failure_result(
        self,
        invocation: _Invocation,
        *,
        warning: str,
        latency_ms: float,
    ) -> BranchResult[Any]:
        status = (
            BranchStatus.FAILED
            if invocation.branch in CORE_BRANCHES
            else BranchStatus.DEGRADED
        )
        return BranchResult(
            branch=invocation.branch,
            candidate_level=_CANDIDATE_LEVELS[invocation.branch],
            query_variant_id=invocation.variant.variant_id,
            candidates=(),
            requested_top_k=invocation.config.top_k,
            latency_ms=latency_ms,
            status=status,
            warnings=(warning,),
        )

    @staticmethod
    def _diagnostics(result: BranchResult[Any]) -> BranchInvocationDiagnostics:
        count = result.returned_count
        return BranchInvocationDiagnostics(
            branch=result.branch,
            query_variant_id=result.query_variant_id,
            metrics=BranchDiagnostics(
                status=result.status,
                latency_ms=result.latency_ms,
                requested_top_k=result.requested_top_k,
                raw_result_count=count,
                output_candidate_count=count,
                mapping_loss_count=0,
                warnings=result.warnings,
            ),
        )

    @staticmethod
    def _normalize_branches(
        branches: Mapping[RetrievalBranch | str, RetrievalBranchRunner],
    ) -> dict[RetrievalBranch, RetrievalBranchRunner]:
        normalized: dict[RetrievalBranch, RetrievalBranchRunner] = {}
        for raw_branch, runner in branches.items():
            try:
                branch = RetrievalBranch(raw_branch)
            except (TypeError, ValueError) as exc:
                raise ValueError("branches contains an unknown branch") from exc
            if branch in normalized:
                raise ValueError(f"duplicate branch registration: {branch.value}")
            if getattr(runner, "branch", None) is not branch:
                raise ValueError(f"runner branch does not match registration: {branch.value}")
            if not callable(getattr(runner, "retrieve_variant", None)):
                raise ValueError(f"runner does not implement retrieve_variant: {branch.value}")
            normalized[branch] = runner
        return normalized

    @staticmethod
    def _normalize_configs(
        configs: Mapping[
            tuple[RetrievalBranch | str, str],
            RetrievalInvocationConfig | Mapping[str, object],
        ],
    ) -> dict[tuple[RetrievalBranch, str], RetrievalInvocationConfig]:
        normalized: dict[tuple[RetrievalBranch, str], RetrievalInvocationConfig] = {}
        for raw_key, raw_config in configs.items():
            if not isinstance(raw_key, tuple) or len(raw_key) != 2:
                raise ValueError("invocation config keys must be (branch, query_variant_id)")
            raw_branch, variant_id = raw_key
            try:
                branch = RetrievalBranch(raw_branch)
            except (TypeError, ValueError) as exc:
                raise ValueError("invocation config contains an unknown branch") from exc
            if not isinstance(variant_id, str) or not variant_id.strip():
                raise ValueError("query_variant_id in invocation config must be non-empty")
            key = (branch, variant_id.strip())
            if key in normalized:
                raise ValueError(
                    f"duplicate invocation config: {branch.value}/{variant_id.strip()}"
                )
            try:
                config = (
                    raw_config
                    if isinstance(raw_config, RetrievalInvocationConfig)
                    else RetrievalInvocationConfig.model_validate(raw_config)
                )
            except ValidationError as exc:
                raise ValueError(
                    f"invalid invocation config: {branch.value}/{variant_id.strip()}"
                ) from exc
            normalized[key] = config
        return normalized

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        elapsed = (perf_counter() - started_at) * 1000.0
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise RuntimeError("monotonic clock returned an invalid duration")
        return elapsed


__all__ = [
    "BranchInvocationDiagnostics",
    "CORE_BRANCHES",
    "MULTI_VARIANT_BRANCHES",
    "RetrievalBranchRunner",
    "RetrievalExecution",
    "RetrievalInvocationConfig",
    "RetrievalService",
    "RetrievalServicePort",
]
