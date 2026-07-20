"""Async fake-ready VQA orchestration over ranked candidates and public ports."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from threading import Condition
from time import perf_counter
from typing import Protocol, runtime_checkable

from online.domain.candidates import FusedFrameCandidate
from online.domain.errors import (
    BranchTimeoutError,
    ContractMismatchError,
    DataInfrastructureError,
    ResourceUnavailableError,
)
from online.domain.vqa import (
    VLMConfidence,
    VLMRequest,
    VLMResponse,
    VLMResponseStatus,
    VQADiagnostics,
    VQAEvidenceBudget,
    VQAQuestion,
    VQAResult,
)
from online.ports.vlm import VLMPort

from .evidence_selector import EvidenceSelectionResult, EvidenceSelector
from .vlm_request import build_vlm_request, validate_vlm_response


@runtime_checkable
class VQACandidateRetrievalPort(Protocol):
    async def retrieve_candidates(
        self,
        question: VQAQuestion,
    ) -> Sequence[FusedFrameCandidate]: ...


@dataclass(frozen=True, slots=True)
class VQAExecution:
    result: VQAResult
    retrieval_latency_ms: float
    evidence_latency_ms: float
    total_latency_ms: float


class VQAOrchestrator:
    def __init__(
        self,
        *,
        candidate_retriever: VQACandidateRetrievalPort,
        evidence_selector: EvidenceSelector,
        vlm: VLMPort,
        total_timeout_sec: float = 30.0,
        vlm_timeout_sec: float = 15.0,
        max_workers: int = 2,
        executor: Executor | None = None,
    ) -> None:
        _validate_timeout("total_timeout_sec", total_timeout_sec)
        _validate_timeout("vlm_timeout_sec", vlm_timeout_sec)
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        self._candidate_retriever = candidate_retriever
        self._evidence_selector = evidence_selector
        self._vlm = vlm
        self._total_timeout_sec = float(total_timeout_sec)
        self._vlm_timeout_sec = float(vlm_timeout_sec)
        self._executor = executor or ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="aic-vqa")
        self._owns_executor = executor is None
        self._state = Condition()
        self._active = 0
        self._closing = False
        self._closed = False

    async def answer(
        self,
        question: VQAQuestion,
        budget: VQAEvidenceBudget = VQAEvidenceBudget(),
    ) -> VQAResult:
        return (await self.execute(question, budget)).result

    async def execute(
        self,
        question: VQAQuestion,
        budget: VQAEvidenceBudget = VQAEvidenceBudget(),
    ) -> VQAExecution:
        if not isinstance(question, VQAQuestion) or not isinstance(budget, VQAEvidenceBudget):
            raise ContractMismatchError("question and budget must be validated public VQA models")
        with self._state:
            if self._closing or self._closed:
                raise ResourceUnavailableError(
                    "VQA orchestrator is closing",
                    details={"resource": "vqa_orchestrator"},
                )
            self._active += 1
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._total_timeout_sec
        started = perf_counter()
        try:
            return await asyncio.wait_for(
                self._execute(question, budget, started, deadline),
                timeout=self._total_timeout_sec,
            )
        except TimeoutError as exc:
            raise BranchTimeoutError("VQA execution exceeded its total timeout") from exc
        finally:
            with self._state:
                self._active -= 1
                self._state.notify_all()

    async def _execute(
        self,
        question: VQAQuestion,
        budget: VQAEvidenceBudget,
        started: float,
        deadline: float,
    ) -> VQAExecution:
        retrieval_started = perf_counter()
        candidates = await self._retrieve_candidates(question)
        retrieval_ms = _elapsed_ms(retrieval_started)
        if not candidates:
            result = self._insufficient_result(question, retrieved_count=0, warnings=("NO_RANKED_FRAMES",))
            return VQAExecution(result, retrieval_ms, 0.0, _elapsed_ms(started))

        loop = asyncio.get_running_loop()
        evidence_started = perf_counter()
        selection = await loop.run_in_executor(
            self._executor,
            partial(self._evidence_selector.select, question, candidates, budget),
        )
        if not isinstance(selection, EvidenceSelectionResult):
            raise ContractMismatchError("evidence selector returned an invalid result")
        evidence_ms = _elapsed_ms(evidence_started)
        if selection.selected_image_count == 0:
            result = self._result_from_selection(
                question,
                selection,
                self._insufficient_response(question),
                vlm_latency_ms=0.0,
                retry_count=0,
                extra_warnings=("NO_IMAGE_EVIDENCE",),
            )
            return VQAExecution(result, retrieval_ms, evidence_ms, _elapsed_ms(started))

        request = build_vlm_request(question, selection.evidence)
        vlm_started = perf_counter()
        retry_count = 0
        while True:
            try:
                response = await self._run_vlm_attempt(request, deadline)
                break
            except BranchTimeoutError:
                raise
            except ContractMismatchError:
                if retry_count >= 1:
                    raise
            except ResourceUnavailableError as exc:
                if retry_count >= 1 or exc.details.get("retryable") is not True:
                    raise
            retry_count += 1
            _remaining_time(deadline)
        vlm_ms = _elapsed_ms(vlm_started)
        result = self._result_from_selection(
            question,
            selection,
            response,
            vlm_latency_ms=vlm_ms,
            retry_count=retry_count,
        )
        return VQAExecution(result, retrieval_ms, evidence_ms, _elapsed_ms(started))

    async def _retrieve_candidates(
        self,
        question: VQAQuestion,
    ) -> tuple[FusedFrameCandidate, ...]:
        try:
            raw = await self._candidate_retriever.retrieve_candidates(question)
        except DataInfrastructureError:
            raise
        except Exception as exc:
            raise ResourceUnavailableError(
                "VQA candidate retrieval failed unexpectedly",
                details={"stage": "candidate_retrieval", "exception_type": type(exc).__name__},
            ) from exc
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise ContractMismatchError("candidate retriever returned a non-sequence value")
        candidates = tuple(raw)
        if any(not isinstance(item, FusedFrameCandidate) for item in candidates):
            raise ContractMismatchError("candidate retriever returned an invalid candidate")
        return candidates

    async def _run_vlm_attempt(self, request: VLMRequest, deadline: float) -> VLMResponse:
        remaining = _remaining_time(deadline)
        attempt_timeout = min(self._vlm_timeout_sec, remaining)
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, self._vlm.answer, request)
        try:
            raw_response = await asyncio.wait_for(future, timeout=attempt_timeout)
        except TimeoutError as exc:
            raise BranchTimeoutError("VLM request exceeded its per-attempt timeout") from exc
        return validate_vlm_response(raw_response, request)

    @staticmethod
    def _insufficient_response(question: VQAQuestion) -> VLMResponse:
        return VLMResponse(
            status=VLMResponseStatus.INSUFFICIENT_EVIDENCE,
            answer_type=question.answer_type,
            confidence=VLMConfidence.LOW,
        )

    @classmethod
    def _insufficient_result(
        cls,
        question: VQAQuestion,
        *,
        retrieved_count: int,
        warnings: tuple[str, ...],
    ) -> VQAResult:
        return VQAResult(
            question_id=question.question_id,
            response=cls._insufficient_response(question),
            diagnostics=VQADiagnostics(retrieved_frame_count=retrieved_count, warnings=warnings),
        )

    @staticmethod
    def _result_from_selection(
        question: VQAQuestion,
        selection: EvidenceSelectionResult,
        response: VLMResponse,
        *,
        vlm_latency_ms: float,
        retry_count: int,
        extra_warnings: tuple[str, ...] = (),
    ) -> VQAResult:
        return VQAResult(
            question_id=question.question_id,
            response=response,
            evidence=selection.evidence,
            diagnostics=VQADiagnostics(
                retrieved_frame_count=selection.retrieved_frame_count,
                selected_image_count=selection.selected_image_count,
                selected_text_evidence_count=selection.selected_text_count,
                dropped_evidence_count=selection.dropped_count,
                missing_evidence_count=selection.missing_count,
                vlm_latency_ms=vlm_latency_ms,
                vlm_retry_count=retry_count,
                warnings=tuple(dict.fromkeys((*selection.warnings, *extra_warnings))),
            ),
        )

    def close(self, *, wait: bool = False) -> None:
        with self._state:
            if self._closed:
                return
            self._closing = True
            if wait:
                while self._active:
                    self._state.wait()
            elif self._active:
                self._closing = False
                raise RuntimeError("VQAOrchestrator cannot close during active execution")
            self._closed = True
        if self._owns_executor:
            self._executor.shutdown(wait=wait, cancel_futures=True)

    async def __aenter__(self) -> "VQAOrchestrator":
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _elapsed_ms(started: float) -> float:
    return max(0.0, (perf_counter() - started) * 1000.0)


def _validate_timeout(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")


def _remaining_time(deadline: float) -> float:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise BranchTimeoutError("VQA execution exceeded its total timeout")
    return remaining


__all__ = ["VQACandidateRetrievalPort", "VQAExecution", "VQAOrchestrator"]
