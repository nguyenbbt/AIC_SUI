"""Async fake-ready VQA orchestration over ranked candidates and public ports."""

from __future__ import annotations

import asyncio
import math
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from threading import Lock
from time import perf_counter
from typing import Protocol, Sequence, runtime_checkable

from online.domain.candidates import FusedFrameCandidate
from online.domain.errors import BranchTimeoutError, ContractMismatchError
from online.domain.vqa import (
    VLMConfidence,
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
        timeout_sec: float = 30.0,
        max_workers: int = 2,
        executor: Executor | None = None,
    ) -> None:
        if (
            isinstance(timeout_sec, bool)
            or not isinstance(timeout_sec, (int, float))
            or not math.isfinite(timeout_sec)
            or timeout_sec <= 0
        ):
            raise ValueError("timeout_sec must be a positive number")
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        self._candidate_retriever = candidate_retriever
        self._evidence_selector = evidence_selector
        self._vlm = vlm
        self._timeout_sec = float(timeout_sec)
        self._executor = executor or ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="aic-vqa")
        self._owns_executor = executor is None
        self._state_lock = Lock()
        self._active = 0
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
        with self._state_lock:
            if self._closed:
                raise RuntimeError("VQAOrchestrator is closed")
            self._active += 1
        started = perf_counter()
        try:
            return await asyncio.wait_for(
                self._execute(question, budget, started),
                timeout=self._timeout_sec,
            )
        except TimeoutError as exc:
            raise BranchTimeoutError("VQA execution exceeded its total timeout") from exc
        finally:
            with self._state_lock:
                self._active -= 1

    async def _execute(
        self,
        question: VQAQuestion,
        budget: VQAEvidenceBudget,
        started: float,
    ) -> VQAExecution:
        retrieval_started = perf_counter()
        candidates = tuple(await self._candidate_retriever.retrieve_candidates(question))
        if any(not isinstance(item, FusedFrameCandidate) for item in candidates):
            raise ContractMismatchError("candidate retriever returned an invalid value")
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
            raw_response = await loop.run_in_executor(self._executor, self._vlm.answer, request)
            try:
                response = validate_vlm_response(raw_response, request)
                break
            except ContractMismatchError:
                if retry_count >= 1:
                    raise
                retry_count += 1
        vlm_ms = _elapsed_ms(vlm_started)
        result = self._result_from_selection(
            question,
            selection,
            response,
            vlm_latency_ms=vlm_ms,
            retry_count=retry_count,
        )
        return VQAExecution(result, retrieval_ms, evidence_ms, _elapsed_ms(started))

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

    def close(self, *, wait: bool = True) -> None:
        with self._state_lock:
            if self._closed:
                return
            if self._active:
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


__all__ = ["VQACandidateRetrievalPort", "VQAExecution", "VQAOrchestrator"]
