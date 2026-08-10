"""Thin VQA mode boundary over the evidence-first Wave-2 orchestrator."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from online.domain.errors import ContractMismatchError
from online.domain.vqa import VQAEvidenceBudget, VQAQuestion, VQAResult


@runtime_checkable
class VQAOrchestratorPort(Protocol):
    async def answer(
        self,
        question: VQAQuestion,
        budget: VQAEvidenceBudget = VQAEvidenceBudget(),
    ) -> VQAResult: ...


class VQAModeAdapter:
    """Validate VQA input and preserve the orchestrator's domain result."""

    def __init__(self, orchestrator: VQAOrchestratorPort) -> None:
        if not isinstance(orchestrator, VQAOrchestratorPort):
            raise TypeError("orchestrator must implement VQAOrchestratorPort")
        self._orchestrator = orchestrator

    async def answer(
        self,
        question: VQAQuestion,
        budget: VQAEvidenceBudget = VQAEvidenceBudget(),
    ) -> VQAResult:
        if not isinstance(question, VQAQuestion) or not isinstance(budget, VQAEvidenceBudget):
            raise ContractMismatchError(
                "question and budget must be validated public VQA models"
            )
        result = await self._orchestrator.answer(question, budget)
        if not isinstance(result, VQAResult):
            raise ContractMismatchError("VQA orchestrator returned an invalid result")
        if result.question_id != question.question_id:
            raise ContractMismatchError("VQA orchestrator changed question_id")
        return result

    async def execute(
        self,
        question: VQAQuestion,
        budget: VQAEvidenceBudget = VQAEvidenceBudget(),
    ) -> VQAResult:
        return await self.answer(question, budget)

    def close(self, *, wait: bool = True) -> None:
        close = getattr(self._orchestrator, "close", None)
        if callable(close):
            close(wait=wait)


__all__ = ["VQAModeAdapter", "VQAOrchestratorPort"]
