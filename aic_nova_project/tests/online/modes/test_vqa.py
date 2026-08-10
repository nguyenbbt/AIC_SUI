from __future__ import annotations

import asyncio

import pytest

from online.domain.errors import ContractMismatchError
from online.domain.vqa import (
    VQAAnswerType,
    VLMConfidence,
    VLMResponse,
    VLMResponseStatus,
    VQAQuestion,
    VQAResult,
)
from online.modes.vqa import VQAModeAdapter


def _question() -> VQAQuestion:
    return VQAQuestion(question_id="vqa-1", question="What is visible?", answer_type=VQAAnswerType.SHORT_TEXT)


class _Orchestrator:
    def __init__(self) -> None:
        self.calls = []

    async def answer(self, question, budget):
        self.calls.append((question, budget))
        return VQAResult(
            question_id=question.question_id,
            response=VLMResponse(
                status=VLMResponseStatus.INSUFFICIENT_EVIDENCE,
                answer_type=question.answer_type,
                confidence=VLMConfidence.LOW,
            ),
        )


def test_vqa_mode_preserves_insufficient_evidence_as_result() -> None:
    orchestrator = _Orchestrator()
    result = asyncio.run(VQAModeAdapter(orchestrator).answer(_question()))
    assert result.response.status is VLMResponseStatus.INSUFFICIENT_EVIDENCE
    assert len(orchestrator.calls) == 1


def test_vqa_mode_rejects_non_domain_input() -> None:
    with pytest.raises(ContractMismatchError):
        asyncio.run(VQAModeAdapter(_Orchestrator()).answer(object()))
