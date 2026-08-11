from __future__ import annotations

import asyncio

import pytest

from online.domain.errors import ResourceUnavailableError
from online.domain.vqa import VQAAnswerType, VQAQuestion
from query_understanding.rewrite import (
    MappingQueryRewriter,
    NoOpQueryRewriter,
    QueryRewriteProposal,
    QueryRewriteRequest,
    QueryRewriteService,
    RewritePurpose,
    RewriteStatus,
)


class _SlowRewriter:
    async def rewrite(self, request):
        await asyncio.sleep(0.05)
        return QueryRewriteProposal(primary_text="too late")


class _UnavailableRewriter:
    async def rewrite(self, request):
        raise ResourceUnavailableError(
            "provider failed with token=secret-value",
            details={"api_token": "secret-value"},
        )


class _MalformedRewriter:
    async def rewrite(self, request):
        return {"primary_text": "unstructured output"}


def _vqa_question(question_id: str = "question-1") -> VQAQuestion:
    return VQAQuestion(
        question_id=question_id,
        question="Người đàn ông đang cầm vật gì?",
        answer_type=VQAAnswerType.SHORT_TEXT,
    )


def test_kis_rewrite_keeps_q0_and_normalizes_at_most_q1_q2() -> None:
    original = "Một người đang đi xe đạp"
    provider = MappingQueryRewriter(
        {
            (RewritePurpose.KIS, original): QueryRewriteProposal(
                primary_text="  người đạp xe trên đường  ",
                paraphrases=(
                    "người đạp xe trên đường",
                    " ",
                    original,
                    "một người điều khiển xe đạp",
                    "variant must be capped",
                ),
                provider_id="fake-provider",
                model_id="fake-model-v1",
                prompt_version="kis-v1",
            )
        }
    )

    result = asyncio.run(
        QueryRewriteService(provider).rewrite_kis(original, request_id="kis-1")
    )

    assert result.status is RewriteStatus.SUCCESS
    assert result.primary_text == original
    assert result.paraphrases == (
        "người đạp xe trên đường",
        "một người điều khiển xe đạp",
    )
    assert result.variants == (original, *result.paraphrases)
    assert result.warnings == ()
    assert result.provider_id == "fake-provider"
    assert provider.calls[0].purpose is RewritePurpose.KIS


def test_vqa_rewrite_uses_visual_evidence_description_without_answering() -> None:
    question = _vqa_question()
    evidence_query = "cảnh người đàn ông và vật đang được cầm trên tay"
    provider = MappingQueryRewriter(
        {
            (RewritePurpose.VQA_EVIDENCE, question.question): QueryRewriteProposal(
                primary_text=evidence_query,
                paraphrases=(
                    "khung hình cho thấy đồ vật trong tay người đàn ông",
                    "cảnh vật thể được người đàn ông cầm",
                ),
            )
        }
    )

    result = asyncio.run(QueryRewriteService(provider).rewrite_vqa(question))

    assert result.status is RewriteStatus.SUCCESS
    assert result.primary_text == evidence_query
    assert result.paraphrases == (
        "khung hình cho thấy đồ vật trong tay người đàn ông",
        "cảnh vật thể được người đàn ông cầm",
    )
    assert "chai nước" not in " ".join(result.variants)
    assert provider.calls[0].answer_type == VQAAnswerType.SHORT_TEXT.value


@pytest.mark.parametrize(
    ("rewriter", "warning"),
    (
        (NoOpQueryRewriter(), "QUERY_REWRITE_NOOP"),
        (_SlowRewriter(), "QUERY_REWRITE_TIMEOUT"),
        (_UnavailableRewriter(), "QUERY_REWRITE_UNAVAILABLE"),
        (_MalformedRewriter(), "QUERY_REWRITE_INVALID_OUTPUT"),
    ),
)
def test_provider_failure_modes_degrade_to_original_q0(rewriter, warning) -> None:
    timeout = 0.001 if isinstance(rewriter, _SlowRewriter) else 1.0
    question = _vqa_question()

    result = asyncio.run(
        QueryRewriteService(rewriter, timeout_sec=timeout).rewrite_vqa(question)
    )

    assert result.status is RewriteStatus.DEGRADED
    assert result.primary_text == question.question
    assert result.paraphrases == ()
    assert warning in result.warnings
    serialized = repr(result)
    assert "secret-value" not in serialized
    assert "token=" not in serialized


def test_invalid_or_secret_like_metadata_is_omitted_from_diagnostics() -> None:
    query = "query"
    provider = MappingQueryRewriter(
        {
            (RewritePurpose.KIS, query): QueryRewriteProposal(
                primary_text="paraphrase",
                provider_id="https://provider.invalid?api_key=secret",
                model_id="model with spaces",
                prompt_version="safe-v1",
            )
        }
    )

    result = asyncio.run(
        QueryRewriteService(provider).rewrite_kis(query, request_id="safe-meta")
    )

    assert result.provider_id is None
    assert result.model_id is None
    assert result.prompt_version == "safe-v1"
    assert result.warnings == ("QUERY_REWRITE_METADATA_OMITTED",)
    assert "secret" not in repr(result)


def test_concurrent_rewrites_keep_request_scoped_results() -> None:
    questions = tuple(_vqa_question(f"question-{index}") for index in range(8))
    mapping = {
        (RewritePurpose.VQA_EVIDENCE, question.question + f" {index}"): QueryRewriteProposal(
            primary_text=f"evidence-{index}"
        )
        for index, question in enumerate(questions)
    }
    rewritten_questions = tuple(
        question.model_copy(update={"question": question.question + f" {index}"})
        for index, question in enumerate(questions)
    )
    service = QueryRewriteService(MappingQueryRewriter(mapping))

    async def scenario():
        return await asyncio.gather(
            *(service.rewrite_vqa(question) for question in rewritten_questions)
        )

    results = asyncio.run(scenario())

    assert tuple(result.request_id for result in results) == tuple(
        question.question_id for question in rewritten_questions
    )
    assert tuple(result.primary_text for result in results) == tuple(
        f"evidence-{index}" for index in range(8)
    )


def test_rewrite_request_and_service_configuration_are_strict() -> None:
    with pytest.raises(ValueError):
        QueryRewriteRequest(request_id=" ", purpose=RewritePurpose.KIS, text="query")
    with pytest.raises(ValueError):
        QueryRewriteRequest(
            request_id="x",
            purpose=RewritePurpose.KIS,
            text="query",
            answer_type="number",
        )
    with pytest.raises(ValueError):
        QueryRewriteService(timeout_sec=float("nan"))
    with pytest.raises(ValueError):
        QueryRewriteService(max_paraphrases=3)


def test_zero_paraphrase_policy_keeps_only_original_kis_query() -> None:
    provider = MappingQueryRewriter(
        {
            (RewritePurpose.KIS, "query"): QueryRewriteProposal(
                primary_text="usable rewrite"
            )
        }
    )

    result = asyncio.run(
        QueryRewriteService(provider, max_paraphrases=0).rewrite_kis(
            "query",
            request_id="no-paraphrases",
        )
    )

    assert result.status is RewriteStatus.DEGRADED
    assert result.variants == ("query",)
    assert result.warnings == ("QUERY_REWRITE_NO_USABLE_VARIANTS",)


def test_prefix_only_and_near_duplicate_variants_are_rejected() -> None:
    original = "Một người mặc áo đỏ đứng cạnh ô tô"
    provider = MappingQueryRewriter(
        {
            (RewritePurpose.KIS, original): QueryRewriteProposal(
                primary_text=f"Khung hình có {original}",
                paraphrases=(
                    f"visual scene: {original}",
                    "A person wearing a red shirt standing next to a car",
                ),
            )
        }
    )

    result = asyncio.run(
        QueryRewriteService(provider).rewrite_kis(original, request_id="dedupe")
    )

    assert result.status is RewriteStatus.SUCCESS
    assert result.paraphrases == (
        "A person wearing a red shirt standing next to a car",
    )


def test_all_semantic_duplicates_degrade_to_q0() -> None:
    original = "người đứng cạnh xe"
    provider = MappingQueryRewriter(
        {
            (RewritePurpose.KIS, original): QueryRewriteProposal(
                primary_text=f"Cảnh cho thấy {original}",
                paraphrases=(f"image showing {original}",),
            )
        }
    )

    result = asyncio.run(
        QueryRewriteService(provider).rewrite_kis(original, request_id="all-duplicate")
    )

    assert result.status is RewriteStatus.DEGRADED
    assert result.variants == (original,)
    assert result.warnings == ("QUERY_REWRITE_NO_USABLE_VARIANTS",)
