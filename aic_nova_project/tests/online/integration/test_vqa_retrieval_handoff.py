from __future__ import annotations

import asyncio

import pytest

from online.config import OnlineDataConfig
from online.domain.candidates import FusedFrameCandidate
from online.domain.enums import RetrievalBranch
from online.domain.errors import BranchTimeoutError
from online.domain.vqa import VQAAnswerType, VQAQuestion
from online.modes.kis import KISRankingService, KISSearchOrchestrator
from online.retrieval import (
    BASELINE_KIS_BRANCHES,
    RetrievalInvocationConfig,
    VQACandidateRetriever,
    build_retrieval_service,
)
from online.testing import (
    FakeBranchBehavior,
    FakeTextEncoder,
    build_integration_fixture,
)
from online.vqa import VQACandidateRetrievalPort
from query_understanding import (
    MappingQueryRewriter,
    QueryRewriteProposal,
    QueryRewriteService,
    RewritePurpose,
    RewriteStatus,
)


class _SlowRewriter:
    async def rewrite(self, request):
        await asyncio.sleep(0.05)
        return QueryRewriteProposal(primary_text="late evidence rewrite")


def _question(
    question_id: str = "vqa-handoff",
    text: str = "Người trong video đang đi bằng phương tiện gì?",
) -> VQAQuestion:
    return VQAQuestion(
        question_id=question_id,
        question=text,
        answer_type=VQAAnswerType.SHORT_TEXT,
    )


def _invocation_configs(
    *,
    timeout_sec: float = 1.0,
) -> dict[tuple[RetrievalBranch, str], RetrievalInvocationConfig]:
    return {
        (branch, f"q{index}"): RetrievalInvocationConfig(
            top_k=5,
            timeout_sec=timeout_sec,
        )
        for branch in BASELINE_KIS_BRANCHES
        for index in range(3)
    }


def _build_runtime(
    rewriter: QueryRewriteService,
    *,
    branch_timeout_sec: float = 1.0,
    milvus_behaviors=None,
):
    fixture = build_integration_fixture()
    metadata = fixture.metadata()
    milvus = fixture.milvus(behaviors=milvus_behaviors)
    elasticsearch = fixture.elasticsearch()
    visual_encoder = FakeTextEncoder(dimension=4)
    vietnamese_encoder = FakeTextEncoder(dimension=6)
    retrieval = build_retrieval_service(
        data_config=OnlineDataConfig(),
        milvus=milvus,
        elasticsearch=elasticsearch,
        metadata=metadata,
        visual_encoder=visual_encoder,
        vietnamese_encoder=vietnamese_encoder,
        invocation_configs=_invocation_configs(timeout_sec=branch_timeout_sec),
        max_workers=8,
    )
    search = KISSearchOrchestrator(
        retrieval=retrieval,
        ranking=KISRankingService(metadata=metadata),
    )
    adapter = VQACandidateRetriever(rewriter=rewriter, kis_search=search)
    return (
        adapter,
        search,
        retrieval,
        milvus,
        elasticsearch,
        visual_encoder,
        vietnamese_encoder,
    )


def _close_runtime(search, retrieval) -> None:
    search.close(wait=True)
    retrieval.close(wait=True)


def test_vqa_question_reaches_real_kis_seven_branch_ranking_handoff() -> None:
    question = _question()
    evidence_query = "cảnh một người đang sử dụng phương tiện để di chuyển"
    rewriter = QueryRewriteService(
        MappingQueryRewriter(
            {
                (RewritePurpose.VQA_EVIDENCE, question.question): QueryRewriteProposal(
                    primary_text=evidence_query,
                    paraphrases=("người đang điều khiển một phương tiện",),
                )
            }
        )
    )
    (
        adapter,
        search,
        retrieval,
        milvus,
        elasticsearch,
        visual_encoder,
        vietnamese_encoder,
    ) = _build_runtime(rewriter)

    try:
        execution = asyncio.run(adapter.execute(question))
    finally:
        _close_runtime(search, retrieval)

    assert isinstance(adapter, VQACandidateRetrievalPort)
    assert execution.rewrite.status is RewriteStatus.SUCCESS
    assert execution.query_bundle.original_query == evidence_query
    assert tuple(item.text for item in execution.query_bundle.text_variants) == (
        evidence_query,
        "người đang điều khiển một phương tiện",
    )
    assert execution.candidates
    assert all(isinstance(item, FusedFrameCandidate) for item in execution.candidates)
    assert tuple(item.final_score for item in execution.candidates) == tuple(
        sorted((item.final_score for item in execution.candidates), reverse=True)
    )
    assert set(execution.kis_diagnostics.branches) == set(BASELINE_KIS_BRANCHES)
    assert execution.kis_diagnostics.query_id == "vqa:vqa-handoff"
    assert {call.query for call in elasticsearch.calls} == {evidence_query}
    assert len(milvus.calls) == 8
    assert len(visual_encoder.calls) == 2
    assert len(vietnamese_encoder.calls) == 6


def test_rewrite_timeout_degrades_to_original_question_and_still_retrieves() -> None:
    question = _question()
    runtime = _build_runtime(
        QueryRewriteService(_SlowRewriter(), timeout_sec=0.001)
    )
    adapter, search, retrieval, _, elasticsearch, _, _ = runtime

    try:
        execution = asyncio.run(adapter.execute(question))
    finally:
        _close_runtime(search, retrieval)

    assert execution.rewrite.status is RewriteStatus.DEGRADED
    assert execution.rewrite.warnings == ("QUERY_REWRITE_TIMEOUT",)
    assert execution.query_bundle.original_query == question.question
    assert len(execution.query_bundle.text_variants) == 1
    assert execution.candidates
    assert {call.query for call in elasticsearch.calls} == {question.question}


def test_concurrent_vqa_retrievals_keep_query_ids_and_rewrites_isolated() -> None:
    questions = tuple(
        _question(f"concurrent-{index}", f"Câu hỏi bằng chứng số {index}?")
        for index in range(4)
    )
    rewriter = QueryRewriteService(
        MappingQueryRewriter(
            {
                (RewritePurpose.VQA_EVIDENCE, question.question): QueryRewriteProposal(
                    primary_text=f"mô tả evidence số {index}"
                )
                for index, question in enumerate(questions)
            }
        )
    )
    runtime = _build_runtime(rewriter)
    adapter, search, retrieval, *_ = runtime

    async def scenario():
        return await asyncio.gather(*(adapter.execute(question) for question in questions))

    try:
        executions = asyncio.run(scenario())
    finally:
        _close_runtime(search, retrieval)

    assert tuple(item.query_bundle.query_id for item in executions) == tuple(
        f"vqa:{question.question_id}" for question in questions
    )
    assert tuple(item.query_bundle.original_query for item in executions) == tuple(
        f"mô tả evidence số {index}" for index in range(4)
    )
    assert all(item.candidates for item in executions)
    assert all(
        item.kis_diagnostics.query_id == item.query_bundle.query_id
        for item in executions
    )


def test_core_visual_timeout_remains_a_typed_kis_failure() -> None:
    question = _question()
    rewriter = QueryRewriteService(
        MappingQueryRewriter(
            {
                (RewritePurpose.VQA_EVIDENCE, question.question): QueryRewriteProposal(
                    primary_text="visual evidence"
                )
            }
        )
    )
    runtime = _build_runtime(
        rewriter,
        branch_timeout_sec=0.001,
        milvus_behaviors={
            RetrievalBranch.VISUAL_DENSE: FakeBranchBehavior(delay_sec=0.05)
        },
    )
    adapter, search, retrieval, *_ = runtime

    try:
        with pytest.raises(BranchTimeoutError):
            asyncio.run(adapter.execute(question))
    finally:
        _close_runtime(search, retrieval)
