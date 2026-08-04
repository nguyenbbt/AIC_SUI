from __future__ import annotations

import asyncio

import pytest

from online.domain.candidates import (
    CandidateDiagnostics,
    CandidateEvidence,
    FusedFrameCandidate,
)
from online.domain.diagnostics import BranchDiagnostics, QueryDiagnostics
from online.domain.enums import BranchStatus, QueryMode, RetrievalBranch
from online.domain.errors import (
    BranchTimeoutError,
    ContractMismatchError,
    ResourceUnavailableError,
)
from online.domain.vqa import VQAAnswerType, VQAQuestion
from online.modes.kis import KISSearchResult
from online.retrieval.vqa import KISSearchPort, VQACandidateRetriever
from online.vqa.orchestrator import VQACandidateRetrievalPort
from query_understanding.rewrite import (
    MappingQueryRewriter,
    NoOpQueryRewriter,
    QueryRewriteProposal,
    QueryRewriteService,
    RewritePurpose,
    RewriteStatus,
)


def _question(question_id: str = "q-vqa") -> VQAQuestion:
    return VQAQuestion(
        question_id=question_id,
        question="Người phụ nữ đang đứng cạnh phương tiện gì?",
        answer_type=VQAAnswerType.SHORT_TEXT,
    )


def _candidate(
    frame_id: str = "L21_V001_001",
    *,
    score: float = 0.9,
) -> FusedFrameCandidate:
    return FusedFrameCandidate(
        frame_id=frame_id,
        video_id="L21_V001",
        keyframe_no=int(frame_id.rsplit("_", 1)[1]),
        local_index=int(frame_id.rsplit("_", 1)[1]) - 1,
        timestamp_sec=1.5,
        source_frame_idx=15,
        final_score=score,
        branch_scores={RetrievalBranch.VISUAL_DENSE: score},
        evidence=(
            CandidateEvidence(
                branch=RetrievalBranch.VISUAL_DENSE,
                query_variant_id="q0",
                raw_score=score,
                normalized_score=score,
                backend="milvus",
                source_resource="visual_features",
            ),
        ),
        diagnostics=CandidateDiagnostics(),
    )


def _diagnostics(query_id: str) -> QueryDiagnostics:
    return QueryDiagnostics(
        query_id=query_id,
        total_latency_ms=1.0,
        stage_latencies_ms={"ranking": 1.0},
        branches={
            RetrievalBranch.VISUAL_DENSE: BranchDiagnostics(
                status=BranchStatus.SUCCESS,
                latency_ms=0.5,
                requested_top_k=10,
                raw_result_count=1,
                output_candidate_count=1,
            )
        },
        normalization_method="test",
        fusion_method="test",
        fusion_weights={RetrievalBranch.VISUAL_DENSE: 1.0},
    )


class _RecordingKISSearch:
    def __init__(self, candidates=()) -> None:
        self._candidates = tuple(candidates)
        self.calls = []

    async def search(self, bundle):
        self.calls.append(bundle)
        return KISSearchResult(
            candidates=self._candidates,
            diagnostics=_diagnostics(bundle.query_id),
        )


class _FailingKISSearch:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def search(self, bundle):
        raise self.error


class _InvalidKISSearch:
    async def search(self, bundle):
        return {"candidates": ()}


def _rewriter(question: VQAQuestion) -> QueryRewriteService:
    return QueryRewriteService(
        MappingQueryRewriter(
            {
                (RewritePurpose.VQA_EVIDENCE, question.question): QueryRewriteProposal(
                    primary_text="cảnh người phụ nữ đứng cạnh một phương tiện",
                    paraphrases=(
                        "khung hình có người phụ nữ và phương tiện bên cạnh",
                        "người phụ nữ đứng gần xe",
                    ),
                    provider_id="fake-vqa-rewriter",
                )
            }
        )
    )


def test_adapter_rewrites_question_and_calls_shared_kis_pipeline_once() -> None:
    question = _question()
    candidates = (_candidate(),)
    kis_search = _RecordingKISSearch(candidates)
    adapter = VQACandidateRetriever(
        rewriter=_rewriter(question),
        kis_search=kis_search,
    )

    execution = asyncio.run(adapter.execute(question))

    assert isinstance(kis_search, KISSearchPort)
    assert isinstance(adapter, VQACandidateRetrievalPort)
    assert len(kis_search.calls) == 1
    bundle = kis_search.calls[0]
    assert bundle.query_id == "vqa:q-vqa"
    assert bundle.mode is QueryMode.KIS_TEXT
    assert bundle.original_query == "cảnh người phụ nữ đứng cạnh một phương tiện"
    assert tuple(item.variant_id for item in bundle.text_variants) == ("q0", "q1", "q2")
    assert tuple(item.text for item in bundle.text_variants) == (
        "cảnh người phụ nữ đứng cạnh một phương tiện",
        "khung hình có người phụ nữ và phương tiện bên cạnh",
        "người phụ nữ đứng gần xe",
    )
    assert execution.rewrite.status is RewriteStatus.SUCCESS
    assert execution.query_bundle is bundle
    assert execution.candidates == candidates
    assert execution.kis_diagnostics.query_id == bundle.query_id


def test_public_handoff_returns_immutable_ranked_candidates() -> None:
    question = _question()
    candidates = (
        _candidate(score=0.95),
        _candidate("L21_V001_002", score=0.75),
    )
    adapter = VQACandidateRetriever(
        rewriter=_rewriter(question),
        kis_search=_RecordingKISSearch(candidates),
    )

    result = asyncio.run(adapter.retrieve_candidates(question))

    assert isinstance(result, tuple)
    assert result == candidates
    assert tuple(item.final_score for item in result) == (0.95, 0.75)


def test_noop_rewrite_degrades_but_still_searches_with_original_question() -> None:
    question = _question()
    kis_search = _RecordingKISSearch()
    adapter = VQACandidateRetriever(
        rewriter=QueryRewriteService(NoOpQueryRewriter()),
        kis_search=kis_search,
    )

    execution = asyncio.run(adapter.execute(question))

    assert execution.rewrite.status is RewriteStatus.DEGRADED
    assert execution.rewrite.warnings == ("QUERY_REWRITE_NOOP",)
    assert execution.query_bundle.original_query == question.question
    assert tuple(execution.query_bundle.text_variants) == (
        execution.query_bundle.text_variants[0],
    )
    assert execution.candidates == ()
    assert len(kis_search.calls) == 1


@pytest.mark.parametrize(
    "error",
    (
        BranchTimeoutError("KIS timed out"),
        ResourceUnavailableError("KIS unavailable"),
    ),
)
def test_typed_kis_failures_are_preserved(error) -> None:
    question = _question()
    adapter = VQACandidateRetriever(
        rewriter=_rewriter(question),
        kis_search=_FailingKISSearch(error),
    )

    with pytest.raises(type(error)) as captured:
        asyncio.run(adapter.execute(question))

    assert captured.value is error


def test_unexpected_kis_failure_is_sanitized_to_resource_unavailable() -> None:
    question = _question()
    adapter = VQACandidateRetriever(
        rewriter=_rewriter(question),
        kis_search=_FailingKISSearch(RuntimeError("password=do-not-leak")),
    )

    with pytest.raises(ResourceUnavailableError) as captured:
        asyncio.run(adapter.execute(question))

    safe = captured.value.to_safe_dict()
    assert safe["details"]["stage"] == "vqa_kis_search"
    assert safe["details"]["exception_type"] == "RuntimeError"
    assert "do-not-leak" not in repr(safe)


def test_invalid_kis_result_is_a_contract_mismatch() -> None:
    question = _question()
    adapter = VQACandidateRetriever(
        rewriter=_rewriter(question),
        kis_search=_InvalidKISSearch(),
    )

    with pytest.raises(ContractMismatchError):
        asyncio.run(adapter.execute(question))


def test_constructor_and_question_boundary_reject_wrong_contracts() -> None:
    question = _question()
    with pytest.raises(TypeError):
        VQACandidateRetriever(rewriter=_rewriter(question), kis_search=object())

    adapter = VQACandidateRetriever(
        rewriter=_rewriter(question),
        kis_search=_RecordingKISSearch(),
    )
    with pytest.raises(ContractMismatchError):
        asyncio.run(adapter.execute("not-a-question"))
