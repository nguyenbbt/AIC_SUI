from __future__ import annotations

import asyncio
import threading
import time
import unittest

from online.domain.candidates import BranchResult
from online.domain.enums import BranchStatus, CandidateLevel, QueryMode, RetrievalBranch
from online.domain.errors import ContractMismatchError, ResourceUnavailableError
from online.domain.query import QueryBundle, TextQueryVariant
from online.retrieval.branches import (
    ASRLexicalBranch,
    ASRSemanticBranch,
    OCRLexicalBranch,
    OCRSemanticBranch,
    SummaryLexicalBranch,
    SummarySemanticBranch,
    VisualSemanticBranch,
)
from online.retrieval.query_builder import BASELINE_KIS_BRANCHES, KISQueryBuilder
from online.retrieval.service import (
    MULTI_VARIANT_BRANCHES,
    RetrievalInvocationConfig,
    RetrievalService,
)
from online.testing import (
    FakeElasticsearchSearchPort,
    FakeMilvusSearchPort,
    FakeTextEncoder,
    build_integration_fixture,
)


LEVELS = {
    RetrievalBranch.VISUAL_DENSE: CandidateLevel.FRAME,
    RetrievalBranch.OCR_DENSE: CandidateLevel.FRAME,
    RetrievalBranch.OCR_BM25: CandidateLevel.FRAME,
    RetrievalBranch.ASR_DENSE: CandidateLevel.ASR_INTERVAL,
    RetrievalBranch.ASR_BM25: CandidateLevel.ASR_INTERVAL,
    RetrievalBranch.SUMMARY_DENSE: CandidateLevel.VIDEO,
    RetrievalBranch.SUMMARY_BM25: CandidateLevel.VIDEO,
}


def make_configs(
    query: QueryBundle,
    *,
    top_k: int = 2,
    timeout_sec: float = 1.0,
):
    configs = {}
    enabled = set(query.enabled_branches)
    for branch in BASELINE_KIS_BRANCHES:
        if branch not in enabled:
            continue
        variants = query.text_variants if branch in MULTI_VARIANT_BRANCHES else query.text_variants[:1]
        for variant in variants:
            configs[(branch, variant.variant_id)] = RetrievalInvocationConfig(
                top_k=top_k,
                timeout_sec=timeout_sec,
            )
    return configs


class ConcurrencyTracker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def enter(self) -> None:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def exit(self) -> None:
        with self.lock:
            self.active -= 1


class ProbeRunner:
    def __init__(
        self,
        branch: RetrievalBranch,
        *,
        delay_sec: float = 0.0,
        error: Exception | None = None,
        tracker: ConcurrencyTracker | None = None,
    ) -> None:
        self.branch = branch
        self.delay_sec = delay_sec
        self.error = error
        self.tracker = tracker
        self.calls: list[tuple[str, int]] = []

    def retrieve_variant(self, variant: TextQueryVariant, *, top_k: int):
        self.calls.append((variant.variant_id, top_k))
        if self.tracker is not None:
            self.tracker.enter()
        try:
            if self.delay_sec:
                time.sleep(self.delay_sec)
            if self.error is not None:
                raise self.error
            return BranchResult(
                branch=self.branch,
                candidate_level=LEVELS[self.branch],
                query_variant_id=variant.variant_id,
                candidates=(),
                requested_top_k=top_k,
                latency_ms=self.delay_sec * 1000.0,
                status=BranchStatus.SUCCESS,
            )
        finally:
            if self.tracker is not None:
                self.tracker.exit()


class WrongResultRunner(ProbeRunner):
    def retrieve_variant(self, variant: TextQueryVariant, *, top_k: int):
        self.calls.append((variant.variant_id, top_k))
        return BranchResult(
            branch=RetrievalBranch.OCR_BM25,
            candidate_level=CandidateLevel.FRAME,
            query_variant_id=variant.variant_id,
            candidates=(),
            requested_top_k=top_k,
            latency_ms=0.0,
            status=BranchStatus.SUCCESS,
        )


class RetrievalServiceTests(unittest.TestCase):
    def test_all_seven_branches_return_canonical_results_and_diagnostics(self) -> None:
        fixture = build_integration_fixture()
        milvus = FakeMilvusSearchPort(
            visual=fixture.visual_hits[:2],
            ocr=fixture.ocr_hits,
            asr=fixture.asr_hits,
            summary=fixture.summary_hits,
        )
        elasticsearch = FakeElasticsearchSearchPort(
            ocr=fixture.ocr_hits,
            asr=fixture.asr_hits,
            summary=fixture.summary_hits,
        )
        visual_encoder = FakeTextEncoder(dimension=4)
        vietnamese_encoder = FakeTextEncoder(dimension=6)
        branches = {
            RetrievalBranch.VISUAL_DENSE: VisualSemanticBranch(
                encoder=visual_encoder,
                milvus=milvus,
                metadata=fixture.metadata(),
            ),
            RetrievalBranch.OCR_DENSE: OCRSemanticBranch(
                encoder=vietnamese_encoder,
                milvus=milvus,
                metadata=fixture.metadata(),
            ),
            RetrievalBranch.OCR_BM25: OCRLexicalBranch(
                elasticsearch=elasticsearch,
                metadata=fixture.metadata(),
            ),
            RetrievalBranch.ASR_DENSE: ASRSemanticBranch(
                encoder=vietnamese_encoder,
                milvus=milvus,
            ),
            RetrievalBranch.ASR_BM25: ASRLexicalBranch(elasticsearch=elasticsearch),
            RetrievalBranch.SUMMARY_DENSE: SummarySemanticBranch(
                encoder=vietnamese_encoder,
                milvus=milvus,
            ),
            RetrievalBranch.SUMMARY_BM25: SummaryLexicalBranch(
                elasticsearch=elasticsearch
            ),
        }
        query = KISQueryBuilder().build(
            "original query",
            mode=QueryMode.KIS_TEXT,
            paraphrases=("one paraphrase",),
            query_id="query-service",
        )
        service = RetrievalService(
            branches=branches,
            invocation_configs=make_configs(query),
            max_workers=7,
        )
        try:
            execution = asyncio.run(service.execute(query))
        finally:
            service.close(wait=True)

        expected_order = (
            (RetrievalBranch.VISUAL_DENSE, "q0"),
            (RetrievalBranch.VISUAL_DENSE, "q1"),
            (RetrievalBranch.OCR_DENSE, "q0"),
            (RetrievalBranch.OCR_DENSE, "q1"),
            (RetrievalBranch.OCR_BM25, "q0"),
            (RetrievalBranch.ASR_DENSE, "q0"),
            (RetrievalBranch.ASR_DENSE, "q1"),
            (RetrievalBranch.ASR_BM25, "q0"),
            (RetrievalBranch.SUMMARY_DENSE, "q0"),
            (RetrievalBranch.SUMMARY_DENSE, "q1"),
            (RetrievalBranch.SUMMARY_BM25, "q0"),
        )
        self.assertEqual(
            tuple((result.branch, result.query_variant_id) for result in execution.results),
            expected_order,
        )
        self.assertEqual(
            tuple((item.branch, item.query_variant_id) for item in execution.invocations),
            expected_order,
        )
        self.assertEqual(execution.query_id, "query-service")
        self.assertGreaterEqual(execution.total_latency_ms, 0.0)
        self.assertTrue(all(result.status is BranchStatus.SUCCESS for result in execution.results))
        self.assertTrue(all(result.returned_count == 2 for result in execution.results))
        self.assertTrue(
            all(
                item.metrics.output_candidate_count == 2
                and item.metrics.raw_result_count == 2
                and item.metrics.mapping_loss_count == 0
                for item in execution.invocations
            )
        )
        for result in execution.results:
            for candidate in result.candidates:
                self.assertEqual(candidate.provenance.branch, result.branch)
                self.assertEqual(
                    candidate.provenance.query_variant_id,
                    result.query_variant_id,
                )

    def test_sync_runners_execute_concurrently_but_results_stay_canonical(self) -> None:
        tracker = ConcurrencyTracker()
        visual = ProbeRunner(
            RetrievalBranch.VISUAL_DENSE,
            delay_sec=0.06,
            tracker=tracker,
        )
        ocr = ProbeRunner(
            RetrievalBranch.OCR_BM25,
            delay_sec=0.02,
            tracker=tracker,
        )
        query = KISQueryBuilder().build(
            "query",
            mode=QueryMode.KIS_TEXT,
            enabled_branches=(RetrievalBranch.OCR_BM25, RetrievalBranch.VISUAL_DENSE),
        )
        service = RetrievalService(
            branches={
                RetrievalBranch.OCR_BM25: ocr,
                RetrievalBranch.VISUAL_DENSE: visual,
            },
            invocation_configs=make_configs(query),
            max_workers=2,
        )
        try:
            results = asyncio.run(service.retrieve(query))
        finally:
            service.close(wait=True)

        self.assertGreaterEqual(tracker.max_active, 2)
        self.assertEqual(
            tuple(result.branch for result in results),
            (RetrievalBranch.VISUAL_DENSE, RetrievalBranch.OCR_BM25),
        )

    def test_timeout_marks_core_failed_and_optional_degraded(self) -> None:
        visual = ProbeRunner(RetrievalBranch.VISUAL_DENSE, delay_sec=0.08)
        ocr = ProbeRunner(RetrievalBranch.OCR_DENSE, delay_sec=0.08)
        query = KISQueryBuilder().build(
            "query",
            mode=QueryMode.KIS_TEXT,
            enabled_branches=(RetrievalBranch.VISUAL_DENSE, RetrievalBranch.OCR_DENSE),
        )
        configs = make_configs(query, timeout_sec=0.01)
        service = RetrievalService(
            branches={
                RetrievalBranch.VISUAL_DENSE: visual,
                RetrievalBranch.OCR_DENSE: ocr,
            },
            invocation_configs=configs,
            max_workers=2,
        )
        try:
            execution = asyncio.run(service.execute(query))
        finally:
            service.close(wait=True)

        self.assertEqual(
            tuple(result.status for result in execution.results),
            (BranchStatus.FAILED, BranchStatus.DEGRADED),
        )
        self.assertTrue(
            all(result.warnings == ("BRANCH_TIMEOUT",) for result in execution.results)
        )
        self.assertTrue(
            all(item.metrics.warnings == ("BRANCH_TIMEOUT",) for item in execution.invocations)
        )

    def test_core_and_optional_failures_are_safe_and_distinct(self) -> None:
        visual = ProbeRunner(
            RetrievalBranch.VISUAL_DENSE,
            error=ResourceUnavailableError("secret visual backend detail"),
        )
        summary = ProbeRunner(
            RetrievalBranch.SUMMARY_BM25,
            error=ResourceUnavailableError("secret summary backend detail"),
        )
        query = KISQueryBuilder().build(
            "query",
            mode=QueryMode.KIS_TEXT,
            enabled_branches=(
                RetrievalBranch.SUMMARY_BM25,
                RetrievalBranch.VISUAL_DENSE,
            ),
        )
        service = RetrievalService(
            branches={
                RetrievalBranch.VISUAL_DENSE: visual,
                RetrievalBranch.SUMMARY_BM25: summary,
            },
            invocation_configs=make_configs(query),
            max_workers=2,
        )
        try:
            results = asyncio.run(service.retrieve(query))
        finally:
            service.close(wait=True)

        self.assertEqual(
            tuple(result.status for result in results),
            (BranchStatus.FAILED, BranchStatus.DEGRADED),
        )
        self.assertTrue(
            all(result.warnings == ("RESOURCE_UNAVAILABLE",) for result in results)
        )
        self.assertTrue(all("secret" not in str(result.warnings) for result in results))

    def test_empty_success_is_not_converted_to_failure(self) -> None:
        visual = ProbeRunner(RetrievalBranch.VISUAL_DENSE)
        query = KISQueryBuilder().build(
            "query",
            mode=QueryMode.KIS_TEXT,
            enabled_branches=(RetrievalBranch.VISUAL_DENSE,),
        )
        service = RetrievalService(
            branches={RetrievalBranch.VISUAL_DENSE: visual},
            invocation_configs=make_configs(query),
            max_workers=1,
        )
        try:
            execution = asyncio.run(service.execute(query))
        finally:
            service.close(wait=True)

        self.assertEqual(execution.results[0].status, BranchStatus.SUCCESS)
        self.assertEqual(execution.results[0].candidates, ())
        self.assertEqual(execution.results[0].warnings, ())
        self.assertEqual(execution.invocations[0].metrics.output_candidate_count, 0)

    def test_invalid_branch_handoff_is_reported_as_contract_mismatch(self) -> None:
        visual = WrongResultRunner(RetrievalBranch.VISUAL_DENSE)
        query = KISQueryBuilder().build(
            "query",
            mode=QueryMode.KIS_TEXT,
            enabled_branches=(RetrievalBranch.VISUAL_DENSE,),
        )
        service = RetrievalService(
            branches={RetrievalBranch.VISUAL_DENSE: visual},
            invocation_configs=make_configs(query),
            max_workers=1,
        )
        try:
            results = asyncio.run(service.retrieve(query))
        finally:
            service.close(wait=True)

        self.assertEqual(results[0].branch, RetrievalBranch.VISUAL_DENSE)
        self.assertEqual(results[0].status, BranchStatus.FAILED)
        self.assertEqual(results[0].warnings, ("CONTRACT_MISMATCH",))

    def test_disabled_branch_is_not_run_and_does_not_need_config(self) -> None:
        visual = ProbeRunner(RetrievalBranch.VISUAL_DENSE)
        disabled = ProbeRunner(RetrievalBranch.OCR_BM25)
        query = KISQueryBuilder().build(
            "query",
            mode=QueryMode.KIS_TEXT,
            enabled_branches=(RetrievalBranch.VISUAL_DENSE,),
        )
        service = RetrievalService(
            branches={
                RetrievalBranch.VISUAL_DENSE: visual,
                RetrievalBranch.OCR_BM25: disabled,
            },
            invocation_configs=make_configs(query),
            max_workers=1,
        )
        try:
            results = asyncio.run(service.retrieve(query))
        finally:
            service.close(wait=True)

        self.assertEqual(len(results), 1)
        self.assertEqual(visual.calls, [("q0", 2)])
        self.assertEqual(disabled.calls, [])

    def test_missing_exact_variant_config_fails_before_any_branch_runs(self) -> None:
        visual = ProbeRunner(RetrievalBranch.VISUAL_DENSE)
        query = KISQueryBuilder().build(
            "query",
            mode=QueryMode.KIS_TEXT,
            paraphrases=("paraphrase",),
            enabled_branches=(RetrievalBranch.VISUAL_DENSE,),
        )
        configs = {
            (RetrievalBranch.VISUAL_DENSE, "q0"): RetrievalInvocationConfig(
                top_k=2,
                timeout_sec=1.0,
            )
        }
        service = RetrievalService(
            branches={RetrievalBranch.VISUAL_DENSE: visual},
            invocation_configs=configs,
            max_workers=1,
        )
        try:
            with self.assertRaises(ContractMismatchError) as raised:
                asyncio.run(service.retrieve(query))
        finally:
            service.close(wait=True)

        self.assertEqual(
            raised.exception.details,
            {"branch": "visual_dense", "query_variant_id": "q1"},
        )
        self.assertEqual(visual.calls, [])

    def test_invalid_service_configuration_is_rejected(self) -> None:
        visual = ProbeRunner(RetrievalBranch.VISUAL_DENSE)
        with self.assertRaises(ValueError):
            RetrievalService(
                branches={RetrievalBranch.VISUAL_DENSE: visual},
                invocation_configs={},
                max_workers=0,
            )
        with self.assertRaises(ValueError):
            RetrievalService(
                branches={RetrievalBranch.OCR_DENSE: visual},
                invocation_configs={},
                max_workers=1,
            )
        with self.assertRaises(ValueError):
            RetrievalInvocationConfig(top_k=0, timeout_sec=1.0)


if __name__ == "__main__":
    unittest.main()
