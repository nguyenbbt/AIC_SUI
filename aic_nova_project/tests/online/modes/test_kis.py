from __future__ import annotations

import asyncio
import threading
import unittest

from online.domain.candidates import (
    ASRIntervalCandidate,
    BranchResult,
    CandidateProvenance,
    FrameCandidate,
    ObjectDetection,
    VideoCandidate,
)
from online.domain.diagnostics import BranchDiagnostics, QueryDiagnostics
from online.domain.enums import BranchStatus, CandidateLevel, CountOperator, FilterMode, QueryMode, RetrievalBranch
from online.domain.errors import (
    BranchTimeoutError,
    ContractMismatchError,
    InvalidQueryError,
    ResourceUnavailableError,
)
from online.domain.query import ObjectConstraint
from online.modes.kis import KISRankingService, KISSearchOrchestrator, KISSearchResult
from online.ports.records import FrameMetadata
from online.ranking.asr_mapper import ASRIntervalFrameMapper, ASRMappingConfig
from online.retrieval.query_builder import KISQueryBuilder
from online.testing import FakeMetadataReaderPort, FakeObjectReaderPort


def provenance(branch: RetrievalBranch, *, variant_id: str = "q0") -> CandidateProvenance:
    return CandidateProvenance(
        branch=branch,
        backend="milvus" if branch.value.endswith("dense") else "elasticsearch",
        source_resource=branch.value,
        query_variant_id=variant_id,
        query_text="query",
    )


def frame_candidate(
    frame_id: str,
    *,
    branch: RetrievalBranch,
    rank: int,
    video_id: str,
    shot_id: int,
    timestamp_sec: float,
) -> FrameCandidate:
    return FrameCandidate(
        frame_id=frame_id,
        video_id=video_id,
        shot_id=shot_id,
        timestamp_sec=timestamp_sec,
        rank=rank,
        raw_score=1.0,
        provenance=provenance(branch),
    )


def frame_result(
    branch: RetrievalBranch,
    candidates: tuple[FrameCandidate, ...],
) -> BranchResult[FrameCandidate]:
    return BranchResult[FrameCandidate](
        branch=branch,
        candidate_level=CandidateLevel.FRAME,
        query_variant_id="q0",
        candidates=candidates,
        requested_top_k=10,
        latency_ms=2.0,
        status=BranchStatus.SUCCESS,
    )


def failed_frame_result(
    branch: RetrievalBranch,
    warning: str,
    *,
    variant_id: str = "q0",
    missing_metadata_count: int = 0,
) -> BranchResult[FrameCandidate]:
    return BranchResult[FrameCandidate](
        branch=branch,
        candidate_level=CandidateLevel.FRAME,
        query_variant_id=variant_id,
        candidates=(),
        requested_top_k=10,
        latency_ms=2.0,
        status=BranchStatus.FAILED,
        warnings=(warning,),
        missing_metadata_count=missing_metadata_count,
    )


def asr_interval(
    interval_id: str,
    *,
    start: float,
    end: float,
    rank: int,
) -> ASRIntervalCandidate:
    return ASRIntervalCandidate(
        video_id="V001",
        interval_id=interval_id,
        start_time_sec=start,
        end_time_sec=end,
        rank=rank,
        raw_score=0.8,
        provenance=provenance(RetrievalBranch.ASR_DENSE),
    )


def asr_result(candidates: tuple[ASRIntervalCandidate, ...]) -> BranchResult[ASRIntervalCandidate]:
    return BranchResult[ASRIntervalCandidate](
        branch=RetrievalBranch.ASR_DENSE,
        candidate_level=CandidateLevel.ASR_INTERVAL,
        query_variant_id="q0",
        candidates=candidates,
        requested_top_k=10,
        latency_ms=3.0,
        status=BranchStatus.SUCCESS,
    )


def video_result(candidates: tuple[VideoCandidate, ...]) -> BranchResult[VideoCandidate]:
    return BranchResult[VideoCandidate](
        branch=RetrievalBranch.SUMMARY_DENSE,
        candidate_level=CandidateLevel.VIDEO,
        query_variant_id="q0",
        candidates=candidates,
        requested_top_k=10,
        latency_ms=1.0,
        status=BranchStatus.SUCCESS,
    )


def video_candidate(video_id: str, *, normalized_score: float) -> VideoCandidate:
    return VideoCandidate(
        video_id=video_id,
        rank=1,
        raw_score=1.0,
        normalized_score=normalized_score,
        summary=f"summary {video_id}",
        provenance=provenance(RetrievalBranch.SUMMARY_DENSE),
    )


class FakeRetrievalService:
    def __init__(self, results: tuple[BranchResult, ...]) -> None:
        self.results = results
        self.calls = []

    async def retrieve(self, bundle):
        self.calls.append(bundle.query_id)
        return self.results


class BlockingRetrievalService:
    def __init__(self, results: tuple[BranchResult, ...]) -> None:
        self.results = results
        self.entered: asyncio.Event | None = None
        self.release: asyncio.Event | None = None

    async def retrieve(self, bundle):
        if self.entered is None or self.release is None:
            raise RuntimeError("events are not configured")
        self.entered.set()
        await self.release.wait()
        return self.results


class ThreadRecordingRanking:
    def __init__(self) -> None:
        self.thread_names: list[str] = []

    def rank(self, bundle, branch_results) -> KISSearchResult:
        self.thread_names.append(threading.current_thread().name)
        return KISSearchResult(
            candidates=(),
            diagnostics=QueryDiagnostics(
                query_id=bundle.query_id,
                total_latency_ms=1.0,
                stage_latencies_ms={"ranking": 1.0},
                branches={
                    RetrievalBranch.VISUAL_DENSE: BranchDiagnostics(
                        status=BranchStatus.SUCCESS,
                        latency_ms=1.0,
                        requested_top_k=1,
                        raw_result_count=0,
                        output_candidate_count=0,
                    )
                },
                normalization_method="fake",
                fusion_method="fake",
                fusion_weights={RetrievalBranch.VISUAL_DENSE: 1.0},
            ),
        )


class KISOrchestrationTests(unittest.TestCase):
    def test_ranking_pipeline_maps_asr_boosts_summary_filters_objects_and_dedups(self) -> None:
        frames = (
            FrameMetadata(frame_id="V001_00000_015", video_id="V001", shot_id=0, timestamp_sec=1.5),
            FrameMetadata(frame_id="V001_00000_050", video_id="V001", shot_id=0, timestamp_sec=5.0),
            FrameMetadata(frame_id="V002_00000_015", video_id="V002", shot_id=0, timestamp_sec=1.5),
        )
        bundle = KISQueryBuilder().build(
            "query",
            mode=QueryMode.KIS_TEXT,
            query_id="kis-orchestrated",
            object_constraints=(
                ObjectConstraint(
                    label="person",
                    count_operator=CountOperator.GTE,
                    count=1,
                    filter_mode=FilterMode.HARD,
                ),
            ),
        )
        ranking = KISRankingService(
            metadata=FakeMetadataReaderPort(frames),
            object_reader=FakeObjectReaderPort(
                {
                    "V001_00000_015": (
                        ObjectDetection(
                            label="person",
                            confidence=0.9,
                            x_min=0,
                            y_min=0,
                            x_max=1,
                            y_max=1,
                        ),
                    ),
                    "V002_00000_015": (),
                }
            ),
        )
        result = ranking.rank(
            bundle,
            (
                frame_result(
                    RetrievalBranch.VISUAL_DENSE,
                    (
                        frame_candidate(
                            "V001_00000_015",
                            branch=RetrievalBranch.VISUAL_DENSE,
                            rank=1,
                            video_id="V001",
                            shot_id=0,
                            timestamp_sec=1.5,
                        ),
                        frame_candidate(
                            "V002_00000_015",
                            branch=RetrievalBranch.VISUAL_DENSE,
                            rank=2,
                            video_id="V002",
                            shot_id=0,
                            timestamp_sec=1.5,
                        ),
                    ),
                ),
                frame_result(
                    RetrievalBranch.OCR_BM25,
                    (
                        frame_candidate(
                            "V001_00000_015",
                            branch=RetrievalBranch.OCR_BM25,
                            rank=1,
                            video_id="V001",
                            shot_id=0,
                            timestamp_sec=1.5,
                        ),
                    ),
                ),
                asr_result(
                    (
                        asr_interval("hit", start=1.0, end=2.0, rank=1),
                        asr_interval("miss", start=20.0, end=21.0, rank=2),
                    )
                ),
                video_result(
                    (
                        video_candidate("V001", normalized_score=1.0),
                        video_candidate("V999", normalized_score=1.0),
                    )
                ),
            ),
        )

        self.assertEqual(tuple(candidate.frame_id for candidate in result.candidates), ("V001_00000_015",))
        self.assertGreater(result.candidates[0].diagnostics.summary_boost, 0.0)
        self.assertEqual(result.candidates[0].diagnostics.object_constraints_satisfied, 1)
        self.assertEqual(result.diagnostics.query_id, "kis-orchestrated")
        self.assertEqual(result.diagnostics.object_filter_removals, 1)
        self.assertEqual(result.diagnostics.branches[RetrievalBranch.ASR_DENSE].mapping_loss_count, 1)
        self.assertEqual(result.diagnostics.branches[RetrievalBranch.ASR_DENSE].output_candidate_count, 1)
        self.assertEqual(result.diagnostics.normalization_method, "rrf")
        self.assertIn("branch_normalization", result.diagnostics.stage_latencies_ms)
        self.assertIn("aggregation_method=weighted_sum_query_variant_v1", result.diagnostics.warnings)
        self.assertIn("fusion", result.diagnostics.stage_latencies_ms)

    def test_visual_paraphrase_failure_degrades_without_dropping_q0(self) -> None:
        frames = (
            FrameMetadata(frame_id="V001_00000_015", video_id="V001", shot_id=0, timestamp_sec=1.5),
        )
        bundle = KISQueryBuilder().build(
            "query",
            mode=QueryMode.KIS_TEXT,
            paraphrases=("variant",),
            query_id="visual-q1-timeout",
        )
        ranking = KISRankingService(metadata=FakeMetadataReaderPort(frames))

        result = ranking.rank(
            bundle,
            (
                frame_result(
                    RetrievalBranch.VISUAL_DENSE,
                    (
                        frame_candidate(
                            "V001_00000_015",
                            branch=RetrievalBranch.VISUAL_DENSE,
                            rank=1,
                            video_id="V001",
                            shot_id=0,
                            timestamp_sec=1.5,
                        ),
                    ),
                ),
                failed_frame_result(
                    RetrievalBranch.VISUAL_DENSE,
                    "BRANCH_TIMEOUT",
                    variant_id="q1",
                ),
            ),
        )

        self.assertEqual(tuple(candidate.frame_id for candidate in result.candidates), ("V001_00000_015",))
        self.assertEqual(result.diagnostics.branches[RetrievalBranch.VISUAL_DENSE].status, BranchStatus.DEGRADED)
        self.assertIn(
            "branch=visual_dense;query_variant_id=q1;code=BRANCH_TIMEOUT",
            result.diagnostics.warnings,
        )

    def test_optional_branch_failure_does_not_fail_core_query(self) -> None:
        frames = (
            FrameMetadata(frame_id="V001_00000_015", video_id="V001", shot_id=0, timestamp_sec=1.5),
        )
        bundle = KISQueryBuilder().build("query", mode=QueryMode.KIS_TEXT, query_id="optional-fail")
        ranking = KISRankingService(metadata=FakeMetadataReaderPort(frames))

        result = ranking.rank(
            bundle,
            (
                frame_result(
                    RetrievalBranch.VISUAL_DENSE,
                    (
                        frame_candidate(
                            "V001_00000_015",
                            branch=RetrievalBranch.VISUAL_DENSE,
                            rank=1,
                            video_id="V001",
                            shot_id=0,
                            timestamp_sec=1.5,
                        ),
                    ),
                ),
                failed_frame_result(RetrievalBranch.OCR_BM25, "RESOURCE_UNAVAILABLE"),
            ),
        )

        self.assertEqual(tuple(candidate.frame_id for candidate in result.candidates), ("V001_00000_015",))
        self.assertIn(
            "branch=ocr_bm25;query_variant_id=q0;code=RESOURCE_UNAVAILABLE",
            result.diagnostics.warnings,
        )

    def test_asr_interval_contribution_is_not_multiplied_in_full_pipeline(self) -> None:
        frames = (
            FrameMetadata(frame_id="V001_00000_040", video_id="V001", shot_id=0, timestamp_sec=4.0),
            FrameMetadata(frame_id="V001_00001_060", video_id="V001", shot_id=1, timestamp_sec=6.0),
            FrameMetadata(frame_id="V001_00002_100", video_id="V001", shot_id=2, timestamp_sec=10.0),
        )
        bundle = KISQueryBuilder().build("query", mode=QueryMode.KIS_TEXT, query_id="asr-score")
        ranking = KISRankingService(
            metadata=FakeMetadataReaderPort(frames),
            asr_mapper=ASRIntervalFrameMapper(
                ASRMappingConfig(max_frames_per_interval=2, interval_rrf_k=10)
            ),
        )

        result = ranking.rank(
            bundle,
            (
                frame_result(RetrievalBranch.VISUAL_DENSE, ()),
                asr_result((asr_interval("long", start=0.0, end=10.0, rank=1),)),
            ),
        )

        asr_total = sum(
            evidence.normalized_score
            for candidate in result.candidates
            for evidence in candidate.evidence
            if evidence.branch is RetrievalBranch.ASR_DENSE
        )
        self.assertAlmostEqual(asr_total, 1 / 11)
        self.assertIn("asr_truncated_interval_count=1", result.diagnostics.warnings)
        asr_evidence = tuple(
            evidence
            for candidate in result.candidates
            for evidence in candidate.evidence
            if evidence.branch is RetrievalBranch.ASR_DENSE
        )
        self.assertTrue(asr_evidence)
        self.assertTrue(all(evidence.source_candidate_id == "long" for evidence in asr_evidence))
        self.assertTrue(all(evidence.source_start_time_sec == 0.0 for evidence in asr_evidence))
        self.assertTrue(all(evidence.source_end_time_sec == 10.0 for evidence in asr_evidence))
        self.assertTrue(all(evidence.source_normalized_score is not None for evidence in asr_evidence))

    def test_missing_metadata_count_is_preserved_in_query_diagnostics(self) -> None:
        ranking = KISRankingService(metadata=FakeMetadataReaderPort(()))
        bundle = KISQueryBuilder().build("query", mode=QueryMode.KIS_TEXT)

        result = ranking.rank(
            bundle,
            (
                frame_result(RetrievalBranch.VISUAL_DENSE, ()),
                failed_frame_result(
                    RetrievalBranch.OCR_BM25,
                    "MISSING_METADATA",
                    missing_metadata_count=3,
                ),
            ),
        )

        self.assertEqual(result.diagnostics.missing_metadata_count, 3)

    def test_visual_dense_failure_is_core_error_not_empty_success(self) -> None:
        ranking = KISRankingService(metadata=FakeMetadataReaderPort(()))
        bundle = KISQueryBuilder().build("query", mode=QueryMode.KIS_TEXT, query_id="core-fail")

        with self.assertRaises(BranchTimeoutError):
            ranking.rank(
                bundle,
                (failed_frame_result(RetrievalBranch.VISUAL_DENSE, "BRANCH_TIMEOUT"),),
            )

        with self.assertRaises(ResourceUnavailableError):
            ranking.rank(bundle, ())

    def test_async_orchestrator_uses_retrieval_service_port(self) -> None:
        bundle = KISQueryBuilder().build("query", mode=QueryMode.KIS_TEXT, query_id="q-orch")
        retrieval = FakeRetrievalService(
            (
                frame_result(
                    RetrievalBranch.VISUAL_DENSE,
                    (
                        frame_candidate(
                            "V001_00000_015",
                            branch=RetrievalBranch.VISUAL_DENSE,
                            rank=1,
                            video_id="V001",
                            shot_id=0,
                            timestamp_sec=1.5,
                        ),
                    ),
                ),
            )
        )
        orchestrator = KISSearchOrchestrator(
            retrieval=retrieval,
            ranking=KISRankingService(
                metadata=FakeMetadataReaderPort(
                    (FrameMetadata(frame_id="V001_00000_015", video_id="V001", shot_id=0, timestamp_sec=1.5),)
                )
            ),
        )

        result = asyncio.run(orchestrator.search(bundle))

        self.assertEqual(retrieval.calls, ["q-orch"])
        self.assertEqual(tuple(candidate.frame_id for candidate in result.candidates), ("V001_00000_015",))

    def test_async_orchestrator_runs_ranking_in_executor(self) -> None:
        bundle = KISQueryBuilder().build("query", mode=QueryMode.KIS_TEXT, query_id="q-exec")
        retrieval = FakeRetrievalService(())
        ranking = ThreadRecordingRanking()
        orchestrator = KISSearchOrchestrator(retrieval=retrieval, ranking=ranking)

        asyncio.run(orchestrator.search(bundle))
        orchestrator.close()

        self.assertEqual(retrieval.calls, ["q-exec"])
        self.assertEqual(len(ranking.thread_names), 1)
        self.assertTrue(ranking.thread_names[0].startswith("aic-ranking"))

    def test_orchestrator_enforces_c_policy_before_retrieval(self) -> None:
        bundle = KISQueryBuilder().build(
            "query",
            mode=QueryMode.KIS_TEXT,
            enabled_branches=(RetrievalBranch.OCR_BM25,),
        )
        retrieval = FakeRetrievalService(())
        orchestrator = KISSearchOrchestrator(
            retrieval=retrieval,
            ranking=KISRankingService(metadata=FakeMetadataReaderPort(())),
        )

        with self.assertRaises(InvalidQueryError):
            asyncio.run(orchestrator.search(bundle))
        orchestrator.close()

        self.assertEqual(retrieval.calls, [])

    def test_orchestrator_close_drains_active_request_and_rejects_new_work(self) -> None:
        async def scenario() -> None:
            bundle = KISQueryBuilder().build("query", mode=QueryMode.KIS_TEXT, query_id="q-drain")
            other = KISQueryBuilder().build("query", mode=QueryMode.KIS_TEXT, query_id="q-rejected")
            retrieval = BlockingRetrievalService(
                (
                    frame_result(
                        RetrievalBranch.VISUAL_DENSE,
                        (
                            frame_candidate(
                                "V001_00000_015",
                                branch=RetrievalBranch.VISUAL_DENSE,
                                rank=1,
                                video_id="V001",
                                shot_id=0,
                                timestamp_sec=1.5,
                            ),
                        ),
                    ),
                )
            )
            retrieval.entered = asyncio.Event()
            retrieval.release = asyncio.Event()
            orchestrator = KISSearchOrchestrator(
                retrieval=retrieval,
                ranking=KISRankingService(
                    metadata=FakeMetadataReaderPort(
                        (FrameMetadata(frame_id="V001_00000_015", video_id="V001", shot_id=0, timestamp_sec=1.5),)
                    )
                ),
            )

            search_task = asyncio.create_task(orchestrator.search(bundle))
            await retrieval.entered.wait()
            close_task = asyncio.create_task(asyncio.to_thread(orchestrator.close))
            await asyncio.sleep(0.01)
            self.assertFalse(close_task.done())
            with self.assertRaises(ResourceUnavailableError):
                await orchestrator.search(other)
            retrieval.release.set()
            result = await search_task
            await close_task
            orchestrator.close()

            self.assertEqual(tuple(candidate.frame_id for candidate in result.candidates), ("V001_00000_015",))

        asyncio.run(scenario())

    def test_object_constraints_require_object_reader(self) -> None:
        bundle = KISQueryBuilder().build(
            "query",
            mode=QueryMode.KIS_TEXT,
            object_constraints=(
                ObjectConstraint(
                    label="person",
                    count_operator=CountOperator.GTE,
                    count=1,
                ),
            ),
        )
        ranking = KISRankingService(metadata=FakeMetadataReaderPort(()))
        with self.assertRaises(ContractMismatchError):
            ranking.rank(
                bundle,
                (
                    frame_result(
                        RetrievalBranch.VISUAL_DENSE,
                        (
                            frame_candidate(
                                "V001_00000_015",
                                branch=RetrievalBranch.VISUAL_DENSE,
                                rank=1,
                                video_id="V001",
                                shot_id=0,
                                timestamp_sec=1.5,
                            ),
                        ),
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
