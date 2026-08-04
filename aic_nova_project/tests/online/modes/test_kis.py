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
        query_text=f"query {variant_id}",
    )


def metadata(
    frame_id: str,
    *,
    timestamp_sec: float,
    source_frame_idx: int,
    fps: float = 25.0,
) -> FrameMetadata:
    video_id, keyframe_text = frame_id.rsplit("_", 1)
    keyframe_no = int(keyframe_text)
    return FrameMetadata(
        frame_id=frame_id,
        video_id=video_id,
        keyframe_no=keyframe_no,
        local_index=keyframe_no - 1,
        timestamp_sec=timestamp_sec,
        fps=fps,
        source_frame_idx=source_frame_idx,
        image_rel_path=f"{video_id}/{keyframe_text}.jpg",
    )


def frame_candidate(
    frame: FrameMetadata,
    *,
    branch: RetrievalBranch,
    rank: int,
    variant_id: str = "q0",
    raw_score: float = 1.0,
) -> FrameCandidate:
    return FrameCandidate(
        frame_id=frame.frame_id,
        video_id=frame.video_id,
        keyframe_no=frame.keyframe_no,
        local_index=frame.local_index,
        timestamp_sec=frame.timestamp_sec,
        source_frame_idx=frame.source_frame_idx,
        rank=rank,
        raw_score=raw_score,
        provenance=provenance(branch, variant_id=variant_id),
    )


def frame_result(
    branch: RetrievalBranch,
    candidates: tuple[FrameCandidate, ...],
    *,
    variant_id: str = "q0",
    status: BranchStatus = BranchStatus.SUCCESS,
    warnings: tuple[str, ...] = (),
    missing_metadata_count: int = 0,
) -> BranchResult[FrameCandidate]:
    return BranchResult[FrameCandidate](
        branch=branch,
        candidate_level=CandidateLevel.FRAME,
        query_variant_id=variant_id,
        candidates=candidates,
        requested_top_k=10,
        latency_ms=2.0,
        status=status,
        warnings=warnings,
        missing_metadata_count=missing_metadata_count,
    )


def failed_frame_result(
    branch: RetrievalBranch,
    warning: str,
    *,
    variant_id: str = "q0",
    missing_metadata_count: int = 0,
) -> BranchResult[FrameCandidate]:
    return frame_result(
        branch,
        (),
        variant_id=variant_id,
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
    video_id: str = "L21_V001",
) -> ASRIntervalCandidate:
    return ASRIntervalCandidate(
        video_id=video_id,
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


def person_detection() -> ObjectDetection:
    return ObjectDetection(
        label_display="Person",
        label_normalized="person",
        class_mid="/m/01g317",
        confidence=0.9,
        x_min=0.0,
        y_min=0.0,
        x_max=1.0,
        y_max=1.0,
        model_source="open_images_v7",
    )


class FakeRetrievalService:
    def __init__(self, results: tuple[BranchResult, ...]) -> None:
        self.results = results
        self.calls: list[tuple[str, QueryMode]] = []

    async def retrieve(self, bundle):
        self.calls.append((bundle.query_id, bundle.mode))
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
    def test_synthetic_pipeline_follows_required_order_and_preserves_organizer_metadata(self) -> None:
        first = metadata("L21_V001_001", timestamp_sec=1.5, source_frame_idx=15, fps=25.0)
        duplicate_source = metadata("L21_V001_002", timestamp_sec=1.6, source_frame_idx=15, fps=50.0)
        other_video = metadata("L21_V002_001", timestamp_sec=1.5, source_frame_idx=15)
        frames = (first, duplicate_source, other_video)
        bundle = KISQueryBuilder().build(
            "query",
            mode=QueryMode.KIS_TEXT,
            paraphrases=("variant",),
            query_id="kis-synthetic",
            object_constraints=(
                ObjectConstraint(
                    label="người",
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
                    first.frame_id: (person_detection(),),
                    duplicate_source.frame_id: (person_detection(),),
                    other_video.frame_id: (),
                }
            ),
        )

        result = ranking.rank(
            bundle,
            (
                frame_result(
                    RetrievalBranch.VISUAL_DENSE,
                    (
                        frame_candidate(first, branch=RetrievalBranch.VISUAL_DENSE, rank=1),
                        frame_candidate(duplicate_source, branch=RetrievalBranch.VISUAL_DENSE, rank=2),
                        frame_candidate(other_video, branch=RetrievalBranch.VISUAL_DENSE, rank=3),
                    ),
                ),
                frame_result(
                    RetrievalBranch.VISUAL_DENSE,
                    (
                        frame_candidate(
                            first,
                            branch=RetrievalBranch.VISUAL_DENSE,
                            rank=1,
                            variant_id="q1",
                        ),
                    ),
                    variant_id="q1",
                ),
                frame_result(
                    RetrievalBranch.OCR_BM25,
                    (frame_candidate(first, branch=RetrievalBranch.OCR_BM25, rank=1),),
                ),
                asr_result(
                    (
                        asr_interval("hit", start=1.5, end=1.5, rank=1),
                        asr_interval("miss", start=20.0, end=21.0, rank=2),
                    )
                ),
                video_result(
                    (
                        video_candidate("L21_V001", normalized_score=1.0),
                        video_candidate("L21_V999", normalized_score=1.0),
                    )
                ),
            ),
        )

        self.assertEqual(tuple(candidate.frame_id for candidate in result.candidates), (first.frame_id,))
        final = result.candidates[0]
        self.assertEqual(
            (
                final.video_id,
                final.keyframe_no,
                final.local_index,
                final.timestamp_sec,
                final.source_frame_idx,
            ),
            ("L21_V001", 1, 0, 1.5, 15),
        )
        self.assertGreater(final.diagnostics.summary_boost, 0.0)
        self.assertEqual(final.diagnostics.object_constraints_satisfied, 1)
        self.assertEqual(tuple(ref.frame_id for ref in final.near_frames), (duplicate_source.frame_id,))
        self.assertEqual(result.diagnostics.object_filter_removals, 1)
        self.assertEqual(result.diagnostics.dedup_removals, 1)
        self.assertEqual(result.diagnostics.branches[RetrievalBranch.ASR_DENSE].mapping_loss_count, 1)
        self.assertEqual(
            tuple(result.diagnostics.stage_latencies_ms),
            (
                "asr_mapping",
                "query_variant_aggregation",
                "branch_normalization",
                "fusion",
                "summary_propagation",
                "object_processing",
                "dedup",
                "final_sort_top_k",
            ),
        )
        self.assertIn("aggregation_method=weighted_sum_query_variant_v1", result.diagnostics.warnings)
        self.assertIn("object_label_normalizer=open_images_vi_en_v1", result.diagnostics.warnings)
        self.assertIn("object_position_policy=bbox_center_in_region_v1", result.diagnostics.warnings)
        self.assertIn("final_top_k=100", result.diagnostics.warnings)

    def test_asr_interval_contribution_is_distributed_and_truncation_is_diagnostic(self) -> None:
        frames = (
            metadata("L21_V001_001", timestamp_sec=4.0, source_frame_idx=40),
            metadata("L21_V001_002", timestamp_sec=6.0, source_frame_idx=60),
            metadata("L21_V001_003", timestamp_sec=10.0, source_frame_idx=100),
        )
        ranking = KISRankingService(
            metadata=FakeMetadataReaderPort(frames),
            asr_mapper=ASRIntervalFrameMapper(ASRMappingConfig(max_frames_per_interval=2, interval_rrf_k=10)),
        )
        bundle = KISQueryBuilder().build("query", mode=QueryMode.KIS_TEXT, query_id="asr-score")

        result = ranking.rank(
            bundle,
            (
                frame_result(RetrievalBranch.VISUAL_DENSE, ()),
                asr_result((asr_interval("long", start=0.0, end=10.0, rank=1),)),
            ),
        )

        evidence = tuple(
            item
            for candidate in result.candidates
            for item in candidate.evidence
            if item.branch is RetrievalBranch.ASR_DENSE
        )
        self.assertAlmostEqual(sum(item.normalized_score for item in evidence), 1 / 11)
        self.assertTrue(all(item.source_candidate_id == "long" for item in evidence))
        self.assertIn("asr_truncated_interval_count=1", result.diagnostics.warnings)

    def test_optional_failure_degrades_but_core_failure_is_explicit(self) -> None:
        frame = metadata("L21_V001_001", timestamp_sec=1.5, source_frame_idx=15)
        ranking = KISRankingService(metadata=FakeMetadataReaderPort((frame,)))
        bundle = KISQueryBuilder().build("query", mode=QueryMode.KIS_TEXT)
        result = ranking.rank(
            bundle,
            (
                frame_result(
                    RetrievalBranch.VISUAL_DENSE,
                    (frame_candidate(frame, branch=RetrievalBranch.VISUAL_DENSE, rank=1),),
                ),
                failed_frame_result(
                    RetrievalBranch.OCR_BM25,
                    "MISSING_METADATA",
                    missing_metadata_count=3,
                ),
            ),
        )
        self.assertEqual(result.diagnostics.missing_metadata_count, 3)
        self.assertIn(
            "branch=ocr_bm25;query_variant_id=q0;code=MISSING_METADATA",
            result.diagnostics.warnings,
        )

        with self.assertRaises(BranchTimeoutError):
            ranking.rank(
                bundle,
                (failed_frame_result(RetrievalBranch.VISUAL_DENSE, "BRANCH_TIMEOUT"),),
            )
        with self.assertRaises(ResourceUnavailableError):
            ranking.rank(bundle, ())

    def test_q1_visual_failure_does_not_drop_successful_q0(self) -> None:
        frame = metadata("L21_V001_001", timestamp_sec=1.5, source_frame_idx=15)
        ranking = KISRankingService(metadata=FakeMetadataReaderPort((frame,)))
        bundle = KISQueryBuilder().build(
            "query",
            mode=QueryMode.KIS_TEXT,
            paraphrases=("variant",),
        )
        result = ranking.rank(
            bundle,
            (
                frame_result(
                    RetrievalBranch.VISUAL_DENSE,
                    (frame_candidate(frame, branch=RetrievalBranch.VISUAL_DENSE, rank=1),),
                ),
                failed_frame_result(
                    RetrievalBranch.VISUAL_DENSE,
                    "BRANCH_TIMEOUT",
                    variant_id="q1",
                ),
            ),
        )
        self.assertEqual(tuple(item.frame_id for item in result.candidates), (frame.frame_id,))
        self.assertIs(
            result.diagnostics.branches[RetrievalBranch.VISUAL_DENSE].status,
            BranchStatus.DEGRADED,
        )

    def test_final_top_k_uses_required_deterministic_order(self) -> None:
        frames = (
            metadata("L21_V002_001", timestamp_sec=1.0, source_frame_idx=10),
            metadata("L21_V001_002", timestamp_sec=2.0, source_frame_idx=20),
            metadata("L21_V001_001", timestamp_sec=1.0, source_frame_idx=10),
        )
        ranking = KISRankingService(metadata=FakeMetadataReaderPort(frames), final_top_k=2)
        bundle = KISQueryBuilder().build("query", mode=QueryMode.KIS_TEXT)
        result = ranking.rank(
            bundle,
            (
                frame_result(
                    RetrievalBranch.VISUAL_DENSE,
                    tuple(
                        frame_candidate(frame, branch=RetrievalBranch.VISUAL_DENSE, rank=1)
                        for frame in frames
                    ),
                ),
            ),
        )
        self.assertEqual(
            tuple(item.frame_id for item in result.candidates),
            ("L21_V001_001", "L21_V001_002"),
        )
        for invalid in (0, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                KISRankingService(metadata=FakeMetadataReaderPort(()), final_top_k=invalid)

    def test_text_and_video_kis_use_the_same_orchestrator(self) -> None:
        frame = metadata("L21_V001_001", timestamp_sec=1.5, source_frame_idx=15)
        branch_results = (
            frame_result(
                RetrievalBranch.VISUAL_DENSE,
                (frame_candidate(frame, branch=RetrievalBranch.VISUAL_DENSE, rank=1),),
            ),
        )
        retrieval = FakeRetrievalService(branch_results)
        orchestrator = KISSearchOrchestrator(
            retrieval=retrieval,
            ranking=KISRankingService(metadata=FakeMetadataReaderPort((frame,))),
        )

        text_result = asyncio.run(
            orchestrator.search(KISQueryBuilder().build("query", mode=QueryMode.KIS_TEXT, query_id="t"))
        )
        video_result_value = asyncio.run(
            orchestrator.search(KISQueryBuilder().build("query", mode=QueryMode.KIS_VIDEO, query_id="v"))
        )
        orchestrator.close()

        self.assertEqual(text_result.candidates, video_result_value.candidates)
        self.assertEqual(retrieval.calls, [("t", QueryMode.KIS_TEXT), ("v", QueryMode.KIS_VIDEO)])

    def test_orchestrator_runs_ranking_in_executor_and_validates_before_retrieval(self) -> None:
        retrieval = FakeRetrievalService(())
        ranking = ThreadRecordingRanking()
        orchestrator = KISSearchOrchestrator(retrieval=retrieval, ranking=ranking)
        asyncio.run(
            orchestrator.search(KISQueryBuilder().build("query", mode=QueryMode.KIS_TEXT, query_id="exec"))
        )
        orchestrator.close()
        self.assertEqual(len(ranking.thread_names), 1)
        self.assertTrue(ranking.thread_names[0].startswith("aic-ranking"))

        invalid_retrieval = FakeRetrievalService(())
        invalid_orchestrator = KISSearchOrchestrator(
            retrieval=invalid_retrieval,
            ranking=KISRankingService(metadata=FakeMetadataReaderPort(())),
        )
        invalid_bundle = KISQueryBuilder().build(
            "query",
            mode=QueryMode.KIS_TEXT,
            enabled_branches=(RetrievalBranch.OCR_BM25,),
        )
        with self.assertRaises(InvalidQueryError):
            asyncio.run(invalid_orchestrator.search(invalid_bundle))
        invalid_orchestrator.close()
        self.assertEqual(invalid_retrieval.calls, [])

    def test_orchestrator_close_drains_active_request_and_rejects_new_work(self) -> None:
        async def scenario() -> None:
            frame = metadata("L21_V001_001", timestamp_sec=1.5, source_frame_idx=15)
            retrieval = BlockingRetrievalService(
                (
                    frame_result(
                        RetrievalBranch.VISUAL_DENSE,
                        (frame_candidate(frame, branch=RetrievalBranch.VISUAL_DENSE, rank=1),),
                    ),
                )
            )
            retrieval.entered = asyncio.Event()
            retrieval.release = asyncio.Event()
            orchestrator = KISSearchOrchestrator(
                retrieval=retrieval,
                ranking=KISRankingService(metadata=FakeMetadataReaderPort((frame,))),
            )
            search_task = asyncio.create_task(
                orchestrator.search(
                    KISQueryBuilder().build("query", mode=QueryMode.KIS_TEXT, query_id="active")
                )
            )
            await retrieval.entered.wait()
            close_task = asyncio.create_task(asyncio.to_thread(orchestrator.close))
            await asyncio.sleep(0.01)
            self.assertFalse(close_task.done())
            with self.assertRaises(ResourceUnavailableError):
                await orchestrator.search(
                    KISQueryBuilder().build("query", mode=QueryMode.KIS_TEXT, query_id="rejected")
                )
            retrieval.release.set()
            result = await search_task
            await close_task
            self.assertEqual(tuple(item.frame_id for item in result.candidates), (frame.frame_id,))

        asyncio.run(scenario())

    def test_object_constraints_require_object_reader(self) -> None:
        frame = metadata("L21_V001_001", timestamp_sec=1.5, source_frame_idx=15)
        bundle = KISQueryBuilder().build(
            "query",
            mode=QueryMode.KIS_TEXT,
            object_constraints=(
                ObjectConstraint(label="person", count_operator=CountOperator.GTE, count=1),
            ),
        )
        with self.assertRaises(ContractMismatchError):
            KISRankingService(metadata=FakeMetadataReaderPort((frame,))).rank(
                bundle,
                (
                    frame_result(
                        RetrievalBranch.VISUAL_DENSE,
                        (frame_candidate(frame, branch=RetrievalBranch.VISUAL_DENSE, rank=1),),
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
