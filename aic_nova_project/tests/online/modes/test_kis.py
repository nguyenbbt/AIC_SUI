from __future__ import annotations

import asyncio
import unittest

from online.domain.candidates import (
    ASRIntervalCandidate,
    BranchResult,
    CandidateProvenance,
    FrameCandidate,
    ObjectDetection,
    VideoCandidate,
)
from online.domain.enums import BranchStatus, CandidateLevel, CountOperator, FilterMode, QueryMode, RetrievalBranch
from online.domain.errors import ContractMismatchError
from online.domain.query import ObjectConstraint
from online.modes.kis import KISRankingService, KISSearchOrchestrator
from online.ports.records import FrameMetadata
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
        self.assertIn("fusion", result.diagnostics.stage_latencies_ms)

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
