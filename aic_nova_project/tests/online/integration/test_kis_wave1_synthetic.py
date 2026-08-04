"""Wave-1 synthetic handoff from fake BranchResults to KIS submission rows."""

from __future__ import annotations

import asyncio

from online.domain.candidates import BranchResult, CandidateProvenance, FrameCandidate
from online.domain.enums import BranchStatus, CandidateLevel, QueryMode, RetrievalBranch
from online.modes.kis import KISRankingService, KISSearchOrchestrator
from online.retrieval.query_builder import KISQueryBuilder
from online.testing import FakeMetadataReaderPort, build_organizer_frame_metadata
from retrieval_api.search_engine import serialize_kis_competition_candidates


class _FakeBranchResultRetrieval:
    def __init__(self, result: BranchResult[FrameCandidate]) -> None:
        self.result = result

    async def retrieve(self, bundle):
        return (self.result,)


def test_fake_branch_results_reach_ranked_competition_rows() -> None:
    frames = build_organizer_frame_metadata()
    candidates = tuple(
        FrameCandidate(
            frame_id=frame.frame_id,
            video_id=frame.video_id,
            keyframe_no=frame.keyframe_no,
            local_index=frame.local_index,
            timestamp_sec=frame.timestamp_sec,
            source_frame_idx=frame.source_frame_idx,
            rank=rank,
            raw_score=1.0 / rank,
            provenance=CandidateProvenance(
                branch=RetrievalBranch.VISUAL_DENSE,
                backend="milvus",
                source_resource="synthetic_visual_features",
                query_variant_id="q0",
                query_text="synthetic organizer query",
            ),
        )
        for rank, frame in enumerate(frames, start=1)
    )
    branch_result = BranchResult[FrameCandidate](
        branch=RetrievalBranch.VISUAL_DENSE,
        candidate_level=CandidateLevel.FRAME,
        query_variant_id="q0",
        candidates=candidates,
        requested_top_k=10,
        latency_ms=1.0,
        status=BranchStatus.SUCCESS,
    )
    orchestrator = KISSearchOrchestrator(
        retrieval=_FakeBranchResultRetrieval(branch_result),
        ranking=KISRankingService(metadata=FakeMetadataReaderPort(frames)),
    )

    try:
        ranked = asyncio.run(
            orchestrator.search(
                KISQueryBuilder().build(
                    "synthetic organizer query",
                    mode=QueryMode.KIS_TEXT,
                    query_id="wave1-synthetic",
                )
            )
        )
    finally:
        orchestrator.close()

    rows = serialize_kis_competition_candidates(ranked.candidates)
    assert [row.model_dump() for row in rows] == [
        {"video_id": "L21_V001", "source_frame_idx": 0},
        {"video_id": "L21_V001", "source_frame_idx": 60},
        {"video_id": "L21_V002", "source_frame_idx": 0},
    ]
    assert tuple(candidate.frame_id for candidate in ranked.candidates) == (
        "L21_V001_001",
        "L21_V001_003",
        "L21_V002_001",
    )
