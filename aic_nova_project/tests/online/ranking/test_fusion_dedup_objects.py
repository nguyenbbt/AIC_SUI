from __future__ import annotations

import unittest

from online.domain.candidates import (
    BranchResult,
    CandidateDiagnostics,
    CandidateProvenance,
    FrameCandidate,
    FusedFrameCandidate,
    ObjectDetection,
)
from online.domain.enums import BranchStatus, CandidateLevel, CountOperator, FilterMode, RetrievalBranch
from online.domain.errors import ContractMismatchError, InvalidQueryError
from online.domain.query import ObjectConstraint
from online.ranking.dedup import ShotDeduplicator
from online.ranking.fusion import FRAME_FUSION_BRANCHES, FusionConfig, WeightedFrameFusion
from online.ranking.object_filter import ObjectConstraintProcessor, ObjectProcessingConfig
from online.testing import FakeObjectReaderPort


def candidate(
    frame_id: str,
    *,
    branch: RetrievalBranch,
    variant_id: str = "q0",
    rank: int = 1,
    normalized_score: float,
    raw_score: float = 1.0,
    shot_id: int = 0,
    video_id: str = "V001",
    timestamp_sec: float = 1.0,
) -> FrameCandidate:
    return FrameCandidate(
        frame_id=frame_id,
        video_id=video_id,
        shot_id=shot_id,
        timestamp_sec=timestamp_sec,
        source_frame_idx=int(timestamp_sec * 30),
        image_rel_path=f"keyframes/{video_id}/{frame_id}.webp",
        rank=rank,
        raw_score=raw_score,
        normalized_score=normalized_score,
        provenance=CandidateProvenance(
            branch=branch,
            backend="milvus" if branch.value.endswith("dense") else "elasticsearch",
            source_resource=branch.value,
            query_variant_id=variant_id,
            query_text="query",
        ),
    )


def result(branch: RetrievalBranch, candidates: tuple[FrameCandidate, ...]) -> BranchResult[FrameCandidate]:
    return BranchResult[FrameCandidate](
        branch=branch,
        candidate_level=CandidateLevel.FRAME,
        query_variant_id=candidates[0].provenance.query_variant_id if candidates else "q0",
        candidates=candidates,
        requested_top_k=10,
        latency_ms=1.0,
        status=BranchStatus.SUCCESS,
    )


def fused(
    frame_id: str,
    *,
    final_score: float,
    shot_id: int,
    video_id: str = "V001",
    timestamp_sec: float = 1.0,
    source_frame_idx: int | None = None,
) -> FusedFrameCandidate:
    return FusedFrameCandidate(
        frame_id=frame_id,
        video_id=video_id,
        shot_id=shot_id,
        timestamp_sec=timestamp_sec,
        source_frame_idx=(
            int(timestamp_sec * 30)
            if source_frame_idx is None
            else source_frame_idx
        ),
        image_rel_path=f"keyframes/{video_id}/{frame_id}.webp",
        final_score=final_score,
        branch_scores={RetrievalBranch.VISUAL_DENSE: min(final_score, 1.0)},
        evidence=(),
        diagnostics=CandidateDiagnostics(),
    )


class FusionDedupAndObjectTests(unittest.TestCase):
    def test_weighted_fusion_merges_by_frame_keeps_evidence_and_sorts_deterministically(self) -> None:
        fusion = WeightedFrameFusion(
            FusionConfig(
                weights={
                    RetrievalBranch.VISUAL_DENSE: 2.0,
                    RetrievalBranch.OCR_BM25: 0.5,
                }
            )
        )

        fused_results = fusion.fuse(
            (
                result(
                    RetrievalBranch.VISUAL_DENSE,
                    (
                        candidate("V001_00000_015", branch=RetrievalBranch.VISUAL_DENSE, normalized_score=0.6),
                        candidate("V001_00000_050", branch=RetrievalBranch.VISUAL_DENSE, normalized_score=0.9),
                    ),
                ),
                result(
                    RetrievalBranch.OCR_BM25,
                    (
                        candidate("V001_00000_015", branch=RetrievalBranch.OCR_BM25, normalized_score=1.0, raw_score=20.0),
                    ),
                ),
            )
        )

        self.assertEqual(tuple(item.frame_id for item in fused_results), ("V001_00000_050", "V001_00000_015"))
        merged = fused_results[1]
        self.assertAlmostEqual(merged.final_score, 1.7)
        self.assertEqual(
            set(merged.branch_scores),
            {RetrievalBranch.VISUAL_DENSE, RetrievalBranch.OCR_BM25},
        )
        self.assertEqual(len(merged.evidence), 2)

    def test_fusion_requires_normalized_scores_and_valid_weights(self) -> None:
        bad = candidate("V001_00000_015", branch=RetrievalBranch.VISUAL_DENSE, normalized_score=0.5)
        bad = bad.model_copy(update={"normalized_score": None})
        with self.assertRaises(ValueError):
            WeightedFrameFusion().fuse((result(RetrievalBranch.VISUAL_DENSE, (bad,)),))
        with self.assertRaises(ValueError):
            FusionConfig(weights={RetrievalBranch.VISUAL_DENSE: -1.0})
        with self.assertRaises(ValueError):
            FusionConfig(
                default_weight=1.0,
                weights={branch: 0.0 for branch in FRAME_FUSION_BRANCHES},
            )

    def test_fusion_rejects_conflicting_metadata_for_same_frame_id(self) -> None:
        with self.assertRaises((ContractMismatchError, ValueError)):
            WeightedFrameFusion().fuse(
                (
                    result(
                        RetrievalBranch.VISUAL_DENSE,
                        (
                            candidate(
                                "V001_00000_015",
                                branch=RetrievalBranch.VISUAL_DENSE,
                                normalized_score=0.6,
                                video_id="V001",
                                shot_id=0,
                                timestamp_sec=1.5,
                            ),
                        ),
                    ),
                    result(
                        RetrievalBranch.OCR_BM25,
                        (
                            candidate(
                                "V001_00000_015",
                                branch=RetrievalBranch.OCR_BM25,
                                normalized_score=0.8,
                                video_id="V001",
                                shot_id=0,
                                timestamp_sec=1.5,
                            ).model_copy(update={"video_id": "V999"}),
                        ),
                    ),
                )
            )

    def test_dedup_groups_by_submission_frame_keeps_near_frames(self) -> None:
        output = ShotDeduplicator().deduplicate(
            (
                fused("V001_00000_050", final_score=0.9, shot_id=0, timestamp_sec=5.0, source_frame_idx=45),
                fused("V001_00000_015", final_score=0.9, shot_id=0, timestamp_sec=1.5, source_frame_idx=45),
                fused("V001_00001_050", final_score=0.7, shot_id=1, timestamp_sec=10.0),
                fused("V002_00000_015", final_score=0.95, shot_id=0, video_id="V002"),
            )
        )

        self.assertEqual(
            tuple(item.frame_id for item in output),
            ("V002_00000_015", "V001_00000_015", "V001_00001_050"),
        )
        representative = output[1]
        self.assertEqual(tuple(ref.frame_id for ref in representative.near_frames), ("V001_00000_050",))
        self.assertNotIn(representative.frame_id, tuple(ref.frame_id for ref in representative.near_frames))

        rerun = ShotDeduplicator().deduplicate(output)
        self.assertEqual(
            tuple(ref.frame_id for ref in rerun[1].near_frames),
            ("V001_00000_050",),
        )

    def test_object_constraints_apply_hard_filter_and_soft_boost(self) -> None:
        objects = {
            "V001_00000_015": (
                ObjectDetection(
                    label="person",
                    confidence=0.95,
                    x_min=0,
                    y_min=0,
                    x_max=0.1,
                    y_max=0.2,
                ),
            ),
            "V001_00000_050": (),
        }
        processor = ObjectConstraintProcessor(
            FakeObjectReaderPort(objects),
            config=ObjectProcessingConfig(soft_boost_per_constraint=0.1, max_total_boost=0.1),
        )
        output = processor.process(
            (
                fused("V001_00000_015", final_score=0.5, shot_id=0),
                fused("V001_00000_050", final_score=0.6, shot_id=0),
            ),
            (
                ObjectConstraint(
                    label="person",
                    count_operator=CountOperator.GTE,
                    count=1,
                    min_confidence=0.8,
                    filter_mode=FilterMode.HARD,
                ),
                ObjectConstraint(
                    label="person",
                    count_operator=CountOperator.GTE,
                    count=1,
                    min_confidence=0.8,
                    filter_mode=FilterMode.SOFT,
                ),
            ),
        )

        self.assertEqual(tuple(item.frame_id for item in output), ("V001_00000_015",))
        self.assertAlmostEqual(output[0].final_score, 0.6)
        self.assertEqual(output[0].diagnostics.object_constraints_satisfied, 2)
        self.assertEqual(len(output[0].objects), 1)

    def test_position_constraints_wait_for_image_size_contract(self) -> None:
        processor = ObjectConstraintProcessor(FakeObjectReaderPort({}))
        with self.assertRaises(InvalidQueryError):
            processor.process(
                (fused("V001_00000_015", final_score=0.5, shot_id=0),),
                (
                    ObjectConstraint.model_validate(
                        {
                            "label": "person",
                            "count_operator": "gte",
                            "count": 1,
                            "position": {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1},
                        }
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
