from __future__ import annotations

import itertools
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
from online.domain.errors import ContractMismatchError
from online.domain.query import ObjectConstraint
from online.ranking.dedup import CompetitionFrameDeduplicator
from online.ranking.fusion import FRAME_FUSION_BRANCHES, FusionConfig, WeightedFrameFusion
from online.ranking.object_filter import ObjectConstraintProcessor, ObjectProcessingConfig
from online.testing import FakeObjectReaderPort
from query_understanding.providers.objects import (
    OBJECT_LABEL_NORMALIZER_VERSION,
    ObjectLabelNormalizer,
)


def candidate(
    frame_id: str,
    *,
    branch: RetrievalBranch,
    variant_id: str = "q0",
    rank: int = 1,
    normalized_score: float,
    raw_score: float = 1.0,
    timestamp_sec: float = 1.0,
    source_frame_idx: int | None = None,
) -> FrameCandidate:
    video_id, keyframe_text = frame_id.rsplit("_", 1)
    keyframe_no = int(keyframe_text)
    return FrameCandidate(
        frame_id=frame_id,
        video_id=video_id,
        keyframe_no=keyframe_no,
        local_index=keyframe_no - 1,
        timestamp_sec=timestamp_sec,
        source_frame_idx=source_frame_idx if source_frame_idx is not None else keyframe_no * 10,
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
    timestamp_sec: float = 1.0,
    source_frame_idx: int | None = None,
) -> FusedFrameCandidate:
    video_id, keyframe_text = frame_id.rsplit("_", 1)
    keyframe_no = int(keyframe_text)
    return FusedFrameCandidate(
        frame_id=frame_id,
        video_id=video_id,
        keyframe_no=keyframe_no,
        local_index=keyframe_no - 1,
        timestamp_sec=timestamp_sec,
        source_frame_idx=source_frame_idx if source_frame_idx is not None else keyframe_no * 10,
        final_score=final_score,
        branch_scores={RetrievalBranch.VISUAL_DENSE: min(final_score, 1.0)},
        evidence=(),
        diagnostics=CandidateDiagnostics(),
    )


def detection(
    *,
    label_normalized: str = "person",
    class_mid: str | None = "/m/01g317",
    confidence: float = 0.9,
    x_min: float = 0.1,
    y_min: float = 0.1,
    x_max: float = 0.3,
    y_max: float = 0.3,
) -> ObjectDetection:
    return ObjectDetection(
        label_display=label_normalized.title(),
        label_normalized=label_normalized,
        class_mid=class_mid,
        confidence=confidence,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        model_source="open_images_v7",
    )


class FusionDedupAndObjectTests(unittest.TestCase):
    def test_weighted_fusion_propagates_metadata_evidence_and_required_tie_order(self) -> None:
        fusion = WeightedFrameFusion(
            FusionConfig(
                weights={
                    RetrievalBranch.VISUAL_DENSE: 2.0,
                    RetrievalBranch.OCR_BM25: 0.5,
                }
            )
        )
        first = candidate(
            "L21_V002_001",
            branch=RetrievalBranch.VISUAL_DENSE,
            normalized_score=0.5,
            source_frame_idx=81,
        )
        second = candidate(
            "L21_V001_002",
            branch=RetrievalBranch.VISUAL_DENSE,
            normalized_score=0.5,
            source_frame_idx=72,
        )
        merged_visual = candidate(
            "L21_V001_001",
            branch=RetrievalBranch.VISUAL_DENSE,
            normalized_score=0.6,
            source_frame_idx=61,
        )
        merged_ocr = candidate(
            "L21_V001_001",
            branch=RetrievalBranch.OCR_BM25,
            normalized_score=0.8,
            raw_score=20.0,
            source_frame_idx=61,
        )

        output = fusion.fuse(
            (
                result(RetrievalBranch.VISUAL_DENSE, (first, second, merged_visual)),
                result(RetrievalBranch.OCR_BM25, (merged_ocr,)),
            )
        )

        self.assertEqual(
            tuple(item.frame_id for item in output),
            ("L21_V001_001", "L21_V001_002", "L21_V002_001"),
        )
        self.assertAlmostEqual(output[0].final_score, 1.6)
        self.assertEqual(output[0].source_frame_idx, 61)
        self.assertEqual(len(output[0].evidence), 2)

    def test_fusion_requires_normalized_scores_and_rejects_metadata_conflicts(self) -> None:
        base = candidate(
            "L21_V001_001",
            branch=RetrievalBranch.VISUAL_DENSE,
            normalized_score=0.5,
            source_frame_idx=15,
        )
        without_score = base.model_copy(update={"normalized_score": None})
        with self.assertRaises(ValueError):
            WeightedFrameFusion().fuse((result(RetrievalBranch.VISUAL_DENSE, (without_score,)),))

        conflicting = candidate(
            "L21_V001_001",
            branch=RetrievalBranch.OCR_BM25,
            normalized_score=0.8,
            source_frame_idx=99,
        )
        with self.assertRaises(ContractMismatchError):
            WeightedFrameFusion().fuse(
                (
                    result(RetrievalBranch.VISUAL_DENSE, (base,)),
                    result(RetrievalBranch.OCR_BM25, (conflicting,)),
                )
            )

        with self.assertRaises(ValueError):
            FusionConfig(weights={RetrievalBranch.VISUAL_DENSE: -1.0})
        with self.assertRaises(ValueError):
            FusionConfig(default_weight=1.0, weights={branch: 0.0 for branch in FRAME_FUSION_BRANCHES})

    def test_competition_dedup_is_stable_and_never_merges_different_videos(self) -> None:
        inputs = (
            fused("L21_V001_002", final_score=0.9, source_frame_idx=150),
            fused("L21_V001_001", final_score=0.9, source_frame_idx=150),
            fused("L21_V001_003", final_score=0.7, source_frame_idx=300),
            fused("L21_V002_001", final_score=0.95, source_frame_idx=150),
        )
        deduplicator = CompetitionFrameDeduplicator()
        expected = ("L21_V002_001", "L21_V001_001", "L21_V001_003")

        for permutation in itertools.permutations(inputs):
            output = deduplicator.deduplicate(permutation)
            self.assertEqual(tuple(item.frame_id for item in output), expected)

        representative = deduplicator.deduplicate(inputs)[1]
        self.assertEqual(tuple(ref.frame_id for ref in representative.near_frames), ("L21_V001_002",))
        self.assertEqual(deduplicator.deduplicate((representative,))[0], representative)

    def test_object_constraints_support_vietnamese_mid_confidence_count_and_policy(self) -> None:
        objects = {
            "L21_V001_001": (
                detection(confidence=0.5),
                detection(confidence=0.9, x_min=0.6, x_max=0.8),
            ),
            "L21_V001_002": (),
        }
        processor = ObjectConstraintProcessor(
            FakeObjectReaderPort(objects),
            config=ObjectProcessingConfig(soft_boost_per_constraint=0.1, max_total_boost=0.2),
        )
        output = processor.process(
            (
                fused("L21_V001_001", final_score=0.5),
                fused("L21_V001_002", final_score=0.6),
            ),
            (
                ObjectConstraint(label="người", count_operator=CountOperator.GTE, count=2, filter_mode=FilterMode.HARD),
                ObjectConstraint(label="/m/01g317", count_operator=CountOperator.EQ, count=2, filter_mode=FilterMode.SOFT),
                ObjectConstraint(label="person", count_operator=CountOperator.LTE, count=2, filter_mode=FilterMode.SOFT),
            ),
        )

        self.assertEqual(tuple(item.frame_id for item in output), ("L21_V001_001",))
        self.assertAlmostEqual(output[0].final_score, 0.7)
        self.assertEqual(output[0].diagnostics.object_constraints_satisfied, 3)
        self.assertEqual(len(output[0].objects), 2)

    def test_position_uses_normalized_bbox_center_and_empty_objects_fail_hard(self) -> None:
        processor = ObjectConstraintProcessor(
            FakeObjectReaderPort(
                {
                    "L21_V001_001": (detection(x_min=0.2, y_min=0.2, x_max=0.4, y_max=0.4),),
                    "L21_V001_002": (),
                }
            )
        )
        constraint = ObjectConstraint.model_validate(
            {
                "label": "person",
                "count_operator": "gte",
                "count": 1,
                "position": {"x_min": 0.3, "y_min": 0.3, "x_max": 0.5, "y_max": 0.5},
                "filter_mode": "hard",
            }
        )
        output = processor.process(
            (
                fused("L21_V001_001", final_score=0.5),
                fused("L21_V001_002", final_score=0.6),
            ),
            (constraint,),
        )
        self.assertEqual(tuple(item.frame_id for item in output), ("L21_V001_001",))

    def test_synonym_normalizer_is_explicitly_versioned(self) -> None:
        normalizer = ObjectLabelNormalizer()
        self.assertEqual(normalizer.version, OBJECT_LABEL_NORMALIZER_VERSION)
        self.assertEqual(normalizer.normalize("  NGƯỜI  "), "person")
        self.assertEqual(normalizer.normalize("Ô tô"), "car")


if __name__ == "__main__":
    unittest.main()
