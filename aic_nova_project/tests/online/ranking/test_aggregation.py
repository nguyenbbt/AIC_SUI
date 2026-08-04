from __future__ import annotations

import unittest

from online.domain.candidates import BranchResult, CandidateProvenance, FrameCandidate
from online.domain.enums import BranchStatus, CandidateLevel, RetrievalBranch
from online.domain.errors import ContractMismatchError
from online.ranking.aggregation import QueryVariantAggregationConfig, RRFQueryVariantAggregator
from online.ranking.fusion import WeightedFrameFusion


def frame(
    frame_id: str,
    *,
    variant_id: str,
    rank: int,
    normalized_score: float | None = None,
) -> FrameCandidate:
    video_id, keyframe_text = frame_id.rsplit("_", 1)
    keyframe_no = int(keyframe_text)
    return FrameCandidate(
        frame_id=frame_id,
        video_id=video_id,
        keyframe_no=keyframe_no,
        local_index=keyframe_no - 1,
        timestamp_sec=1.0,
        source_frame_idx=keyframe_no * 10,
        rank=rank,
        raw_score=100.0 - rank,
        normalized_score=normalized_score,
        provenance=CandidateProvenance(
            branch=RetrievalBranch.VISUAL_DENSE,
            backend="milvus",
            source_resource="visual_features",
            query_variant_id=variant_id,
            query_text=f"query {variant_id}",
        ),
    )


def branch_result(
    variant_id: str,
    candidates: tuple[FrameCandidate, ...],
    *,
    status: BranchStatus = BranchStatus.SUCCESS,
    warnings: tuple[str, ...] = (),
) -> BranchResult[FrameCandidate]:
    return BranchResult[FrameCandidate](
        branch=RetrievalBranch.VISUAL_DENSE,
        candidate_level=CandidateLevel.FRAME,
        query_variant_id=variant_id,
        candidates=candidates,
        requested_top_k=10,
        latency_ms=1.0,
        status=status,
        warnings=warnings,
    )


class QueryVariantAggregationTests(unittest.TestCase):
    def test_weighted_aggregation_preserves_variants_and_boosts_repeated_frame_in_fusion(self) -> None:
        aggregator = RRFQueryVariantAggregator(
            QueryVariantAggregationConfig(query_variant_weights={"q0": 1.0, "q1": 0.5})
        )
        prepared = aggregator.aggregate(
            (
                branch_result(
                    "q0",
                    (
                        frame("L21_V001_001", variant_id="q0", rank=1, normalized_score=0.4),
                        frame("L21_V001_002", variant_id="q0", rank=2, normalized_score=0.7),
                    ),
                ),
                branch_result(
                    "q1",
                    (
                        frame("L21_V001_001", variant_id="q1", rank=2, normalized_score=0.4),
                    ),
                ),
            )
        )
        aggregated = aggregator.apply_normalized_weights(prepared)

        self.assertEqual(tuple(result.query_variant_id for result in aggregated), ("q0", "q1"))
        repeated_scores = [
            candidate.normalized_score
            for result in aggregated
            for candidate in result.candidates
            if candidate.frame_id == "L21_V001_001"
        ]
        self.assertEqual(repeated_scores, [0.4, 0.2])

        fused = WeightedFrameFusion().fuse(aggregated)
        self.assertEqual(fused[0].frame_id, "L21_V001_002")
        self.assertAlmostEqual(
            fused[1].branch_scores[RetrievalBranch.VISUAL_DENSE],
            0.6,
        )
        self.assertEqual(
            tuple(evidence.query_variant_id for evidence in fused[1].evidence),
            ("q0", "q1"),
        )

    def test_failed_or_missing_variant_does_not_remove_successful_evidence(self) -> None:
        failed = branch_result(
            "q1",
            (),
            status=BranchStatus.FAILED,
            warnings=("BRANCH_TIMEOUT",),
        )
        aggregator = RRFQueryVariantAggregator()
        prepared = aggregator.aggregate(
            (
                branch_result(
                    "q0",
                    (frame("L21_V001_001", variant_id="q0", rank=1, normalized_score=0.3),),
                ),
                failed,
            )
        )
        aggregated = aggregator.apply_normalized_weights(prepared)

        self.assertEqual(aggregated[0].candidates[0].normalized_score, 0.3)
        self.assertIs(aggregated[1], failed)

    def test_query_variant_weights_have_observable_effect(self) -> None:
        aggregator = RRFQueryVariantAggregator(
            QueryVariantAggregationConfig(query_variant_weights={"q0": 0.25})
        )
        aggregated = aggregator.apply_normalized_weights(
            aggregator.aggregate(
                (
                    branch_result(
                        "q0",
                        (
                            frame(
                                "L21_V001_001",
                                variant_id="q0",
                                rank=1,
                                normalized_score=0.42,
                            ),
                        ),
                    ),
                )
            )
        )

        self.assertEqual(aggregated[0].candidates[0].normalized_score, 0.105)

    def test_invalid_aggregator_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            QueryVariantAggregationConfig(query_variant_weights={"q0": -1.0})
        with self.assertRaises(ContractMismatchError):
            aggregator = RRFQueryVariantAggregator()
            aggregator.apply_normalized_weights(
                aggregator.aggregate(
                    (
                        branch_result(
                            "q0",
                            (frame("L21_V001_001", variant_id="q0", rank=1),),
                        ),
                    )
                )
            )
        with self.assertRaises(ContractMismatchError):
            duplicate = branch_result(
                "q0",
                (frame("L21_V001_001", variant_id="q0", rank=1),),
            )
            RRFQueryVariantAggregator().aggregate((duplicate, duplicate))
        with self.assertRaises(TypeError):
            RRFQueryVariantAggregator().aggregate(("not-a-result",))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
