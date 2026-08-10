from __future__ import annotations

import unittest

from online.domain.candidates import (
    BranchResult,
    CandidateDiagnostics,
    CandidateProvenance,
    FusedFrameCandidate,
    VideoCandidate,
)
from online.domain.enums import BranchStatus, CandidateLevel, RetrievalBranch
from online.ranking.summary import SummaryPropagationConfig, SummaryScorePropagator


def video_candidate(
    video_id: str,
    *,
    branch: RetrievalBranch,
    rank: int = 1,
    normalized_score: float | None = None,
    variant_id: str = "q0",
) -> VideoCandidate:
    return VideoCandidate(
        video_id=video_id,
        rank=rank,
        raw_score=10.0 - rank,
        normalized_score=normalized_score,
        summary=f"summary for {video_id}",
        provenance=CandidateProvenance(
            branch=branch,
            backend="milvus" if branch is RetrievalBranch.SUMMARY_DENSE else "elasticsearch",
            source_resource=branch.value,
            query_variant_id=variant_id,
            query_text="query",
        ),
    )


def summary_result(
    branch: RetrievalBranch,
    candidates: tuple[VideoCandidate, ...],
    *,
    status: BranchStatus = BranchStatus.SUCCESS,
    warnings: tuple[str, ...] = (),
    variant_id: str = "q0",
) -> BranchResult[VideoCandidate]:
    return BranchResult[VideoCandidate](
        branch=branch,
        candidate_level=CandidateLevel.VIDEO,
        query_variant_id=variant_id,
        candidates=candidates,
        requested_top_k=10,
        latency_ms=1.0,
        status=status,
        warnings=warnings,
    )


def frame(frame_id: str, *, video_id: str, final_score: float) -> FusedFrameCandidate:
    return FusedFrameCandidate(
        frame_id=frame_id,
        video_id=video_id,
        shot_id=0,
        timestamp_sec=1.0,
        source_frame_idx=30,
        image_rel_path=f"keyframes/{video_id}/{frame_id}.webp",
        final_score=final_score,
        branch_scores={RetrievalBranch.VISUAL_DENSE: min(final_score, 1.0)},
        evidence=(),
        diagnostics=CandidateDiagnostics(object_boost=0.03),
    )


class SummaryPropagationTests(unittest.TestCase):
    def test_aggregates_summary_scores_and_boosts_only_existing_frames(self) -> None:
        propagator = SummaryScorePropagator(
            SummaryPropagationConfig(weight=0.5, max_boost=0.2)
        )
        output = propagator.propagate(
            (
                frame("V001_00000_015", video_id="V001", final_score=0.5),
                frame("V002_00000_015", video_id="V002", final_score=0.55),
            ),
            (
                summary_result(
                    RetrievalBranch.SUMMARY_DENSE,
                    (
                        video_candidate(
                            "V001",
                            branch=RetrievalBranch.SUMMARY_DENSE,
                            normalized_score=0.3,
                        ),
                        video_candidate(
                            "V999",
                            branch=RetrievalBranch.SUMMARY_DENSE,
                            normalized_score=1.0,
                        ),
                    ),
                ),
                summary_result(
                    RetrievalBranch.SUMMARY_BM25,
                    (
                        video_candidate(
                            "V001",
                            branch=RetrievalBranch.SUMMARY_BM25,
                            normalized_score=0.4,
                        ),
                    ),
                ),
            ),
        )

        self.assertEqual(tuple(item.video_id for item in output), ("V001", "V002"))
        self.assertAlmostEqual(output[0].final_score, 0.7)
        self.assertAlmostEqual(output[0].diagnostics.summary_boost, 0.2)
        self.assertEqual(
            tuple(evidence.branch for evidence in output[0].evidence),
            (RetrievalBranch.SUMMARY_DENSE, RetrievalBranch.SUMMARY_BM25),
        )
        self.assertAlmostEqual(output[0].evidence[0].normalized_score, 0.15)
        self.assertAlmostEqual(output[0].evidence[1].normalized_score, 0.05)
        self.assertAlmostEqual(
            sum(evidence.normalized_score for evidence in output[0].evidence),
            output[0].diagnostics.summary_boost,
        )
        self.assertAlmostEqual(output[0].diagnostics.object_boost, 0.03)
        self.assertAlmostEqual(output[1].final_score, 0.55)
        self.assertEqual(len(output), 2)

    def test_missing_normalized_score_falls_back_to_rrf_and_failed_branch_is_ignored(self) -> None:
        propagator = SummaryScorePropagator(
            SummaryPropagationConfig(weight=1.0, max_boost=1.0)
        )
        output = propagator.propagate(
            (frame("V001_00000_015", video_id="V001", final_score=0.5),),
            (
                summary_result(
                    RetrievalBranch.SUMMARY_DENSE,
                    (video_candidate("V001", branch=RetrievalBranch.SUMMARY_DENSE, rank=1),),
                ),
                summary_result(
                    RetrievalBranch.SUMMARY_BM25,
                    (),
                    status=BranchStatus.FAILED,
                    warnings=("BRANCH_TIMEOUT",),
                ),
            ),
        )

        self.assertAlmostEqual(output[0].final_score, 0.5 + (1 / 61))
        self.assertAlmostEqual(output[0].diagnostics.summary_boost, 1 / 61)

    def test_empty_frames_do_not_create_summary_only_results_and_config_is_validated(self) -> None:
        propagator = SummaryScorePropagator()
        output = propagator.propagate(
            (),
            (
                summary_result(
                    RetrievalBranch.SUMMARY_DENSE,
                    (video_candidate("V001", branch=RetrievalBranch.SUMMARY_DENSE, normalized_score=1.0),),
                ),
            ),
        )
        self.assertEqual(output, ())
        with self.assertRaises(ValueError):
            SummaryPropagationConfig(weight=-0.1)
        with self.assertRaises(ValueError):
            SummaryPropagationConfig(method_name=" ")
        with self.assertRaises(ValueError):
            SummaryPropagationConfig(weight=2.0)
        with self.assertRaises(ValueError):
            SummaryPropagationConfig(max_boost=2.0)

    def test_propagate_is_idempotent_for_summary_evidence(self) -> None:
        propagator = SummaryScorePropagator(
            SummaryPropagationConfig(weight=0.1, max_boost=0.1)
        )
        summary_results = (
            summary_result(
                RetrievalBranch.SUMMARY_DENSE,
                (
                    video_candidate(
                        "V001",
                        branch=RetrievalBranch.SUMMARY_DENSE,
                        normalized_score=0.7,
                    ),
                ),
            ),
            summary_result(
                RetrievalBranch.SUMMARY_BM25,
                (
                    video_candidate(
                        "V001",
                        branch=RetrievalBranch.SUMMARY_BM25,
                        normalized_score=0.7,
                    ),
                ),
            ),
        )
        once = propagator.propagate(
            (frame("V001_00000_015", video_id="V001", final_score=0.5),),
            summary_results,
        )
        twice = propagator.propagate(once, summary_results)

        self.assertEqual(twice, once)

    def test_query_variant_weight_and_configured_rrf_apply_to_summary(self) -> None:
        disabled_q1 = SummaryScorePropagator(
            SummaryPropagationConfig(
                weight=1.0,
                max_boost=1.0,
                fallback_rrf_k=10,
                query_variant_weights={"q0": 1.0, "q1": 0.0},
            )
        )
        q1_result = summary_result(
            RetrievalBranch.SUMMARY_DENSE,
            (
                video_candidate(
                    "V001",
                    branch=RetrievalBranch.SUMMARY_DENSE,
                    rank=1,
                    variant_id="q1",
                ),
            ),
            variant_id="q1",
        )

        disabled = disabled_q1.propagate(
            (frame("V001_00000_015", video_id="V001", final_score=0.5),),
            (q1_result,),
        )
        self.assertEqual(disabled[0].diagnostics.summary_boost, 0.0)

        enabled_q1 = SummaryScorePropagator(
            SummaryPropagationConfig(
                weight=1.0,
                max_boost=1.0,
                fallback_rrf_k=10,
                query_variant_weights={"q1": 0.5},
            )
        ).propagate(
            (frame("V001_00000_015", video_id="V001", final_score=0.5),),
            (q1_result,),
        )[0]
        self.assertAlmostEqual(enabled_q1.diagnostics.summary_boost, (1 / 11) * 0.5)
        self.assertAlmostEqual(enabled_q1.evidence[0].source_normalized_score, 1 / 11)

    def test_invalid_inputs_are_rejected(self) -> None:
        propagator = SummaryScorePropagator()
        with self.assertRaises(TypeError):
            propagator.propagate(("not-a-frame",), ())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            propagator.aggregate_video_scores(("not-a-result",))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
