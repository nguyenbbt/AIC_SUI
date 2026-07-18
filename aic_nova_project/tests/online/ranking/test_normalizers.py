from __future__ import annotations

import unittest

from online.domain.candidates import CandidateProvenance, FrameCandidate
from online.domain.enums import RetrievalBranch
from online.ranking.normalizers import MinMaxScoreNormalizer, RRFScoreNormalizer


def frame(frame_id: str, *, rank: int, raw_score: float) -> FrameCandidate:
    return FrameCandidate(
        frame_id=frame_id,
        video_id="V001",
        shot_id=0,
        timestamp_sec=float(rank),
        rank=rank,
        raw_score=raw_score,
        provenance=CandidateProvenance(
            branch=RetrievalBranch.VISUAL_DENSE,
            backend="milvus",
            source_resource="visual_features",
            query_variant_id="q0",
            query_text="query",
        ),
    )


class ScoreNormalizerTests(unittest.TestCase):
    def test_rrf_uses_rank_not_raw_backend_scale(self) -> None:
        candidates = (
            frame("V001_00000_015", rank=1, raw_score=-100.0),
            frame("V001_00000_050", rank=2, raw_score=10000.0),
        )

        normalized = RRFScoreNormalizer(k=10).normalize(candidates)

        self.assertGreater(normalized[0].normalized_score, normalized[1].normalized_score)
        self.assertAlmostEqual(normalized[0].normalized_score, 1 / 11)
        self.assertEqual(candidates[0].normalized_score, None)

    def test_min_max_is_branch_local_and_handles_equal_scores(self) -> None:
        normalized = MinMaxScoreNormalizer().normalize(
            (
                frame("V001_00000_015", rank=1, raw_score=4.0),
                frame("V001_00000_050", rank=2, raw_score=7.0),
                frame("V001_00001_050", rank=3, raw_score=10.0),
            )
        )
        self.assertEqual(
            tuple(candidate.normalized_score for candidate in normalized),
            (0.0, 0.5, 1.0),
        )

        equal = MinMaxScoreNormalizer().normalize(
            (
                frame("V001_00000_015", rank=1, raw_score=5.0),
                frame("V001_00000_050", rank=2, raw_score=5.0),
            )
        )
        self.assertEqual(tuple(candidate.normalized_score for candidate in equal), (1.0, 1.0))

    def test_invalid_normalizer_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RRFScoreNormalizer(k=0)
        with self.assertRaises(TypeError):
            MinMaxScoreNormalizer().normalize(("not-a-candidate",))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
