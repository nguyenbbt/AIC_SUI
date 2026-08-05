from __future__ import annotations

import unittest

from online.domain.candidates import ASRIntervalCandidate, BranchResult, CandidateProvenance
from online.domain.enums import BranchStatus, CandidateLevel, RetrievalBranch
from online.domain.errors import ContractMismatchError
from online.ports.records import FrameMetadata
from online.ranking.asr_mapper import ASRIntervalFrameMapper, ASRMappingConfig
from online.testing import FakeMetadataReaderPort


def interval(
    interval_id: str,
    *,
    start: float,
    end: float,
    rank: int = 1,
    raw_score: float = 0.9,
    normalized_score: float | None = 0.6,
    video_id: str = "V001",
) -> ASRIntervalCandidate:
    return ASRIntervalCandidate(
        video_id=video_id,
        interval_id=interval_id,
        start_time_sec=start,
        end_time_sec=end,
        rank=rank,
        raw_score=raw_score,
        normalized_score=normalized_score,
        text="speech evidence",
        provenance=CandidateProvenance(
            branch=RetrievalBranch.ASR_DENSE,
            backend="milvus",
            source_resource="asr_features",
            query_variant_id="q0",
            query_text="query",
        ),
    )


def frame(
    frame_id: str,
    *,
    timestamp: float,
    shot_id: int = 0,
    video_id: str = "V001",
) -> FrameMetadata:
    return FrameMetadata(
        frame_id=frame_id,
        video_id=video_id,
        shot_id=shot_id,
        timestamp_sec=timestamp,
        source_frame_idx=max(0, int(timestamp * 30)),
        image_rel_path=f"keyframes/{video_id}/{frame_id}.webp",
    )


def asr_result(candidates: tuple[ASRIntervalCandidate, ...]) -> BranchResult[ASRIntervalCandidate]:
    return BranchResult[ASRIntervalCandidate](
        branch=RetrievalBranch.ASR_DENSE,
        candidate_level=CandidateLevel.ASR_INTERVAL,
        query_variant_id="q0",
        candidates=candidates,
        requested_top_k=10,
        latency_ms=1.0,
        status=BranchStatus.SUCCESS,
    )


class ASRIntervalFrameMapperTests(unittest.TestCase):
    def test_maps_single_and_multiple_frames_inside_interval_with_distributed_scores(self) -> None:
        mapper = ASRIntervalFrameMapper()
        mapped = mapper.map_interval(
            interval("0", start=1.0, end=6.0, raw_score=0.9, normalized_score=0.6),
            (
                frame("V001_00000_015", timestamp=1.5),
                frame("V001_00000_050", timestamp=5.0),
                frame("V001_00001_050", timestamp=10.0, shot_id=1),
            ),
        )

        self.assertEqual(tuple(item.frame_id for item in mapped), ("V001_00000_015", "V001_00000_050"))
        self.assertEqual(tuple(item.provenance.backend for item in mapped), ("milvus", "milvus"))
        self.assertEqual(tuple(item.provenance.source_candidate_id for item in mapped), ("0", "0"))
        self.assertEqual(tuple(item.provenance.source_start_time_sec for item in mapped), (1.0, 1.0))
        self.assertEqual(tuple(item.provenance.source_end_time_sec for item in mapped), (6.0, 6.0))
        self.assertEqual(tuple(item.provenance.source_normalized_score for item in mapped), (0.6, 0.6))
        self.assertEqual(tuple(item.raw_score for item in mapped), (0.45, 0.45))
        self.assertEqual(tuple(item.normalized_score for item in mapped), (0.3, 0.3))

    def test_no_frame_in_interval_counts_mapping_loss_in_branch_result(self) -> None:
        mapper = ASRIntervalFrameMapper()
        result = mapper.map_result(
            asr_result((interval("0", start=20.0, end=21.0),)),
            FakeMetadataReaderPort((frame("V001_00000_015", timestamp=1.5),)),
        )

        self.assertEqual(result.branch_result.candidate_level, CandidateLevel.FRAME)
        self.assertEqual(result.branch_result.candidates, ())
        self.assertEqual(result.mapping_loss_count, 1)
        self.assertEqual(result.policy_name, "timestamp_inclusive_distributed_v1")

    def test_interval_boundaries_are_inclusive_and_output_is_reranked_deterministically(self) -> None:
        mapper = ASRIntervalFrameMapper()
        result = mapper.map_result(
            asr_result(
                (
                    interval("1", start=10.0, end=10.0, rank=2),
                    interval("0", start=1.5, end=1.5, rank=1),
                )
            ),
            FakeMetadataReaderPort(
                (
                    frame("V001_00000_015", timestamp=1.5),
                    frame("V001_00001_050", timestamp=10.0, shot_id=1),
                )
            ),
        )

        self.assertEqual(
            tuple(candidate.frame_id for candidate in result.branch_result.candidates),
            ("V001_00000_015", "V001_00001_050"),
        )
        self.assertEqual(tuple(candidate.rank for candidate in result.branch_result.candidates), (1, 2))

    def test_overlapping_intervals_keep_distinct_evidence_for_later_fusion(self) -> None:
        mapper = ASRIntervalFrameMapper()
        result = mapper.map_result(
            asr_result(
                (
                    interval("0", start=1.0, end=6.0, rank=1),
                    interval("1", start=5.0, end=11.0, rank=2),
                )
            ),
            FakeMetadataReaderPort(
                (
                    frame("V001_00000_015", timestamp=1.5),
                    frame("V001_00000_050", timestamp=5.0),
                    frame("V001_00001_050", timestamp=10.0, shot_id=1),
                )
            ),
        )

        self.assertEqual(
            tuple(candidate.frame_id for candidate in result.branch_result.candidates),
            (
                "V001_00000_015",
                "V001_00000_050",
                "V001_00000_050",
                "V001_00001_050",
            ),
        )
        self.assertEqual(result.mapping_loss_count, 0)

    def test_long_interval_is_limited_and_scores_use_kept_frame_count(self) -> None:
        mapper = ASRIntervalFrameMapper(ASRMappingConfig(max_frames_per_interval=2))
        mapped = mapper.map_interval(
            interval("0", start=0.0, end=10.0, raw_score=1.0, normalized_score=0.8),
            (
                frame("V001_00000_000", timestamp=0.0),
                frame("V001_00000_040", timestamp=4.0),
                frame("V001_00000_060", timestamp=6.0),
                frame("V001_00000_100", timestamp=10.0),
            ),
        )

        self.assertEqual(tuple(item.frame_id for item in mapped), ("V001_00000_040", "V001_00000_060"))
        self.assertEqual(tuple(item.raw_score for item in mapped), (0.5, 0.5))
        self.assertEqual(tuple(item.normalized_score for item in mapped), (0.4, 0.4))

    def test_missing_interval_normalized_score_uses_interval_level_rrf_before_mapping(self) -> None:
        mapper = ASRIntervalFrameMapper(ASRMappingConfig(interval_rrf_k=10))
        mapped = mapper.map_interval(
            interval("42", start=1.0, end=6.0, rank=1, normalized_score=None),
            (
                frame("V001_00000_015", timestamp=1.5),
                frame("V001_00000_050", timestamp=5.0),
            ),
        )

        self.assertAlmostEqual(sum(item.normalized_score for item in mapped), 1 / 11)
        self.assertEqual(
            tuple(item.provenance.source_resource for item in mapped),
            ("asr_features", "asr_features"),
        )
        self.assertEqual(
            tuple(item.provenance.source_candidate_id for item in mapped),
            ("42", "42"),
        )
        self.assertEqual(
            tuple(item.provenance.source_normalized_score for item in mapped),
            (1 / 11, 1 / 11),
        )

    def test_video_without_metadata_is_mapping_loss_not_cross_video_mapping(self) -> None:
        mapper = ASRIntervalFrameMapper()
        result = mapper.map_result(
            asr_result((interval("0", start=1.0, end=2.0),)),
            FakeMetadataReaderPort(()),
        )

        self.assertEqual(result.branch_result.candidates, ())
        self.assertEqual(result.mapping_loss_count, 1)

    def test_rejects_wrong_level_wrong_video_metadata_and_invalid_policy(self) -> None:
        mapper = ASRIntervalFrameMapper()
        with self.assertRaises(ContractMismatchError):
            mapper.map_result(
                BranchResult(
                    branch=RetrievalBranch.ASR_DENSE,
                    candidate_level=CandidateLevel.FRAME,
                    query_variant_id="q0",
                    candidates=(),
                    requested_top_k=10,
                    latency_ms=1.0,
                    status=BranchStatus.SUCCESS,
                ),
                FakeMetadataReaderPort(()),
            )
        with self.assertRaises(ContractMismatchError):
            mapper.map_interval(
                interval("0", start=1.0, end=2.0),
                (frame("V002_00000_015", timestamp=1.5, video_id="V002"),),
            )
        with self.assertRaises(ValueError):
            ASRMappingConfig(policy_name=" ")
        with self.assertRaises(ValueError):
            ASRMappingConfig(max_frames_per_interval=0)


if __name__ == "__main__":
    unittest.main()
