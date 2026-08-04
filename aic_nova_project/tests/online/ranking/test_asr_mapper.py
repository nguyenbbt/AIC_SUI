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
    video_id: str = "L21_V001",
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
    keyframe_no: int,
    *,
    timestamp: float,
    video_id: str = "L21_V001",
    fps: float = 30.0,
    source_frame_idx: int | None = None,
) -> FrameMetadata:
    return FrameMetadata(
        frame_id=f"{video_id}_{keyframe_no:03d}",
        video_id=video_id,
        keyframe_no=keyframe_no,
        local_index=keyframe_no - 1,
        timestamp_sec=timestamp,
        fps=fps,
        source_frame_idx=(
            round(timestamp * fps)
            if source_frame_idx is None
            else source_frame_idx
        ),
        image_rel_path=f"keyframes/{video_id}/{keyframe_no:03d}.jpg",
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
            interval("speech", start=1.0, end=6.0, raw_score=0.9, normalized_score=0.6),
            (
                frame(1, timestamp=1.5),
                frame(2, timestamp=5.0),
                frame(3, timestamp=10.0),
            ),
        )

        self.assertEqual(tuple(item.frame_id for item in mapped), ("L21_V001_001", "L21_V001_002"))
        self.assertEqual(tuple(item.keyframe_no for item in mapped), (1, 2))
        self.assertEqual(tuple(item.local_index for item in mapped), (0, 1))
        self.assertEqual(tuple(item.source_frame_idx for item in mapped), (45, 150))
        self.assertEqual(tuple(item.provenance.backend for item in mapped), ("milvus", "milvus"))
        self.assertEqual(tuple(item.provenance.source_candidate_id for item in mapped), ("speech", "speech"))
        self.assertEqual(tuple(item.provenance.source_start_time_sec for item in mapped), (1.0, 1.0))
        self.assertEqual(tuple(item.provenance.source_end_time_sec for item in mapped), (6.0, 6.0))
        self.assertEqual(tuple(item.provenance.source_normalized_score for item in mapped), (0.6, 0.6))
        self.assertEqual(tuple(item.raw_score for item in mapped), (0.45, 0.45))
        self.assertEqual(tuple(item.normalized_score for item in mapped), (0.3, 0.3))

    def test_no_frame_in_interval_counts_mapping_loss_in_branch_result(self) -> None:
        mapper = ASRIntervalFrameMapper()
        result = mapper.map_result(
            asr_result((interval("silent", start=20.0, end=21.0),)),
            FakeMetadataReaderPort((frame(1, timestamp=1.5),)),
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
                    interval("later", start=10.0, end=10.0, rank=2),
                    interval("early", start=1.5, end=1.5, rank=1),
                )
            ),
            FakeMetadataReaderPort(
                (
                    frame(1, timestamp=1.5),
                    frame(2, timestamp=10.0),
                )
            ),
        )

        self.assertEqual(
            tuple(candidate.frame_id for candidate in result.branch_result.candidates),
            ("L21_V001_001", "L21_V001_002"),
        )
        self.assertEqual(tuple(candidate.rank for candidate in result.branch_result.candidates), (1, 2))

    def test_preserves_timestamp_and_source_index_without_recomputing_from_fps(self) -> None:
        source = frame(
            1,
            timestamp=1.25,
            fps=120.0,
            source_frame_idx=7,
        )
        mapped = ASRIntervalFrameMapper().map_interval(
            interval("metadata-source", start=1.25, end=1.25),
            (source,),
        )

        self.assertEqual(mapped[0].timestamp_sec, 1.25)
        self.assertEqual(mapped[0].source_frame_idx, 7)
        self.assertNotEqual(mapped[0].source_frame_idx, round(source.timestamp_sec * source.fps))

    def test_overlapping_intervals_keep_distinct_evidence_for_later_fusion(self) -> None:
        mapper = ASRIntervalFrameMapper()
        result = mapper.map_result(
            asr_result(
                (
                    interval("first", start=1.0, end=6.0, rank=1),
                    interval("second", start=5.0, end=11.0, rank=2),
                )
            ),
            FakeMetadataReaderPort(
                (
                    frame(1, timestamp=1.5),
                    frame(2, timestamp=5.0),
                    frame(3, timestamp=10.0),
                )
            ),
        )

        self.assertEqual(
            tuple(candidate.frame_id for candidate in result.branch_result.candidates),
            (
                "L21_V001_001",
                "L21_V001_002",
                "L21_V001_002",
                "L21_V001_003",
            ),
        )
        self.assertEqual(result.mapping_loss_count, 0)

    def test_long_interval_is_limited_and_scores_use_kept_frame_count(self) -> None:
        mapper = ASRIntervalFrameMapper(ASRMappingConfig(max_frames_per_interval=2))
        mapped = mapper.map_interval(
            interval("long", start=0.0, end=10.0, raw_score=1.0, normalized_score=0.8),
            (
                frame(1, timestamp=0.0),
                frame(2, timestamp=4.0),
                frame(3, timestamp=6.0),
                frame(4, timestamp=10.0),
            ),
        )

        self.assertEqual(tuple(item.frame_id for item in mapped), ("L21_V001_002", "L21_V001_003"))
        self.assertEqual(tuple(item.raw_score for item in mapped), (0.5, 0.5))
        self.assertEqual(tuple(item.normalized_score for item in mapped), (0.4, 0.4))

    def test_missing_interval_normalized_score_uses_interval_level_rrf_before_mapping(self) -> None:
        mapper = ASRIntervalFrameMapper(ASRMappingConfig(interval_rrf_k=10))
        mapped = mapper.map_interval(
            interval("speech-42", start=1.0, end=6.0, rank=1, normalized_score=None),
            (
                frame(1, timestamp=1.5),
                frame(2, timestamp=5.0),
            ),
        )

        self.assertAlmostEqual(sum(item.normalized_score for item in mapped), 1 / 11)
        self.assertEqual(
            tuple(item.provenance.source_resource for item in mapped),
            ("asr_features", "asr_features"),
        )
        self.assertEqual(
            tuple(item.provenance.source_candidate_id for item in mapped),
            ("speech-42", "speech-42"),
        )
        self.assertEqual(
            tuple(item.provenance.source_normalized_score for item in mapped),
            (1 / 11, 1 / 11),
        )

    def test_video_without_metadata_is_mapping_loss_not_cross_video_mapping(self) -> None:
        mapper = ASRIntervalFrameMapper()
        result = mapper.map_result(
            asr_result((interval("missing-video", start=1.0, end=2.0),)),
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
                interval("speech", start=1.0, end=2.0),
                (frame(1, timestamp=1.5, video_id="L21_V002"),),
            )
        with self.assertRaises(ValueError):
            ASRMappingConfig(policy_name=" ")
        with self.assertRaises(ValueError):
            ASRMappingConfig(max_frames_per_interval=0)


if __name__ == "__main__":
    unittest.main()
