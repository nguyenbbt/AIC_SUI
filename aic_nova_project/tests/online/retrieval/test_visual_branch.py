from __future__ import annotations

import math
import unittest

from online.domain.candidates import FrameCandidate
from online.domain.enums import BranchStatus, CandidateLevel, QueryMode, RetrievalBranch
from online.domain.errors import (
    ContractMismatchError,
    InvalidQueryError,
    MissingMetadataError,
    ResourceUnavailableError,
)
from online.domain.query import TextQueryVariant
from online.ports.records import FrameMetadata, FrameSearchHit
from online.retrieval.branches import VisualSemanticBranch
from online.retrieval.query_builder import KISQueryBuilder
from online.testing import FakeMilvusSearchPort, FakeTextEncoder, build_integration_fixture


class StepClock:
    def __init__(self, step_sec: float = 0.01) -> None:
        self.value = 100.0
        self.step_sec = step_sec

    def __call__(self) -> float:
        value = self.value
        self.value += self.step_sec
        return value


class RecordingMetadataPort:
    def __init__(self, frames: tuple[FrameMetadata, ...]) -> None:
        self.frames = {frame.frame_id: frame for frame in frames}
        self.calls: list[tuple[str, ...]] = []

    def get_frames_by_ids(self, frame_ids: tuple[str, ...]):
        self.calls.append(tuple(frame_ids))
        return {frame_id: self.frames[frame_id] for frame_id in frame_ids if frame_id in self.frames}

    def get_ordered_frames_by_video(self, video_id: str):
        return tuple(frame for frame in self.frames.values() if frame.video_id == video_id)


class BrokenEncoder:
    dimension = 3

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def encode_texts(self, texts):
        if self.error is not None:
            raise self.error
        return ()


class BrokenMilvusPort(FakeMilvusSearchPort):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def search_visual(self, vector, top_k):
        raise self.error


class BrokenMetadataPort(RecordingMetadataPort):
    def __init__(self, error: Exception) -> None:
        super().__init__(())
        self.error = error

    def get_frames_by_ids(self, frame_ids):
        raise self.error


class VisualSemanticBranchTests(unittest.TestCase):
    def test_success_hydrates_batch_and_preserves_rank_score_and_provenance(self) -> None:
        fixture = build_integration_fixture()
        encoder = FakeTextEncoder(dimension=4)
        milvus = FakeMilvusSearchPort(visual=fixture.visual_hits)
        metadata = RecordingMetadataPort(fixture.frames)
        branch = VisualSemanticBranch(
            encoder=encoder,
            milvus=milvus,
            metadata=metadata,
            source_resource="configured_visual",
            clock=StepClock(),
        )

        result = branch.retrieve_variant(
            TextQueryVariant(variant_id="q0", text="a person riding a bicycle"),
            top_k=2,
        )

        self.assertEqual(result.branch, RetrievalBranch.VISUAL_DENSE)
        self.assertEqual(result.candidate_level, CandidateLevel.FRAME)
        self.assertEqual(result.query_variant_id, "q0")
        self.assertEqual(result.status, BranchStatus.SUCCESS)
        self.assertEqual(result.requested_top_k, 2)
        self.assertAlmostEqual(result.latency_ms, 10.0)
        self.assertEqual(result.returned_count, 2)
        self.assertEqual(metadata.calls, [tuple(hit.frame_id for hit in fixture.visual_hits[:2])])
        self.assertEqual(len(milvus.visual_calls), 1)
        self.assertEqual(milvus.visual_calls[0][1], 2)
        self.assertAlmostEqual(
            math.sqrt(sum(value * value for value in milvus.visual_calls[0][0])),
            1.0,
        )

        first = result.candidates[0]
        self.assertIsInstance(first, FrameCandidate)
        self.assertEqual(first.rank, 1)
        self.assertEqual(first.raw_score, fixture.visual_hits[0].raw_score)
        self.assertIsNone(first.normalized_score)
        self.assertEqual(first.timestamp_sec, fixture.frames[0].timestamp_sec)
        self.assertEqual(first.provenance.backend, "milvus")
        self.assertEqual(first.provenance.source_resource, "configured_visual")
        self.assertEqual(first.provenance.query_variant_id, "q0")
        self.assertEqual(first.provenance.query_text, "a person riding a bicycle")

    def test_bundle_retrieves_q0_q1_q2_independently_without_aggregation(self) -> None:
        fixture = build_integration_fixture()
        encoder = FakeTextEncoder(dimension=4)
        milvus = FakeMilvusSearchPort(visual=fixture.visual_hits[:1])
        branch = VisualSemanticBranch(
            encoder=encoder,
            milvus=milvus,
            metadata=RecordingMetadataPort(fixture.frames),
            clock=StepClock(),
        )
        query = KISQueryBuilder().build(
            "original query",
            mode=QueryMode.KIS_TEXT,
            paraphrases=("first paraphrase", "second paraphrase"),
            query_id="query-visual",
        )

        results = branch.retrieve(query, top_k=1)

        self.assertEqual(tuple(result.query_variant_id for result in results), ("q0", "q1", "q2"))
        self.assertEqual(
            encoder.calls,
            [("original query",), ("first paraphrase",), ("second paraphrase",)],
        )
        self.assertEqual(len(milvus.visual_calls), 3)
        self.assertEqual([call[1] for call in milvus.visual_calls], [1, 1, 1])
        self.assertEqual(len({call[0] for call in milvus.visual_calls}), 3)
        self.assertTrue(all(result.returned_count == 1 for result in results))

    def test_empty_search_is_success_and_does_not_query_metadata(self) -> None:
        metadata = RecordingMetadataPort(())
        branch = VisualSemanticBranch(
            encoder=FakeTextEncoder(),
            milvus=FakeMilvusSearchPort(),
            metadata=metadata,
            clock=StepClock(),
        )

        result = branch.retrieve_variant(TextQueryVariant(variant_id="q0", text="query"), top_k=5)

        self.assertEqual(result.status, BranchStatus.SUCCESS)
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.warnings, ())
        self.assertEqual(metadata.calls, [])

    def test_missing_metadata_surfaces_typed_error_until_policy_is_approved(self) -> None:
        fixture = build_integration_fixture()
        metadata = RecordingMetadataPort(fixture.frames)
        branch = VisualSemanticBranch(
            encoder=FakeTextEncoder(),
            milvus=FakeMilvusSearchPort(visual=(fixture.missing_metadata_hit,)),
            metadata=metadata,
            clock=StepClock(),
        )

        with self.assertRaises(MissingMetadataError) as raised:
            branch.retrieve_variant(TextQueryVariant(variant_id="q0", text="query"), top_k=1)
        self.assertEqual(
            raised.exception.details,
            {"branch": "visual_dense", "missing_count": 1, "hit_count": 1},
        )

    def test_cross_database_identity_mismatch_is_not_hidden(self) -> None:
        hit = FrameSearchHit(frame_id="L21_V001_001", video_id="L21_V001", raw_score=0.9)
        conflicting = FrameMetadata(
            frame_id="L21_V999_001",
            video_id="L21_V999",
            keyframe_no=1,
            local_index=0,
            timestamp_sec=2.0,
            fps=25.0,
            source_frame_idx=50,
            image_rel_path="L21_V999/001.jpg",
        )
        metadata = RecordingMetadataPort((conflicting,))
        metadata.frames[hit.frame_id] = conflicting
        branch = VisualSemanticBranch(
            encoder=FakeTextEncoder(),
            milvus=FakeMilvusSearchPort(visual=(hit,)),
            metadata=metadata,
            clock=StepClock(),
        )

        with self.assertRaises(ContractMismatchError):
            branch.retrieve_variant(TextQueryVariant(variant_id="q0", text="query"), top_k=1)

    def test_invalid_top_k_is_rejected_and_query_builder_keeps_branch_policy_neutral(self) -> None:
        encoder = FakeTextEncoder()
        milvus = FakeMilvusSearchPort()
        branch = VisualSemanticBranch(
            encoder=encoder,
            milvus=milvus,
            metadata=RecordingMetadataPort(()),
        )
        variant = TextQueryVariant(variant_id="q0", text="query")
        for top_k in (0, -1, True, 1.5):
            with self.subTest(top_k=top_k):
                with self.assertRaises(InvalidQueryError):
                    branch.retrieve_variant(variant, top_k=top_k)  # type: ignore[arg-type]

        bundle = KISQueryBuilder().build(
            "query",
            mode=QueryMode.KIS_TEXT,
            enabled_branches=(RetrievalBranch.OCR_DENSE,),
        )
        self.assertEqual(bundle.enabled_branches, (RetrievalBranch.OCR_DENSE,))
        self.assertEqual(encoder.calls, [])
        self.assertEqual(milvus.visual_calls, [])

    def test_encoder_milvus_and_metadata_failures_surface(self) -> None:
        no_vector_branch = VisualSemanticBranch(
            encoder=BrokenEncoder(),
            milvus=FakeMilvusSearchPort(),
            metadata=RecordingMetadataPort(()),
        )
        variant = TextQueryVariant(variant_id="q0", text="query")
        with self.assertRaises(ContractMismatchError):
            no_vector_branch.retrieve_variant(variant, top_k=1)

        backend_error = ResourceUnavailableError("Milvus unavailable")
        broken_encoder_branch = VisualSemanticBranch(
            encoder=BrokenEncoder(backend_error),
            milvus=FakeMilvusSearchPort(),
            metadata=RecordingMetadataPort(()),
        )
        with self.assertRaises(ResourceUnavailableError) as raised:
            broken_encoder_branch.retrieve_variant(variant, top_k=1)
        self.assertIs(raised.exception, backend_error)

        milvus_error = ResourceUnavailableError("Milvus unavailable")
        broken_milvus_branch = VisualSemanticBranch(
            encoder=FakeTextEncoder(),
            milvus=BrokenMilvusPort(milvus_error),
            metadata=RecordingMetadataPort(()),
        )
        with self.assertRaises(ResourceUnavailableError) as raised:
            broken_milvus_branch.retrieve_variant(variant, top_k=1)
        self.assertIs(raised.exception, milvus_error)

        hit = FrameSearchHit(frame_id="L21_V001_001", video_id="L21_V001", raw_score=0.8)
        metadata_error = ResourceUnavailableError("SQLite unavailable")
        broken_metadata_branch = VisualSemanticBranch(
            encoder=FakeTextEncoder(),
            milvus=FakeMilvusSearchPort(visual=(hit,)),
            metadata=BrokenMetadataPort(metadata_error),
        )
        with self.assertRaises(ResourceUnavailableError) as raised:
            broken_metadata_branch.retrieve_variant(variant, top_k=1)
        self.assertIs(raised.exception, metadata_error)


if __name__ == "__main__":
    unittest.main()
