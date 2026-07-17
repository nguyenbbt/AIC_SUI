from __future__ import annotations

import unittest

from online.domain.enums import BranchStatus, QueryMode, RetrievalBranch
from online.domain.errors import InvalidQueryError, MissingMetadataError, ResourceUnavailableError
from online.domain.query import TextQueryVariant
from online.ports.records import FrameMetadata, FrameSearchHit
from online.retrieval.branches import OCRLexicalBranch, OCRSemanticBranch
from online.retrieval.query_builder import KISQueryBuilder
from online.testing import (
    FakeElasticsearchSearchPort,
    FakeMilvusSearchPort,
    FakeTextEncoder,
    build_integration_fixture,
)


class StepClock:
    def __init__(self, step_sec: float = 0.01) -> None:
        self.value = 10.0
        self.step_sec = step_sec

    def __call__(self) -> float:
        value = self.value
        self.value += self.step_sec
        return value


class RecordingMetadataPort:
    def __init__(self, frames: tuple[FrameMetadata, ...]) -> None:
        self.frames = {frame.frame_id: frame for frame in frames}
        self.calls: list[tuple[str, ...]] = []

    def get_frames_by_ids(self, frame_ids):
        values = tuple(frame_ids)
        self.calls.append(values)
        return {frame_id: self.frames[frame_id] for frame_id in values if frame_id in self.frames}

    def get_ordered_frames_by_video(self, video_id: str):
        return tuple(frame for frame in self.frames.values() if frame.video_id == video_id)


class BrokenElasticsearchPort(FakeElasticsearchSearchPort):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def search_ocr(self, query, top_k, *, fuzzy=None):
        raise self.error


class BrokenOCRMilvusPort(FakeMilvusSearchPort):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def search_ocr(self, vector, top_k):
        raise self.error


class OCRBranchTests(unittest.TestCase):
    def test_lexical_uses_only_q0_fuzzy_config_and_elasticsearch_provenance(self) -> None:
        fixture = build_integration_fixture()
        elasticsearch = FakeElasticsearchSearchPort(ocr=fixture.ocr_hits)
        metadata = RecordingMetadataPort(fixture.frames)
        branch = OCRLexicalBranch(
            elasticsearch=elasticsearch,
            metadata=metadata,
            fuzzy=True,
            source_resource="configured_ocr_index",
            clock=StepClock(),
        )
        query = KISQueryBuilder().build(
            "store sign",
            mode=QueryMode.KIS_TEXT,
            paraphrases=("shop text", "words on storefront"),
        )

        results = branch.retrieve(query, top_k=2)

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.branch, RetrievalBranch.OCR_BM25)
        self.assertEqual(result.query_variant_id, "q0")
        self.assertEqual(result.status, BranchStatus.SUCCESS)
        self.assertEqual(elasticsearch.ocr_calls, [("store sign", 2, True)])
        self.assertEqual(result.returned_count, 2)
        self.assertEqual(result.candidates[0].raw_score, fixture.ocr_hits[0].raw_score)
        self.assertIsNone(result.candidates[0].normalized_score)
        self.assertEqual(result.candidates[0].provenance.backend, "elasticsearch")
        self.assertEqual(
            result.candidates[0].provenance.source_resource,
            "configured_ocr_index",
        )
        self.assertEqual(result.candidates[0].provenance.query_text, "store sign")
        self.assertEqual(len(metadata.calls), 1)

    def test_semantic_runs_all_variants_independently_and_hydrates_shot_id(self) -> None:
        fixture = build_integration_fixture()
        semantic_hits = tuple(
            FrameSearchHit(
                frame_id=hit.frame_id,
                video_id=hit.video_id,
                shot_id=None,
                raw_score=hit.raw_score,
            )
            for hit in fixture.ocr_hits
        )
        encoder = FakeTextEncoder(dimension=6)
        milvus = FakeMilvusSearchPort(ocr=semantic_hits)
        branch = OCRSemanticBranch(
            encoder=encoder,
            milvus=milvus,
            metadata=RecordingMetadataPort(fixture.frames),
            clock=StepClock(),
        )
        query = KISQueryBuilder().build(
            "original OCR query",
            mode=QueryMode.KIS_VIDEO,
            paraphrases=("OCR paraphrase one", "OCR paraphrase two"),
        )

        results = branch.retrieve(query, top_k=2)

        self.assertEqual(tuple(result.query_variant_id for result in results), ("q0", "q1", "q2"))
        self.assertEqual(
            encoder.calls,
            [("original OCR query",), ("OCR paraphrase one",), ("OCR paraphrase two",)],
        )
        self.assertEqual(len(milvus.ocr_calls), 3)
        self.assertEqual(len({call[0] for call in milvus.ocr_calls}), 3)
        self.assertTrue(all(call[1] == 2 for call in milvus.ocr_calls))
        self.assertTrue(all(result.branch is RetrievalBranch.OCR_DENSE for result in results))
        self.assertTrue(all(result.returned_count == 2 for result in results))
        self.assertEqual(results[0].candidates[0].shot_id, fixture.frames[1].shot_id)
        self.assertEqual(results[0].candidates[0].provenance.backend, "milvus")
        self.assertEqual(results[0].candidates[0].provenance.source_resource, "ocr_features")

    def test_empty_lexical_and_semantic_searches_are_success_without_hydration(self) -> None:
        metadata = RecordingMetadataPort(())
        query = KISQueryBuilder().build("query", mode=QueryMode.KIS_TEXT)
        lexical = OCRLexicalBranch(
            elasticsearch=FakeElasticsearchSearchPort(),
            metadata=metadata,
            clock=StepClock(),
        )
        semantic = OCRSemanticBranch(
            encoder=FakeTextEncoder(),
            milvus=FakeMilvusSearchPort(),
            metadata=metadata,
            clock=StepClock(),
        )

        lexical_result = lexical.retrieve(query, top_k=3)[0]
        semantic_result = semantic.retrieve(query, top_k=3)[0]

        self.assertEqual(lexical_result.status, BranchStatus.SUCCESS)
        self.assertEqual(semantic_result.status, BranchStatus.SUCCESS)
        self.assertEqual(lexical_result.candidates, ())
        self.assertEqual(semantic_result.candidates, ())
        self.assertEqual(metadata.calls, [])

    def test_missing_metadata_is_typed_and_identifies_each_ocr_branch(self) -> None:
        fixture = build_integration_fixture()
        missing_hit = fixture.missing_metadata_hit
        metadata = RecordingMetadataPort(fixture.frames)
        query = KISQueryBuilder().build("query", mode=QueryMode.KIS_TEXT)
        branches = (
            (
                OCRLexicalBranch(
                    elasticsearch=FakeElasticsearchSearchPort(ocr=(missing_hit,)),
                    metadata=metadata,
                ),
                "ocr_bm25",
            ),
            (
                OCRSemanticBranch(
                    encoder=FakeTextEncoder(),
                    milvus=FakeMilvusSearchPort(ocr=(missing_hit,)),
                    metadata=metadata,
                ),
                "ocr_dense",
            ),
        )

        for branch, expected_name in branches:
            with self.subTest(branch=expected_name):
                with self.assertRaises(MissingMetadataError) as raised:
                    branch.retrieve(query, top_k=1)
                self.assertEqual(raised.exception.details["branch"], expected_name)
                self.assertEqual(raised.exception.details["missing_count"], 1)

    def test_lexical_rejects_q1_and_disabled_branches_do_no_work(self) -> None:
        lexical_backend = FakeElasticsearchSearchPort()
        semantic_backend = FakeMilvusSearchPort()
        encoder = FakeTextEncoder()
        metadata = RecordingMetadataPort(())
        lexical = OCRLexicalBranch(elasticsearch=lexical_backend, metadata=metadata)
        semantic = OCRSemanticBranch(
            encoder=encoder,
            milvus=semantic_backend,
            metadata=metadata,
        )

        with self.assertRaises(InvalidQueryError):
            lexical.retrieve_variant(
                TextQueryVariant(variant_id="q1", text="paraphrase"),
                top_k=1,
            )
        disabled = KISQueryBuilder().build(
            "query",
            mode=QueryMode.KIS_TEXT,
            enabled_branches=(RetrievalBranch.VISUAL_DENSE,),
        )
        with self.assertRaises(InvalidQueryError):
            lexical.retrieve(disabled, top_k=1)
        with self.assertRaises(InvalidQueryError):
            semantic.retrieve(disabled, top_k=1)
        self.assertEqual(lexical_backend.ocr_calls, [])
        self.assertEqual(semantic_backend.ocr_calls, [])
        self.assertEqual(encoder.calls, [])

    def test_ocr_backend_failures_propagate_without_becoming_empty_success(self) -> None:
        query = KISQueryBuilder().build("query", mode=QueryMode.KIS_TEXT)
        elasticsearch_error = ResourceUnavailableError("Elasticsearch unavailable")
        lexical = OCRLexicalBranch(
            elasticsearch=BrokenElasticsearchPort(elasticsearch_error),
            metadata=RecordingMetadataPort(()),
        )
        with self.assertRaises(ResourceUnavailableError) as raised:
            lexical.retrieve(query, top_k=1)
        self.assertIs(raised.exception, elasticsearch_error)

        milvus_error = ResourceUnavailableError("Milvus unavailable")
        semantic = OCRSemanticBranch(
            encoder=FakeTextEncoder(),
            milvus=BrokenOCRMilvusPort(milvus_error),
            metadata=RecordingMetadataPort(()),
        )
        with self.assertRaises(ResourceUnavailableError) as raised:
            semantic.retrieve(query, top_k=1)
        self.assertIs(raised.exception, milvus_error)

    def test_invalid_fuzzy_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OCRLexicalBranch(
                elasticsearch=FakeElasticsearchSearchPort(),
                metadata=RecordingMetadataPort(()),
                fuzzy="AUTO",  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
