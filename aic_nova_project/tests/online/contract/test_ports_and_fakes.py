from __future__ import annotations

import unittest

from online.ports import (
    ElasticsearchSearchPort,
    MetadataReaderPort,
    MilvusSearchPort,
    ObjectReaderPort,
)
from online.adapters.elasticsearch import ElasticsearchSearchAdapter
from online.adapters.milvus import MilvusSearchAdapter
from online.adapters.sqlite import SQLiteReadAdapter
from online.config import ElasticsearchResourceConfig, MilvusResourceConfig, SQLiteResourceConfig
from online.domain.enums import RetrievalBranch
from online.domain.errors import (
    BranchTimeoutError,
    InvalidQueryError,
    ResourceUnavailableError,
)
from online.testing import FakeBranchBehavior, build_integration_fixture
from online.testing import FakeElasticsearchSearchPort


class PortConformanceTests(unittest.TestCase):
    def test_shared_fakes_implement_runtime_protocols(self) -> None:
        fixture = build_integration_fixture()
        self.assertIsInstance(fixture.milvus(), MilvusSearchPort)
        self.assertIsInstance(fixture.metadata(), MetadataReaderPort)
        self.assertIsInstance(fixture.object_reader(), ObjectReaderPort)
        self.assertIsInstance(
            FakeElasticsearchSearchPort(
                ocr=fixture.ocr_hits,
                asr=fixture.asr_hits,
                summary=fixture.summary_hits,
            ),
            ElasticsearchSearchPort,
        )

    def test_real_adapters_expose_the_same_ports_without_connecting(self) -> None:
        self.assertIsInstance(
            MilvusSearchAdapter(MilvusResourceConfig(), backend=object()), MilvusSearchPort
        )
        self.assertIsInstance(
            ElasticsearchSearchAdapter(ElasticsearchResourceConfig(), client=object()),
            ElasticsearchSearchPort,
        )
        sqlite = SQLiteReadAdapter(SQLiteResourceConfig(), connection=None)
        self.assertIsInstance(sqlite, MetadataReaderPort)
        self.assertIsInstance(sqlite, ObjectReaderPort)

    def test_fixture_has_two_videos_asr_edges_objects_and_missing_metadata(self) -> None:
        fixture = build_integration_fixture()
        self.assertEqual(
            {frame.video_id for frame in fixture.frames},
            {"L21_V001", "L21_V002"},
        )
        self.assertEqual({hit.interval_id for hit in fixture.asr_hits}, {"0", "1", "2"})
        self.assertNotIn(
            fixture.missing_metadata_hit.frame_id,
            fixture.metadata().get_frames_by_ids([fixture.missing_metadata_hit.frame_id]),
        )
        filtered = fixture.object_reader().get_objects_by_frame_ids(
            ["L21_V001_00002_085"], label="person", min_confidence=0.5
        )
        self.assertEqual(len(filtered["L21_V001_00002_085"]), 1)
        self.assertEqual(filtered["L21_V001_00002_085"][0].label, "person")
        by_label = fixture.object_reader().get_objects_by_frame_ids(
            ["L21_V001_00002_085"], label="car", min_confidence=0.5
        )
        self.assertEqual(by_label["L21_V001_00002_085"][0].label, "car")

    def test_fakes_validate_inputs_record_calls_and_preserve_empty_success(self) -> None:
        fixture = build_integration_fixture()
        fake = fixture.milvus()
        with self.assertRaises(InvalidQueryError):
            fake.search_visual((1.0, 0.0), 0)
        with self.assertRaises(InvalidQueryError):
            fake.search_visual((float("nan"), 0.0), 2)
        result = fake.search_visual((1.0, 0.0), 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(fake.calls[-1].branch, RetrievalBranch.VISUAL_DENSE)
        self.assertEqual(fake.calls[-1].top_k, 1)
        empty = fixture.milvus(
            behaviors={
                RetrievalBranch.SUMMARY_DENSE: FakeBranchBehavior(
                    error=BranchTimeoutError("simulated timeout")
                )
            }
        )
        with self.assertRaises(BranchTimeoutError):
            empty.search_summary((1.0, 0.0), 1)
        es = fixture.elasticsearch(
            behaviors={
                RetrievalBranch.OCR_BM25: FakeBranchBehavior(
                    error=ResourceUnavailableError("simulated unavailable")
                )
            }
        )
        with self.assertRaises(ResourceUnavailableError):
            es.search_ocr("query", 1, fuzzy=False)

    def test_fake_metadata_and_objects_match_adapter_input_validation(self) -> None:
        fixture = build_integration_fixture()
        metadata = fixture.metadata()
        objects = fixture.object_reader()
        with self.assertRaises(InvalidQueryError):
            metadata.get_frames_by_ids("L21_V001_001")  # type: ignore[arg-type]
        with self.assertRaises(InvalidQueryError):
            objects.get_objects_by_frame_ids(
                ["L21_V001_001"], label=123  # type: ignore[arg-type]
            )
        with self.assertRaises(InvalidQueryError):
            objects.get_objects_by_frame_ids(
                ["L21_V001_001"], min_confidence="bad"  # type: ignore[arg-type]
            )
        with self.assertRaises(InvalidQueryError):
            fixture.milvus().search_visual((True, 0.0), 1)


if __name__ == "__main__":
    unittest.main()
