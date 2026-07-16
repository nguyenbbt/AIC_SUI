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
from online.testing import build_integration_fixture
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
        self.assertEqual({frame.video_id for frame in fixture.frames}, {"V001", "V002"})
        self.assertEqual({hit.interval_id for hit in fixture.asr_hits}, {"overlap", "boundary", "no_overlap"})
        self.assertNotIn(
            fixture.missing_metadata_hit.frame_id,
            fixture.metadata().get_frames_by_ids([fixture.missing_metadata_hit.frame_id]),
        )
        filtered = fixture.object_reader().get_objects_by_frame_ids(
            ["V001_00001_050"], label="person", min_confidence=0.5
        )
        self.assertEqual(len(filtered["V001_00001_050"]), 1)


if __name__ == "__main__":
    unittest.main()
