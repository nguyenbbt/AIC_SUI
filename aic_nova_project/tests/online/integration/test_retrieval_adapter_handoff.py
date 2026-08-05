from __future__ import annotations

import asyncio
import sqlite3
import unittest

from online.adapters.elasticsearch import ElasticsearchSearchAdapter
from online.adapters.milvus import MilvusSearchAdapter
from online.adapters.sqlite import SQLiteReadAdapter
from online.config import (
    ElasticsearchResourceConfig,
    MilvusResourceConfig,
    OnlineDataConfig,
    SQLiteResourceConfig,
)
from online.domain.candidates import BranchResult
from online.domain.enums import BranchStatus, CandidateLevel, QueryMode, RetrievalBranch
from online.retrieval import (
    BASELINE_KIS_BRANCHES,
    RetrievalInvocationConfig,
    RetrievalServicePort,
    build_retrieval_service,
)
from online.testing import FakeTextEncoder
from query_understanding import parse_kis_query


class _MilvusBackend:
    def __init__(self, config: MilvusResourceConfig) -> None:
        self.connected = False
        self.searches: list[tuple[str, int]] = []
        self.descriptions = {
            config.visual_collection: {
                "fields": {"embedding": "FLOAT_VECTOR"},
                "dimension": 4,
                "metric_type": "IP",
                "index_type": "HNSW",
            },
            config.ocr_collection: {
                "fields": {"embedding": "FLOAT_VECTOR"},
                "dimension": 6,
                "metric_type": "IP",
                "index_type": "HNSW",
            },
            config.asr_collection: {
                "fields": {"embedding": "FLOAT_VECTOR"},
                "dimension": 6,
                "metric_type": "IP",
                "index_type": "HNSW",
            },
            config.summary_collection: {
                "fields": {"embedding": "FLOAT_VECTOR"},
                "dimension": 6,
                "metric_type": "IP",
                "index_type": "HNSW",
            },
        }
        self.hits = {
            config.visual_collection: (
                {
                    "entity": {
                        "frame_id": "V001_00000_015",
                        "video_id": "V001",
                        "shot_id": 0,
                    },
                    "distance": 0.95,
                },
            ),
            config.ocr_collection: (
                {
                    "entity": {
                        "frame_id": "V001_00000_015",
                        "video_id": "V001",
                    },
                    "distance": 0.85,
                },
            ),
            config.asr_collection: (
                {
                    "entity": {
                        "video_id": "V001",
                        "interval_id": "0",
                        "start_time_sec": 1.0,
                        "end_time_sec": 3.0,
                    },
                    "distance": 0.75,
                },
            ),
            config.summary_collection: (
                {
                    "entity": {"video_id": "V001"},
                    "distance": 0.65,
                },
            ),
        }

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def collection_exists(self, name: str) -> bool:
        return name in self.descriptions

    def describe_collection(self, name: str):
        return self.descriptions[name]

    def search(
        self,
        name,
        vector,
        output_fields,
        top_k,
        search_params,
        timeout_sec,
    ):
        self.searches.append((name, top_k))
        return self.hits[name][:top_k]

    def sample_records(self, name, output_fields, limit):
        return ()


class _ElasticsearchClient:
    def __init__(self, config: ElasticsearchResourceConfig) -> None:
        self.config = config
        self.searches: list[tuple[str, int]] = []

    def search(self, *, index, body, request_timeout):
        self.searches.append((index, body["size"]))
        if index == self.config.ocr_index:
            source = {
                "frame_id": "V001_00000_015",
                "video_id": "V001",
                "shot_id": 0,
            }
            score = 5.5
        elif index == self.config.asr_index:
            source = {
                "video_id": "V001",
                "interval_id": "0",
                "start_time_sec": 1.0,
                "end_time_sec": 3.0,
                "cleaned_text": "người mặc áo đỏ đi xe đạp",
            }
            score = 4.5
        elif index == self.config.summary_index:
            source = {
                "video_id": "V001",
                "summary": "Một người mặc áo đỏ đang đi xe đạp.",
            }
            score = 3.5
        else:
            raise AssertionError(f"unexpected index: {index}")
        return {"hits": {"hits": [{"_score": score, "_source": source}]}}

    def ping(self):
        return True


class RetrievalAdapterHandoffTests(unittest.TestCase):
    def test_seven_branches_cross_concrete_a_adapters_and_return_c_handoff(self) -> None:
        config = OnlineDataConfig(
            milvus=MilvusResourceConfig(
                visual_collection="integration_visual",
                ocr_collection="integration_ocr",
                asr_collection="integration_asr",
                summary_collection="integration_summary",
            ),
            elasticsearch=ElasticsearchResourceConfig(
                ocr_index="integration_ocr_texts",
                asr_index="integration_asr_transcripts",
                summary_index="integration_video_summaries",
            ),
            sqlite=SQLiteResourceConfig(),
        )
        milvus_backend = _MilvusBackend(config.milvus)
        milvus = MilvusSearchAdapter(config.milvus, backend=milvus_backend)
        elasticsearch_client = _ElasticsearchClient(config.elasticsearch)
        elasticsearch = ElasticsearchSearchAdapter(
            config.elasticsearch,
            client=elasticsearch_client,
        )
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        connection.execute(
            "CREATE TABLE metadata ("
            "frame_id TEXT PRIMARY KEY, video_id TEXT, shot_id INTEGER, "
            "source_frame_idx INTEGER, timestamp REAL, image_rel_path TEXT)"
        )
        connection.execute(
            "INSERT INTO metadata(frame_id, video_id, shot_id, source_frame_idx, timestamp, image_rel_path) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("V001_00000_015", "V001", 0, 45, 1.5, "keyframes/V001/V001_00000_015.jpg"),
        )
        connection.commit()
        metadata = SQLiteReadAdapter(config.sqlite, connection=connection)
        milvus.connect()

        query = parse_kis_query(
            "người mặc áo đỏ đang đi xe đạp",
            mode=QueryMode.KIS_TEXT,
            query_id="adapter-handoff",
        )
        invocation_configs = {
            (branch, "q0"): RetrievalInvocationConfig(
                top_k=1,
                timeout_sec=1.0,
            )
            for branch in BASELINE_KIS_BRANCHES
        }
        service = build_retrieval_service(
            data_config=config,
            milvus=milvus,
            elasticsearch=elasticsearch,
            metadata=metadata,
            visual_encoder=FakeTextEncoder(dimension=4),
            vietnamese_encoder=FakeTextEncoder(dimension=6),
            invocation_configs=invocation_configs,
            max_workers=7,
        )
        try:
            results = asyncio.run(service.retrieve(query))
        finally:
            service.close(wait=True)
            milvus.close()
            elasticsearch.close()
            metadata.close()
            connection.close()

        self.assertIsInstance(service, RetrievalServicePort)
        self.assertTrue(all(isinstance(result, BranchResult) for result in results))
        self.assertEqual(tuple(result.branch for result in results), BASELINE_KIS_BRANCHES)
        self.assertTrue(all(result.status is BranchStatus.SUCCESS for result in results))
        self.assertTrue(all(result.returned_count == 1 for result in results))
        self.assertEqual(
            tuple(result.candidate_level for result in results),
            (
                CandidateLevel.FRAME,
                CandidateLevel.FRAME,
                CandidateLevel.FRAME,
                CandidateLevel.ASR_INTERVAL,
                CandidateLevel.ASR_INTERVAL,
                CandidateLevel.VIDEO,
                CandidateLevel.VIDEO,
            ),
        )
        expected_resources = {
            RetrievalBranch.VISUAL_DENSE: config.milvus.visual_collection,
            RetrievalBranch.OCR_DENSE: config.milvus.ocr_collection,
            RetrievalBranch.OCR_BM25: config.elasticsearch.ocr_index,
            RetrievalBranch.ASR_DENSE: config.milvus.asr_collection,
            RetrievalBranch.ASR_BM25: config.elasticsearch.asr_index,
            RetrievalBranch.SUMMARY_DENSE: config.milvus.summary_collection,
            RetrievalBranch.SUMMARY_BM25: config.elasticsearch.summary_index,
        }
        for result in results:
            self.assertEqual(
                result.candidates[0].provenance.source_resource,
                expected_resources[result.branch],
            )
            self.assertEqual(result.candidates[0].provenance.query_variant_id, "q0")

        self.assertEqual(
            {name for name, _ in milvus_backend.searches},
            {
                config.milvus.visual_collection,
                config.milvus.ocr_collection,
                config.milvus.asr_collection,
                config.milvus.summary_collection,
            },
        )
        self.assertEqual(
            {name for name, _ in elasticsearch_client.searches},
            {
                config.elasticsearch.ocr_index,
                config.elasticsearch.asr_index,
                config.elasticsearch.summary_index,
            },
        )


if __name__ == "__main__":
    unittest.main()
