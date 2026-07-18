from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from online.config import OnlineDataConfig
from online.domain.enums import RetrievalBranch
from online.ports.records import FrameMetadata, FrameSearchHit
from online.testing import (
    FakeElasticsearchSearchPort,
    FakeMetadataReaderPort,
    FakeMilvusSearchPort,
    FakeObjectReaderPort,
    FakeTextEncoder,
)
from retrieval_api.composition import (
    RuntimeCompositionConfig,
    build_invocation_configs,
    build_online_runtime,
    create_runtime_app_from_env,
)


class ManagedMilvus(FakeMilvusSearchPort):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.connected = False
        self.closed = False

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.closed = True
        self.connected = False

    def health_check(self) -> None:
        if not self.connected:
            raise RuntimeError("milvus is not connected")


class ManagedElasticsearch(FakeElasticsearchSearchPort):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.connected = False
        self.closed = False

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.closed = True
        self.connected = False

    def health_check(self) -> None:
        if not self.connected:
            raise RuntimeError("elasticsearch is not connected")


def runtime_with_fakes() -> tuple[object, ManagedMilvus, ManagedElasticsearch]:
    frame = FrameMetadata(
        frame_id="V001_00000_015",
        video_id="V001",
        shot_id=0,
        timestamp_sec=1.5,
    )
    milvus = ManagedMilvus(
        visual=(
            FrameSearchHit(
                frame_id=frame.frame_id,
                video_id=frame.video_id,
                shot_id=frame.shot_id,
                raw_score=0.9,
            ),
        )
    )
    elasticsearch = ManagedElasticsearch()
    runtime = build_online_runtime(
        data_config=OnlineDataConfig(),
        runtime_config=RuntimeCompositionConfig(default_top_k=3, default_timeout_sec=1.0),
        milvus=milvus,
        elasticsearch=elasticsearch,
        metadata=FakeMetadataReaderPort((frame,)),
        object_reader=FakeObjectReaderPort({}),
        visual_encoder=FakeTextEncoder(dimension=4),
        vietnamese_encoder=FakeTextEncoder(dimension=4),
    )
    return runtime, milvus, elasticsearch


class RuntimeCompositionTests(unittest.TestCase):
    def test_runtime_config_reads_env_and_builds_exact_invocation_configs(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "AIC_ONLINE_RETRIEVAL_TOP_K": "11",
                "AIC_ONLINE_VISUAL_DENSE_TOP_K": "7",
                "AIC_ONLINE_RETRIEVAL_TIMEOUT_SEC": "2.5",
                "AIC_ONLINE_OCR_BM25_TIMEOUT_SEC": "1.25",
                "AIC_ONLINE_RETRIEVAL_MAX_WORKERS": "3",
                "AIC_ONLINE_VISUAL_ENCODER_DIMENSION": "1024",
            },
            clear=False,
        ):
            config = RuntimeCompositionConfig.from_env()

        self.assertEqual(config.max_workers, 3)
        self.assertEqual(config.visual_expected_dimension, 1024)
        invocation_configs = build_invocation_configs(config)
        self.assertEqual(invocation_configs[(RetrievalBranch.VISUAL_DENSE, "q0")].top_k, 7)
        self.assertEqual(invocation_configs[(RetrievalBranch.OCR_DENSE, "q1")].top_k, 11)
        self.assertEqual(invocation_configs[(RetrievalBranch.OCR_BM25, "q0")].timeout_sec, 1.25)

    def test_runtime_build_does_not_connect_until_lifespan_start(self) -> None:
        runtime, milvus, elasticsearch = runtime_with_fakes()

        self.assertFalse(milvus.connected)
        self.assertFalse(elasticsearch.connected)

        health = runtime.start()
        runtime.close()

        self.assertEqual(health.status.value, "healthy")
        self.assertTrue(milvus.closed)
        self.assertTrue(elasticsearch.closed)

    def test_runtime_app_lifespan_wires_search_end_to_end_with_fakes(self) -> None:
        runtime, milvus, elasticsearch = runtime_with_fakes()

        with TestClient(create_runtime_app_from_env(runtime_factory=lambda: runtime)) as client:
            ready = client.get("/health/ready")
            self.assertEqual(ready.status_code, 200)
            self.assertEqual(ready.json()["status"], "ready")

            response = client.post(
                "/search",
                json={
                    "query": "query",
                    "enabled_branches": ["visual_dense"],
                    "query_id": "wired-query",
                    "include_diagnostics": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["query_id"], "wired-query")
        self.assertEqual(response.json()["candidates"][0]["frame_id"], "V001_00000_015")
        self.assertTrue(milvus.closed)
        self.assertTrue(elasticsearch.closed)


if __name__ == "__main__":
    unittest.main()
