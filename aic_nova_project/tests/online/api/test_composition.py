from __future__ import annotations

import math
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from online.config import DatasetResourceConfig, OnlineDataConfig
from online.domain.enums import RetrievalBranch
from online.ports.records import FrameMetadata, FrameSearchHit
from online.testing import (
    AdvancedRuntimeState,
    FakeElasticsearchSearchPort,
    FakeMetadataReaderPort,
    FakeMilvusSearchPort,
    FakeObjectReaderPort,
    FakeTextEncoder,
    build_advanced_runtime_bundle,
)
from online.testing.advanced_composition import attach_advanced_fake_modes
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


class BrokenTextEncoder(FakeTextEncoder):
    def encode_texts(self, texts):
        raise RuntimeError("encoder unavailable")


def runtime_with_fakes() -> tuple[object, ManagedMilvus, ManagedElasticsearch]:
    frame = FrameMetadata(
        frame_id="V001_00000_015",
        video_id="V001",
        shot_id=0,
        timestamp_sec=1.5,
        source_frame_idx=45,
        image_rel_path="keyframes/V001/V001_00000_015.jpg",
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
        data_config=OnlineDataConfig(
            dataset=DatasetResourceConfig(manifest_required=False)
        ),
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
                "AIC_ONLINE_RANKING_MAX_WORKERS": "4",
                "AIC_ONLINE_VISUAL_ENCODER_DIMENSION": "1024",
                "AIC_ONLINE_RANKING_NORMALIZATION_RRF_K": "17",
                "AIC_ONLINE_RANKING_QUERY_Q1_WEIGHT": "0.25",
                "AIC_ONLINE_RANKING_ASR_MAX_FRAMES_PER_INTERVAL": "9",
                "AIC_ONLINE_TRAKE_ENABLED": "true",
            },
            clear=True,
        ):
            config = RuntimeCompositionConfig.from_env()

        self.assertEqual(config.max_workers, 3)
        self.assertEqual(config.ranking_max_workers, 4)
        self.assertEqual(config.visual_expected_dimension, 1024)
        self.assertEqual(config.ranking_policy.normalization_rrf_k, 17)
        self.assertEqual(config.ranking_policy.query_variant_weights["q1"], 0.25)
        self.assertEqual(config.ranking_policy.asr_max_frames_per_interval, 9)
        self.assertTrue(config.trake_enabled)
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
        with self.assertRaises(RuntimeError):
            runtime.ranking_executor.submit(lambda: None)

    def test_required_encoder_probe_marks_runtime_unhealthy(self) -> None:
        frame = FrameMetadata(
            frame_id="V001_00000_015",
            video_id="V001",
            shot_id=0,
            timestamp_sec=1.5,
            source_frame_idx=45,
            image_rel_path="keyframes/V001/V001_00000_015.jpg",
        )
        runtime = build_online_runtime(
            data_config=OnlineDataConfig(),
            runtime_config=RuntimeCompositionConfig(),
            milvus=ManagedMilvus(),
            elasticsearch=ManagedElasticsearch(),
            metadata=FakeMetadataReaderPort((frame,)),
            object_reader=FakeObjectReaderPort({}),
            visual_encoder=BrokenTextEncoder(),
            vietnamese_encoder=FakeTextEncoder(),
        )

        health = runtime.start()
        runtime.close()

        self.assertEqual(health.status.value, "unhealthy")
        visual = next(component for component in health.components if component.name == "visual_encoder")
        self.assertFalse(visual.healthy)

    def test_runtime_app_lifespan_wires_search_end_to_end_with_fakes(self) -> None:
        runtime, milvus, elasticsearch = runtime_with_fakes()

        with TestClient(create_runtime_app_from_env(runtime_factory=lambda: runtime)) as client:
            ready = client.get("/health/ready")
            self.assertEqual(ready.status_code, 200)
            self.assertEqual(ready.json()["status"], "ready")
            self.assertEqual(
                ready.json()["checks"],
                {
                    "kis.enabled": "true",
                    "kis.readiness": "ready",
                    "trake.enabled": "false",
                    "trake.readiness": "disabled",
                    "vqa.enabled": "false",
                    "vqa.readiness": "disabled",
                    "rewrite.enabled": "false",
                    "ui_resources.enabled": "true",
                },
            )

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

    def test_production_rejects_experimental_ranking_policy(self) -> None:
        with patch.dict(
            "os.environ",
            {"AIC_ONLINE_DEPLOYMENT_MODE": "production"},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                RuntimeCompositionConfig.from_env()

        with patch.dict(
            "os.environ",
            {"AIC_ONLINE_DEPLOYMENT_MODE": "Production"},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                RuntimeCompositionConfig.from_env()

        with self.assertRaises(ValueError):
            RuntimeCompositionConfig(deployment_mode="prodution")

    def test_runtime_config_rejects_non_finite_timeouts(self) -> None:
        for invalid_timeout in (math.nan, math.inf, -math.inf, 0.0):
            with self.subTest(invalid_timeout=invalid_timeout):
                with self.assertRaises(ValueError):
                    RuntimeCompositionConfig(default_timeout_sec=invalid_timeout)

        with patch.dict(
            "os.environ",
            {"AIC_ONLINE_TRAKE_ENABLED": "sometimes"},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                RuntimeCompositionConfig.from_env()

    def test_vqa_runtime_flag_requires_explicit_vlm(self) -> None:
        with self.assertRaisesRegex(ValueError, "VLMPort"):
            build_online_runtime(
                data_config=OnlineDataConfig(
                    dataset=DatasetResourceConfig(manifest_required=False)
                ),
                runtime_config=RuntimeCompositionConfig(vqa_enabled=True),
                milvus=ManagedMilvus(),
                elasticsearch=ManagedElasticsearch(),
                metadata=FakeMetadataReaderPort(()),
                object_reader=FakeObjectReaderPort({}),
                visual_encoder=FakeTextEncoder(),
                vietnamese_encoder=FakeTextEncoder(),
            )

    def test_advanced_readiness_is_mode_specific_and_does_not_call_vlm(self) -> None:
        runtime, _, _ = runtime_with_fakes()
        bundle = build_advanced_runtime_bundle(
            vlm_state=AdvancedRuntimeState.UNAVAILABLE,
        )
        attach_advanced_fake_modes(runtime, bundle)

        with TestClient(create_runtime_app_from_env(runtime_factory=lambda: runtime)) as client:
            ready = client.get("/health/ready")
            self.assertEqual(ready.status_code, 503)
            checks = ready.json()["checks"]
            self.assertEqual(checks["trake.enabled"], "true")
            self.assertEqual(checks["trake.readiness"], "ready")
            self.assertEqual(checks["vqa.enabled"], "true")
            self.assertEqual(checks["vqa.readiness"], "unavailable")
        self.assertFalse(any(call.operation == "answer" for call in bundle.calls))
        self.assertTrue(bundle.closed)


if __name__ == "__main__":
    unittest.main()
