from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from online.config import OnlineDataConfig, SQLiteResourceConfig
from online.lifecycle import HealthStatus, InfrastructureLifecycle


class Resource:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.connected = False

    def connect(self):
        if self.fail:
            raise ConnectionError("down")
        self.connected = True

    def health_check(self):
        if not self.connected:
            raise ConnectionError("down")

    def close(self):
        self.connected = False


class ConfigLifecycleTests(unittest.TestCase):
    def test_default_manifest_path_matches_offline_publisher(self) -> None:
        config = OnlineDataConfig()

        self.assertEqual(
            config.dataset.manifest_path,
            Path("data/processed/dataset-manifest.json"),
        )
        self.assertEqual(config.dataset.data_root, Path("data/processed"))

    def test_config_loads_environment_and_rejects_sql_identifier_injection(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AIC_ONLINE_MILVUS_SEARCH_EF": "64",
                "AIC_ONLINE_ES_FUZZY_ENABLED": "true",
                "AIC_ONLINE_SQLITE_BATCH_SIZE": "100",
                "AIC_ONLINE_DATASET_EXPECTED_FINGERPRINT": "sha256:" + "a" * 64,
                "AIC_ONLINE_DATASET_MANIFEST_REQUIRED": "true",
                "AIC_ONLINE_DATASET_AUDIT_BATCH_SIZE": "250",
            },
            clear=False,
        ):
            config = OnlineDataConfig.from_env()
        self.assertEqual(config.milvus.search_ef, 64)
        self.assertTrue(config.elasticsearch.fuzzy_enabled)
        self.assertEqual(config.sqlite.batch_size, 100)
        self.assertEqual(config.dataset.expected_fingerprint, "sha256:" + "a" * 64)
        self.assertTrue(config.dataset.manifest_required)
        self.assertEqual(config.dataset.audit_batch_size, 250)
        with self.assertRaises(ValidationError):
            SQLiteResourceConfig(metadata_table="metadata; DROP TABLE objects")
        with patch.dict(
            os.environ,
            {"AIC_ONLINE_DATASET_MANIFEST_REQUIRED": "maybe"},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                OnlineDataConfig.from_env()

    def test_health_distinguishes_required_and_optional_failures(self) -> None:
        lifecycle = InfrastructureLifecycle()
        required = Resource()
        optional = Resource(fail=True)
        lifecycle.register("required", required, required=True)
        lifecycle.register("optional", optional, required=False)
        self.assertEqual(lifecycle.start().status, HealthStatus.DEGRADED)
        lifecycle.close()

        lifecycle = InfrastructureLifecycle()
        lifecycle.register("required", Resource(fail=True), required=True)
        self.assertEqual(lifecycle.start().status, HealthStatus.UNHEALTHY)
        self.assertEqual(lifecycle.start().status, HealthStatus.UNHEALTHY)
        lifecycle.close()


if __name__ == "__main__":
    unittest.main()
