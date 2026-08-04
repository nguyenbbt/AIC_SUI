from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from online.adapters.manifest import JsonManifestAdapter
from online.config import ManifestResourceConfig
from online.domain.errors import ContractMismatchError, ResourceUnavailableError


def _payload() -> dict[str, object]:
    return {
        "contract_version": "organizer-v1",
        "visual_model_id": "ViT-B-32::openai",
        "visual_dimension": 512,
        "visual_normalized": True,
        "frame_id_contract_version": "organizer-v1",
        "object_threshold": 0.25,
        "object_nms_iou": 0.45,
        "record_counts": {"metadata": 3},
        "dataset_fingerprint": "sha256:fixture",
        "offline_only_extra": "accepted",
    }


def test_reads_organizer_manifest_without_exposing_offline_extras() -> None:
    with (
        patch.object(Path, "is_file", return_value=True),
        patch.object(Path, "read_text", return_value=json.dumps(_payload())),
    ):
        manifest = JsonManifestAdapter(
            ManifestResourceConfig(path=Path("private/manifest.json"))
        ).read_manifest()
    assert manifest.visual_model_id == "ViT-B-32::openai"
    assert manifest.visual_dimension == 512
    assert "offline_only_extra" not in type(manifest).model_fields


def test_missing_and_invalid_manifest_are_typed_and_path_safe() -> None:
    secret_path = Path("credential-secret/manifest.json")
    with patch.object(Path, "is_file", return_value=False):
        with pytest.raises(ResourceUnavailableError) as missing:
            JsonManifestAdapter(
                ManifestResourceConfig(path=secret_path)
            ).read_manifest()
    assert "credential-secret" not in str(missing.value)

    with (
        patch.object(Path, "is_file", return_value=True),
        patch.object(Path, "read_text", return_value="not-json"),
    ):
        with pytest.raises(ContractMismatchError) as malformed:
            JsonManifestAdapter(
                ManifestResourceConfig(path=secret_path)
            ).read_manifest()
    assert "credential-secret" not in str(malformed.value)
