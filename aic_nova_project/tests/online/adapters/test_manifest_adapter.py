from __future__ import annotations

import json

import pytest

from online.adapters.manifest import DatasetManifestGate
from online.config import DatasetResourceConfig
from online.domain.errors import ContractMismatchError, ResourceUnavailableError
from online.domain.manifest import DatasetManifest


def manifest_payload(*, fingerprint: str = "sha256:" + "a" * 64) -> dict[str, object]:
    return {
        "contract_version": "self-indexed-v2",
        "dataset_id": "fixture-run-001",
        "dataset_fingerprint": fingerprint,
        "status": "READY",
        "frame_index_base": 0,
        "bbox_space": "absolute_pixel_xyxy",
        "visual_model_id": "ViT-B-32::openai",
        "visual_dimension": 512,
        "visual_normalized": True,
        "text_model_name": "dangvantuan/vietnamese-embedding",
        "text_model_revision": "4ab46e46ba5902328ba0742e489e75f787932f2b",
        "text_dimension": 768,
        "text_max_length": 256,
        "record_counts": {
            "videos": 1,
            "metadata": 2,
            "objects": 0,
            "visual_features": 2,
            "ocr_features": 1,
            "asr_features": 1,
            "summary_features": 1,
            "ocr_texts": 1,
            "asr_transcripts": 1,
            "video_summaries": 1,
        },
        "created_at_utc": "2026-08-05T00:00:00Z",
    }


def test_manifest_model_is_strict_and_serializable() -> None:
    manifest = DatasetManifest.model_validate(manifest_payload())

    assert manifest.status == "READY"
    assert manifest.model_dump(mode="json")["record_counts"]["metadata"] == 2

    bad = manifest_payload()
    bad["status"] = "BUILDING"
    with pytest.raises(ValueError):
        DatasetManifest.model_validate(bad)


def test_manifest_gate_locks_fingerprint_and_detects_atomic_switch(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest_payload()), encoding="utf-8")
    gate = DatasetManifestGate(
        DatasetResourceConfig(
            manifest_path=path,
            data_root=tmp_path,
            expected_fingerprint="sha256:" + "a" * 64,
        )
    )

    gate.connect()
    gate.health_check()
    assert gate.manifest.dataset_id == "fixture-run-001"

    path.write_text(
        json.dumps(manifest_payload(fingerprint="sha256:" + "b" * 64)),
        encoding="utf-8",
    )
    with pytest.raises(ContractMismatchError):
        gate.health_check()


def test_manifest_gate_rejects_missing_and_wrong_fingerprint(tmp_path) -> None:
    missing = DatasetManifestGate(
        DatasetResourceConfig(manifest_path=tmp_path / "missing.json", data_root=tmp_path)
    )
    with pytest.raises(ResourceUnavailableError):
        missing.connect()

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest_payload()), encoding="utf-8")
    wrong = DatasetManifestGate(
        DatasetResourceConfig(
            manifest_path=path,
            data_root=tmp_path,
            expected_fingerprint="sha256:" + "c" * 64,
        )
    )
    with pytest.raises(ContractMismatchError):
        wrong.connect()
