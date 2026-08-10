import json

import pytest
from pydantic import ValidationError

from src.indexing.dataset_manifest import (
    RECORD_COUNT_KEYS,
    DatasetManifestDraft,
    build_manifest_draft,
)


def _record_counts() -> dict[str, int]:
    return {
        key: (
            2
            if key in {"videos", "summary_features", "video_summaries"}
            else 6
        )
        for key in RECORD_COUNT_KEYS
    }


def _write_metadata(data_dir, *, source_hash: str = "a" * 64) -> None:
    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for video_id in ("V002", "V001"):
        (metadata_dir / f"{video_id}.json").write_text(
            json.dumps(
                {
                    "contract_version": "self-indexed-v2",
                    "video_id": video_id,
                    "source_video_rel_path": f"videos/{video_id}.mp4",
                    "source_fingerprint": source_hash,
                    "producer_config_fingerprint": "b" * 64,
                }
            ),
            encoding="utf-8",
        )


def test_manifest_fingerprint_is_deterministic_and_run_id_independent(
    tmp_path,
):
    _write_metadata(tmp_path)
    counts = _record_counts()

    first = build_manifest_draft(
        data_dir=tmp_path,
        dataset_id="run-001",
        record_counts=counts,
        created_at_utc="2026-08-10T00:00:00Z",
        producer_config={"object_threshold": 0.25},
    )
    second = build_manifest_draft(
        data_dir=tmp_path,
        dataset_id="run-002",
        record_counts=dict(reversed(list(counts.items()))),
        created_at_utc="2026-08-11T00:00:00Z",
        producer_config={"object_threshold": 0.25},
    )

    assert first.status == "BUILDING"
    assert first.dataset_fingerprint == second.dataset_fingerprint
    assert first.dataset_fingerprint.startswith("sha256:")


def test_manifest_fingerprint_changes_with_source_or_producer_config(
    tmp_path,
):
    _write_metadata(tmp_path)
    counts = _record_counts()
    baseline = build_manifest_draft(
        data_dir=tmp_path,
        dataset_id="run",
        record_counts=counts,
        producer_config={"object_threshold": 0.25},
    )

    (tmp_path / "metadata" / "V001.json").unlink()
    (tmp_path / "metadata" / "V002.json").unlink()
    _write_metadata(tmp_path, source_hash="c" * 64)
    changed_source = build_manifest_draft(
        data_dir=tmp_path,
        dataset_id="run",
        record_counts=counts,
        producer_config={"object_threshold": 0.25},
    )
    changed_config = build_manifest_draft(
        data_dir=tmp_path,
        dataset_id="run",
        record_counts=counts,
        producer_config={"object_threshold": 0.5},
    )

    assert changed_source.dataset_fingerprint != baseline.dataset_fingerprint
    assert changed_config.dataset_fingerprint != changed_source.dataset_fingerprint


def test_manifest_draft_cannot_claim_ready():
    with pytest.raises(ValidationError):
        DatasetManifestDraft.model_validate(
            {
                "contract_version": "self-indexed-v2",
                "dataset_id": "run",
                "dataset_fingerprint": f"sha256:{'a' * 64}",
                "status": "READY",
                "frame_index_base": 0,
                "bbox_space": "absolute_pixel_xyxy",
                "visual_model_id": "ViT-B-32::openai",
                "visual_dimension": 512,
                "visual_normalized": True,
                "text_model_name": "dangvantuan/vietnamese-embedding",
                "text_model_revision": (
                    "4ab46e46ba5902328ba0742e489e75f787932f2b"
                ),
                "text_dimension": 768,
                "text_max_length": 256,
                "record_counts": _record_counts(),
                "created_at_utc": "2026-08-10T00:00:00Z",
            }
        )
