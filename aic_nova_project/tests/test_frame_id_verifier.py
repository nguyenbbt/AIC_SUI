from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from verify_frame_id_consistency import (
    FullVerificationSnapshot,
    VerificationSnapshot,
    build_consistency_report,
    build_full_contract_report,
    collect_sqlite_records,
    main,
    record_counts,
    _validate_and_compact_milvus_record,
)
import verify_frame_id_consistency as verifier_module


def test_verifier_compares_the_same_cross_db_record():
    snapshot = VerificationSnapshot(
        visual_frame_ids={"V001_00000_015"},
        ocr_vector_frame_ids=set(),
        ocr_text_frame_ids=set(),
        metadata_frame_ids={"V001_00000_050"},
        object_frame_ids=set(),
        asr_vector_ids=set(),
        asr_text_ids=set(),
        summary_vector_ids=set(),
        summary_text_ids=set(),
    )

    report = build_consistency_report(snapshot)

    assert any("visual" in error and "metadata" in error for error in report)


def test_verifier_accepts_joinable_records_across_all_backends():
    frame_id = "V001_00000_015"
    snapshot = VerificationSnapshot(
        visual_frame_ids={frame_id},
        ocr_vector_frame_ids={frame_id},
        ocr_text_frame_ids={frame_id},
        metadata_frame_ids={frame_id},
        object_frame_ids={frame_id},
        asr_vector_ids={("V001", "0")},
        asr_text_ids={("V001", "0")},
        summary_vector_ids={"V001"},
        summary_text_ids={"V001"},
    )

    assert build_consistency_report(snapshot) == []


def test_streaming_milvus_validation_discards_vector_payload() -> None:
    compact = _validate_and_compact_milvus_record(
        {
            "frame_id": "V001_00000_015",
            "video_id": "V001",
            "embedding": [1.0, 0.0],
        },
        stream_name="visual_features",
        expected_dimension=2,
        index=0,
    )

    assert "embedding" not in compact
    assert compact["_embedding_validated"] is True


def test_streaming_milvus_validation_rejects_invalid_vector() -> None:
    with np.testing.assert_raises_regex(
        ValueError,
        "not L2-normalized",
    ):
        _validate_and_compact_milvus_record(
            {
                "frame_id": "V001_00000_015",
                "embedding": [2.0, 0.0],
            },
            stream_name="visual_features",
            expected_dimension=2,
            index=0,
        )


def _full_snapshot(tmp_path) -> FullVerificationSnapshot:
    frame_id = "V001_00000_015"
    image_path = tmp_path / "keyframes" / "V001" / "frame.webp"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 12), color="red").save(image_path, "webp")
    joins = VerificationSnapshot(
        visual_frame_ids={frame_id},
        ocr_vector_frame_ids={frame_id},
        ocr_text_frame_ids={frame_id},
        metadata_frame_ids={frame_id},
        object_frame_ids={frame_id},
        asr_vector_ids={("V001", "0")},
        asr_text_ids={("V001", "0")},
        summary_vector_ids={"V001"},
        summary_text_ids={"V001"},
    )
    return FullVerificationSnapshot(
        joins=joins,
        videos=(
            {
                "video_id": "V001",
                "source_video_rel_path": "videos/V001.mp4",
                "fps": 25.0,
                "duration_sec": 4.0,
                "frame_count": 100,
                "width": 16,
                "height": 12,
            },
        ),
        metadata=(
            {
                "frame_id": frame_id,
                "video_id": "V001",
                "shot_id": 0,
                "source_frame_idx": 15,
                "timestamp": 0.6,
                "image_rel_path": "keyframes/V001/frame.webp",
            },
        ),
        objects=(
            {
                "frame_id": frame_id,
                "label": "person",
                "confidence": 0.9,
                "x_min": 1.0,
                "y_min": 1.0,
                "x_max": 10.0,
                "y_max": 10.0,
                "model_source": "yolo",
            },
        ),
        milvus={
            "visual_features": (
                {
                    "frame_id": frame_id,
                    "video_id": "V001",
                    "shot_id": 0,
                    "embedding": [1.0] + [0.0] * 511,
                },
            ),
            "ocr_features": (
                {
                    "frame_id": frame_id,
                    "video_id": "V001",
                    "embedding": [1.0] + [0.0] * 767,
                },
            ),
            "asr_features": (
                {
                    "video_id": "V001",
                    "interval_id": "0",
                    "start_time_sec": 0.0,
                    "end_time_sec": 1.0,
                    "embedding": [1.0] + [0.0] * 767,
                },
            ),
            "summary_features": (
                {"video_id": "V001", "embedding": [1.0] + [0.0] * 767},
            ),
        },
        elasticsearch={
            "ocr_texts": (
                {
                    "frame_id": frame_id,
                    "video_id": "V001",
                    "shot_id": "0",
                },
            ),
            "asr_transcripts": (
                {
                    "video_id": "V001",
                    "interval_id": "0",
                    "start_time_sec": 0.0,
                    "end_time_sec": 1.0,
                },
            ),
            "video_summaries": ({"video_id": "V001"},),
        },
    )


def _manifest(snapshot: FullVerificationSnapshot) -> dict:
    return {
        "contract_version": "self-indexed-v2",
        "dataset_id": "fixture-run",
        "dataset_fingerprint": f"sha256:{'a' * 64}",
        "status": "BUILDING",
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
        "record_counts": record_counts(snapshot),
        "created_at_utc": "2026-08-10T00:00:00Z",
    }


def test_full_verifier_accepts_valid_contract(tmp_path):
    snapshot = _full_snapshot(tmp_path)

    assert build_full_contract_report(
        snapshot,
        data_root=tmp_path,
        manifest=_manifest(snapshot),
    ) == []


def test_full_verifier_accepts_float32_asr_timestamp_roundtrip(tmp_path):
    snapshot = _full_snapshot(tmp_path)
    milvus = dict(snapshot.milvus)
    vector_record = dict(milvus["asr_features"][0])
    vector_record["end_time_sec"] = float(np.float32(599.97))
    milvus["asr_features"] = (vector_record,)
    elasticsearch = dict(snapshot.elasticsearch)
    transcript = dict(elasticsearch["asr_transcripts"][0])
    transcript["end_time_sec"] = 599.97
    elasticsearch["asr_transcripts"] = (transcript,)
    video = dict(snapshot.videos[0])
    video["duration_sec"] = 600.0
    roundtripped = replace(
        snapshot,
        videos=(video,),
        milvus=milvus,
        elasticsearch=elasticsearch,
    )

    assert build_full_contract_report(
        roundtripped,
        data_root=tmp_path,
        manifest=_manifest(roundtripped),
    ) == []


def test_full_verifier_reports_domain_vector_and_count_failures(tmp_path):
    snapshot = _full_snapshot(tmp_path)
    broken_metadata = dict(snapshot.metadata[0])
    broken_metadata.update(
        source_frame_idx=100,
        image_rel_path="../escape.webp",
    )
    broken_object = dict(snapshot.objects[0])
    broken_object["x_max"] = 17.0
    broken_milvus = dict(snapshot.milvus)
    broken_milvus["visual_features"] = (
        {
            "frame_id": "V001_00000_015",
            "video_id": "V999",
            "shot_id": 0,
            "embedding": [1.0, 0.0],
        },
    )
    broken = replace(
        snapshot,
        metadata=(broken_metadata,),
        objects=(broken_object,),
        milvus=broken_milvus,
    )
    manifest = _manifest(snapshot)
    manifest["record_counts"] = {
        **manifest["record_counts"],
        "objects": 99,
    }

    report = build_full_contract_report(
        broken,
        data_root=tmp_path,
        manifest=manifest,
    )

    assert any("source_frame_idx" in error for error in report)
    assert any("image_rel_path" in error for error in report)
    assert any("bbox" in error for error in report)
    assert any("visual_features" in error and "dimension" in error for error in report)
    assert any("visual_features" in error and "video_id" in error for error in report)
    assert any("record_counts.objects" in error for error in report)


def test_sqlite_collector_reads_rows_and_audits_schema(
    tmp_path,
    monkeypatch,
):
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project_root / "indexing"))
    from src.indexing.clients.tabular_client import TabularClient

    db_path = tmp_path / "metadata.db"
    client = TabularClient(str(db_path))
    client.connect()
    client.create_tables()
    client.insert_video_batch(
        [
            {
                "video_id": "V001",
                "source_video_rel_path": "videos/V001.mp4",
                "fps": 25.0,
                "duration_sec": 4.0,
                "frame_count": 100,
                "width": 16,
                "height": 12,
            }
        ]
    )
    client.insert_metadata_batch(
        [
            {
                "frame_id": "V001_00000_015",
                "video_id": "V001",
                "shot_id": 0,
                "source_frame_idx": 15,
                "timestamp": 0.6,
                "image_rel_path": "keyframes/V001/frame.webp",
            }
        ]
    )
    client.disconnect()

    records = collect_sqlite_records(f"sqlite:///{db_path}")

    assert records["videos"][0]["video_id"] == "V001"
    assert records["metadata"][0]["source_frame_idx"] == 15
    assert records["schema_errors"] == ()
    assert records["foreign_key_errors"] == ()


def test_full_verifier_rejects_duplicate_vector_domain_keys(tmp_path):
    snapshot = _full_snapshot(tmp_path)
    milvus = dict(snapshot.milvus)
    milvus["visual_features"] = (
        snapshot.milvus["visual_features"]
        + snapshot.milvus["visual_features"]
    )
    duplicated = replace(snapshot, milvus=milvus)

    report = build_full_contract_report(
        duplicated,
        data_root=tmp_path,
        manifest=_manifest(duplicated),
    )

    assert any(
        "visual_features contains duplicate domain keys" in error
        for error in report
    )


def test_verifier_cli_returns_nonzero_for_manifest_mismatch(
    tmp_path,
    monkeypatch,
):
    snapshot = _full_snapshot(tmp_path)
    manifest = _manifest(snapshot)
    manifest["record_counts"]["objects"] = 99
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        verifier_module,
        "collect_full_snapshot",
        lambda **kwargs: snapshot,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify-frame-id-consistency",
            "--data-root",
            str(tmp_path),
            "--manifest-path",
            str(manifest_path),
        ],
    )

    assert main() == 1
