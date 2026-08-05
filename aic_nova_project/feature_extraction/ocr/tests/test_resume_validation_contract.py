import json

from ocr_module.resume_validation import (
    OCR_SCHEMA_VERSION,
    build_ocr_provenance,
    is_valid_ocr_artifact,
)


def test_ocr_resume_requires_complete_frames_and_matching_provenance(
    tmp_path,
):
    output_path = tmp_path / "V001.json"
    provenance = build_ocr_provenance(
        backbone="vgg_transformer",
        confidence_threshold=0.4,
        width_ths=0.7,
        mag_ratio=1.5,
    )
    artifact = {
        "schema_version": OCR_SCHEMA_VERSION,
        "video_id": "V001",
        "provenance": provenance,
        "frames": [
            {"frame_id": "V001_00000_015"},
            {"frame_id": "V001_00001_050"},
        ],
    }
    output_path.write_text(json.dumps(artifact), encoding="utf-8")

    assert is_valid_ocr_artifact(
        output_path,
        "V001",
        ["V001_00000_015", "V001_00001_050"],
        provenance,
    )

    artifact["frames"].pop()
    output_path.write_text(json.dumps(artifact), encoding="utf-8")
    assert not is_valid_ocr_artifact(
        output_path,
        "V001",
        ["V001_00000_015", "V001_00001_050"],
        provenance,
    )


def test_ocr_resume_rejects_corrupt_or_stale_artifact(tmp_path):
    output_path = tmp_path / "V001.json"
    provenance = build_ocr_provenance(
        backbone="vgg_transformer",
        confidence_threshold=0.4,
        width_ths=0.7,
        mag_ratio=1.5,
    )
    output_path.write_text("{broken", encoding="utf-8")
    assert not is_valid_ocr_artifact(
        output_path,
        "V001",
        ["V001_00000_015"],
        provenance,
    )

    stale = {
        "schema_version": OCR_SCHEMA_VERSION,
        "video_id": "V001",
        "provenance": {
            **provenance,
            "recognizer_backbone": "vgg_seq2seq",
        },
        "frames": [{"frame_id": "V001_00000_015"}],
    }
    output_path.write_text(json.dumps(stale), encoding="utf-8")
    assert not is_valid_ocr_artifact(
        output_path,
        "V001",
        ["V001_00000_015"],
        provenance,
    )
