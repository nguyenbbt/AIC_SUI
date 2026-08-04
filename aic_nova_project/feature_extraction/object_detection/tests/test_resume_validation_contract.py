import json

from src.object_detection.resume_validation import (
    OBJECT_SCHEMA_VERSION,
    build_object_provenance,
    is_valid_object_artifact,
)


def test_object_resume_requires_complete_frames_and_current_config(
    tmp_path,
):
    output_path = tmp_path / "V001.json"
    provenance = build_object_provenance(
        yolo_world_model="weights/yolo.pt",
        custom_vocab_file=None,
        co_detr_backbone="resnet50",
        confidence_threshold=0.25,
        nms_threshold=0.5,
    )
    artifact = {
        "schema_version": OBJECT_SCHEMA_VERSION,
        "video_id": "V001",
        "provenance": provenance,
        "frames": [
            {"frame_id": "V001_00000_015", "objects": []},
            {"frame_id": "V001_00001_050", "objects": []},
        ],
    }
    output_path.write_text(json.dumps(artifact), encoding="utf-8")

    assert is_valid_object_artifact(
        output_path,
        "V001",
        ["V001_00000_015", "V001_00001_050"],
        provenance,
    )

    artifact["frames"].pop()
    output_path.write_text(json.dumps(artifact), encoding="utf-8")
    assert not is_valid_object_artifact(
        output_path,
        "V001",
        ["V001_00000_015", "V001_00001_050"],
        provenance,
    )


def test_object_resume_rejects_corrupt_and_stale_output(tmp_path):
    output_path = tmp_path / "V001.json"
    provenance = build_object_provenance(
        yolo_world_model="weights/yolo.pt",
        custom_vocab_file=None,
        co_detr_backbone=None,
        confidence_threshold=0.25,
        nms_threshold=0.5,
    )
    output_path.write_text("{broken", encoding="utf-8")
    assert not is_valid_object_artifact(
        output_path,
        "V001",
        ["V001_00000_015"],
        provenance,
    )

    stale = {
        "schema_version": OBJECT_SCHEMA_VERSION,
        "video_id": "V001",
        "provenance": {**provenance, "nms_threshold": 0.7},
        "frames": [{"frame_id": "V001_00000_015", "objects": []}],
    }
    output_path.write_text(json.dumps(stale), encoding="utf-8")
    assert not is_valid_object_artifact(
        output_path,
        "V001",
        ["V001_00000_015"],
        provenance,
    )
