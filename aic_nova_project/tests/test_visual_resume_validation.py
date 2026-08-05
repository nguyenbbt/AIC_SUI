import os

import pandas as pd
import pytest

from feature_extraction.visual_embedding.resume_validation import (
    visual_output_is_valid,
)


def _write_artifact(output_path, image_path, **overrides):
    image_stat = image_path.stat()
    row = {
        "frame_id": "V001_00000_015",
        "video_id": "V001",
        "shot_id": 0,
        "position": 0.15,
        "file_path": str(image_path),
        "model_id": "model-a",
        "precision": "fp32",
        "source_size_bytes": image_stat.st_size,
        "source_mtime_ns": image_stat.st_mtime_ns,
        "embedding_dim": 2,
        "embedding": [1.0, 0.0],
    }
    row.update(overrides)
    pd.DataFrame([row]).to_parquet(output_path, index=False)


def _expected_records(image_path):
    return [
        {
            "frame_id": "V001_00000_015",
            "video_id": "V001",
            "shot_id": 0,
            "position": 0.15,
            "file_path": str(image_path),
        }
    ]


def test_visual_resume_accepts_complete_matching_artifact(tmp_path):
    image_path = tmp_path / "frame.webp"
    image_path.write_bytes(b"source-image")
    output_path = tmp_path / "V001.parquet"
    _write_artifact(output_path, image_path)

    assert visual_output_is_valid(
        output_path,
        _expected_records(image_path),
        model_id="model-a",
        precision="fp32",
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"frame_id": "shot_00000_pos_015"},
        {"model_id": "model-b"},
        {"precision": "fp16"},
        {"embedding_dim": 3},
        {"embedding": [2.0, 0.0]},
        {"embedding": [float("nan"), 0.0]},
    ],
)
def test_visual_resume_rejects_stale_or_invalid_artifact(
    tmp_path,
    overrides,
):
    image_path = tmp_path / "frame.webp"
    image_path.write_bytes(b"source-image")
    output_path = tmp_path / "V001.parquet"
    _write_artifact(output_path, image_path, **overrides)

    assert not visual_output_is_valid(
        output_path,
        _expected_records(image_path),
        model_id="model-a",
        precision="fp32",
    )


def test_visual_resume_rejects_changed_source_image(tmp_path):
    image_path = tmp_path / "frame.webp"
    image_path.write_bytes(b"source-image")
    output_path = tmp_path / "V001.parquet"
    _write_artifact(output_path, image_path)

    image_path.write_bytes(b"changed-source-image")
    os.utime(image_path, None)

    assert not visual_output_is_valid(
        output_path,
        _expected_records(image_path),
        model_id="model-a",
        precision="fp32",
    )
