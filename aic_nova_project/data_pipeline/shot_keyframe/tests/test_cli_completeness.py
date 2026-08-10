import json
import sys

import pandas as pd
import pytest

from data_pipeline.shot_keyframe import cli


def _valid_metadata(video_id: str) -> dict:
    return {
        "contract_version": "self-indexed-v2",
        "video_id": video_id,
        "source_path": f"raw_videos/{video_id}.mp4",
        "source_video_rel_path": f"raw_videos/{video_id}.mp4",
        "fps": 30.0,
        "duration_sec": 1.0,
        "frame_count": 30,
        "width": 320,
        "height": 240,
        "num_shots": 1,
        "shots": [
            {
                "shot_id": 0,
                "keyframes": [
                    {
                        "position": 0.15,
                        "position_code": 15,
                        "frame_index": 4,
                        "source_frame_idx": 4,
                        "time_sec": 0.133,
                        "file_path": (
                            f"keyframes/{video_id}/"
                            "shot_00000_pos_015.webp"
                        ),
                        "image_rel_path": (
                            f"keyframes/{video_id}/"
                            "shot_00000_pos_015.webp"
                        ),
                    }
                ],
            }
        ],
    }


def test_build_parquet_index_refuses_partial_metadata(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "V001.json").write_text(
        json.dumps(_valid_metadata("V001")),
        encoding="utf-8",
    )
    (metadata_dir / "V002.json").write_text("{invalid", encoding="utf-8")
    output_path = tmp_path / "metadata_index.parquet"

    with pytest.raises(RuntimeError, match="V002.json"):
        cli.build_parquet_index(str(metadata_dir), str(output_path))

    assert not output_path.exists()


def test_build_parquet_index_publishes_complete_artifact(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "V001.json").write_text(
        json.dumps(_valid_metadata("V001")),
        encoding="utf-8",
    )
    output_path = tmp_path / "metadata_index.parquet"

    record_count = cli.build_parquet_index(
        str(metadata_dir),
        str(output_path),
    )

    assert record_count == 1
    assert len(pd.read_parquet(output_path)) == 1


def test_cli_returns_failure_when_any_video_fails(
    tmp_path,
    monkeypatch,
):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "V001.mp4").write_bytes(b"placeholder")
    monkeypatch.setattr(cli, "process_single_video", lambda _: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shot-keyframe",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--device",
            "cpu",
        ],
    )

    assert cli.main() == 1
