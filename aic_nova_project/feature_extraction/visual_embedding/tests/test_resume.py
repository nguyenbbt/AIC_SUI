import json

from PIL import Image

from feature_extraction.visual_embedding import pipeline
from feature_extraction.visual_embedding.config import DEFAULT_VISUAL_MODEL_ID


def test_resume_skip_existing(tmp_path, monkeypatch):
    metadata_dir = tmp_path / "metadata"
    keyframe_dir = tmp_path / "keyframes"
    output_dir = tmp_path / "output"
    image_dir = keyframe_dir / "V001"
    metadata_dir.mkdir()
    image_dir.mkdir(parents=True)

    image_path = image_dir / "shot_00000_pos_015.webp"
    Image.new("RGB", (8, 8), color="red").save(image_path, format="WEBP")
    (metadata_dir / "V001.json").write_text(
        json.dumps(
            {
                "video_id": "V001",
                "shots": [
                    {
                        "shot_id": 0,
                        "keyframes": [
                            {
                                "position": 0.15,
                                "frame_index": 13,
                                "time_sec": 0.52,
                                "file_path": (
                                    "keyframes/V001/"
                                    "shot_00000_pos_015.webp"
                                ),
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    processed_video_ids = []
    original_process_video_batch = pipeline.process_video_batch

    def tracked_process_video_batch(*args, **kwargs):
        processed_video_ids.append(kwargs["video_id"])
        return original_process_video_batch(*args, **kwargs)

    monkeypatch.setattr(
        pipeline,
        "process_video_batch",
        tracked_process_video_batch,
    )

    arguments = {
        "metadata_dir": str(metadata_dir),
        "keyframe_dir": str(keyframe_dir),
        "output_dir": str(output_dir),
        "model_id": DEFAULT_VISUAL_MODEL_ID,
        "device": "cpu",
        "precision": "fp32",
        "batch_size": 1,
        "num_workers": 0,
    }

    pipeline.run_pipeline(**arguments, force=False)
    assert (output_dir / "V001.parquet").is_file()
    assert processed_video_ids == ["V001"]

    pipeline.run_pipeline(**arguments, force=False)
    assert processed_video_ids == ["V001"]

    pipeline.run_pipeline(**arguments, force=True)
    assert processed_video_ids == ["V001", "V001"]
