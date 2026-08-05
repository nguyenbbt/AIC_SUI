import numpy as np
import pytest
from PIL import Image

from feature_extraction.visual_embedding import pipeline


class FakeEncoder:
    model_id = "model-a"
    precision = "fp32"

    def encode_batch(self, images):
        return np.asarray(
            [[1.0, 0.0] for _ in images],
            dtype=np.float32,
        )


def _record(frame_id, file_path):
    return {
        "frame_id": frame_id,
        "video_id": "V001",
        "shot_id": 0,
        "position": 0.15,
        "file_path": str(file_path),
    }


def test_visual_batch_does_not_publish_when_an_image_is_missing(tmp_path):
    valid_image = tmp_path / "valid.webp"
    Image.new("RGB", (10, 10)).save(valid_image)
    missing_image = tmp_path / "missing.webp"

    with pytest.raises(RuntimeError, match="incomplete"):
        pipeline.process_video_batch(
            video_id="V001",
            records=[
                _record("V001_00000_015", valid_image),
                _record("V001_00000_050", missing_image),
            ],
            encoder=FakeEncoder(),
            output_dir=str(tmp_path),
            batch_size=2,
            num_workers=0,
        )

    assert not (tmp_path / "V001.parquet").exists()


def test_visual_batch_does_not_publish_after_oom_retries(
    tmp_path,
    monkeypatch,
):
    image_path = tmp_path / "frame.webp"
    Image.new("RGB", (10, 10)).save(image_path)

    class OOMEncoder(FakeEncoder):
        def encode_batch(self, images):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(
        pipeline.torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        pipeline.torch.cuda,
        "empty_cache",
        lambda: None,
    )

    with pytest.raises(RuntimeError, match="incomplete"):
        pipeline.process_video_batch(
            video_id="V001",
            records=[_record("V001_00000_015", image_path)],
            encoder=OOMEncoder(),
            output_dir=str(tmp_path),
            batch_size=1,
            num_workers=0,
        )

    assert not (tmp_path / "V001.parquet").exists()
