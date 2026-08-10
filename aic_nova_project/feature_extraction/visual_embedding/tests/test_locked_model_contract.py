from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from feature_extraction.visual_embedding.config import (
    DEFAULT_VISUAL_MODEL_ID,
)
from feature_extraction.visual_embedding.pipeline import process_video_batch


class WrongDimensionEncoder:
    model_id = DEFAULT_VISUAL_MODEL_ID
    precision = "fp32"

    def encode_batch(self, images):
        return np.asarray(
            [[1.0, 0.0] for _ in images],
            dtype=np.float32,
        )


def test_canonical_visual_model_rejects_wrong_dimension(
    tmp_path: Path,
):
    image_path = tmp_path / "frame.webp"
    Image.new("RGB", (8, 8), color="red").save(image_path, "webp")
    records = [
        {
            "frame_id": "V001_00000_015",
            "video_id": "V001",
            "shot_id": 0,
            "file_path": str(image_path),
        }
    ]

    with pytest.raises(ValueError, match="expected 512"):
        process_video_batch(
            video_id="V001",
            records=records,
            encoder=WrongDimensionEncoder(),
            output_dir=str(tmp_path / "output"),
            batch_size=1,
            num_workers=0,
        )

    assert not (tmp_path / "output" / "V001.parquet").exists()
