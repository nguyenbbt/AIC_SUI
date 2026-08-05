import numpy as np
import pandas as pd
from PIL import Image

from feature_extraction.visual_embedding.pipeline import process_video_batch


def test_visual_artifact_records_actual_configured_model_id(tmp_path):
    image_path = tmp_path / "frame.webp"
    Image.new("RGB", (10, 10)).save(image_path)

    class FakeEncoder:
        model_id = "hf-hub:organization/custom-vision-model"
        precision = "fp32"

        def encode_batch(self, images):
            return np.asarray(
                [[1.0, 0.0] for _ in images],
                dtype=np.float32,
            )

    processed_count = process_video_batch(
        video_id="V001",
        records=[
            {
                "frame_id": "V001_00000_015",
                "video_id": "V001",
                "shot_id": 0,
                "position": 0.15,
                "file_path": str(image_path),
            }
        ],
        encoder=FakeEncoder(),
        output_dir=str(tmp_path),
        batch_size=1,
        num_workers=0,
    )

    dataframe = pd.read_parquet(tmp_path / "V001.parquet")
    assert processed_count == 1
    assert dataframe.loc[0, "model_name"] == FakeEncoder.model_id
    assert dataframe.loc[0, "model_id"] == FakeEncoder.model_id
