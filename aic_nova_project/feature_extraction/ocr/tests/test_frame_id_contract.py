import json
from unittest.mock import patch

import numpy as np

from ocr_module.pipeline import OCRPipeline


def test_pipeline_opens_local_filename_and_writes_global_frame_id(tmp_path):
    video_id = "V001"
    keyframe_dir = tmp_path / "keyframes"
    video_keyframe_dir = keyframe_dir / video_id
    metadata_dir = tmp_path / "metadata"
    output_dir = tmp_path / "ocr"
    video_keyframe_dir.mkdir(parents=True)
    metadata_dir.mkdir()

    local_filename = "shot_00000_pos_015.webp"
    (video_keyframe_dir / local_filename).write_bytes(b"placeholder")
    (metadata_dir / f"{video_id}.json").write_text(
        json.dumps(
            {
                "video_id": video_id,
                "shots": [
                    {
                        "shot_id": 0,
                        "keyframes": [
                            {
                                "position": 0.15,
                                "file_path": (
                                    f"keyframes/{video_id}/{local_filename}"
                                ),
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with patch("ocr_module.pipeline.TextDetector"), patch(
        "ocr_module.pipeline.TextRecognizer"
    ):
        pipeline = OCRPipeline(use_gpu=False)
    pipeline.detector.detect.return_value = []

    with patch(
        "ocr_module.pipeline.cv2.imread",
        return_value=np.zeros((10, 10, 3), dtype=np.uint8),
    ):
        pipeline.process_video(
            video_id,
            keyframe_dir,
            metadata_dir,
            output_dir,
        )

    output = json.loads(
        (output_dir / f"{video_id}.json").read_text(encoding="utf-8")
    )
    assert output["frames"][0]["frame_id"] == "V001_00000_015"
