import json
from unittest.mock import MagicMock, patch

from src.object_detection.pipeline import ObjectDetectionPipeline


class EmptyDetector:
    def detect_batch(self, images):
        return [[] for _ in images]


def test_pipeline_closes_source_and_converted_images(tmp_path):
    metadata_path = tmp_path / "V001.json"
    keyframe_dir = tmp_path / "keyframes"
    output_path = tmp_path / "objects" / "V001.json"
    keyframe_dir.mkdir()
    metadata_path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_id": "V001_000",
                        "shot_id": 0,
                        "position": 0.1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (keyframe_dir / "V001_000.webp").write_bytes(b"placeholder")

    source_image = MagicMock()
    source_image.__enter__.return_value = source_image
    converted_image = MagicMock()
    source_image.convert.return_value = converted_image

    with patch(
        "src.object_detection.pipeline.YOLOWorldDetector",
        return_value=EmptyDetector(),
    ), patch(
        "src.object_detection.pipeline.Image.open",
        return_value=source_image,
    ):
        pipeline = ObjectDetectionPipeline(yolo_world_model="dummy")
        pipeline.process_video(
            "V001",
            metadata_path,
            keyframe_dir,
            output_path,
        )

    source_image.__exit__.assert_called_once()
    converted_image.close.assert_called_once_with()
