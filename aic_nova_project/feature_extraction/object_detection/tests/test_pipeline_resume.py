import json

from src.object_detection.pipeline import ObjectDetectionPipeline


class DummyDetector:
    def __init__(self):
        self.call_count = 0

    def detect_batch(self, images):
        self.call_count += 1
        return [[{"label": "dummy", "confidence": 0.99, "bbox": [0,0,10,10], "model_source": "dummy"}] for _ in images]


def test_pipeline_resume(tmp_path):
    # Setup paths
    metadata_dir = tmp_path / "metadata"
    keyframe_dir = tmp_path / "keyframes" / "V001"
    output_dir = tmp_path / "objects"
    
    metadata_dir.mkdir(parents=True)
    keyframe_dir.mkdir(parents=True)
    
    # Create metadata through the schema used by Module 1.
    metadata_path = metadata_dir / "V001.json"
    metadata_content = {
        "video_id": "V001",
        "source_path": "videos/V001.mp4",
        "fps": 30.0,
        "duration_sec": 1.0,
        "num_shots": 1,
        "shots": [
            {
                "shot_id": 0,
                "start_frame": 0,
                "end_frame": 30,
                "start_time_sec": 0.0,
                "end_time_sec": 1.0,
                "keyframes": [
                    {
                        "position": 0.15,
                        "frame_index": 4,
                        "time_sec": 0.133,
                        "file_path": "keyframes/V001/shot_00000_pos_015.webp",
                    },
                    {
                        "position": 0.50,
                        "frame_index": 15,
                        "time_sec": 0.5,
                        "file_path": "keyframes/V001/shot_00000_pos_050.webp",
                    },
                ],
            }
        ],
    }
    metadata_path.write_text(json.dumps(metadata_content, indent=2), encoding="utf-8")
        
    # Create fake images
    from PIL import Image
    Image.new('RGB', (100, 100)).save(keyframe_dir / "shot_00000_pos_015.webp")
    Image.new('RGB', (100, 100)).save(keyframe_dir / "shot_00000_pos_050.webp")
    
    # Create pipeline with mocked detector
    from unittest.mock import patch
    detector = DummyDetector()
    with patch('src.object_detection.pipeline.YOLOWorldDetector', return_value=detector):
        pipeline = ObjectDetectionPipeline(yolo_world_model="dummy", confidence_threshold=0.5, nms_threshold=0.5)
    
    output_path = output_dir / "V001.json"
    
    # Pre-create a partial output; resume must reject and regenerate it.
    output_dir.mkdir()
    with open(output_path, "w") as f:
        json.dump({"video_id": "V001", "frames": [{"frame_id": "V001_000"}]}, f)
        
    # Run pipeline without force
    pipeline.process_video("V001", metadata_path, keyframe_dir, output_path, force=False)
    
    # Partial output should be overwritten.
    with open(output_path, "r") as f:
        data = json.load(f)
    assert len(data["frames"]) == 2
    calls_after_regeneration = detector.call_count

    # The now-complete artifact should be safely skipped.
    pipeline.process_video(
        "V001",
        metadata_path,
        keyframe_dir,
        output_path,
        force=False,
    )
    assert detector.call_count == calls_after_regeneration
    
    # Run pipeline with force
    pipeline.process_video("V001", metadata_path, keyframe_dir, output_path, force=True)
    
    # Should overwrite
    with open(output_path, "r") as f:
        data = json.load(f)
    assert len(data["frames"]) == 2
    assert [frame["frame_id"] for frame in data["frames"]] == [
        "V001_00000_015",
        "V001_00000_050",
    ]
    assert data["frames"][0]["objects"][0]["label"] == "dummy"
