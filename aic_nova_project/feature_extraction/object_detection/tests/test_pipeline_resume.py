import json
from pathlib import Path
from src.object_detection.pipeline import ObjectDetectionPipeline

class DummyDetector:
    def detect_batch(self, images):
        return [[{"label": "dummy", "confidence": 0.99, "bbox": [0,0,10,10], "model_source": "dummy"}] for _ in images]

def test_pipeline_resume(tmp_path):
    # Setup paths
    metadata_dir = tmp_path / "metadata"
    keyframe_dir = tmp_path / "keyframes" / "V001"
    output_dir = tmp_path / "objects"
    
    metadata_dir.mkdir(parents=True)
    keyframe_dir.mkdir(parents=True)
    
    # Create metadata
    metadata_path = metadata_dir / "V001.json"
    metadata_content = {
        "frames": [
            {"frame_id": "V001_000", "shot_id": 0, "position": 0.1},
            {"frame_id": "V001_001", "shot_id": 0, "position": 0.2}
        ]
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata_content, f)
        
    # Create fake images
    from PIL import Image
    Image.new('RGB', (100, 100)).save(keyframe_dir / "V001_000.webp")
    Image.new('RGB', (100, 100)).save(keyframe_dir / "V001_001.webp")
    
    # Create pipeline with mocked detector
    from unittest.mock import patch
    with patch('src.object_detection.pipeline.YOLOWorldDetector', return_value=DummyDetector()):
        pipeline = ObjectDetectionPipeline(yolo_world_model="dummy", confidence_threshold=0.5, nms_threshold=0.5)
    
    output_path = output_dir / "V001.json"
    
    # Pre-create output to trigger resume logic
    output_dir.mkdir()
    with open(output_path, "w") as f:
        json.dump({"video_id": "V001", "frames": [{"frame_id": "V001_000"}]}, f)
        
    # Run pipeline without force
    pipeline.process_video("V001", metadata_path, keyframe_dir, output_path, force=False)
    
    # Should not overwrite
    with open(output_path, "r") as f:
        data = json.load(f)
    assert len(data["frames"]) == 1
    
    # Run pipeline with force
    pipeline.process_video("V001", metadata_path, keyframe_dir, output_path, force=True)
    
    # Should overwrite
    with open(output_path, "r") as f:
        data = json.load(f)
    assert len(data["frames"]) == 2
    assert data["frames"][0]["objects"][0]["label"] == "dummy"
