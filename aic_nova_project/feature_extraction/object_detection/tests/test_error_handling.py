import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.object_detection.pipeline import ObjectDetectionPipeline

class OOMDetector:
    def __init__(self):
        self.call_count = 0
        
    def detect_batch(self, images):
        self.call_count += 1
        # Fail on first call if batch size > 1
        if len(images) > 1 and self.call_count == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate...")
            
        return [[{"label": "obj", "confidence": 0.9, "bbox": [0,0,10,10], "model_source": "oom_test"}] for _ in images]

def test_oom_handling(tmp_path):
    metadata_dir = tmp_path / "metadata"
    keyframe_dir = tmp_path / "keyframes" / "V001"
    output_dir = tmp_path / "objects"
    
    metadata_dir.mkdir(parents=True)
    keyframe_dir.mkdir(parents=True)
    
    metadata_content = {
        "frames": [
            {"frame_id": "V001_000", "shot_id": 0, "position": 0.1},
            {"frame_id": "V001_001", "shot_id": 0, "position": 0.2},
            {"frame_id": "V001_002", "shot_id": 0, "position": 0.3},
            {"frame_id": "V001_003", "shot_id": 0, "position": 0.4}
        ]
    }
    with open(metadata_dir / "V001.json", "w") as f:
        json.dump(metadata_content, f)
        
    from PIL import Image
    for i in range(4):
        Image.new('RGB', (100, 100)).save(keyframe_dir / f"V001_00{i}.webp")
        
    # Corrupt one image
    with open(keyframe_dir / "V001_002.webp", "wb") as f:
        f.write(b"corrupt data")
        
    with patch('src.object_detection.pipeline.YOLOWorldDetector', return_value=OOMDetector()):
        pipeline = ObjectDetectionPipeline(yolo_world_model="dummy", confidence_threshold=0.5, nms_threshold=0.5)
        
    output_path = output_dir / "V001.json"
    
    # Process batch_size 4. It should OOM on the first call of 3 valid images (since 1 is corrupt).
    # Valid images: 000, 001, 003. Total = 3.
    # Call 1: len=3 -> OOM.
    # Call 2: len=1 -> success
    # Call 3: len=2 -> success
    pipeline.process_video("V001", metadata_dir / "V001.json", keyframe_dir, output_path, batch_size=4)
    
    with open(output_path, "r") as f:
        data = json.load(f)
        
    # Should have 3 frames processed (corrupt one skipped)
    assert len(data["frames"]) == 3
    for frame in data["frames"]:
        assert len(frame["objects"]) == 1
        assert frame["objects"][0]["label"] == "obj"
