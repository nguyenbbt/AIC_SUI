import pytest
import os
import tempfile
from data_pipeline.shot_keyframe.keyframe_extractor import KeyframeExtractor

def test_keyframe_extractor_short_shot(mock_video_path):
    # Test shot with only 1 frame
    extractor = KeyframeExtractor(positions=[0.15, 0.5, 0.85], webp_quality=90)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        keyframes_dir = os.path.join(temp_dir, "keyframes")
        # shot from frame 10 to 10 (duration 0)
        shots = [(10, 10)]
        shots_metadata, fps = extractor.extract_keyframes(mock_video_path, "short_test", shots, keyframes_dir)
        
        assert len(shots_metadata) == 1
        assert len(shots_metadata[0]["keyframes"]) == 3
        
        # All 3 keyframes should map to frame 10
        for kf in shots_metadata[0]["keyframes"]:
            assert kf["frame_index"] == 10
            abs_path = os.path.join(temp_dir, kf["file_path"])
            assert os.path.exists(abs_path)

def test_keyframe_extractor_normal_shot(mock_video_path):
    extractor = KeyframeExtractor(positions=[0.15, 0.5, 0.85], webp_quality=90)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        keyframes_dir = os.path.join(temp_dir, "keyframes")
        # shot from 0 to 100
        shots = [(0, 100)]
        shots_metadata, fps = extractor.extract_keyframes(mock_video_path, "normal_test", shots, keyframes_dir)
        
        assert len(shots_metadata) == 1
        kfs = shots_metadata[0]["keyframes"]
        assert len(kfs) == 3
        assert kfs[0]["frame_index"] == 15 # 0 + 0.15 * 100
        assert kfs[1]["frame_index"] == 50 # 0 + 0.5 * 100
        assert kfs[2]["frame_index"] == 85 # 0 + 0.85 * 100
