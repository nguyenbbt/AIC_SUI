import pytest
import os
import tempfile
import numpy as np
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


def test_keyframe_extractor_records_the_frame_that_fallback_decoded(
    tmp_path,
    monkeypatch,
):
    class FallbackCapture:
        def __init__(self):
            self.frame_index = 0

        def isOpened(self):
            return True

        def get(self, property_id):
            return 25.0

        def set(self, property_id, value):
            self.frame_index = int(value)
            return True

        def read(self):
            if self.frame_index == 15:
                return False, None
            return True, np.full((24, 32, 3), 127, dtype=np.uint8)

        def release(self):
            return None

    monkeypatch.setattr(
        "data_pipeline.shot_keyframe.keyframe_extractor.cv2.VideoCapture",
        lambda _: FallbackCapture(),
    )
    extractor = KeyframeExtractor(positions=[0.15])

    metadata, fps = extractor.extract_keyframes(
        "fallback.mp4",
        "V001",
        [(0, 100)],
        str(tmp_path / "keyframes"),
    )

    keyframe = metadata[0]["keyframes"][0]
    assert keyframe["frame_index"] == 0
    assert keyframe["source_frame_idx"] == 0
    assert keyframe["time_sec"] == 0.0
    assert keyframe["image_rel_path"] == keyframe["file_path"]
    assert fps == 25.0


def test_keyframe_extractor_fails_closed_without_placeholder(
    tmp_path,
    monkeypatch,
):
    class FailedCapture:
        def isOpened(self):
            return True

        def get(self, property_id):
            return 25.0

        def set(self, property_id, value):
            return True

        def read(self):
            return False, None

        def release(self):
            return None

    monkeypatch.setattr(
        "data_pipeline.shot_keyframe.keyframe_extractor.cv2.VideoCapture",
        lambda _: FailedCapture(),
    )
    extractor = KeyframeExtractor(positions=[0.15])

    with pytest.raises(RuntimeError, match="Could not decode a real keyframe"):
        extractor.extract_keyframes(
            "unreadable.mp4",
            "V001",
            [(0, 10)],
            str(tmp_path / "keyframes"),
        )

    assert list(tmp_path.rglob("*.webp")) == []
