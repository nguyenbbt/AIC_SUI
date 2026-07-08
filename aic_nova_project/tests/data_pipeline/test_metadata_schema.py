import pytest
from pydantic import ValidationError
from data_pipeline.shot_keyframe.metadata_schema import KeyframeMetadata, ShotMetadata, VideoMetadata

def test_keyframe_metadata_valid():
    kf = KeyframeMetadata(position=0.15, frame_index=15, time_sec=0.5, file_path="path.webp")
    assert kf.position == 0.15

def test_keyframe_metadata_invalid_position():
    with pytest.raises(ValidationError):
        KeyframeMetadata(position=1.5, frame_index=15, time_sec=0.5, file_path="path.webp")

def test_shot_metadata_valid():
    shot = ShotMetadata(
        shot_id=0,
        start_frame=0,
        end_frame=30,
        start_time_sec=0.0,
        end_time_sec=1.0,
        keyframes=[
            KeyframeMetadata(position=0.15, frame_index=4, time_sec=0.15, file_path="1.webp"),
            KeyframeMetadata(position=0.5, frame_index=15, time_sec=0.5, file_path="2.webp"),
            KeyframeMetadata(position=0.85, frame_index=25, time_sec=0.85, file_path="3.webp")
        ]
    )
    assert len(shot.keyframes) == 3
