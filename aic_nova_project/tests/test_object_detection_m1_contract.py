from data_pipeline.shot_keyframe.metadata_schema import (
    KeyframeMetadata,
    ShotMetadata,
    VideoMetadata,
)
from feature_extraction.object_detection.src.object_detection.metadata_reader import (
    read_metadata,
)


def test_object_detection_reads_module1_metadata(tmp_path):
    metadata = VideoMetadata(
        video_id="V001",
        source_path="videos/V001.mp4",
        fps=30.0,
        duration_sec=1.0,
        num_shots=1,
        shots=[
            ShotMetadata(
                shot_id=0,
                start_frame=0,
                end_frame=30,
                start_time_sec=0.0,
                end_time_sec=1.0,
                keyframes=[
                    KeyframeMetadata(
                        position=0.15,
                        frame_index=4,
                        time_sec=0.133,
                        file_path="keyframes/V001/shot_00000_pos_015.webp",
                    ),
                    KeyframeMetadata(
                        position=0.50,
                        frame_index=15,
                        time_sec=0.5,
                        file_path="keyframes/V001/shot_00000_pos_050.webp",
                    ),
                ],
            )
        ],
    )
    metadata_path = tmp_path / "V001.json"
    metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

    assert read_metadata(metadata_path) == [
        {
            "frame_id": "V001_00000_015",
            "shot_id": 0,
            "position": 0.15,
            "file_path": "keyframes/V001/shot_00000_pos_015.webp",
        },
        {
            "frame_id": "V001_00000_050",
            "shot_id": 0,
            "position": 0.50,
            "file_path": "keyframes/V001/shot_00000_pos_050.webp",
        },
    ]
