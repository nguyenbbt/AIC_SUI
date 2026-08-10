from PIL import Image

from data_pipeline.shot_keyframe.metadata_schema import (
    KeyframeMetadata,
    ShotMetadata,
    VideoMetadata,
)
from data_pipeline.shot_keyframe.fingerprints import sha256_file
from data_pipeline.shot_keyframe.resume_validation import (
    keyframe_artifacts_are_valid,
)


def _metadata(image_sha256: str = "0" * 64) -> VideoMetadata:
    return VideoMetadata(
        video_id="V001",
        source_path="videos/V001.mp4",
        source_video_rel_path="videos/V001.mp4",
        fps=30.0,
        duration_sec=1.0,
        frame_count=31,
        width=320,
        height=240,
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
                        file_path=(
                            "keyframes/V001/shot_00000_pos_015.webp"
                        ),
                        image_sha256=image_sha256,
                    )
                ],
            )
        ],
    )


def test_resume_accepts_readable_keyframe_artifacts(tmp_path):
    image_path = (
        tmp_path / "keyframes" / "V001" / "shot_00000_pos_015.webp"
    )
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(image_path)

    assert keyframe_artifacts_are_valid(
        _metadata(sha256_file(image_path)),
        tmp_path,
    )


def test_resume_rejects_missing_keyframe_artifacts(tmp_path):
    assert not keyframe_artifacts_are_valid(_metadata(), tmp_path)


def test_resume_rejects_corrupt_keyframe_artifacts(tmp_path):
    image_path = (
        tmp_path / "keyframes" / "V001" / "shot_00000_pos_015.webp"
    )
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"not a webp image")

    assert not keyframe_artifacts_are_valid(_metadata(), tmp_path)
