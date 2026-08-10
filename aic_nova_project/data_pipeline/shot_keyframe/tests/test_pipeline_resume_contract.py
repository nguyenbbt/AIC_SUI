import json
import os
from pathlib import Path

from PIL import Image
import pytest

import data_pipeline.shot_keyframe.pipeline as pipeline_module
from data_pipeline.shot_keyframe.fingerprints import (
    build_processing_config_fingerprint,
    sha256_file,
)
from data_pipeline.shot_keyframe.metadata_schema import VideoMetadata
from data_pipeline.shot_keyframe.pipeline import VideoProcessor
from data_pipeline.shot_keyframe.resume_validation import (
    keyframe_artifacts_are_valid,
)


def _write_image(path: Path, color: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color=color).save(path, "webp")
    return sha256_file(path)


def _metadata(
    *,
    source_fingerprint: str,
    config_fingerprint: str,
    image_fingerprint: str,
) -> VideoMetadata:
    return VideoMetadata.model_validate(
        {
            "video_id": "V001",
            "source_path": "videos/V001.mp4",
            "source_video_rel_path": "videos/V001.mp4",
            "source_fingerprint": source_fingerprint,
            "producer_config_fingerprint": config_fingerprint,
            "fps": 25.0,
            "duration_sec": 4.0,
            "frame_count": 100,
            "width": 1920,
            "height": 1080,
            "num_shots": 1,
            "shots": [
                {
                    "shot_id": 0,
                    "start_frame": 0,
                    "end_frame": 99,
                    "start_time_sec": 0.0,
                    "end_time_sec": 3.96,
                    "keyframes": [
                        {
                            "position": 0.15,
                            "frame_index": 15,
                            "time_sec": 0.6,
                            "file_path": (
                                "keyframes/V001/"
                                "shot_00000_pos_015.webp"
                            ),
                            "image_sha256": image_fingerprint,
                        }
                    ],
                }
            ],
        }
    )


def test_resume_requires_matching_source_config_and_image_content(tmp_path):
    source_path = tmp_path / "videos" / "V001.mp4"
    source_path.parent.mkdir()
    source_path.write_bytes(b"original-video")
    source_fingerprint = sha256_file(source_path)
    config_fingerprint = build_processing_config_fingerprint(
        threshold=0.5,
        positions=[0.15, 0.5, 0.85],
        webp_quality=90,
    )
    image_path = (
        tmp_path / "keyframes" / "V001" /
        "shot_00000_pos_015.webp"
    )
    image_fingerprint = _write_image(image_path, "red")
    metadata = _metadata(
        source_fingerprint=source_fingerprint,
        config_fingerprint=config_fingerprint,
        image_fingerprint=image_fingerprint,
    )

    assert keyframe_artifacts_are_valid(
        metadata,
        tmp_path,
        expected_source_fingerprint=source_fingerprint,
        expected_config_fingerprint=config_fingerprint,
        expected_source_video_rel_path="videos/V001.mp4",
    )

    source_path.write_bytes(b"changed-video")
    assert not keyframe_artifacts_are_valid(
        metadata,
        tmp_path,
        expected_source_fingerprint=sha256_file(source_path),
        expected_config_fingerprint=config_fingerprint,
        expected_source_video_rel_path="videos/V001.mp4",
    )

    assert not keyframe_artifacts_are_valid(
        metadata,
        tmp_path,
        expected_source_fingerprint=source_fingerprint,
        expected_config_fingerprint=build_processing_config_fingerprint(
            threshold=0.6,
            positions=[0.15, 0.5, 0.85],
            webp_quality=90,
        ),
        expected_source_video_rel_path="videos/V001.mp4",
    )

    _write_image(image_path, "blue")
    assert not keyframe_artifacts_are_valid(
        metadata,
        tmp_path,
        expected_source_fingerprint=source_fingerprint,
        expected_config_fingerprint=config_fingerprint,
        expected_source_video_rel_path="videos/V001.mp4",
    )


def test_failed_rebuild_preserves_last_known_good_artifacts(
    tmp_path,
    mock_video_path,
):
    processor = VideoProcessor(output_dir=str(tmp_path), device="cpu")

    class MockTransNet:
        def predict_shots(self, video_path, threshold):
            return [(0, 89)]

    processor.transnet = MockTransNet()
    assert processor.process_video(mock_video_path)

    metadata_path = tmp_path / "metadata" / "test_video.json"
    original_metadata = metadata_path.read_bytes()
    final_keyframe_dir = tmp_path / "keyframes" / "test_video"
    original_images = {
        path.name: path.read_bytes()
        for path in final_keyframe_dir.glob("*.webp")
    }

    processor.threshold = 0.6

    def fail_after_partial_write(
        *,
        video_path,
        video_id,
        shots,
        output_dir,
    ):
        partial_dir = Path(output_dir) / video_id
        partial_dir.mkdir(parents=True, exist_ok=True)
        (partial_dir / "poison.webp").write_bytes(b"partial")
        raise RuntimeError("simulated extraction failure")

    processor.extractor.extract_keyframes = fail_after_partial_write

    assert not processor.process_video(mock_video_path)
    assert metadata_path.read_bytes() == original_metadata
    assert {
        path.name: path.read_bytes()
        for path in final_keyframe_dir.glob("*.webp")
    } == original_images
    assert not list((tmp_path / "keyframes").glob(".test_video.*"))

    metadata = json.loads(original_metadata)
    assert metadata["source_fingerprint"]
    assert metadata["producer_config_fingerprint"]


@pytest.mark.parametrize("failure_point", ["backup", "metadata"])
def test_publish_failure_restores_last_known_good_artifacts(
    tmp_path,
    mock_video_path,
    monkeypatch,
    failure_point,
):
    processor = VideoProcessor(output_dir=str(tmp_path), device="cpu")

    class MockTransNet:
        def predict_shots(self, video_path, threshold):
            return [(0, 89)]

    processor.transnet = MockTransNet()
    assert processor.process_video(mock_video_path)

    metadata_path = tmp_path / "metadata" / "test_video.json"
    final_keyframe_dir = tmp_path / "keyframes" / "test_video"
    original_metadata = metadata_path.read_bytes()
    original_images = {
        path.name: path.read_bytes()
        for path in final_keyframe_dir.glob("*.webp")
    }
    processor.threshold = 0.6

    real_replace = os.replace

    def fail_selected_replace(source, destination):
        destination_path = Path(destination)
        should_fail = (
            failure_point == "backup"
            and ".test_video.backup-" in destination_path.name
        ) or (
            failure_point == "metadata"
            and destination_path == metadata_path
        )
        if should_fail:
            raise OSError(f"simulated {failure_point} publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(pipeline_module.os, "replace", fail_selected_replace)

    assert not processor.process_video(mock_video_path)
    assert metadata_path.read_bytes() == original_metadata
    assert {
        path.name: path.read_bytes()
        for path in final_keyframe_dir.glob("*.webp")
    } == original_images
    assert not list((tmp_path / "keyframes").glob(".test_video.*"))
