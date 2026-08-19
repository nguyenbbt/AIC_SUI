import json
import threading
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from ocr_module.pipeline import run_pipeline
from ocr_module.recognizer import TextRecognizer


def test_vietocr_legacy_antialias_uses_pillow_resampling_lanczos():
    assert Image.ANTIALIAS == Image.Resampling.LANCZOS


def test_recognizer_uses_vietocr_batch_api():
    recognizer = TextRecognizer.__new__(TextRecognizer)
    recognizer.detector = MagicMock()
    recognizer.detector.predict_batch.return_value = (
        ["first", "second"],
        [0.9, 0.8],
    )
    images = [MagicMock(), MagicMock()]

    assert recognizer.recognize_batch(images) == [
        ("first", 0.9),
        ("second", 0.8),
    ]
    recognizer.detector.predict_batch.assert_called_once_with(
        images,
        return_prob=True,
    )


@patch("ocr_module.pipeline.OCRPipeline")
def test_run_pipeline_processes_cpu_videos_concurrently(
    mock_pipeline_class,
    tmp_path,
):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    for video_id in ("V001", "V002"):
        (metadata_dir / f"{video_id}.json").write_text(
            json.dumps({"video_id": video_id}),
            encoding="utf-8",
        )

    barrier = threading.Barrier(2)
    pipeline = MagicMock()

    def process_video(**kwargs):
        barrier.wait(timeout=2)

    pipeline.process_video.side_effect = process_video
    mock_pipeline_class.return_value = pipeline

    run_pipeline(
        keyframe_dir=str(tmp_path / "keyframes"),
        metadata_dir=str(metadata_dir),
        output_dir=str(tmp_path / "output"),
        use_gpu=False,
        workers=2,
        batch_size=8,
    )

    assert pipeline.process_video.call_count == 2
    assert {
        call.kwargs["batch_size"]
        for call in pipeline.process_video.call_args_list
    } == {8}


def test_run_pipeline_rejects_multiple_gpu_workers(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()

    with pytest.raises(ValueError, match="GPU"):
        run_pipeline(
            keyframe_dir=str(tmp_path / "keyframes"),
            metadata_dir=str(metadata_dir),
            output_dir=str(tmp_path / "output"),
            use_gpu=True,
            workers=2,
        )


@patch("ocr_module.pipeline.OCRPipeline")
def test_run_pipeline_processes_only_its_deterministic_shard(
    mock_pipeline_class,
    tmp_path,
):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    for index in range(6):
        video_id = f"V{index:03d}"
        (metadata_dir / f"{video_id}.json").write_text(
            json.dumps({"video_id": video_id}),
            encoding="utf-8",
        )

    pipeline = MagicMock()
    mock_pipeline_class.return_value = pipeline

    run_pipeline(
        keyframe_dir=str(tmp_path / "keyframes"),
        metadata_dir=str(metadata_dir),
        output_dir=str(tmp_path / "output"),
        use_gpu=True,
        workers=1,
        shard_index=1,
        shard_count=2,
    )

    assert [
        call.kwargs["video_id"]
        for call in pipeline.process_video.call_args_list
    ] == ["V001", "V003", "V005"]


@pytest.mark.parametrize(
    ("shard_index", "shard_count"),
    [(-1, 2), (2, 2), (0, 0)],
)
def test_run_pipeline_rejects_invalid_shard_bounds(
    tmp_path,
    shard_index,
    shard_count,
):
    with pytest.raises(ValueError, match="shard"):
        run_pipeline(
            keyframe_dir=str(tmp_path / "keyframes"),
            metadata_dir=str(tmp_path / "metadata"),
            output_dir=str(tmp_path / "output"),
            shard_index=shard_index,
            shard_count=shard_count,
        )


@patch("ocr_module.cli.run_pipeline")
def test_cli_forwards_batch_size(mock_run_pipeline, monkeypatch):
    from ocr_module.cli import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "ocr",
            "--keyframe-dir",
            "keyframes",
            "--metadata-dir",
            "metadata",
            "--output-dir",
            "output",
            "--batch-size",
            "16",
            "--shard-index",
            "2",
            "--shard-count",
            "5",
        ],
    )

    assert main() is None
    assert mock_run_pipeline.call_args.kwargs["batch_size"] == 16
    assert mock_run_pipeline.call_args.kwargs["shard_index"] == 2
    assert mock_run_pipeline.call_args.kwargs["shard_count"] == 5
