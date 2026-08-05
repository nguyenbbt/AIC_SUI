from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from feature_extraction.asr_transcript.asr_engine import ASREngine
from feature_extraction.asr_transcript.pipeline import ASRTranscriptPipeline
from feature_extraction.asr_transcript.summarizer import VideoSummarizer


def test_asr_engine_propagates_transcription_failure() -> None:
    engine = ASREngine.__new__(ASREngine)
    engine.window_duration_sec = 30.0
    engine._read_wav = MagicMock(
        return_value=(np.zeros(16000, dtype=np.float32), 16000)
    )
    engine.transcriber = MagicMock(side_effect=RuntimeError("provider unavailable"))

    with pytest.raises(RuntimeError, match="provider unavailable"):
        engine.transcribe("audio.wav")


def test_asr_engine_uses_exact_audio_window_boundaries() -> None:
    engine = ASREngine.__new__(ASREngine)
    engine.window_duration_sec = 30.0
    engine._read_wav = MagicMock(
        return_value=(np.zeros(45 * 10, dtype=np.float32), 10)
    )
    engine.transcriber = MagicMock(
        side_effect=[{"text": "đoạn một"}, {"text": "đoạn hai"}]
    )

    assert engine.transcribe("audio.wav") == [
        {"timestamp": (0.0, 30.0), "text": "đoạn một"},
        {"timestamp": (30.0, 45.0), "text": "đoạn hai"},
    ]
    first_input = engine.transcriber.call_args_list[0].args[0]
    second_input = engine.transcriber.call_args_list[1].args[0]
    assert len(first_input["array"]) == 300
    assert len(second_input["array"]) == 150


def test_asr_engine_disables_provider_timestamp_chunking() -> None:
    engine = ASREngine.__new__(ASREngine)
    engine.window_duration_sec = 30.0
    engine._read_wav = MagicMock(
        return_value=(np.zeros(25 * 10, dtype=np.float32), 10)
    )
    engine.transcriber = MagicMock(
        return_value={"text": "nội dung"}
    )

    assert engine.transcribe("audio.wav") == [
        {"timestamp": (0.0, 25.0), "text": "nội dung"}
    ]
    call_kwargs = engine.transcriber.call_args.kwargs
    assert call_kwargs["return_timestamps"] is False
    assert "chunk_length_s" not in call_kwargs
    assert "batch_size" not in call_kwargs


def test_summarizer_propagates_provider_failure() -> None:
    llm = MagicMock()
    llm.summarize.side_effect = RuntimeError("quota exceeded")
    summarizer = VideoSummarizer(llm)

    with pytest.raises(RuntimeError, match="quota exceeded"):
        summarizer.summarize_video([{"cleaned_text": "usable transcript"}])


def test_summarizer_rejects_empty_provider_response() -> None:
    llm = MagicMock()
    llm.summarize.return_value = "  "
    summarizer = VideoSummarizer(llm)

    with pytest.raises(ValueError, match="empty summary"):
        summarizer.summarize_video([{"cleaned_text": "usable transcript"}])


def test_pipeline_run_reports_failures_after_processing_all_videos() -> None:
    pipeline = ASRTranscriptPipeline.__new__(ASRTranscriptPipeline)
    pipeline.llm = SimpleNamespace(total_tokens_used=None)
    pipeline._get_video_ids = MagicMock(return_value=["V001", "V002"])
    pipeline.process_video = MagicMock(
        side_effect=[RuntimeError("ASR failed"), None]
    )

    with pytest.raises(RuntimeError, match=r"V001.*ASR failed"):
        pipeline.run()

    assert pipeline.process_video.call_count == 2


@patch("feature_extraction.asr_transcript.cli.ASRTranscriptPipeline")
def test_cli_returns_failure_exit_code(mock_pipeline_class: MagicMock) -> None:
    from feature_extraction.asr_transcript.cli import main

    mock_pipeline_class.return_value.run.side_effect = RuntimeError(
        "1 video failed"
    )
    argv = [
        "asr-transcript",
        "--video-dir",
        "videos",
        "--metadata-dir",
        "metadata",
        "--caption-dir",
        "captions",
        "--output-dir",
        "output",
    ]

    with patch("sys.argv", argv):
        assert main() == 1
