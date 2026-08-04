from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from feature_extraction.asr_transcript.asr_engine import ASREngine
from feature_extraction.asr_transcript.pipeline import ASRTranscriptPipeline
from feature_extraction.asr_transcript.summarizer import VideoSummarizer


def test_asr_engine_propagates_transcription_failure() -> None:
    engine = ASREngine.__new__(ASREngine)
    engine.batch_size = 4
    engine.transcriber = MagicMock(side_effect=RuntimeError("provider unavailable"))

    with pytest.raises(RuntimeError, match="provider unavailable"):
        engine.transcribe("audio.wav")


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
