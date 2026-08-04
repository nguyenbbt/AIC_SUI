from unittest.mock import MagicMock, patch

from feature_extraction.asr_transcript.summarizer import VideoSummarizer


def test_long_transcript_is_summarized_hierarchically_with_bounded_requests() -> None:
    llm = MagicMock()
    llm.summarize.return_value = "brief"
    summarizer = VideoSummarizer(llm, max_chunk_chars=24)
    intervals = [
        {
            "cleaned_text": (
                "alpha bravo charlie delta echo foxtrot golf hotel india"
            )
        }
    ]

    assert summarizer.summarize_video(intervals) == "brief"

    request_texts = [call.args[0] for call in llm.summarize.call_args_list]
    assert len(request_texts) > 1
    assert all(0 < len(text) <= 24 for text in request_texts)


@patch("feature_extraction.asr_transcript.cli.ASRTranscriptPipeline")
def test_cli_forwards_summary_chunk_budget(
    mock_pipeline_class: MagicMock,
) -> None:
    from feature_extraction.asr_transcript.cli import main

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
        "--summary-chunk-chars",
        "4096",
    ]

    with patch("sys.argv", argv):
        assert main() == 0

    assert mock_pipeline_class.call_args.kwargs["summary_chunk_chars"] == 4096
