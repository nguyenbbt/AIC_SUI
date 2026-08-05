import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from feature_extraction.asr_transcript.pipeline import ASRTranscriptPipeline


def _write_cached_transcripts(output_dir: Path) -> None:
    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(parents=True)
    (output_dir / "summaries").mkdir(parents=True)

    (transcripts_dir / "V001_raw.json").write_text(
        json.dumps(
            {
                "video_id": "V001",
                "source": "asr",
                "segments": [{"timestamp": [0.0, 1.0], "text": "raw"}],
            }
        ),
        encoding="utf-8",
    )
    (transcripts_dir / "V001_cleaned.json").write_text(
        json.dumps(
            {
                "video_id": "V001",
                "source": "asr",
                "intervals": [
                    {
                        "interval_id": "0",
                        "start_time_sec": 0.0,
                        "end_time_sec": 1.0,
                        "raw_text": "raw",
                        "cleaned_text": "cleaned",
                        "cleaning_failed": False,
                        "segment_ids": [0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "cached_summary",
    [
        '{"video_id": "V001", "summary": ""}',
        '{"video_id": "OTHER", "summary": "wrong video"}',
        "{not valid json",
    ],
)
@patch("feature_extraction.asr_transcript.llm.gemini_llm.GeminiTranscriptLLM")
def test_invalid_cached_summary_is_regenerated(
    mock_llm_class: MagicMock,
    cached_summary: str,
    tmp_path: Path,
) -> None:
    video_dir = tmp_path / "videos"
    metadata_dir = tmp_path / "metadata"
    caption_dir = tmp_path / "captions"
    output_dir = tmp_path / "output"
    for directory in (video_dir, metadata_dir, caption_dir):
        directory.mkdir()
    _write_cached_transcripts(output_dir)

    summary_path = output_dir / "summaries" / "V001.json"
    summary_path.write_text(cached_summary, encoding="utf-8")

    llm = MagicMock()
    llm.summarize.return_value = "new summary"
    mock_llm_class.return_value = llm
    pipeline = ASRTranscriptPipeline(
        video_dir=str(video_dir),
        metadata_dir=str(metadata_dir),
        caption_dir=str(caption_dir),
        output_dir=str(output_dir),
        llm_provider="gemini",
    )

    pipeline.process_video("V001")

    llm.summarize.assert_called_once_with("cleaned")
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["video_id"] == "V001"
    assert saved["summary"] == "new summary"
