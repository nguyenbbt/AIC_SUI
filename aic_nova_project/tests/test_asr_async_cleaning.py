import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from feature_extraction.asr_transcript.pipeline import ASRTranscriptPipeline


@patch("feature_extraction.asr_transcript.llm.gemini_llm.GeminiTranscriptLLM")
def test_process_video_can_clean_inside_running_event_loop(
    mock_llm_class: MagicMock,
    tmp_path: Path,
) -> None:
    video_dir = tmp_path / "videos"
    metadata_dir = tmp_path / "metadata"
    caption_dir = tmp_path / "captions"
    output_dir = tmp_path / "output"
    transcripts_dir = output_dir / "transcripts"
    summaries_dir = output_dir / "summaries"
    for directory in (
        video_dir,
        metadata_dir,
        caption_dir,
        transcripts_dir,
        summaries_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

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
    (summaries_dir / "V001.json").write_text(
        json.dumps({"video_id": "V001", "summary": "cached summary"}),
        encoding="utf-8",
    )

    llm = MagicMock()
    llm.clean.return_value = "cleaned"
    mock_llm_class.return_value = llm
    pipeline = ASRTranscriptPipeline(
        video_dir=str(video_dir),
        metadata_dir=str(metadata_dir),
        caption_dir=str(caption_dir),
        output_dir=str(output_dir),
        llm_provider="gemini",
    )

    async def invoke_from_async_caller() -> None:
        pipeline.process_video("V001")

    asyncio.run(invoke_from_async_caller())

    cleaned_path = transcripts_dir / "V001_cleaned.json"
    cleaned = json.loads(cleaned_path.read_text(encoding="utf-8"))
    assert cleaned["intervals"][0]["cleaned_text"] == "cleaned"
