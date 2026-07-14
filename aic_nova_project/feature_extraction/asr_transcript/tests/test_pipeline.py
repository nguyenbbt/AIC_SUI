import pytest
import os
import json
from unittest.mock import patch, MagicMock
from feature_extraction.asr_transcript.pipeline import ASRTranscriptPipeline

@pytest.fixture
def mock_dirs(tmp_path):
    video_dir = tmp_path / "raw_videos"
    metadata_dir = tmp_path / "metadata"
    caption_dir = tmp_path / "captions"
    output_dir = tmp_path / "output"
    
    video_dir.mkdir()
    metadata_dir.mkdir()
    caption_dir.mkdir()
    
    # Create fake metadata for a video
    with open(metadata_dir / "V001.json", "w") as f:
        json.dump({"video_id": "V001"}, f)
        
    # Create fake video file
    with open(video_dir / "V001.mp4", "w") as f:
        f.write("fake video data")
        
    return {
        "video_dir": str(video_dir),
        "metadata_dir": str(metadata_dir),
        "caption_dir": str(caption_dir),
        "output_dir": str(output_dir)
    }

@patch("feature_extraction.asr_transcript.pipeline.ASREngine")
@patch("feature_extraction.asr_transcript.llm.gemini_llm.GeminiTranscriptLLM")
def test_pipeline_resume(mock_llm_class, mock_asr_class, mock_dirs):
    # Mock LLM and ASR to do nothing or track calls
    mock_llm = MagicMock()
    mock_llm.clean.return_value = "clean text"
    mock_llm.summarize.return_value = "summary"
    mock_llm_class.return_value = mock_llm
    
    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = [{"timestamp": (0.0, 1.0), "text": "hello"}]
    mock_asr_class.return_value = mock_asr
    
    # Create output dir with pre-existing raw transcript
    output_dir = mock_dirs["output_dir"]
    os.makedirs(os.path.join(output_dir, "transcripts"), exist_ok=True)
    raw_path = os.path.join(output_dir, "transcripts", "V001_raw.json")
    with open(raw_path, "w") as f:
        json.dump({"video_id": "V001", "source": "asr", "segments": [{"timestamp": [0, 1], "text": "pre-existing"}]}, f)
        
    # Also create a pre-existing summary
    os.makedirs(os.path.join(output_dir, "summaries"), exist_ok=True)
    summary_path = os.path.join(output_dir, "summaries", "V001.json")
    with open(summary_path, "w") as f:
        json.dump({"video_id": "V001", "summary": "pre-existing summary"}, f)
        
    # Run pipeline
    pipeline = ASRTranscriptPipeline(
        video_dir=mock_dirs["video_dir"],
        metadata_dir=mock_dirs["metadata_dir"],
        caption_dir=mock_dirs["caption_dir"],
        output_dir=mock_dirs["output_dir"],
        llm_provider="gemini",
        force=False
    )
    
    with patch("feature_extraction.asr_transcript.pipeline.AudioExtractor.extract_audio") as mock_extract:
        pipeline.process_video("V001")
        
        # AudioExtractor should NOT be called because raw transcript exists
        mock_extract.assert_not_called()
        
        # ASR engine transcribe should NOT be called
        mock_asr.transcribe.assert_not_called()
        
        # Summarizer should NOT be called because summary exists
        mock_llm.summarize.assert_not_called()
        
        # However, cleaned transcript does NOT exist, so clean SHOULD be called
        # The mock LLM needs to return a string for clean
        mock_llm.clean.assert_called_once()
