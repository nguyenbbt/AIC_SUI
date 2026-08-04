import pytest
import os
from unittest.mock import patch, mock_open, MagicMock
from feature_extraction.asr_transcript.audio_extractor import AudioExtractor
from feature_extraction.asr_transcript.caption_parser import CaptionParser
from feature_extraction.asr_transcript.segment_grouper import SegmentGrouper

# --- AudioExtractor Tests ---

@patch("subprocess.run")
@patch("os.makedirs")
@patch("os.path.exists")
def test_audio_extractor_success(mock_exists, mock_makedirs, mock_subprocess_run):
    # Setup
    mock_exists.return_value = False # Output file doesn't exist
    mock_run_result = MagicMock()
    mock_subprocess_run.return_value = mock_run_result
    
    # Execute
    result = AudioExtractor.extract_audio("fake_video.mp4", "fake_audio.wav")
    
    # Assert
    assert result is True
    mock_subprocess_run.assert_called_once()
    args, kwargs = mock_subprocess_run.call_args
    assert "ffmpeg" in args[0]
    assert "-vn" in args[0]
    assert "pcm_s16le" in args[0]
    assert "16000" in args[0]
    assert "fake_audio.wav" in args[0]

# --- CaptionParser Tests ---

def test_parse_time_to_seconds():
    assert CaptionParser.parse_time_to_seconds("00:01:23,456") == 83.456
    assert CaptionParser.parse_time_to_seconds("00:01:23.456") == 83.456
    assert CaptionParser.parse_time_to_seconds("01:23.456") == 83.456

def test_parse_srt():
    srt_content = (
        "1\n"
        "00:00:01,000 --> 00:00:03,500\n"
        "Xin chào các bạn.\n"
        "\n"
        "2\n"
        "00:00:04,000 --> 00:00:05,000\n"
        "Hôm nay trời đẹp.\n"
        "Thật sự rất đẹp.\n"
    )
    
    with patch("builtins.open", mock_open(read_data=srt_content)):
        segments = CaptionParser.parse_srt("fake.srt")
        
        assert len(segments) == 2
        assert segments[0]["timestamp"] == (1.0, 3.5)
        assert segments[0]["text"] == "Xin chào các bạn."
        assert segments[1]["timestamp"] == (4.0, 5.0)
        assert segments[1]["text"] == "Hôm nay trời đẹp. Thật sự rất đẹp."

# --- SegmentGrouper Tests ---

def test_segment_grouper():
    segments = [
        {"timestamp": (0.0, 2.0), "text": "A"},
        {"timestamp": (2.0, 4.0), "text": "B"},
        {"timestamp": (4.0, 6.0), "text": "C"},
        {"timestamp": (6.0, 8.0), "text": "D"},
    ]
    
    intervals = SegmentGrouper.group_segments(segments, group_size=2)
    
    assert len(intervals) == 2
    assert intervals[0]["interval_id"] == "0"
    assert intervals[0]["start_time_sec"] == 0.0
    assert intervals[0]["end_time_sec"] == 4.0
    assert intervals[0]["raw_text"] == "A B"
    assert intervals[0]["segment_ids"] == [0, 1]
    
    assert intervals[1]["interval_id"] == "1"
    assert intervals[1]["start_time_sec"] == 4.0
    assert intervals[1]["end_time_sec"] == 8.0
    assert intervals[1]["raw_text"] == "C D"
    assert intervals[1]["segment_ids"] == [2, 3]

def test_segment_grouper_none_end():
    segments = [
        {"timestamp": (0.0, 2.0), "text": "A"},
        {"timestamp": (2.0, None), "text": "B"},
    ]
    intervals = SegmentGrouper.group_segments(segments, group_size=2)
    assert intervals[0]["end_time_sec"] == 4.0 # fallback is start_time + 2.0
