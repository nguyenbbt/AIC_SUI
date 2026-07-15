import pytest
from pathlib import Path
import json
import tempfile
import os

from src.text_embedding.data_readers import parse_asr_file, parse_summary_file, parse_ocr_file

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)

def test_parse_asr_file(temp_dir):
    data = [
        {"interval_id": "0", "start_time": 0.0, "end_time": 2.5, "cleaned_text": "Xin chào"},
        {"interval_id": "1", "start_time": 2.5, "end_time": 5.0, "cleaned_text": "  "}, # empty text, should be filtered
        {"interval_id": "2", "start_time": 5.0, "end_time": 6.0, "cleaned_text": "tất cả các bạn"}
    ]
    file_path = temp_dir / "V_001_cleaned.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
        
    records = parse_asr_file(file_path)
    assert len(records) == 2
    assert records[0]["video_id"] == "V_001"
    assert records[0]["text"] == "Xin chào"
    assert records[1]["text"] == "tất cả các bạn"

def test_parse_summary_file(temp_dir):
    data = {"summary": "Đây là video nói về trí tuệ nhân tạo."}
    file_path = temp_dir / "V_002.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
        
    records = parse_summary_file(file_path)
    assert len(records) == 1
    assert records[0]["video_id"] == "V_002"
    assert records[0]["text"] == "Đây là video nói về trí tuệ nhân tạo."

def test_parse_summary_empty(temp_dir):
    data = {"summary": ""}
    file_path = temp_dir / "V_003.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
        
    records = parse_summary_file(file_path)
    assert len(records) == 0

def test_parse_ocr_file(temp_dir):
    data = {
        "frames": [
            {"frame_id": "001", "shot_id": "s1", "ocr_text_concat": "CHÚ Ý"},
            {"frame_id": "002", "shot_id": "s1", "ocr_text_concat": ""}, # empty
        ]
    }
    file_path = temp_dir / "V_004.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
        
    records = parse_ocr_file(file_path)
    assert len(records) == 1
    assert records[0]["video_id"] == "V_004"
    assert records[0]["frame_id"] == "001"
    assert records[0]["text"] == "CHÚ Ý"
