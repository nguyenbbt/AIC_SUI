"""
Tests for data_loader module.
Verifies parsing of JSON and Parquet fixtures, dynamic dimension detection,
and graceful handling of missing/empty data.
"""

import pytest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.indexing.data_loader import (
    detect_embedding_dim,
    discover_video_ids,
    load_ocr_texts,
    load_asr_transcripts,
    load_video_summary,
    load_metadata_and_objects,
    load_visual_embeddings,
    load_text_asr_embeddings,
)


@pytest.fixture
def data_dir():
    """Create a temporary data directory with fixture files."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        video_id = "TEST_VIDEO_001"

        # --- Metadata JSON (Module 1) ---
        meta_dir = root / "metadata"
        meta_dir.mkdir()
        meta = {
            "video_id": video_id,
            "shots": [
                {
                    "shot_id": 0,
                    "keyframes": [
                        {
                            "file_path": f"keyframes/{video_id}/shot_00000_pos_050.webp",
                            "time_sec": 2.5,
                        }
                    ],
                },
                {
                    "shot_id": 1,
                    "keyframes": [
                        {
                            "file_path": f"keyframes/{video_id}/shot_00001_pos_050.webp",
                            "time_sec": 7.0,
                        }
                    ],
                },
            ],
        }
        with open(meta_dir / f"{video_id}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        # --- OCR JSON (Module 4) ---
        ocr_dir = root / "ocr"
        ocr_dir.mkdir()
        ocr = {
            "video_id": video_id,
            "frames": [
                {
                    "frame_id": "shot_00000_pos_050",
                    "shot_id": 0,
                    "ocr_text_concat": "Xin chào Việt Nam",
                },
                {
                    "frame_id": "shot_00001_pos_050",
                    "shot_id": 1,
                    "ocr_text_concat": "",  # empty — should be skipped
                },
            ],
        }
        with open(ocr_dir / f"{video_id}.json", "w", encoding="utf-8") as f:
            json.dump(ocr, f)

        # --- ASR Transcripts JSON (Module 3) ---
        transcript_dir = root / "transcripts"
        transcript_dir.mkdir()
        asr = [
            {
                "interval_id": "0",
                "start_time": 0.0,
                "end_time": 5.0,
                "cleaned_text": "Đây là tin tức hôm nay.",
            },
            {
                "interval_id": "1",
                "start_time": 5.0,
                "end_time": 10.0,
                "cleaned_text": "",  # empty — should be skipped
            },
        ]
        with open(
            transcript_dir / f"{video_id}_cleaned.json", "w", encoding="utf-8"
        ) as f:
            json.dump(asr, f)

        # --- Summary JSON (Module 3) ---
        summary_dir = root / "summaries"
        summary_dir.mkdir()
        summary = {"summary": "Video nói về tình hình thời tiết tại TP.HCM."}
        with open(summary_dir / f"{video_id}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f)

        # --- Object Detection JSON (Module 5) ---
        obj_dir = root / "object_detection"
        obj_dir.mkdir()
        obj_det = {
            "video_id": video_id,
            "frames": [
                {
                    "frame_id": "shot_00000_pos_050",
                    "shot_id": 0,
                    "objects": [
                        {
                            "label": "person",
                            "confidence": 0.95,
                            "bbox": [10, 20, 100, 200],
                            "model_source": "yolo_world",
                        },
                        {
                            "label": "car",
                            "confidence": 0.8,
                            "bbox": [50, 60, 300, 400],
                            "model_source": "co_detr",
                        },
                    ],
                }
            ],
        }
        with open(obj_dir / f"{video_id}.json", "w", encoding="utf-8") as f:
            json.dump(obj_det, f)

        # --- Visual Embedding Parquet (Module 2) ---
        emb_dir = root / "embeddings" / "visual"
        emb_dir.mkdir(parents=True)
        vis_df = pd.DataFrame(
            [
                {
                    "frame_id": "shot_00000_pos_050",
                    "video_id": video_id,
                    "shot_id": 0,
                    "embedding": np.random.rand(512).tolist(),
                },
                {
                    "frame_id": "shot_00001_pos_050",
                    "video_id": video_id,
                    "shot_id": 1,
                    "embedding": np.random.rand(512).tolist(),
                },
            ]
        )
        vis_df.to_parquet(emb_dir / f"{video_id}.parquet", index=False)

        # --- Text ASR Embedding Parquet (Module 6) ---
        text_asr_dir = root / "embeddings" / "text_asr"
        text_asr_dir.mkdir(parents=True)
        asr_df = pd.DataFrame(
            [
                {
                    "video_id": video_id,
                    "interval_id": "0",
                    "start_time_sec": 0.0,
                    "end_time_sec": 5.0,
                    "text": "Đây là tin tức hôm nay.",
                    "embedding": np.random.rand(768).tolist(),
                }
            ]
        )
        asr_df.to_parquet(text_asr_dir / f"{video_id}.parquet", index=False)

        yield root, video_id


class TestDiscoverVideoIds:
    def test_discovers_from_metadata(self, data_dir):
        root, video_id = data_dir
        ids = discover_video_ids(root)
        assert video_id in ids

    def test_empty_dir(self, tmp_path):
        ids = discover_video_ids(tmp_path)
        assert ids == []


class TestDetectEmbeddingDim:
    def test_detects_visual_dim(self, data_dir):
        root, _ = data_dir
        dim = detect_embedding_dim(root / "embeddings" / "visual")
        assert dim == 512

    def test_detects_text_dim(self, data_dir):
        root, _ = data_dir
        dim = detect_embedding_dim(root / "embeddings" / "text_asr")
        assert dim == 768

    def test_returns_none_for_missing_dir(self, tmp_path):
        dim = detect_embedding_dim(tmp_path / "nonexistent")
        assert dim is None


class TestLoadOcrTexts:
    def test_loads_and_filters_empty(self, data_dir):
        root, video_id = data_dir
        records = load_ocr_texts(root, video_id)
        assert len(records) == 1  # second frame had empty text
        assert records[0]["frame_id"] == "shot_00000_pos_050"
        assert records[0]["ocr_text_concat"] == "Xin chào Việt Nam"


class TestLoadAsrTranscripts:
    def test_loads_and_filters_empty(self, data_dir):
        root, video_id = data_dir
        records = load_asr_transcripts(root, video_id)
        assert len(records) == 1  # second interval had empty text
        assert records[0]["cleaned_text"] == "Đây là tin tức hôm nay."


class TestLoadVideoSummary:
    def test_loads_summary(self, data_dir):
        root, video_id = data_dir
        records = load_video_summary(root, video_id)
        assert len(records) == 1
        assert "thời tiết" in records[0]["summary"]


class TestLoadMetadataAndObjects:
    def test_loads_metadata_and_objects(self, data_dir):
        root, video_id = data_dir
        meta, objs = load_metadata_and_objects(root, video_id)

        assert len(meta) == 2  # 2 keyframes across 2 shots
        assert meta[0]["frame_id"] == "shot_00000_pos_050"
        assert meta[0]["shot_id"] == 0
        assert meta[0]["timestamp"] == 2.5

        assert len(objs) == 2  # 2 objects in shot 0
        assert objs[0]["label"] == "person"
        assert objs[0]["x_min"] == 10.0
        assert objs[1]["label"] == "car"


class TestLoadVisualEmbeddings:
    def test_loads_from_parquet(self, data_dir):
        root, video_id = data_dir
        records = load_visual_embeddings(root, video_id)
        assert len(records) == 2
        assert len(records[0]["embedding"]) == 512


class TestLoadTextAsrEmbeddings:
    def test_loads_from_parquet(self, data_dir):
        root, video_id = data_dir
        records = load_text_asr_embeddings(root, video_id)
        assert len(records) == 1
        assert len(records[0]["embedding"]) == 768
