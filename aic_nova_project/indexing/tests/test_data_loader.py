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
    normalize_frame_id,
    load_ocr_texts,
    load_asr_transcripts,
    load_video_summary,
    load_video_metadata,
    load_metadata_and_objects,
    load_visual_embeddings,
    load_text_asr_embeddings,
    load_text_ocr_embeddings,
)
from src.indexing.clients.es_client import ASR_MAPPING


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
            "contract_version": "self-indexed-v2",
            "video_id": video_id,
            "source_path": f"raw_videos/{video_id}.mp4",
            "source_video_rel_path": f"raw_videos/{video_id}.mp4",
            "fps": 25.0,
            "duration_sec": 10.0,
            "frame_count": 250,
            "width": 640,
            "height": 480,
            "num_shots": 2,
            "shots": [
                {
                    "shot_id": 0,
                    "keyframes": [
                        {
                            "file_path": f"keyframes/{video_id}/shot_00000_pos_050.webp",
                            "image_rel_path": f"keyframes/{video_id}/shot_00000_pos_050.webp",
                            "position": 0.5,
                            "position_code": 50,
                            "frame_index": 62,
                            "source_frame_idx": 62,
                            "time_sec": 2.5,
                        }
                    ],
                },
                {
                    "shot_id": 1,
                    "keyframes": [
                        {
                            "file_path": f"keyframes/{video_id}/shot_00001_pos_050.webp",
                            "image_rel_path": f"keyframes/{video_id}/shot_00001_pos_050.webp",
                            "position": 0.5,
                            "position_code": 50,
                            "frame_index": 175,
                            "source_frame_idx": 175,
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
                    "frame_id": f"{video_id}_00000_050",
                    "shot_id": 0,
                    "ocr_text_concat": "Xin chào Việt Nam",
                },
                {
                    "frame_id": f"{video_id}_00001_050",
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
        asr = {
            "video_id": video_id,
            "source": "asr",
            "llm_provider": "MockTranscriptLLM",
            "intervals": [
                {
                    "interval_id": "0",
                    "start_time_sec": 0.0,
                    "end_time_sec": 5.0,
                    "cleaned_text": "Đây là tin tức hôm nay.",
                },
                {
                    "interval_id": "1",
                    "start_time_sec": 5.0,
                    "end_time_sec": 10.0,
                    "cleaned_text": "",  # empty — should be skipped
                },
            ],
        }
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
                    "frame_id": f"{video_id}_00000_050",
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
                    "embedding": (
                        lambda vector: (
                            vector / np.linalg.norm(vector)
                        ).tolist()
                    )(np.random.rand(512)),
                },
                {
                    "frame_id": "shot_00001_pos_050",
                    "video_id": video_id,
                    "shot_id": 1,
                    "embedding": (
                        lambda vector: (
                            vector / np.linalg.norm(vector)
                        ).tolist()
                    )(np.random.rand(512)),
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
                    "embedding": (
                        lambda vector: (
                            vector / np.linalg.norm(vector)
                        ).tolist()
                    )(np.random.rand(768)),
                }
            ]
        )
        asr_df.to_parquet(text_asr_dir / f"{video_id}.parquet", index=False)

        # --- Text OCR Embedding Parquet (Module 6) ---
        text_ocr_dir = root / "embeddings" / "text_ocr"
        text_ocr_dir.mkdir(parents=True)
        ocr_emb_df = pd.DataFrame(
            [
                {
                    "frame_id": f"{video_id}_00000_050",
                    "video_id": video_id,
                    "embedding": (
                        lambda vector: (
                            vector / np.linalg.norm(vector)
                        ).tolist()
                    )(np.random.rand(768)),
                }
            ]
        )
        ocr_emb_df.to_parquet(text_ocr_dir / f"{video_id}.parquet", index=False)

        yield root, video_id


class TestDiscoverVideoIds:
    def test_discovers_from_metadata(self, data_dir):
        root, video_id = data_dir
        ids = discover_video_ids(root)
        assert video_id in ids

    def test_empty_dir(self, tmp_path):
        ids = discover_video_ids(tmp_path)
        assert ids == []

    def test_discovers_ids_from_every_artifact_family(self, tmp_path):
        parquet_dirs = {
            "embeddings/visual": "V_VISUAL",
            "embeddings/text_asr": "V_ASR_EMB",
            "embeddings/text_summary": "V_SUM_EMB",
            "embeddings/text_ocr": "V_OCR_EMB",
        }
        for relative_dir, video_id in parquet_dirs.items():
            directory = tmp_path / relative_dir
            directory.mkdir(parents=True)
            (directory / f"{video_id}.parquet").write_bytes(b"fixture")

        json_artifacts = {
            "transcripts/renamed_cleaned.json": "V_TRANSCRIPT",
            "summaries/renamed.json": "V_SUMMARY",
            "object_detection/renamed.json": "V_OBJECT",
            "ocr/renamed.json": "V_OCR",
        }
        for relative_path, video_id in json_artifacts.items():
            path = tmp_path / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"video_id": video_id}),
                encoding="utf-8",
            )

        assert set(discover_video_ids(tmp_path)) == {
            "V_VISUAL",
            "V_ASR_EMB",
            "V_SUM_EMB",
            "V_OCR_EMB",
            "V_TRANSCRIPT",
            "V_SUMMARY",
            "V_OBJECT",
            "V_OCR",
        }


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

    def test_rejects_dimension_mismatch_in_later_file(self, tmp_path):
        pd.DataFrame(
            [{"embedding": [1.0, 0.0]}]
        ).to_parquet(tmp_path / "V001.parquet", index=False)
        pd.DataFrame(
            [{"embedding": [1.0, 0.0, 0.0]}]
        ).to_parquet(tmp_path / "V002.parquet", index=False)

        with pytest.raises(ValueError, match="dimension"):
            detect_embedding_dim(tmp_path)

    @pytest.mark.parametrize(
        "embedding",
        [
            [2.0, 0.0],
            [float("nan"), 0.0],
        ],
    )
    def test_rejects_non_normalized_or_non_finite_vector(
        self,
        tmp_path,
        embedding,
    ):
        pd.DataFrame(
            [{"embedding": embedding}]
        ).to_parquet(tmp_path / "V001.parquet", index=False)

        with pytest.raises(ValueError, match="embedding"):
            detect_embedding_dim(tmp_path)


class TestLoadOcrTexts:
    def test_loads_and_filters_empty(self, data_dir):
        root, video_id = data_dir
        records = load_ocr_texts(root, video_id)
        assert len(records) == 1  # second frame had empty text
        # frame_id should be in Global ID format (normalized)
        assert records[0]["frame_id"] == f"{video_id}_00000_050"
        assert records[0]["ocr_text_concat"] == "Xin chào Việt Nam"


class TestLoadAsrTranscripts:
    def test_loads_and_filters_empty(self, data_dir):
        root, video_id = data_dir
        records = load_asr_transcripts(root, video_id)
        assert len(records) == 1  # second interval had empty text
        assert records[0]["video_id"] == video_id
        assert records[0]["interval_id"] == "0"
        assert records[0]["start_time_sec"] == 0.0
        assert records[0]["end_time_sec"] == 5.0
        assert records[0]["cleaned_text"] == "Đây là tin tức hôm nay."

    def test_normalizes_legacy_integer_interval_id(self, data_dir):
        root, video_id = data_dir
        transcript_path = root / "transcripts" / f"{video_id}_cleaned.json"
        with open(transcript_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["intervals"][0]["interval_id"] = 0
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        records = load_asr_transcripts(root, video_id)

        assert records[0]["interval_id"] == "0"

    def test_rejects_noncanonical_envelope(self, data_dir):
        root, video_id = data_dir
        transcript_path = root / "transcripts" / f"{video_id}_cleaned.json"
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump([{"interval_id": "0", "cleaned_text": "text"}], f)

        assert load_asr_transcripts(root, video_id) == []

    def test_rejects_video_id_mismatch(self, data_dir):
        root, video_id = data_dir
        transcript_path = root / "transcripts" / f"{video_id}_cleaned.json"
        with open(transcript_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["video_id"] = "ANOTHER_VIDEO"
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        assert load_asr_transcripts(root, video_id) == []

    def test_rejects_legacy_timestamp_names(self, data_dir):
        root, video_id = data_dir
        transcript_path = root / "transcripts" / f"{video_id}_cleaned.json"
        with open(transcript_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        interval = payload["intervals"][0]
        interval["start_time"] = interval.pop("start_time_sec")
        interval["end_time"] = interval.pop("end_time_sec")
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        assert load_asr_transcripts(root, video_id) == []


def test_asr_mapping_uses_canonical_timestamp_names():
    properties = ASR_MAPPING["properties"]

    assert properties["_doc_id"] == {"type": "keyword"}
    assert properties["start_time_sec"] == {"type": "float"}
    assert properties["end_time_sec"] == {"type": "float"}
    assert "start_time" not in properties
    assert "end_time" not in properties


class TestLoadVideoSummary:
    def test_loads_summary(self, data_dir):
        root, video_id = data_dir
        records = load_video_summary(root, video_id)
        assert len(records) == 1
        assert "thời tiết" in records[0]["summary"]


class TestLoadMetadataAndObjects:
    def test_loads_video_contract(self, data_dir):
        root, video_id = data_dir

        video = load_video_metadata(root, video_id)

        assert video == {
            "video_id": video_id,
            "source_video_rel_path": f"raw_videos/{video_id}.mp4",
            "fps": 25.0,
            "duration_sec": 10.0,
            "frame_count": 250,
            "width": 640,
            "height": 480,
        }

    def test_loads_metadata_and_objects(self, data_dir):
        root, video_id = data_dir
        meta, objs = load_metadata_and_objects(root, video_id)

        assert len(meta) == 2  # 2 keyframes across 2 shots
        # frame_id should be normalized to Global ID format
        assert meta[0]["frame_id"] == f"{video_id}_00000_050"
        assert meta[0]["shot_id"] == 0
        assert meta[0]["source_frame_idx"] == 62
        assert meta[0]["timestamp"] == 2.5
        assert meta[0]["image_rel_path"] == (
            f"keyframes/{video_id}/shot_00000_pos_050.webp"
        )

        assert len(objs) == 2  # 2 objects in shot 0
        assert objs[0]["label"] == "person"
        # Object records should also use normalized frame_id
        assert objs[0]["frame_id"] == f"{video_id}_00000_050"
        assert objs[0]["x_min"] == 10.0
        assert objs[1]["label"] == "car"

    def test_rejects_object_bbox_outside_video_bounds(self, data_dir):
        root, video_id = data_dir
        object_path = root / "object_detection" / f"{video_id}.json"
        payload = json.loads(object_path.read_text(encoding="utf-8"))
        payload["frames"][0]["objects"][0]["bbox"] = [10, 20, 641, 200]
        object_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="bbox"):
            load_metadata_and_objects(root, video_id)

    def test_rejects_unsafe_image_relative_path(self, data_dir):
        root, video_id = data_dir
        metadata_path = root / "metadata" / f"{video_id}.json"
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["shots"][0]["keyframes"][0]["image_rel_path"] = "../escape.webp"
        metadata_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="image_rel_path"):
            load_metadata_and_objects(root, video_id)


class TestLoadVisualEmbeddings:
    def test_loads_from_parquet(self, data_dir):
        root, video_id = data_dir
        records = load_visual_embeddings(root, video_id)
        assert len(records) == 2
        assert len(records[0]["embedding"]) == 512
        assert records[0]["frame_id"] == f"{video_id}_00000_050"
        assert records[1]["frame_id"] == f"{video_id}_00001_050"


class TestLoadTextAsrEmbeddings:
    def test_loads_from_parquet(self, data_dir):
        root, video_id = data_dir
        records = load_text_asr_embeddings(root, video_id)
        assert len(records) == 1
        assert len(records[0]["embedding"]) == 768


class TestLoadTextOcrEmbeddings:
    def test_loads_from_parquet(self, data_dir):
        root, video_id = data_dir
        records = load_text_ocr_embeddings(root, video_id)
        assert len(records) == 1
        assert len(records[0]["embedding"]) == 768
        # frame_id should be normalized
        assert records[0]["frame_id"] == f"{video_id}_00000_050"

    def test_returns_empty_for_missing(self, tmp_path):
        records = load_text_ocr_embeddings(tmp_path, "NONEXISTENT")
        assert records == []


class TestNormalizeFrameId:
    def test_shot_pos_format(self):
        result = normalize_frame_id("shot_00000_pos_015", "V001")
        assert result == "V001_00000_015"

    def test_already_normalized(self):
        result = normalize_frame_id("V001_00000_015", "V001")
        assert result == "V001_00000_015"

    def test_different_video_id(self):
        result = normalize_frame_id("shot_00012_pos_050", "TEST_VIDEO_001")
        assert result == "TEST_VIDEO_001_00012_050"

    @pytest.mark.parametrize(
        "raw_frame_id",
        [
            "",
            "garbage",
            "V001_",
            "V001_1_2",
            "V001_00000_015_extra",
            "V002_00000_015",
            "shot_1_pos_15",
            "shot_00000_pos_015.webp",
        ],
    )
    def test_rejects_invalid_or_wrong_video_ids(self, raw_frame_id):
        with pytest.raises(ValueError, match="frame_id"):
            normalize_frame_id(raw_frame_id, "V001")

    def test_consistency_across_loaders(self, data_dir):
        """Verify that frame_id from metadata, OCR, and objects are identical."""
        root, video_id = data_dir
        meta, objs = load_metadata_and_objects(root, video_id)
        visual_records = load_visual_embeddings(root, video_id)
        ocr_records = load_ocr_texts(root, video_id)
        ocr_emb_records = load_text_ocr_embeddings(root, video_id)

        # All sources should produce the same normalized frame_id for frame 0
        expected = f"{video_id}_00000_050"
        assert meta[0]["frame_id"] == expected
        assert objs[0]["frame_id"] == expected
        assert visual_records[0]["frame_id"] == expected
        assert ocr_records[0]["frame_id"] == expected
        assert ocr_emb_records[0]["frame_id"] == expected
