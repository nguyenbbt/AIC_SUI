import json
from pathlib import Path

import numpy as np

from src.text_embedding.encoders.base import BaseTextEncoder
from src.text_embedding.pipeline import TextEmbeddingPipeline


class UnitEncoder(BaseTextEncoder):
    model_name = "test/model"
    model_revision = "revision"
    max_length = 128
    embedding_dim = 2

    def encode_batch(self, texts):
        return np.tile(
            np.array([[1.0, 0.0]], dtype=np.float32),
            (len(texts), 1),
        )

    def encode_long_text(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)


def test_asr_discovers_only_cleaned_files_and_uses_payload_video_id(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "asr"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    payload = {
        "video_id": "V001",
        "intervals": [
            {
                "interval_id": "0",
                "start_time_sec": 0.0,
                "end_time_sec": 1.0,
                "cleaned_text": "cleaned",
            }
        ],
    }
    (input_dir / "renamed_cleaned.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    (input_dir / "V999_raw.json").write_text(
        json.dumps({**payload, "video_id": "V999"}),
        encoding="utf-8",
    )

    TextEmbeddingPipeline(UnitEncoder()).process_asr(
        input_dir,
        output_dir,
    )

    assert (output_dir / "V001.parquet").exists()
    assert not (output_dir / "renamed.parquet").exists()
    assert not (output_dir / "V999.parquet").exists()


def test_summary_and_ocr_outputs_use_payload_video_id(
    tmp_path: Path,
) -> None:
    summary_dir = tmp_path / "summary"
    ocr_dir = tmp_path / "ocr"
    output_dir = tmp_path / "output"
    summary_dir.mkdir()
    ocr_dir.mkdir()
    (summary_dir / "renamed.json").write_text(
        json.dumps({"video_id": "V001", "summary": "summary"}),
        encoding="utf-8",
    )
    (ocr_dir / "renamed.json").write_text(
        json.dumps(
            {
                "video_id": "V002",
                "frames": [
                    {
                        "frame_id": "V002_00000_015",
                        "shot_id": 0,
                        "ocr_text_concat": "text",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    pipeline = TextEmbeddingPipeline(UnitEncoder())
    pipeline.process_summary(summary_dir, output_dir / "summary")
    pipeline.process_ocr(ocr_dir, output_dir / "ocr")

    assert (output_dir / "summary" / "V001.parquet").exists()
    assert (output_dir / "ocr" / "V002.parquet").exists()
    assert not (output_dir / "summary" / "renamed.parquet").exists()
    assert not (output_dir / "ocr" / "renamed.parquet").exists()
