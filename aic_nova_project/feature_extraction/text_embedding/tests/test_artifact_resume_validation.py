import json
from pathlib import Path

import numpy as np

from src.text_embedding.encoders.base import BaseTextEncoder
from src.text_embedding.pipeline import TextEmbeddingPipeline


class CountingEncoder(BaseTextEncoder):
    model_name = "test/model"
    model_revision = "revision-1"
    max_length = 128
    embedding_dim = 3

    def __init__(self) -> None:
        self.calls = 0

    def encode_batch(self, texts):
        self.calls += 1
        return np.tile(
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            (len(texts), 1),
        )

    def encode_long_text(self, text):
        self.calls += 1
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def _write_asr(path: Path, text: str) -> None:
    path.write_text(
        json.dumps(
            {
                "video_id": "V001",
                "intervals": [
                    {
                        "interval_id": "0",
                        "start_time_sec": 0.0,
                        "end_time_sec": 1.0,
                        "cleaned_text": text,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_resume_reuses_only_current_complete_text_artifact(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    input_path = input_dir / "V001_cleaned.json"
    _write_asr(input_path, "first text")

    encoder = CountingEncoder()
    pipeline = TextEmbeddingPipeline(encoder)
    pipeline.process_asr(input_dir, output_dir)
    assert encoder.calls == 1

    pipeline.process_asr(input_dir, output_dir)
    assert encoder.calls == 1

    _write_asr(input_path, "changed text")
    pipeline.process_asr(input_dir, output_dir)
    assert encoder.calls == 2

    encoder.max_length = 256
    pipeline.process_asr(input_dir, output_dir)
    assert encoder.calls == 3


def test_resume_regenerates_corrupt_parquet(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    input_path = input_dir / "V001_cleaned.json"
    _write_asr(input_path, "text")

    encoder = CountingEncoder()
    pipeline = TextEmbeddingPipeline(encoder)
    pipeline.process_asr(input_dir, output_dir)
    output_path = output_dir / "V001.parquet"
    output_path.write_bytes(b"corrupt")

    pipeline.process_asr(input_dir, output_dir)

    assert encoder.calls == 2
