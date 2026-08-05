import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.text_embedding.encoders.base import BaseTextEncoder
from src.text_embedding.encoders.sbert_encoder import (
    SentenceTransformerEncoder,
)
from src.text_embedding.pipeline import TextEmbeddingPipeline


class ProvenanceEncoder(BaseTextEncoder):
    model_name = "org/model"
    model_revision = "commit-abc"
    max_length = 384
    embedding_dim = 2

    def encode_batch(self, texts):
        return np.tile(
            np.array([[1.0, 0.0]], dtype=np.float32),
            (len(texts), 1),
        )

    def encode_long_text(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)


def test_parquet_rows_include_full_encoder_provenance(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "summary"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "V001.json").write_text(
        json.dumps({"video_id": "V001", "summary": "summary text"}),
        encoding="utf-8",
    )

    TextEmbeddingPipeline(ProvenanceEncoder()).process_summary(
        input_dir,
        output_dir,
    )

    dataframe = pd.read_parquet(output_dir / "V001.parquet")
    row = dataframe.iloc[0]
    assert row["model_name"] == "org/model"
    assert row["model_revision"] == "commit-abc"
    assert row["pooling_strategy"] == "chunk_mean_l2"
    assert row["max_length"] == 384
    assert row["embedding_dimension"] == 2
    assert bool(row["normalized"]) is True


@patch("src.text_embedding.encoders.sbert_encoder.SentenceTransformer")
def test_sentence_transformer_revision_is_pinned(
    mock_sentence_transformer: MagicMock,
) -> None:
    mock_sentence_transformer.return_value.encode.return_value = np.ones(
        (1, 3),
        dtype=np.float32,
    )

    encoder = SentenceTransformerEncoder(
        model_name="org/model",
        model_revision="commit-abc",
        device="cpu",
    )

    assert encoder.model_revision == "commit-abc"
    mock_sentence_transformer.assert_called_once_with(
        "org/model",
        device="cpu",
        revision="commit-abc",
    )
