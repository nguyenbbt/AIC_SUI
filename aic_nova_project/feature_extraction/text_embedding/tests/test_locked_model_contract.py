import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.text_embedding import cli
from src.text_embedding.artifact_contract import build_encoder_provenance
from src.text_embedding.config import (
    TEXT_EMBEDDING_DIMENSION,
    TEXT_MAX_LENGTH,
    TEXT_MODEL_NAME,
    TEXT_MODEL_REVISION,
)
from src.text_embedding.encoders.base import BaseTextEncoder
from src.text_embedding.encoders.sbert_encoder import (
    SentenceTransformerEncoder,
)


def test_cli_defaults_are_the_locked_online_query_contract(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["text-embedding", "--output-dir", "embeddings"],
    )

    with patch.object(cli, "SentenceTransformerEncoder") as encoder:
        cli.main()

    assert encoder.call_args.kwargs["model_name"] == TEXT_MODEL_NAME
    assert encoder.call_args.kwargs["model_revision"] == TEXT_MODEL_REVISION
    assert encoder.call_args.kwargs["max_length"] == TEXT_MAX_LENGTH


def test_cli_rejects_a_different_text_revision(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "text-embedding",
            "--output-dir",
            "embeddings",
            "--model-revision",
            "main",
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()


@patch("src.text_embedding.encoders.sbert_encoder.SentenceTransformer")
def test_default_encoder_loads_exact_revision_and_dimension(
    sentence_transformer: MagicMock,
):
    sentence_transformer.return_value.encode.return_value = np.ones(
        (1, TEXT_EMBEDDING_DIMENSION),
        dtype=np.float32,
    )

    encoder = SentenceTransformerEncoder(
        model_name=TEXT_MODEL_NAME,
        device="cpu",
    )

    sentence_transformer.assert_called_once_with(
        TEXT_MODEL_NAME,
        device="cpu",
        revision=TEXT_MODEL_REVISION,
    )
    assert encoder.model_revision == TEXT_MODEL_REVISION
    assert encoder.max_length == TEXT_MAX_LENGTH
    assert encoder.embedding_dim == TEXT_EMBEDDING_DIMENSION


@patch("src.text_embedding.encoders.sbert_encoder.SentenceTransformer")
def test_default_encoder_rejects_wrong_dimension(
    sentence_transformer: MagicMock,
):
    sentence_transformer.return_value.encode.return_value = np.ones(
        (1, TEXT_EMBEDDING_DIMENSION - 1),
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="expected 768"):
        SentenceTransformerEncoder(
            model_name=TEXT_MODEL_NAME,
            device="cpu",
        )


class MissingRevisionEncoder(BaseTextEncoder):
    model_name = TEXT_MODEL_NAME
    max_length = TEXT_MAX_LENGTH
    embedding_dim = TEXT_EMBEDDING_DIMENSION

    def encode_batch(self, texts):
        raise NotImplementedError

    def encode_long_text(self, text):
        raise NotImplementedError


def test_artifact_provenance_rejects_missing_model_revision():
    with pytest.raises(ValueError, match="model_revision"):
        build_encoder_provenance(MissingRevisionEncoder(), "asr")
