import sys
from unittest.mock import patch

from feature_extraction.visual_embedding.cli import main


def test_cli_defaults_to_openai_clip_vit_b32(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visual-embedding",
            "--metadata-dir",
            "metadata",
            "--keyframe-dir",
            "keyframes",
            "--output-dir",
            "embeddings",
        ],
    )

    with patch(
        "feature_extraction.visual_embedding.cli.run_pipeline"
    ) as run_pipeline:
        main()

    assert run_pipeline.call_args.kwargs["model_id"] == "ViT-B-32::openai"
