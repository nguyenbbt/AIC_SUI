from unittest.mock import patch

from scripts.download_visual_model import download_default_visual_model


def test_download_script_caches_openai_clip_vit_b32():
    with patch(
        "scripts.download_visual_model.open_clip.create_model_and_transforms"
    ) as create_model:
        download_default_visual_model()

    create_model.assert_called_once_with("ViT-B-32", pretrained="openai")
