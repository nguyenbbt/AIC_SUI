"""Cache the default visual embedding model for offline execution."""

import open_clip

from feature_extraction.visual_embedding.config import (
    DEFAULT_VISUAL_MODEL_ID,
    parse_open_clip_model_id,
)


def download_default_visual_model() -> None:
    """Download the configured OpenCLIP weights into the local model cache."""
    model_name, pretrained = parse_open_clip_model_id(DEFAULT_VISUAL_MODEL_ID)
    if pretrained is None:
        open_clip.create_model_and_transforms(model_name)
        return
    open_clip.create_model_and_transforms(model_name, pretrained=pretrained)


if __name__ == "__main__":
    print(f"Downloading {DEFAULT_VISUAL_MODEL_ID} weights...")
    download_default_visual_model()
    print("Download complete.")
