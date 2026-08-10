"""Configuration shared by the visual embedding entry points."""

DEFAULT_VISUAL_MODEL_ID = "ViT-B-32::openai"
EXPECTED_VISUAL_EMBEDDING_DIMENSION = 512


def parse_open_clip_model_id(model_id: str) -> tuple[str, str | None]:
    """Return the OpenCLIP architecture and optional pretrained tag."""
    model_name, separator, pretrained = model_id.partition("::")
    return model_name, pretrained if separator else None
