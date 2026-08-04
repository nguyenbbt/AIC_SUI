import pytest
import numpy as np
from PIL import Image
import torch
import sys
import os
from unittest.mock import MagicMock, patch

# Add src to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from feature_extraction.visual_embedding.encoders import PECoreEncoder


def _mock_open_clip_model():
    model = MagicMock()
    model.to.return_value = model
    model.encode_image.return_value = torch.tensor(
        [[3.0, 4.0], [0.0, 5.0]],
        dtype=torch.float32,
    )
    preprocess = MagicMock(
        side_effect=lambda image: torch.zeros((3, 8, 8))
    )
    return model, preprocess

def test_encoder_output_shape_and_norm():
    """
    Test that the encoder returns the correct shape and L2 normalized vectors.
    """
    model, preprocess = _mock_open_clip_model()
    with patch(
        "feature_extraction.visual_embedding.encoders.pe_core_encoder."
        "open_clip.create_model_and_transforms",
        return_value=(model, None, preprocess),
    ):
        encoder = PECoreEncoder(
            device="cpu",
            precision="fp32",
            model_id="hf-hub:organization/custom-vision-model",
        )
    
    # Create dummy images
    img1 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    embeddings = encoder.encode_batch([img1, img2])
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    assert embeddings.shape == (2, 2)
    
    # Check L2 norm is approx 1.0
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5, atol=1e-5)

def test_encoder_empty_batch():
    model, preprocess = _mock_open_clip_model()
    with patch(
        "feature_extraction.visual_embedding.encoders.pe_core_encoder."
        "open_clip.create_model_and_transforms",
        return_value=(model, None, preprocess),
    ):
        encoder = PECoreEncoder(device="cpu", precision="fp32")
    embeddings = encoder.encode_batch([])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (0,)
    model.encode_image.assert_not_called()


def test_encoder_defaults_to_openai_clip_vit_b32():
    model, preprocess = _mock_open_clip_model()
    with patch(
        "feature_extraction.visual_embedding.encoders.pe_core_encoder."
        "open_clip.create_model_and_transforms",
        return_value=(model, None, preprocess),
    ) as create_model:
        encoder = PECoreEncoder(device="cpu", precision="fp32")

    assert encoder.model_id == "ViT-B-32::openai"
    create_model.assert_called_once_with("ViT-B-32", pretrained="openai")
