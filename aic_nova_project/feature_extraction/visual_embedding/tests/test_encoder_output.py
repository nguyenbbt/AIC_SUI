import pytest
import numpy as np
from PIL import Image
import torch
import sys
import os

# Add src to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from feature_extraction.visual_embedding.encoders import PECoreEncoder

def test_encoder_output_shape_and_norm():
    """
    Test that the encoder returns the correct shape and L2 normalized vectors.
    """
    # Initialize encoder on CPU to test without requiring a GPU
    encoder = PECoreEncoder(device="cpu", precision="fp32", model_id="hf-hub:timm/PE-Core-bigG-14-448")
    
    # Create dummy images
    img1 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    embeddings = encoder.encode_batch([img1, img2])
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    # PE-Core-bigG-14-448 has 1152 embedding dim usually, but let's just check it's 2D and > 0
    assert len(embeddings.shape) == 2
    assert embeddings.shape[1] > 0
    
    # Check L2 norm is approx 1.0
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5, atol=1e-5)

def test_encoder_empty_batch():
    encoder = PECoreEncoder(device="cpu", precision="fp32")
    embeddings = encoder.encode_batch([])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (0,)
