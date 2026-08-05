import pytest
import numpy as np
from src.text_embedding.encoders.sbert_encoder import SentenceTransformerEncoder

@pytest.fixture(scope="module")
def encoder():
    # Use a small model for fast testing if possible, but testing the actual model is fine since it's only ~400MB
    # For CI/CD, a smaller mock model could be used. Here we use the real one.
    return SentenceTransformerEncoder(model_name="dangvantuan/vietnamese-embedding", device="cpu", max_length=128)

def test_encode_batch_shape_and_norm(encoder):
    texts = ["Xin chào Việt Nam", "Đây là một đoạn test ngắn.", "Một câu khác."]
    embeddings = encoder.encode_batch(texts)
    
    assert embeddings.shape == (3, 768), f"Expected shape (3, 768), got {embeddings.shape}"
    
    # Check L2 norm is approx 1.0
    norms = np.linalg.norm(embeddings, axis=1)
    for i, norm in enumerate(norms):
        assert np.isclose(norm, 1.0, atol=1e-5), f"Norm of embedding {i} is {norm}, expected 1.0"
        
def test_encode_batch_empty(encoder):
    embeddings = encoder.encode_batch([])
    assert embeddings.shape == (0, 768)

def test_encode_long_text_shape_and_norm(encoder):
    # A long text that will be chunked
    text = " ".join(["từ"] * 300) # 300 words, > max_length (128)
    
    embedding = encoder.encode_long_text(text)
    
    assert embedding.shape == (768,), f"Expected shape (768,), got {embedding.shape}"
    
    # Check L2 norm is approx 1.0
    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-5), f"Norm of long text embedding is {norm}, expected 1.0"

def test_encode_long_text_empty(encoder):
    embedding = encoder.encode_long_text("")
    assert embedding.shape == (768,)
    assert np.all(embedding == 0.0), "Expected all zeros for empty string"
