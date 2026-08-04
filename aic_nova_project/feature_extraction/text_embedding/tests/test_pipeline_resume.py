import pytest
from pathlib import Path
import json
import tempfile
import os
import numpy as np

from src.text_embedding.encoders.base import BaseTextEncoder
from src.text_embedding.pipeline import TextEmbeddingPipeline

class MockEncoder(BaseTextEncoder):
    model_name = "test/mock-encoder"
    model_revision = "test"
    max_length = 256
    embedding_dim = 768

    def encode_batch(self, texts):
        embeddings = np.random.rand(len(texts), self.embedding_dim)
        return embeddings / np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

    def encode_long_text(self, text):
        embedding = np.random.rand(self.embedding_dim)
        return embedding / np.linalg.norm(embedding)

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)

def test_pipeline_idempotency(temp_dir):
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    input_dir.mkdir()
    
    # Create mock ASR file
    data = {
        "video_id": "V_001",
        "source": "asr",
        "llm_provider": "MockTranscriptLLM",
        "intervals": [
            {
                "interval_id": "0",
                "start_time_sec": 0.0,
                "end_time_sec": 2.5,
                "cleaned_text": "Xin chào",
            }
        ],
    }
    with open(input_dir / "V_001_cleaned.json", "w", encoding="utf-8") as f:
        json.dump(data, f)
        
    pipeline = TextEmbeddingPipeline(MockEncoder())
    
    # Run first time
    pipeline.process_asr(input_dir, output_dir)
    assert (output_dir / "V_001.parquet").exists()
    
    # Run second time, should skip (fast and not modify modification time if we checked it, but here we just ensure it doesn't crash)
    mtime = (output_dir / "V_001.parquet").stat().st_mtime
    pipeline.process_asr(input_dir, output_dir)
    assert (output_dir / "V_001.parquet").stat().st_mtime == mtime
    
    # Run with force
    pipeline.process_asr(input_dir, output_dir, force=True)
    assert (output_dir / "V_001.parquet").stat().st_mtime >= mtime
