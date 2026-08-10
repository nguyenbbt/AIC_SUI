import numpy as np
import pytest


@pytest.fixture(autouse=True)
def isolate_visual_pipeline_from_remote_model(monkeypatch):
    """Keep pipeline tests deterministic and free of multi-GB model loads."""
    from feature_extraction.visual_embedding import pipeline

    class FixtureEncoder:
        def __init__(self, device, precision, model_id):
            del device
            self.model_id = model_id
            self.precision = "fp32" if precision == "fp32" else precision

        def encode_batch(self, images):
            embeddings = np.zeros((len(images), 512), dtype=np.float32)
            embeddings[:, 0] = 1.0
            return embeddings

    monkeypatch.setattr(pipeline, "OpenCLIPEncoder", FixtureEncoder)
