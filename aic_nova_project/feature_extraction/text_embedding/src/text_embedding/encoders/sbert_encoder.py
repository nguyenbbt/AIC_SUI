import numpy as np
import logging
from typing import List
from sentence_transformers import SentenceTransformer

from ..config import (
    TEXT_EMBEDDING_DIMENSION,
    TEXT_MAX_LENGTH,
    TEXT_MODEL_NAME,
    TEXT_MODEL_REVISION,
)
from .base import BaseTextEncoder

logger = logging.getLogger(__name__)

class SentenceTransformerEncoder(BaseTextEncoder):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_length: int = TEXT_MAX_LENGTH,
        batch_size: int = 32,
        model_revision: str = TEXT_MODEL_REVISION,
    ):
        if not isinstance(model_revision, str) or not model_revision.strip():
            raise ValueError("model_revision must be an explicit revision")
        if model_name == TEXT_MODEL_NAME:
            if model_revision != TEXT_MODEL_REVISION:
                raise ValueError(
                    "Default text model must use the locked revision "
                    f"{TEXT_MODEL_REVISION}"
                )
            if max_length != TEXT_MAX_LENGTH:
                raise ValueError(
                    "Default text model must use max_length "
                    f"{TEXT_MAX_LENGTH}"
                )
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        
        logger.info(f"Loading SentenceTransformer model '{model_name}' on {device}...")
        self.model = SentenceTransformer(
            model_name,
            device=device,
            revision=model_revision,
        )
        self.model_revision = model_revision
        self.model.max_seq_length = max_length
        
        # Test shape
        dummy_embedding = self.model.encode(["test"])
        self.embedding_dim = dummy_embedding.shape[1]
        if (
            model_name == TEXT_MODEL_NAME
            and self.embedding_dim != TEXT_EMBEDDING_DIMENSION
        ):
            raise ValueError(
                "Default text model returned dimension "
                f"{self.embedding_dim}; expected {TEXT_EMBEDDING_DIMENSION}"
            )
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encodes a batch of texts using standard truncation and normalization.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
            
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True # Crucial for Cosine Similarity
        )
        return embeddings

    def encode_long_text(self, text: str) -> np.ndarray:
        """
        Encodes a long text using Chunking -> Mean-Pooling -> L2-Normalization.
        """
        if not text or not text.strip():
            return np.zeros(self.embedding_dim, dtype=np.float32)
            
        # Basic chunking by words to avoid breaking words.
        # sentence-transformers does subword tokenization, but simple word split is robust enough for chunking.
        words = text.split()
        
        # Approximate words per chunk to fit max_length tokens.
        # Assuming ~1.5 tokens per word for Vietnamese.
        words_per_chunk = max(10, int(self.max_length / 1.5))
        
        chunks = []
        for i in range(0, len(words), words_per_chunk):
            chunk = " ".join(words[i:i + words_per_chunk])
            if chunk.strip():
                chunks.append(chunk)
                
        if not chunks:
            return np.zeros(self.embedding_dim, dtype=np.float32)
            
        # Encode chunks
        chunk_embeddings = self.model.encode(
            chunks,
            batch_size=len(chunks), # Small batch usually
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False # Normalize later after mean-pooling
        )
        
        # Mean pooling
        mean_embedding = np.mean(chunk_embeddings, axis=0)
        
        # L2 Normalize
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm
            
        return mean_embedding
