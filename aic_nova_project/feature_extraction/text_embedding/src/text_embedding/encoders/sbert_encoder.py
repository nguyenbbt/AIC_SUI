import numpy as np
import logging
from typing import List
from sentence_transformers import SentenceTransformer

from .base import BaseTextEncoder

logger = logging.getLogger(__name__)

class SentenceTransformerEncoder(BaseTextEncoder):
    def __init__(self, model_name: str, device: str = "cuda", max_length: int = 256, batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        
        logger.info(f"Loading SentenceTransformer model '{model_name}' on {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_length
        
        # Test shape
        dummy_embedding = self.model.encode(["test"])
        self.embedding_dim = dummy_embedding.shape[1]
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
