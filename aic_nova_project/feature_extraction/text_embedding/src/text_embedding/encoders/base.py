from abc import ABC, abstractmethod
import numpy as np

class BaseTextEncoder(ABC):
    @abstractmethod
    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Encodes a list of texts into embeddings.
        Should return a numpy array of shape (N, D).
        """
        pass

    @abstractmethod
    def encode_long_text(self, text: str) -> np.ndarray:
        """
        Encodes a single long text (e.g. video summary) by chunking and mean-pooling.
        Should return a 1D numpy array of shape (D,).
        """
        pass
