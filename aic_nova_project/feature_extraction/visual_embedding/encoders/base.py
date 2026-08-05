from abc import ABC, abstractmethod
from typing import List
from PIL.Image import Image
import numpy as np

class VisualEncoder(ABC):
    """
    Abstract base class for visual embedding models.
    """
    @abstractmethod
    def __init__(self, device: str = "auto", precision: str = "fp16", **kwargs):
        """
        Initialize the encoder.
        
        Args:
            device: 'cuda', 'cpu', or 'auto'
            precision: 'fp16', 'bf16', or 'fp32'
            **kwargs: Additional model-specific arguments
        """
        pass
        
    @abstractmethod
    def encode_batch(self, images: List[Image]) -> np.ndarray:
        """
        Takes a batch of PIL Images and returns normalized embeddings.
        The output MUST be L2-normalized.
        
        Args:
            images: List of PIL.Image objects.
            
        Returns:
            np.ndarray: Matrix of size (Batch_Size, Embedding_Dimension), dtype = float32.
        """
        pass
