from abc import ABC, abstractmethod
from typing import List, Dict, Any
from PIL import Image

class BaseDetector(ABC):
    @abstractmethod
    def detect_batch(self, images: List[Image.Image]) -> List[List[Dict[str, Any]]]:
        """
        Run object detection on a batch of images.
        
        Args:
            images: List of PIL.Image objects.
            
        Returns:
            A list where each element corresponds to an image.
            Each element is a list of detected objects.
            Object dictionary should follow:
            {
                "label": str,
                "confidence": float,
                "bbox": [x_min, y_min, x_max, y_max],  # Absolute integer coordinates
                "model_source": str
            }
        """
        pass
