import torch
import torch.nn.functional as F
from typing import List
from PIL.Image import Image
import numpy as np

try:
    import open_clip
except ImportError:
    raise ImportError("Please install open_clip_torch to use PECoreEncoder: pip install open_clip_torch")

from .base import VisualEncoder

class PECoreEncoder(VisualEncoder):
    """
    Encoder implementation using open_clip and PE-Core-bigG-14-448.
    """
    def __init__(self, device: str = "auto", precision: str = "fp16", model_id: str = "hf-hub:timm/PE-Core-bigG-14-448"):
        """
        Initialize the PE-Core model.
        """
        self.model_id = model_id
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device == "cpu":
            # Force fp32 on CPU as half precision operations might not be fully supported
            self.precision = "fp32"
            self.dtype = torch.float32
        else:
            self.precision = precision
            if precision == "fp16":
                self.dtype = torch.float16
            elif precision == "bf16":
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float32

        # Load model and transforms
        if "::" in model_id:
            model_name, pretrained = model_id.split("::")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        else:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_id)
        
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

    def encode_batch(self, images: List[Image]) -> np.ndarray:
        """
        Encode a batch of images and return L2-normalized embeddings.
        """
        if not images:
            return np.array([])
            
        # Preprocess images
        tensors = []
        for img in images:
            tensor = self.preprocess(img)
            tensors.append(tensor)
            
        batch_tensor = torch.stack(tensors).to(self.device, dtype=self.dtype)
        
        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=self.dtype) if self.device == 'cuda' else torch.autocast(device_type='cpu', enabled=False):
                image_features = self.model.encode_image(batch_tensor)
                # L2 normalize
                image_features = F.normalize(image_features, p=2, dim=-1)
                
        # Move to CPU and convert to numpy float32
        embeddings = image_features.cpu().to(torch.float32).numpy()
        return embeddings
