from typing import List

import numpy as np
from PIL.Image import Image
import torch
import torch.nn.functional as F

try:
    import open_clip
except ImportError:
    raise ImportError(
        "Please install open_clip_torch to use OpenCLIPEncoder: "
        "pip install open_clip_torch"
    )

from ..config import DEFAULT_VISUAL_MODEL_ID, parse_open_clip_model_id
from .base import VisualEncoder


class OpenCLIPEncoder(VisualEncoder):
    """Encode images with an OpenCLIP-compatible model."""

    def __init__(
        self,
        device: str = "auto",
        precision: str = "fp16",
        model_id: str = DEFAULT_VISUAL_MODEL_ID,
    ):
        """Initialize the configured OpenCLIP model."""
        self.model_id = model_id
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device == "cpu":
            # Half precision operations are not fully supported on CPU.
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

        model_name, pretrained = parse_open_clip_model_id(model_id)
        if pretrained is not None:
            self.model, _, self.preprocess = (
                open_clip.create_model_and_transforms(
                    model_name,
                    pretrained=pretrained,
                )
            )
        else:
            self.model, _, self.preprocess = (
                open_clip.create_model_and_transforms(model_name)
            )

        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

    def encode_batch(self, images: List[Image]) -> np.ndarray:
        """Encode images and return L2-normalized float32 embeddings."""
        if not images:
            return np.array([])

        tensors = [self.preprocess(image) for image in images]
        batch_tensor = torch.stack(tensors).to(
            self.device,
            dtype=self.dtype,
        )

        with torch.no_grad():
            autocast = (
                torch.autocast(device_type="cuda", dtype=self.dtype)
                if self.device == "cuda"
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with autocast:
                image_features = self.model.encode_image(batch_tensor)
                image_features = F.normalize(image_features, p=2, dim=-1)

        return image_features.cpu().to(torch.float32).numpy()
