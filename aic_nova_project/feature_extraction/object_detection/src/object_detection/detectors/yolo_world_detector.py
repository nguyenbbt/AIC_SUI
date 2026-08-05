import os
from typing import List, Dict, Any
import numpy as np
from PIL import Image
from .base import BaseDetector


def _tensor_scalar(value: Any, field_name: str) -> float:
    """Extract exactly one scalar from a Torch-like tensor value."""
    cpu_value = value.cpu() if hasattr(value, "cpu") else value
    raw_value = (
        cpu_value.numpy()
        if hasattr(cpu_value, "numpy")
        else cpu_value
    )
    array = np.asarray(raw_value)
    if array.size != 1:
        raise ValueError(
            f"YOLO-World {field_name} must contain exactly one value; "
            f"received shape {array.shape}."
        )
    return float(array.item())

class YOLOWorldDetector(BaseDetector):
    def __init__(
        self,
        model_path: str = "weights/yolov8s-world.pt",
        custom_vocab_file: str = None,
        confidence_threshold: float = 0.25,
        device: str = "cuda"
    ):
        try:
            from ultralytics import YOLOWorld
        except ImportError:
            raise ImportError("Please install ultralytics to use YOLO-World.")
            
        self.model = YOLOWorld(model_path)
        self.model.to(device)
        self.confidence_threshold = confidence_threshold
        
        if custom_vocab_file and os.path.exists(custom_vocab_file):
            with open(custom_vocab_file, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
            if classes:
                self.model.set_classes(classes)
        # If no custom_vocab_file or it's empty, YOLO-World defaults to COCO 80 classes.

    def detect_batch(self, images: List[Image.Image]) -> List[List[Dict[str, Any]]]:
        if not images:
            return []
            
        # ultralytics model handles batch inference by taking a list of PIL images
        results = self.model.predict(
            source=images,
            conf=self.confidence_threshold,
            verbose=False
        )
        
        batch_results = []
        for result in results:
            img_results = []
            orig_shape = result.orig_shape  # (height, width)
            max_y, max_x = orig_shape
            
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # coords are [x_min, y_min, x_max, y_max]
                    coords = box.xyxy[0].cpu().numpy()
                    conf = _tensor_scalar(box.conf[0], "confidence")
                    cls_idx = int(_tensor_scalar(box.cls[0], "class index"))
                    label = result.names[cls_idx]
                    
                    # Absolute integer coordinates, clipped to image boundaries
                    x_min = max(0, int(round(coords[0])))
                    y_min = max(0, int(round(coords[1])))
                    x_max = min(max_x, int(round(coords[2])))
                    y_max = min(max_y, int(round(coords[3])))
                    
                    if x_max > x_min and y_max > y_min:
                        img_results.append({
                            "label": label,
                            "confidence": conf,
                            "bbox": [x_min, y_min, x_max, y_max],
                            "model_source": "yolo_world"
                        })
                        
            batch_results.append(img_results)
            
        return batch_results
