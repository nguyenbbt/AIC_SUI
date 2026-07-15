import os
import glob
import numpy as np
from typing import List, Dict, Any
from PIL import Image
from .base import BaseDetector

class CoDETRDetector(BaseDetector):
    def __init__(
        self,
        backbone: str = "resnet50",
        weights_dir: str = "weights/",
        confidence_threshold: float = 0.25,
        device: str = "cuda"
    ):
        try:
            from mmdet.apis import init_detector, inference_detector
            import mmcv
        except ImportError:
            raise ImportError("Please install mmdet and mmcv to use Co-DETR.")
            
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.inference_detector = inference_detector
        
        if backbone == "resnet50":
            config_name = "co_dino_5scale_r50_8xb2_1x_coco"
        elif backbone == "swin_l":
            config_name = "co_dino_5scale_swin_l_16e_o365tococo"
        else:
            raise ValueError(f"Unsupported Co-DETR backbone: {backbone}. Choose 'resnet50' or 'swin_l'.")
            
        # Find config and checkpoint
        config_path = os.path.join(weights_dir, f"{config_name}.py")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}. Did you run download script?")
            
        checkpoint_pattern = os.path.join(weights_dir, f"{config_name}*.pth")
        checkpoint_files = glob.glob(checkpoint_pattern)
        if not checkpoint_files:
            raise FileNotFoundError(f"Checkpoint file not found for {config_name} in {weights_dir}.")
        checkpoint_path = checkpoint_files[0]
        
        self.model = init_detector(config_path, checkpoint_path, device=device)

    def detect_batch(self, images: List[Image.Image]) -> List[List[Dict[str, Any]]]:
        if not images:
            return []
            
        # Convert PIL images to numpy arrays (RGB to BGR as expected by mmcv usually, 
        # but mmcv/mmdet inference_detector can take np.array in BGR format or path)
        # inference_detector takes list of np.ndarray
        np_images = [np.array(img)[:, :, ::-1] for img in images] # RGB to BGR
        
        # MMDetection inference
        results = self.inference_detector(self.model, np_images)
        
        batch_results = []
        for img, result in zip(images, results):
            img_results = []
            max_x, max_y = img.size
            
            # result is a DetDataSample in mmdet v3
            pred_instances = result.pred_instances
            
            # Move to CPU and numpy
            bboxes = pred_instances.bboxes.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()
            labels = pred_instances.labels.cpu().numpy()
            
            for bbox, score, label_idx in zip(bboxes, scores, labels):
                if score >= self.confidence_threshold:
                    class_name = self.model.dataset_meta['classes'][label_idx]
                    
                    # Absolute integer coordinates, clipped to image boundaries
                    x_min = max(0, int(round(bbox[0])))
                    y_min = max(0, int(round(bbox[1])))
                    x_max = min(max_x, int(round(bbox[2])))
                    y_max = min(max_y, int(round(bbox[3])))
                    
                    if x_max > x_min and y_max > y_min:
                        img_results.append({
                            "label": class_name,
                            "confidence": float(score),
                            "bbox": [x_min, y_min, x_max, y_max],
                            "model_source": "co_detr"
                        })
                        
            batch_results.append(img_results)
            
        return batch_results
