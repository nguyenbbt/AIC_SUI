import numpy as np
from typing import List, Dict, Any

def compute_iou(box1: List[int], box2: List[int]) -> float:
    """Compute Intersection over Union of two bounding boxes (absolute integer format)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    if union == 0:
        return 0.0
        
    return intersection / union

def apply_nms(objects: List[Dict[str, Any]], nms_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Apply Non-Maximum Suppression to filter overlapping bounding boxes with the same label.
    
    Args:
        objects: List of object dictionaries, each containing 'label', 'confidence', 'bbox', 'model_source'.
        nms_threshold: IoU threshold above which overlapping boxes are suppressed.
        
    Returns:
        List of filtered objects.
    """
    if not objects:
        return []
        
    # Group objects by label
    grouped_objects = {}
    for obj in objects:
        label = obj["label"].lower()
        if label not in grouped_objects:
            grouped_objects[label] = []
        grouped_objects[label].append(obj)
        
    filtered_objects = []
    
    for label, group in grouped_objects.items():
        # Sort objects by confidence in descending order
        group = sorted(group, key=lambda x: x["confidence"], reverse=True)
        keep = []
        
        for i, obj in enumerate(group):
            is_suppressed = False
            for kept_obj in keep:
                iou = compute_iou(obj["bbox"], kept_obj["bbox"])
                if iou > nms_threshold:
                    is_suppressed = True
                    break
            
            if not is_suppressed:
                keep.append(obj)
                
        filtered_objects.extend(keep)
        
    # Restore original order of keeping, sorted by confidence descending
    return sorted(filtered_objects, key=lambda x: x["confidence"], reverse=True)
