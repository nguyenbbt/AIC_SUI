from src.object_detection.box_fusion import compute_iou, apply_nms

def test_compute_iou():
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    iou = compute_iou(box1, box2)
    assert 0 < iou < 0.5
    
    # Same box
    assert compute_iou(box1, box1) == 1.0
    
    # Non-overlapping
    box3 = [20, 20, 30, 30]
    assert compute_iou(box1, box3) == 0.0
    
def test_apply_nms():
    objects = [
        {"label": "person", "confidence": 0.9, "bbox": [0, 0, 100, 100], "model_source": "m1"},
        {"label": "person", "confidence": 0.8, "bbox": [5, 5, 95, 95], "model_source": "m2"}, # High overlap, lower conf -> suppressed
        {"label": "car", "confidence": 0.95, "bbox": [200, 200, 300, 300], "model_source": "m1"},
        {"label": "car", "confidence": 0.99, "bbox": [205, 205, 295, 295], "model_source": "m2"}, # High overlap, higher conf -> keeps m2, suppresses m1
        {"label": "person", "confidence": 0.5, "bbox": [500, 500, 600, 600], "model_source": "m1"} # No overlap -> kept
    ]
    
    filtered = apply_nms(objects, nms_threshold=0.5)
    
    # Expected: 
    # person from m1 (0.9)
    # car from m2 (0.99)
    # person from m1 (0.5)
    assert len(filtered) == 3
    
    person_boxes = [obj for obj in filtered if obj["label"] == "person"]
    car_boxes = [obj for obj in filtered if obj["label"] == "car"]
    
    assert len(person_boxes) == 2
    assert person_boxes[0]["confidence"] == 0.9
    assert person_boxes[1]["confidence"] == 0.5
    
    assert len(car_boxes) == 1
    assert car_boxes[0]["confidence"] == 0.99
