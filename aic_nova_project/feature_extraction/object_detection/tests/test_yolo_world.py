import numpy as np
from PIL import Image
import sys
from unittest.mock import MagicMock

# Mock ultralytics module
mock_ultralytics = MagicMock()
sys.modules['ultralytics'] = mock_ultralytics

def test_yolo_world_detector():
    mock_model_instance = MagicMock()
    mock_ultralytics.YOLOWorld.return_value = mock_model_instance
    
    # Mock result object
    mock_result = MagicMock()
    mock_result.orig_shape = (100, 100) # (height, width)
    
    mock_box = MagicMock()
    # Mocking xyxy to behave like torch tensor
    class MockTensor:
        def __init__(self, data):
            self.data = np.array(data)
        def cpu(self):
            return self
        def numpy(self):
            return self.data
        def __getitem__(self, item):
            return MockTensor(self.data[item])
            
    mock_box.xyxy = MockTensor([[10.4, 20.6, 50.1, 80.9]])
    mock_box.conf = MockTensor([[0.95]])
    mock_box.cls = MockTensor([[0]])
    
    mock_result.boxes = [mock_box]
    mock_result.names = {0: "dog"}
    
    # model.predict returns a list of results (one per image)
    mock_model_instance.predict.return_value = [mock_result]
    
    from src.object_detection.detectors.yolo_world_detector import YOLOWorldDetector
    detector = YOLOWorldDetector(model_path="dummy.pt", confidence_threshold=0.5)
    
    img = Image.new('RGB', (100, 100))
    results = detector.detect_batch([img])
    
    assert len(results) == 1
    assert len(results[0]) == 1
    
    obj = results[0][0]
    assert obj["label"] == "dog"
    assert round(obj["confidence"], 2) == 0.95
    assert obj["model_source"] == "yolo_world"
    assert obj["bbox"] == [10, 21, 50, 81] # Rounded
