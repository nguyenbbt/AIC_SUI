import numpy as np
from PIL import Image
import sys
from unittest.mock import patch, MagicMock

# Mock mmdet and mmcv modules
mock_mmdet = MagicMock()
mock_mmcv = MagicMock()
sys.modules['mmdet'] = mock_mmdet
sys.modules['mmdet.apis'] = mock_mmdet.apis
sys.modules['mmcv'] = mock_mmcv

@patch('os.path.exists')
@patch('glob.glob')
def test_co_detr_detector(mock_glob, mock_exists):
    # Mock file existence checks
    mock_exists.return_value = True
    mock_glob.return_value = ["dummy_weights.pth"]
    
    # Mock model
    mock_model = MagicMock()
    mock_model.dataset_meta = {'classes': ['cat', 'dog']}
    mock_mmdet.apis.init_detector.return_value = mock_model
    
    # Mock result object (DetDataSample)
    mock_result = MagicMock()
    mock_pred = MagicMock()
    
    class MockTensor:
        def __init__(self, data):
            self.data = np.array(data)
        def cpu(self):
            return self
        def numpy(self):
            return self.data
            
    mock_pred.bboxes = MockTensor([[10.1, 10.1, 50.9, 50.9]])
    mock_pred.scores = MockTensor([0.9])
    mock_pred.labels = MockTensor([1]) # index 1 -> dog
    
    mock_result.pred_instances = mock_pred
    mock_mmdet.apis.inference_detector.return_value = [mock_result]
    
    from src.object_detection.detectors.co_detr_detector import CoDETRDetector
    detector = CoDETRDetector(backbone="resnet50", confidence_threshold=0.5)
    
    img = Image.new('RGB', (100, 100))
    results = detector.detect_batch([img])
    
    assert len(results) == 1
    assert len(results[0]) == 1
    
    obj = results[0][0]
    assert obj["label"] == "dog"
    assert round(obj["confidence"], 2) == 0.9
    assert obj["model_source"] == "co_detr"
    assert obj["bbox"] == [10, 10, 51, 51]
