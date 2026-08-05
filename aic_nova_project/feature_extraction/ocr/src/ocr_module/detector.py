import easyocr
import numpy as np
from typing import List, Tuple

class TextDetector:
    """Wrapper around EasyOCR to perform text detection only."""
    
    def __init__(self, use_gpu: bool = True):
        """
        Initializes the EasyOCR detector.
        
        Args:
            use_gpu (bool): Whether to use GPU for detection.
        """
        self.reader = easyocr.Reader(['vi'], gpu=use_gpu)

    def detect(self, image: np.ndarray, width_ths: float = 0.7, mag_ratio: float = 1.5) -> List[List[List[float]]]:
        """
        Detects text bounding boxes in an image.
        
        Args:
            image (np.ndarray): The input image as a numpy array (e.g., loaded by cv2 or PIL).
            width_ths (float): Threshold for merging boxes horizontally.
            mag_ratio (float): Image magnification ratio before processing.
            
        Returns:
            List[List[List[float]]]: A list of polygons. Each polygon is a list of 4 points 
                                     [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
        """
        # We use the detect method of easyocr which only runs the CRAFT detector.
        # It returns a tuple of (horizontal_list, free_list).
        # Typically we just use the first element of the first tuple for standard text detection.
        # We must return it in a uniform format.
        
        # easyocr.detect returns (horizontal_list, free_list)
        # We need a unified list of polygons.
        detection_result = self.reader.detect(image, width_ths=width_ths, mag_ratio=mag_ratio)
        
        # horizontal_list format: [x_min, x_max, y_min, y_max]
        horizontal_list = detection_result[0][0] if len(detection_result) > 0 and len(detection_result[0]) > 0 else []
        free_list = detection_result[1][0] if len(detection_result) > 1 and len(detection_result[1]) > 0 else []
        
        polygons = []
        
        # Convert horizontal_list [x_min, x_max, y_min, y_max] to polygon
        for box in horizontal_list:
            x_min, x_max, y_min, y_max = box
            polygons.append([
                [float(x_min), float(y_min)],
                [float(x_max), float(y_min)],
                [float(x_max), float(y_max)],
                [float(x_min), float(y_max)]
            ])
            
        # Add free_list polygons directly
        for box in free_list:
            polygons.append([[float(pt[0]), float(pt[1])] for pt in box])
            
        return polygons
