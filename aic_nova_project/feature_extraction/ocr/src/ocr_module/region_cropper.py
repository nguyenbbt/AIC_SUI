import cv2
import numpy as np
from PIL import Image
from typing import List

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points to be: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # top-left will have the smallest sum, bottom-right the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # top-right will have the smallest difference, bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def crop_polygon(image: np.ndarray, polygon: List[List[float]]) -> Image.Image:
    """
    Crops a polygon from an image using perspective transform to un-tilt the box.
    Clips coordinates to image boundaries to prevent errors.
    
    Args:
        image (np.ndarray): Original image (BGR format from cv2).
        polygon (List[List[float]]): 4 points of the bounding polygon.
        
    Returns:
        Image.Image: The cropped, perspective-transformed image as a PIL Image.
    """
    pts = np.array(polygon, dtype="float32")
    
    # Clip coordinates to image boundaries
    h_img, w_img = image.shape[:2]
    pts[:, 0] = np.clip(pts[:, 0], 0, w_img - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h_img - 1)
    
    # Order points: tl, tr, br, bl
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute width of new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute height of new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Construct destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Perspective transform
    if maxWidth == 0 or maxHeight == 0:
        # Fallback to an empty small image if dimensions are 0 due to extreme clipping
        return Image.new('RGB', (1, 1))
        
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # Convert BGR to RGB for PIL
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(warped_rgb)
