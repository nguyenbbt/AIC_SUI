import numpy as np
from PIL import Image
from ocr_module.region_cropper import crop_polygon

def test_crop_polygon_straight():
    # Create a dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = 255 # White rectangle in the middle
    
    # Define polygon
    polygon = [[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]]
    
    cropped = crop_polygon(img, polygon)
    assert cropped.size == (60, 60)
    
def test_crop_polygon_out_of_bounds():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Polygon goes outside the image boundaries
    polygon = [[-10.0, -10.0], [110.0, -10.0], [110.0, 110.0], [-10.0, 110.0]]
    
    cropped = crop_polygon(img, polygon)
    # The clipper should limit it to 0-99
    assert cropped.size[0] > 0
    assert cropped.size[1] > 0
    assert cropped.size[0] <= 100
    assert cropped.size[1] <= 100
