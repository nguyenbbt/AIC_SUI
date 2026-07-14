import pytest
import os
import cv2
import numpy as np

@pytest.fixture(scope="session")
def fixture_dir():
    d = os.path.join(os.path.dirname(__file__), "fixtures")
    os.makedirs(d, exist_ok=True)
    return d

@pytest.fixture(scope="session")
def mock_video_path(fixture_dir):
    """
    Generate a 3-second 30FPS video (90 frames) with 3 'scenes':
    - frames 0-29: Red
    - frames 30-59: Green
    - frames 60-89: Blue
    """
    path = os.path.join(fixture_dir, "test_video.mp4")
    if os.path.exists(path):
        return path
        
    fps = 30.0
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    # Red scene
    red_frame = np.zeros((height, width, 3), dtype=np.uint8)
    red_frame[:] = (0, 0, 255) # BGR
    for _ in range(30): out.write(red_frame)
        
    # Green scene
    green_frame = np.zeros((height, width, 3), dtype=np.uint8)
    green_frame[:] = (0, 255, 0)
    for _ in range(30): out.write(green_frame)
        
    # Blue scene
    blue_frame = np.zeros((height, width, 3), dtype=np.uint8)
    blue_frame[:] = (255, 0, 0)
    for _ in range(30): out.write(blue_frame)
        
    out.release()
    return path
