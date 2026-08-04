import os
import torch
from transnetv2_pytorch import TransNetV2
import logging

logger = logging.getLogger(__name__)

class TransNetPredictor:
    def __init__(self, weights_path: str = "weights/transnetv2-pytorch-weights.pth", device: str = None):
        """
        Initialize the TransNetV2 model.
        Args:
            weights_path: Path to the PyTorch weights file.
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing TransNetV2 on {self.device}...")
        try:
            self.model = TransNetV2()
            
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded weights from {weights_path}")
            else:
                logger.warning(f"Weights file not found at {weights_path}. Relying on internal package weights if available.")
            
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to initialize TransNetV2: {e}")
            raise

    def predict_shots(self, video_path: str, threshold: float = 0.5):
        """
        Predict shot boundaries for a video.
        Args:
            video_path: Path to the video file.
            threshold: Threshold for shot boundary detection (not strictly used by high-level API but kept for signature).
        Returns:
            List of tuples (start_frame, end_frame)
        """
        logger.info(f"Running TransNetV2 inference on {video_path}")
        try:
            with torch.no_grad():
                try:
                    scenes = self.model.detect_scenes(video_path, threshold=threshold)
                except TypeError as exc:
                    message = str(exc)
                    if (
                        "unexpected keyword" not in message
                        or "threshold" not in message
                    ):
                        raise
                    scenes = self.model.detect_scenes(video_path)
                    
            shots = []
            for scene in scenes:
                if isinstance(scene, dict):
                    start_frame = int(scene.get('start_frame', 0))
                    end_frame = int(scene.get('end_frame', 0))
                else:
                    start_frame = int(scene[0])
                    end_frame = int(scene[1])
                shots.append((start_frame, end_frame))
                
            # Edge case: if no shots detected or video is a single shot
            if len(shots) == 0:
                logger.warning(f"No shots detected in {video_path}, assuming single shot.")
                import cv2
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    if total_frames > 0:
                        shots.append((0, total_frames - 1))
                    else:
                        shots.append((0, 0))
                else:
                    shots.append((0, 0))
                    
            return shots
            
        except Exception as e:
            logger.error(f"Error during shot detection on {video_path}: {e}")
            raise
