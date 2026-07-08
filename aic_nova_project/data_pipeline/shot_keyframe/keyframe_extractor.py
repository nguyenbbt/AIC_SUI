import os
import cv2
import logging
from PIL import Image
from typing import List, Dict

logger = logging.getLogger(__name__)

class KeyframeExtractor:
    def __init__(self, positions: List[float] = [0.15, 0.50, 0.85], webp_quality: int = 90):
        """
        Initialize the extractor.
        Args:
            positions: List of normalized positions (0.0 to 1.0) to extract.
            webp_quality: Quality of the saved WebP image (0-100).
        """
        self.positions = positions
        self.webp_quality = webp_quality
        
    def extract_keyframes(self, video_path: str, video_id: str, shots: List[tuple], output_dir: str) -> List[dict]:
        """
        Extract keyframes for all shots in a video.
        Args:
            video_path: Path to the video file.
            video_id: Identifier for the video.
            shots: List of (start_frame, end_frame) tuples.
            output_dir: Base directory to save the keyframes.
        Returns:
            List of dictionaries containing shot metadata (matching ShotMetadata schema).
        """
        video_out_dir = os.path.join(output_dir, video_id)
        os.makedirs(video_out_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video {video_path}")
            raise ValueError(f"Cannot open video {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            # Fallback for some weird videos
            fps = 25.0 
            logger.warning(f"Invalid FPS detected for {video_path}, falling back to {fps}")
            
        shots_metadata = []
        
        for shot_id, (start_frame, end_frame) in enumerate(shots):
            # Calculate target frame indices
            duration_frames = max(0, end_frame - start_frame)
            
            keyframes_meta = []
            for pos in self.positions:
                target_idx = start_frame + round(pos * duration_frames)
                # Ensure we don't go out of bounds of the shot
                target_idx = min(end_frame, max(start_frame, target_idx))
                
                # Seek and read
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"Could not read frame {target_idx} for video {video_id}. Trying to fallback to start_frame.")
                    # Fallback to start frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    ret, frame = cap.read()
                    if not ret:
                        logger.error(f"Total failure reading frames for shot {shot_id} in {video_id}. Creating empty frame.")
                        frame = cv2.Mat.zeros((224, 224, 3), dtype="uint8") # fallback empty frame
                        
                # Save as WebP
                # Convert BGR to RGB for PIL
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    
                    # Format position for filename (e.g., 0.15 -> 015)
                    pos_str = f"{int(pos * 100):03d}"
                    file_name = f"shot_{shot_id:05d}_pos_{pos_str}.webp"
                    rel_file_path = f"keyframes/{video_id}/{file_name}"
                    abs_file_path = os.path.join(video_out_dir, file_name)
                    
                    pil_img.save(abs_file_path, "webp", quality=self.webp_quality)
                    
                    time_sec = target_idx / fps
                    keyframes_meta.append({
                        "position": pos,
                        "frame_index": target_idx,
                        "time_sec": round(time_sec, 3),
                        "file_path": rel_file_path
                    })
                    
            start_time_sec = start_frame / fps
            end_time_sec = end_frame / fps
            
            shots_metadata.append({
                "shot_id": shot_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_sec": round(start_time_sec, 3),
                "end_time_sec": round(end_time_sec, 3),
                "keyframes": keyframes_meta
            })
            
        cap.release()
        return shots_metadata, fps
