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
            
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25.0
                logger.warning(
                    "Invalid FPS detected for %s, falling back to %s",
                    video_path,
                    fps,
                )

            shots_metadata = []
            for shot_id, (start_frame, end_frame) in enumerate(shots):
                duration_frames = max(0, end_frame - start_frame)
                keyframes_meta = []

                for pos in self.positions:
                    target_idx = start_frame + round(pos * duration_frames)
                    target_idx = min(end_frame, max(start_frame, target_idx))
                    decoded_idx = target_idx

                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        logger.warning(
                            "Could not read frame %s for video %s; "
                            "trying shot start frame %s.",
                            target_idx,
                            video_id,
                            start_frame,
                        )
                        decoded_idx = start_frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                        ret, frame = cap.read()

                    if not ret or frame is None:
                        raise RuntimeError(
                            "Could not decode a real keyframe for "
                            f"video={video_id}, shot_id={shot_id}, "
                            f"target_frame={target_idx}, fallback_frame={start_frame}"
                        )

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    position_code = round(pos * 100)
                    file_name = (
                        f"shot_{shot_id:05d}_pos_{position_code:03d}.webp"
                    )
                    relative_path = f"keyframes/{video_id}/{file_name}"
                    absolute_path = os.path.join(video_out_dir, file_name)
                    pil_img.save(
                        absolute_path,
                        "webp",
                        quality=self.webp_quality,
                    )

                    keyframes_meta.append(
                        {
                            "position": pos,
                            "position_code": position_code,
                            "frame_index": decoded_idx,
                            "source_frame_idx": decoded_idx,
                            "time_sec": round(decoded_idx / fps, 3),
                            "file_path": relative_path,
                            "image_rel_path": relative_path,
                        }
                    )

                shots_metadata.append(
                    {
                        "shot_id": shot_id,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "start_time_sec": round(start_frame / fps, 3),
                        "end_time_sec": round(end_frame / fps, 3),
                        "keyframes": keyframes_meta,
                    }
                )

            return shots_metadata, fps
        finally:
            cap.release()
