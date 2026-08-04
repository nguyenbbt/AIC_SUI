import os
import json
import logging
from typing import List

from .transnet_wrapper import TransNetPredictor
from .keyframe_extractor import KeyframeExtractor
from .metadata_schema import VideoMetadata
from .resume_validation import keyframe_artifacts_are_valid

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, output_dir: str, device: str = None, webp_quality: int = 90, threshold: float = 0.5):
        """
        Orchestrator for processing a single video.
        """
        self.output_dir = output_dir
        self.threshold = threshold
        
        # Initialize sub-modules
        self.transnet = TransNetPredictor(device=device)
        self.extractor = KeyframeExtractor(webp_quality=webp_quality)
        
        self.keyframes_dir = os.path.join(output_dir, "keyframes")
        self.metadata_dir = os.path.join(output_dir, "metadata")
        
        os.makedirs(self.keyframes_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
    def process_video(self, video_path: str) -> bool:
        """
        Process a single video: shot boundary detection + keyframe extraction + save metadata.
        Returns True if successful, False if failed/skipped.
        """
        try:
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            metadata_path = os.path.join(self.metadata_dir, f"{video_id}.json")
            
            # Idempotency / Resume check
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        metadata = VideoMetadata.model_validate(data)
                    if not keyframe_artifacts_are_valid(
                        metadata,
                        self.output_dir,
                    ):
                        raise ValueError(
                            "One or more keyframe artifacts are missing or unreadable"
                        )
                    logger.info(f"Skipping {video_id} - already processed successfully.")
                    return True
                except Exception as e:
                    logger.warning(f"Metadata for {video_id} is corrupted or invalid, re-processing. Error: {e}")
            
            logger.info(f"Processing video: {video_id} ({video_path})")
            
            # Step 1: Detect Shots
            shots = self.transnet.predict_shots(video_path, threshold=self.threshold)
            
            # Step 2: Extract Keyframes
            shots_metadata, fps = self.extractor.extract_keyframes(
                video_path=video_path,
                video_id=video_id,
                shots=shots,
                output_dir=self.keyframes_dir
            )
            
            # Calculate duration
            # Duration can be estimated from the last shot's end_time_sec or open video again
            duration_sec = 0.0
            if shots_metadata:
                duration_sec = shots_metadata[-1]["end_time_sec"]
                
            # Step 3: Build and Validate Metadata
            video_meta = VideoMetadata(
                video_id=video_id,
                source_path=video_path,
                fps=fps,
                duration_sec=duration_sec,
                num_shots=len(shots_metadata),
                shots=shots_metadata
            )
            
            # Step 4: Save Metadata
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(video_meta.model_dump_json(indent=2))
                
            logger.info(f"Successfully processed video: {video_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process video {video_path}. Error: {e}", exc_info=True)
            return False
