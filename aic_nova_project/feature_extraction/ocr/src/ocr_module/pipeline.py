import json
import logging
import cv2
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from .detector import TextDetector
from .recognizer import TextRecognizer
from .region_cropper import crop_polygon
from .text_ordering import group_and_order_regions, concat_text
from .metadata_reader import get_keyframes_from_metadata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCRPipeline:
    def __init__(self, use_gpu: bool = True, backbone: str = 'vgg_transformer', 
                 confidence_threshold: float = 0.4):
        self.detector = TextDetector(use_gpu=use_gpu)
        self.recognizer = TextRecognizer(backbone=backbone, use_gpu=use_gpu)
        self.confidence_threshold = confidence_threshold

    def process_video(self, video_id: str, keyframe_dir: Path, metadata_dir: Path, 
                      output_dir: Path, width_ths: float = 0.7, mag_ratio: float = 1.5,
                      force: bool = False):
        """Processes all keyframes for a single video."""
        output_path = output_dir / f"{video_id}.json"
        
        if output_path.exists() and not force:
            logger.info(f"Output for {video_id} already exists. Skipping.")
            return

        metadata_path = metadata_dir / f"{video_id}.json"
        frames = get_keyframes_from_metadata(metadata_path)
        
        if not frames:
            logger.warning(f"No metadata frames found for {video_id}.")
            return

        video_keyframes_dir = keyframe_dir / video_id
        if not video_keyframes_dir.exists():
            logger.warning(f"Keyframe directory not found: {video_keyframes_dir}")
            return
            
        result_frames = []
        
        # We can implement batching later if needed, currently iterating through frames
        for frame in tqdm(frames, desc=f"Processing {video_id}"):
            frame_id = frame.get("frame_id")
            if not frame_id:
                continue
                
            image_path = video_keyframes_dir / f"{frame_id}.webp"
            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                continue
                
            # Read image using OpenCV
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                continue
                
            # 1. Detection
            polygons = self.detector.detect(image, width_ths=width_ths, mag_ratio=mag_ratio)
            
            ocr_regions = []
            for poly in polygons:
                # 2. Crop
                cropped_img = crop_polygon(image, poly)
                
                # Check if cropped image is valid (not 1x1 fallback)
                if cropped_img.width <= 1 or cropped_img.height <= 1:
                    continue
                    
                # 3. Recognition
                text, conf = self.recognizer.recognize(cropped_img)
                
                # 4. Filtering
                if conf >= self.confidence_threshold and text.strip():
                    ocr_regions.append({
                        "bbox": poly,
                        "text": text.strip(),
                        "confidence": round(conf, 4)
                    })
            
            # 5. Ordering
            ordered_regions = group_and_order_regions(ocr_regions)
            concat_txt = concat_text(ordered_regions)
            
            # Append result
            frame_result = {
                "frame_id": frame_id,
                "shot_id": frame.get("shot_id", 0),
                "position": frame.get("position", 0.0),
                "ocr_regions": ordered_regions,
                "ocr_text_concat": concat_txt
            }
            result_frames.append(frame_result)
            
        # Write output
        output_dir.mkdir(parents=True, exist_ok=True)
        result_json = {
            "video_id": video_id,
            "frames": result_frames
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Successfully processed {video_id}. Saved to {output_path}")

def run_pipeline(keyframe_dir: str, metadata_dir: str, output_dir: str, 
                 width_ths: float = 0.7, mag_ratio: float = 1.5,
                 confidence_threshold: float = 0.4, backbone: str = 'vgg_transformer',
                 use_gpu: bool = True, force: bool = False, workers: int = 1):
    """Entry point for CLI or multiprocessing wrapper."""
    k_dir = Path(keyframe_dir)
    m_dir = Path(metadata_dir)
    o_dir = Path(output_dir)
    
    # Setup pipeline
    pipeline = OCRPipeline(use_gpu=use_gpu, backbone=backbone, confidence_threshold=confidence_threshold)
    
    # Identify videos to process based on metadata files
    video_ids = [p.stem for p in m_dir.glob("*.json")]
    
    if workers > 1:
        # Multiprocessing placeholder for future extension
        logger.info("Multiprocessing requested. Currently executing sequentially.")
    
    for video_id in video_ids:
        pipeline.process_video(
            video_id=video_id,
            keyframe_dir=k_dir,
            metadata_dir=m_dir,
            output_dir=o_dir,
            width_ths=width_ths,
            mag_ratio=mag_ratio,
            force=force
        )
