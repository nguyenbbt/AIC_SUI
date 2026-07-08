import os
import glob
import json
import logging
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from tqdm import tqdm

from .pipeline import VideoProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_single_video(args_tuple):
    video_path, output_dir, device, webp_quality, threshold = args_tuple
    processor = VideoProcessor(
        output_dir=output_dir,
        device=device,
        webp_quality=webp_quality,
        threshold=threshold
    )
    return processor.process_video(video_path)

def build_parquet_index(metadata_dir: str, output_parquet: str):
    """
    Combine all individual JSON metadata into a single flat Parquet file.
    """
    logger.info(f"Building parquet index from {metadata_dir}...")
    json_files = glob.glob(os.path.join(metadata_dir, "*.json"))
    
    all_keyframes = []
    
    for jf in tqdm(json_files, desc="Parsing JSONs"):
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
                video_id = data["video_id"]
                for shot in data["shots"]:
                    shot_id = shot["shot_id"]
                    for kf in shot["keyframes"]:
                        all_keyframes.append({
                            "video_id": video_id,
                            "shot_id": shot_id,
                            "position": kf["position"],
                            "frame_index": kf["frame_index"],
                            "timestamp_sec": kf["time_sec"],
                            "file_path": kf["file_path"]
                        })
        except Exception as e:
            logger.error(f"Failed to parse {jf} for parquet: {e}")
            
    if all_keyframes:
        df = pd.DataFrame(all_keyframes)
        df.to_parquet(output_parquet, index=False)
        logger.info(f"Saved parquet index to {output_parquet} with {len(df)} records.")
    else:
        logger.warning("No keyframes found to build parquet index.")

def main():
    parser = argparse.ArgumentParser(description="Shot Detection and Keyframe Extraction Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing videos")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Device to use for TransNetV2 (default: auto)")
    parser.add_argument("--quality", type=int, default=90, help="WebP quality (0-100)")
    parser.add_argument("--threshold", type=float, default=0.5, help="TransNetV2 threshold")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Setup multiprocessing start method for PyTorch
    if args.workers > 1:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
            
    # Find all videos
    video_extensions = ('.mp4', '.mkv', '.avi', '.webm')
    video_paths = []
    for root, _, files in os.walk(args.input):
        for f in files:
            if f.lower().endswith(video_extensions):
                video_paths.append(os.path.join(root, f))
                
    logger.info(f"Found {len(video_paths)} videos in {args.input}")
    
    # Prepare arguments for multiprocessing
    tasks = [(vp, args.output, args.device, args.quality, args.threshold) for vp in video_paths]
    
    success_count = 0
    if args.workers <= 1:
        for t in tqdm(tasks, desc="Processing videos"):
            if process_single_video(t):
                success_count += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(tqdm(executor.map(process_single_video, tasks), total=len(tasks), desc="Processing videos"))
            success_count = sum(results)
            
    logger.info(f"Successfully processed {success_count}/{len(video_paths)} videos.")
    
    # Build parquet
    metadata_dir = os.path.join(args.output, "metadata")
    parquet_path = os.path.join(args.output, "metadata_index.parquet")
    build_parquet_index(metadata_dir, parquet_path)
    
    logger.info("Pipeline finished.")

if __name__ == "__main__":
    main()
