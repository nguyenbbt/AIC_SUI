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


def resolve_worker_count(
    requested_workers: int,
    device: str | None,
    cuda_available: bool | None = None,
) -> int:
    """Return a worker count that cannot replicate a CUDA model in VRAM."""
    if requested_workers < 1:
        raise ValueError("--workers must be at least 1")

    if cuda_available is None:
        import torch

        cuda_available = torch.cuda.is_available()

    uses_cuda = device == "cuda" or (device is None and cuda_available)
    if uses_cuda and requested_workers > 1:
        logger.warning(
            "CUDA execution uses one worker to avoid loading multiple "
            "TransNetV2 copies into VRAM."
        )
        return 1

    return requested_workers


def process_single_video(args_tuple):
    video_path, output_dir, device, webp_quality, threshold, data_root = args_tuple
    processor = VideoProcessor(
        output_dir=output_dir,
        device=device,
        webp_quality=webp_quality,
        threshold=threshold,
        data_root=data_root,
    )
    return processor.process_video(video_path)

def build_parquet_index(metadata_dir: str, output_parquet: str):
    """
    Combine all individual JSON metadata into a single flat Parquet file.
    """
    logger.info(f"Building parquet index from {metadata_dir}...")
    json_files = sorted(glob.glob(os.path.join(metadata_dir, "*.json")))
    
    all_keyframes = []
    parse_failures = []
    
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
                            "position_code": kf["position_code"],
                            "frame_index": kf["frame_index"],
                            "source_frame_idx": kf["source_frame_idx"],
                            "timestamp_sec": kf["time_sec"],
                            "file_path": kf["file_path"],
                            "image_rel_path": kf["image_rel_path"],
                        })
        except Exception as e:
            logger.error(f"Failed to parse {jf} for parquet: {e}")
            parse_failures.append(f"{os.path.basename(jf)}: {e}")

    if parse_failures:
        raise RuntimeError(
            "Metadata index build failed: " + "; ".join(parse_failures)
        )

    if all_keyframes:
        df = pd.DataFrame(all_keyframes)
        temporary_path = f"{output_parquet}.tmp"
        try:
            df.to_parquet(temporary_path, index=False)
            os.replace(temporary_path, output_parquet)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        logger.info(f"Saved parquet index to {output_parquet} with {len(df)} records.")
    else:
        logger.warning("No keyframes found to build parquet index.")
    return len(all_keyframes)

def main():
    parser = argparse.ArgumentParser(description="Shot Detection and Keyframe Extraction Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing videos")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Device to use for TransNetV2 (default: auto)")
    parser.add_argument("--quality", type=int, default=90, help="WebP quality (0-100)")
    parser.add_argument("--threshold", type=float, default=0.5, help="TransNetV2 threshold")
    
    args = parser.parse_args()
    worker_count = resolve_worker_count(args.workers, args.device)
    
    os.makedirs(args.output, exist_ok=True)
    
    # Setup multiprocessing start method for PyTorch
    if worker_count > 1:
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
    data_root = os.path.dirname(os.path.abspath(args.input))
    tasks = [
        (
            vp,
            args.output,
            args.device,
            args.quality,
            args.threshold,
            data_root,
        )
        for vp in video_paths
    ]
    
    success_count = 0
    if worker_count <= 1:
        for t in tqdm(tasks, desc="Processing videos"):
            if process_single_video(t):
                success_count += 1
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results = list(tqdm(executor.map(process_single_video, tasks), total=len(tasks), desc="Processing videos"))
            success_count = sum(results)
            
    logger.info(f"Successfully processed {success_count}/{len(video_paths)} videos.")

    if success_count != len(video_paths):
        logger.error(
            "Processing incomplete: %s of %s videos succeeded.",
            success_count,
            len(video_paths),
        )
        return 1
    
    # Build parquet
    metadata_dir = os.path.join(args.output, "metadata")
    parquet_path = os.path.join(args.output, "metadata_index.parquet")
    try:
        build_parquet_index(metadata_dir, parquet_path)
    except Exception as e:
        logger.error(f"Failed to build complete metadata index: {e}")
        return 1
    
    logger.info("Pipeline finished.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
