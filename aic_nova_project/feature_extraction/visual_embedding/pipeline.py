import os
import gc
import logging
from typing import List, Dict, Any
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import time
from datetime import timedelta

from .encoders import PECoreEncoder
from .metadata_reader import read_metadata
from .embedding_writer import write_embeddings_to_parquet
from .resume_validation import visual_output_is_valid

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class KeyframeDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        file_path = record["file_path"]
        
        try:
            # We open the image. PIL Image.open is lazy, so we ensure it's loaded 
            # and converted to RGB to prevent issues with RGBA/Grayscale
            img = Image.open(file_path).convert('RGB')
            # Load to memory so we can close the file descriptor safely? 
            # Not strictly required but good practice for multiprocessing
            img.load()
            return record, img
        except Exception as e:
            logger.warning(f"Error loading image {file_path}: {e}")
            return record, None

def custom_collate(batch):
    """
    Custom collate function to filter out records where image loading failed (None).
    """
    valid_batch = [item for item in batch if item[1] is not None]
    if not valid_batch:
        return [], []
    records, images = zip(*valid_batch)
    return list(records), list(images)

def process_video_batch(
    video_id: str,
    records: List[Dict[str, Any]],
    encoder: PECoreEncoder,
    output_dir: str,
    batch_size: int,
    num_workers: int
):
    """
    Process all keyframes for a single video.
    Returns number of successful encoded frames.
    """
    dataset = KeyframeDataset(records)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=custom_collate
    )
    
    encoded_records = []
    
    for batch_records, batch_images in dataloader:
        if not batch_images:
            continue
            
        current_batch_size = len(batch_images)
        
        # OOM Handling loop
        max_retries = 3
        current_bs = current_batch_size
        success = False
        
        # We need to sub-batch if OOM occurs
        sub_batches = [(batch_records, batch_images)]
        
        while not success and max_retries > 0:
            try:
                # We will process sub_batches
                batch_encoded_records = []
                for sub_records, sub_images in sub_batches:
                    embeddings = encoder.encode_batch(sub_images)
                    
                    for i in range(len(sub_records)):
                        record_out = sub_records[i].copy()
                        source_stat = os.stat(record_out["file_path"])
                        record_out["model_name"] = encoder.model_id
                        record_out["model_id"] = encoder.model_id
                        record_out["precision"] = encoder.precision
                        record_out["source_size_bytes"] = source_stat.st_size
                        record_out["source_mtime_ns"] = source_stat.st_mtime_ns
                        record_out["embedding_dim"] = embeddings.shape[1]
                        record_out["embedding"] = embeddings[i]
                        batch_encoded_records.append(record_out)
                
                # If we get here, no OOM
                encoded_records.extend(batch_encoded_records)
                success = True
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and torch.cuda.is_available():
                    logger.warning(f"OOM encountered for batch size {current_bs}. Halving batch size.")
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    max_retries -= 1
                    if max_retries == 0:
                        logger.error(f"Failed to process batch after multiple OOM retries.")
                        break
                        
                    # Split each sub_batch in half
                    new_sub_batches = []
                    for sr, si in sub_batches:
                        mid = len(si) // 2
                        if mid == 0: # Can't split further
                            new_sub_batches.append((sr, si))
                        else:
                            new_sub_batches.append((sr[:mid], si[:mid]))
                            new_sub_batches.append((sr[mid:], si[mid:]))
                    sub_batches = new_sub_batches
                    current_bs = max(1, current_bs // 2)
                else:
                    raise e
                    
    if len(encoded_records) != len(records):
        raise RuntimeError(
            f"Visual embedding output is incomplete for {video_id}: "
            f"encoded {len(encoded_records)} of {len(records)} frames"
        )

    if encoded_records:
        write_embeddings_to_parquet(video_id, encoded_records, output_dir)
        
    return len(encoded_records)

def run_pipeline(
    metadata_dir: str,
    keyframe_dir: str,
    output_dir: str,
    model_id: str,
    device: str,
    precision: str,
    batch_size: int,
    num_workers: int,
    force: bool
):
    logger.info("Initializing encoder...")
    encoder = PECoreEncoder(device=device, precision=precision, model_id=model_id)
    
    logger.info(f"Reading metadata from {metadata_dir}...")
    # Assume keyframe_base_dir is parent of keyframe_dir if relative, 
    # but based on requirements, keyframe_dir is likely the parent containing the video folders.
    # We pass keyframe_dir as the base.
    base_dir = str(Path(keyframe_dir).parent) # if metadata says "keyframes/V001/..." we should join with "data/"
    if Path(keyframe_dir).name in ["keyframes"]:
        # if keyframe_dir is /data/keyframes and path is keyframes/V..., base is /data
        # Let's handle it smartly.
        pass
    
    # Let's adjust metadata_reader logic if we just pass keyframe_dir as base it might be fine,
    # if paths are relative to base. For now, let's pass a safe base.
    # According to module 1, "file_path": "keyframes/V001/...", so passing the directory containing 'keyframes' is best.
    # Usually this is the parent of keyframe_dir.
    keyframe_base = str(Path(keyframe_dir).parent) 
    
    records = read_metadata(metadata_dir, keyframe_base)
    logger.info(f"Found {len(records)} keyframes to process.")
    
    # Group by video_id
    video_records = {}
    for r in records:
        vid = r["video_id"]
        if vid not in video_records:
            video_records[vid] = []
        video_records[vid].append(r)
        
    total_videos = len(video_records)
    logger.info(f"Total videos to process: {total_videos}")
    
    start_time = time.time()
    total_processed = 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, (video_id, v_records) in enumerate(video_records.items()):
        output_file = os.path.join(output_dir, f"{video_id}.parquet")
        
        # Check if already processed
        if not force and os.path.exists(output_file):
            if visual_output_is_valid(
                output_file,
                v_records,
                model_id=encoder.model_id,
                precision=encoder.precision,
            ):
                logger.info(f"Skipping {video_id} (already processed {len(v_records)} frames).")
                total_processed += len(v_records)
                continue
            logger.warning(
                f"Existing visual artifact for {video_id} is stale or invalid; "
                "re-processing."
            )
                
        logger.info(f"Processing video {video_id} ({idx+1}/{total_videos}) with {len(v_records)} frames...")
        
        processed_count = process_video_batch(
            video_id=video_id,
            records=v_records,
            encoder=encoder,
            output_dir=output_dir,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        total_processed += processed_count
        
        # ETA calculation
        elapsed = time.time() - start_time
        avg_time_per_frame = elapsed / max(1, total_processed)
        remaining_frames = len(records) - total_processed
        eta = remaining_frames * avg_time_per_frame
        
        logger.info(f"Progress: {total_processed}/{len(records)} frames | Throughput: {max(1, total_processed)/elapsed:.2f} fps | ETA: {timedelta(seconds=int(eta))}")
        
    logger.info("Pipeline completed successfully.")
