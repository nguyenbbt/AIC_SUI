import argparse
import logging
from pathlib import Path

from src.object_detection.pipeline import ObjectDetectionPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def select_metadata_files(
    metadata_dir: Path,
    shard_index: int,
    shard_count: int,
) -> list[Path]:
    """Select one deterministic, non-overlapping metadata shard."""
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1.")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            "shard_index must be between 0 and shard_count - 1."
        )
    metadata_files = sorted(metadata_dir.glob("*.json"))
    return metadata_files[shard_index::shard_count]


def main():
    parser = argparse.ArgumentParser(description="AI Challenge 2026 - Object Detection")
    parser.add_argument("--keyframe-dir", type=Path, required=True, help="Directory containing input keyframes (.webp)")
    parser.add_argument("--metadata-dir", type=Path, required=True, help="Directory containing metadata JSONs")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save output JSONs")
    
    # Models
    parser.add_argument("--run-yolo-world", action="store_true", help="Enable YOLO-World detector")
    parser.add_argument("--yolo-world-model", type=str, default="weights/yolov8s-world.pt", help="Path to YOLO-World model weights")
    parser.add_argument("--custom-vocab-file", type=str, default=None, help="Path to custom vocabulary TXT. If not set, defaults to COCO 80.")
    
    parser.add_argument("--run-co-detr", action="store_true", help="Enable Co-DETR detector")
    parser.add_argument("--co-detr-backbone", type=str, choices=["resnet50", "swin_l"], default=None, help="Backbone for Co-DETR")
    
    # Thresholds
    parser.add_argument("--confidence-threshold", type=float, default=0.25, help="Confidence threshold for detection")
    parser.add_argument("--nms-threshold", type=float, default=0.5, help="IoU threshold for Box Fusion (NMS)")
    
    # Hardware & Batch
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index for distributed execution",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total number of distributed execution shards",
    )
    parser.add_argument("--force", action="store_true", help="Force re-processing if output exists")
    
    args = parser.parse_args()
    
    if not args.run_yolo_world and not args.run_co_detr:
        logger.error("At least one detector must be enabled using --run-yolo-world or --run-co-detr")
        return
        
    co_detr_backbone = args.co_detr_backbone if args.run_co_detr else None
    if args.run_co_detr and not co_detr_backbone:
        co_detr_backbone = "resnet50"
        
    yolo_model = args.yolo_world_model if args.run_yolo_world else None

    pipeline = ObjectDetectionPipeline(
        yolo_world_model=yolo_model,
        custom_vocab_file=args.custom_vocab_file,
        co_detr_backbone=co_detr_backbone,
        confidence_threshold=args.confidence_threshold,
        nms_threshold=args.nms_threshold,
        device=args.device
    )
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_files = select_metadata_files(
        args.metadata_dir,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    logger.info(
        "Selected %d video(s) for Object Detection shard %d/%d.",
        len(metadata_files),
        args.shard_index,
        args.shard_count,
    )

    # Process videos
    for metadata_file in metadata_files:
        video_id = metadata_file.stem
        output_file = args.output_dir / f"{video_id}.json"
        
        video_keyframe_dir = args.keyframe_dir / video_id
        if not video_keyframe_dir.exists():
            raise FileNotFoundError(
                f"Keyframe directory missing for {video_id}: "
                f"{video_keyframe_dir}"
            )
            
        pipeline.process_video(
            video_id=video_id,
            metadata_path=metadata_file,
            keyframe_dir=video_keyframe_dir,
            output_path=output_file,
            batch_size=args.batch_size,
            force=args.force
        )
        
    logger.info("Processing complete.")

if __name__ == "__main__":
    main()
