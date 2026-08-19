import argparse
import sys
from ocr_module.pipeline import run_pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="OCR Module for AI Challenge (Module 4)")
    
    parser.add_argument("--keyframe-dir", type=str, required=True,
                        help="Path to the directory containing video keyframes")
    parser.add_argument("--metadata-dir", type=str, required=True,
                        help="Path to the directory containing video metadata JSON files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to the output directory for OCR JSON results")
                        
    # EasyOCR params
    parser.add_argument("--width-ths", type=float, default=0.7,
                        help="Threshold for merging boxes horizontally (default: 0.7)")
    parser.add_argument("--mag-ratio", type=float, default=1.5,
                        help="Image magnification ratio for text detection (default: 1.5)")
                        
    # VietOCR params
    parser.add_argument("--vietocr-backbone", type=str, default="vgg_transformer",
                        choices=["vgg_transformer", "vgg_seq2seq"],
                        help="VietOCR backbone model (default: vgg_transformer)")
    parser.add_argument("--confidence-threshold", type=float, default=0.4,
                        help="Minimum confidence threshold to keep recognized text (default: 0.4)")
                        
    # System params
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use for inference (e.g., cuda:0 or cpu)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="VietOCR recognition crops per inference batch")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers to process videos")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Zero-based shard index for distributed execution")
    parser.add_argument("--shard-count", type=int, default=1,
                        help="Total number of distributed execution shards")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output files")
                        
    return parser.parse_args()

def main():
    args = parse_args()
    
    use_gpu = args.device.startswith("cuda")
    
    run_pipeline(
        keyframe_dir=args.keyframe_dir,
        metadata_dir=args.metadata_dir,
        output_dir=args.output_dir,
        width_ths=args.width_ths,
        mag_ratio=args.mag_ratio,
        confidence_threshold=args.confidence_threshold,
        backbone=args.vietocr_backbone,
        use_gpu=use_gpu,
        force=args.force,
        workers=args.workers,
        batch_size=args.batch_size,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )

if __name__ == "__main__":
    sys.exit(main())
