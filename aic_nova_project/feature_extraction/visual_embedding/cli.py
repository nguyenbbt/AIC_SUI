import argparse
import sys
from .pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Visual Embedding Module for AI Challenge 2026")
    
    parser.add_argument("--metadata-dir", type=str, required=True, help="Directory containing input metadata JSON files.")
    parser.add_argument("--keyframe-dir", type=str, required=True, help="Directory containing extracted keyframes.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save output Parquet files.")
    parser.add_argument("--model-id", type=str, default="hf-hub:timm/PE-Core-bigG-14-448", help="Model ID for open_clip.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device to use for inference.")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"], help="Precision for inference.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--force", action="store_true", help="Force re-processing of all videos.")
    
    args = parser.parse_args()
    
    try:
        run_pipeline(
            metadata_dir=args.metadata_dir,
            keyframe_dir=args.keyframe_dir,
            output_dir=args.output_dir,
            model_id=args.model_id,
            device=args.device,
            precision=args.precision,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            force=args.force
        )
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
