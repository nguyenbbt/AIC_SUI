import argparse
import logging
from pathlib import Path
import torch

from src.text_embedding.encoders import SentenceTransformerEncoder
from src.text_embedding.config import (
    TEXT_MAX_LENGTH,
    TEXT_MODEL_NAME,
    TEXT_MODEL_REVISION,
)
from src.text_embedding.pipeline import TextEmbeddingPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="AI Challenge 2026 - Vietnamese Text Embedding")
    parser.add_argument("--asr-dir", type=Path, help="Directory containing ASR JSON files")
    parser.add_argument("--summary-dir", type=Path, help="Directory containing Summary JSON files")
    parser.add_argument("--ocr-dir", type=Path, help="Directory containing OCR JSON files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Base directory for Parquet outputs")
    
    parser.add_argument(
        "--model-name",
        type=str,
        default=TEXT_MODEL_NAME,
        choices=[TEXT_MODEL_NAME],
        help="Locked Hugging Face model name",
    )
    parser.add_argument(
        "--model-revision",
        type=str,
        default=TEXT_MODEL_REVISION,
        choices=[TEXT_MODEL_REVISION],
        help="Locked immutable Hugging Face commit",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu). Auto-detects if None.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference")
    parser.add_argument(
        "--max-length",
        type=int,
        default=TEXT_MAX_LENGTH,
        choices=[TEXT_MAX_LENGTH],
        help="Locked sequence length for truncation/chunking",
    )
    parser.add_argument("--force", action="store_true", help="Force re-processing if output exists")
    
    args = parser.parse_args()
    
    device = args.device
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    logger.info(f"Using device: {device}")
    
    encoder = SentenceTransformerEncoder(
        model_name=args.model_name,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        model_revision=args.model_revision,
    )
    
    pipeline = TextEmbeddingPipeline(encoder)
    
    if args.asr_dir and args.asr_dir.exists():
        logger.info("Processing ASR...")
        pipeline.process_asr(args.asr_dir, args.output_dir / "text_asr", args.force)
        
    if args.summary_dir and args.summary_dir.exists():
        logger.info("Processing Summaries...")
        pipeline.process_summary(args.summary_dir, args.output_dir / "text_summary", args.force)
        
    if args.ocr_dir and args.ocr_dir.exists():
        logger.info("Processing OCR...")
        pipeline.process_ocr(args.ocr_dir, args.output_dir / "text_ocr", args.force)
        
    logger.info("Text embedding complete.")

if __name__ == "__main__":
    main()
