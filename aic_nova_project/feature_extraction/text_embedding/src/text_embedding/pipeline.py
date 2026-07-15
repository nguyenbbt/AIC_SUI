import logging
from pathlib import Path
from typing import Optional

from .encoders import BaseTextEncoder
from .data_readers import parse_asr_file, parse_summary_file, parse_ocr_file
from .embedding_writer import write_embeddings_to_parquet

logger = logging.getLogger(__name__)

class TextEmbeddingPipeline:
    def __init__(self, encoder: BaseTextEncoder):
        self.encoder = encoder

    def process_asr(self, input_dir: Path, output_dir: Path, force: bool = False):
        """Processes ASR JSONs to Parquet embeddings using batch encoding."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for json_path in input_dir.glob("*.json"):
            video_id = json_path.stem.replace("_cleaned", "")
            output_path = output_dir / f"{video_id}.parquet"
            
            if output_path.exists() and not force:
                logger.info(f"Skipping {video_id} (ASR) - already exists.")
                continue
                
            records = parse_asr_file(json_path)
            if not records:
                logger.debug(f"No valid ASR records for {video_id}")
                continue
                
            texts = [r["text"] for r in records]
            embeddings = self.encoder.encode_batch(texts)
            
            for i, record in enumerate(records):
                record["embedding"] = embeddings[i].tolist()
                
            write_embeddings_to_parquet(records, output_path)

    def process_ocr(self, input_dir: Path, output_dir: Path, force: bool = False):
        """Processes OCR JSONs to Parquet embeddings using batch encoding."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for json_path in input_dir.glob("*.json"):
            video_id = json_path.stem
            output_path = output_dir / f"{video_id}.parquet"
            
            if output_path.exists() and not force:
                logger.info(f"Skipping {video_id} (OCR) - already exists.")
                continue
                
            records = parse_ocr_file(json_path)
            if not records:
                logger.debug(f"No valid OCR records for {video_id}")
                continue
                
            texts = [r["text"] for r in records]
            embeddings = self.encoder.encode_batch(texts)
            
            for i, record in enumerate(records):
                record["embedding"] = embeddings[i].tolist()
                
            write_embeddings_to_parquet(records, output_path)

    def process_summary(self, input_dir: Path, output_dir: Path, force: bool = False):
        """Processes Summary JSONs to Parquet embeddings using chunking & mean-pooling."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for json_path in input_dir.glob("*.json"):
            video_id = json_path.stem
            output_path = output_dir / f"{video_id}.parquet"
            
            if output_path.exists() and not force:
                logger.info(f"Skipping {video_id} (Summary) - already exists.")
                continue
                
            records = parse_summary_file(json_path)
            if not records:
                logger.debug(f"No valid Summary records for {video_id}")
                continue
                
            # Summary uses encode_long_text for chunking + mean pooling
            for record in records:
                embedding = self.encoder.encode_long_text(record["text"])
                record["embedding"] = embedding.tolist()
                
            write_embeddings_to_parquet(records, output_path)
