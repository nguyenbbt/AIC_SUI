import torch
import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

class ASREngine:
    """
    Wrapper for Hugging Face's automatic-speech-recognition pipeline.
    Uses PhoWhisper to transcribe Vietnamese audio.
    """
    def __init__(self, model_size: str = "medium", device: str = "auto", batch_size: int = 8):
        self.model_name = f"vinai/PhoWhisper-{model_size}"
        self.batch_size = batch_size
        
        # Determine device
        if device == "auto":
            self.device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device_name = device
            
        if self.device_name == "cpu":
            logger.warning("Running ASR on CPU. This will be very slow.")
        else:
            logger.info(f"Running ASR on {self.device_name}.")
            
        logger.info(f"Loading ASR model {self.model_name}...")
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            device=self.device_name,
            torch_dtype=torch.float16 if "cuda" in self.device_name else torch.float32
        )
        logger.info("ASR model loaded successfully.")

    def transcribe(self, audio_path: str) -> list:
        """
        Transcribes the given audio file and returns a list of segments.
        Handles long audio via chunking (chunk_length_s=30).
        """
        logger.info(f"Transcribing audio: {audio_path}")
        try:
            # return_timestamps=True and chunk_length_s=30 instruct the pipeline
            # to chunk long files and return absolute timestamps.
            result = self.transcriber(
                audio_path,
                chunk_length_s=30,
                batch_size=self.batch_size,
                return_timestamps=True
            )
            
            # Format output to match our expected schema
            segments = []
            if "chunks" in result:
                for chunk in result["chunks"]:
                    segments.append({
                        "timestamp": chunk["timestamp"],
                        "text": chunk["text"].strip()
                    })
            else:
                # Fallback if chunks are not returned (rare for this config)
                segments.append({
                    "timestamp": (0.0, None),
                    "text": result.get("text", "").strip()
                })
                
            return segments
        except Exception:
            logger.exception("Error during transcription of %s", audio_path)
            raise
