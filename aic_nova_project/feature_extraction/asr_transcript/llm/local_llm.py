import logging
import torch
import re
from transformers import pipeline
from .base import TranscriptLLM

logger = logging.getLogger(__name__)

class LocalTranscriptLLM(TranscriptLLM):
    """
    Implementation of TranscriptLLM using a local open-source model (e.g. Qwen2.5-1.5B-Instruct).
    Requires transformers and accelerate. Runs completely offline.
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "auto"):
        self.model_name = model_name
        
        if device == "auto":
            self.device_map = "auto" if torch.cuda.is_available() else "cpu"
        else:
            self.device_map = device

        logger.info(f"Loading local LLM {self.model_name} on {self.device_map}...")
        try:
            self.generator = pipeline(
                "text-generation",
                model=self.model_name,
                device_map=self.device_map,
                torch_dtype=torch.float16 if "cpu" not in str(self.device_map) else torch.float32,
            )
            logger.info("Local LLM loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load local LLM: {e}")
            raise e

    def _extract_json_field(self, text: str, field_name: str) -> str:
        """
        Attempts to extract a JSON field value from the raw model output text.
        Small models might not format JSON perfectly, so we use regex as a fallback.
        """
        import json
        
        # Try finding a JSON block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if field_name in data:
                    return str(data[field_name])
            except Exception:
                pass
                
        # Try direct parsing if the whole output is JSON
        try:
            data = json.loads(text)
            if field_name in data:
                return str(data[field_name])
        except Exception:
            pass
            
        # Fallback regex extraction
        pattern = rf'"{field_name}"\s*:\s*"((?:\\"|[^"])*)"'
        match = re.search(pattern, text)
        if match:
            # Unescape basic json escapes
            val = match.group(1).replace('\\"', '"').replace('\\n', '\n')
            return val
            
        # If all parsing fails, return the raw text generated (hoping it's just the answer)
        return text.strip()

    def clean(self, raw_text: str, context: str = "") -> str:
        if not raw_text.strip():
            return ""
            
        system_prompt = (
            "You are an expert Vietnamese transcriber. Clean the noisy ASR text. "
            "Fix typos, add punctuation. Output ONLY valid JSON with a single key 'cleaned_text'."
        )
        
        user_prompt = f"Raw text: {raw_text}\n"
        if context:
            user_prompt = f"Context: {context}\n" + user_prompt
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # We use max_new_tokens proportional to input length + buffer
            max_tokens = max(128, int(len(raw_text) * 1.5))
            outputs = self.generator(
                messages,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=False
            )
            
            generated_text = outputs[0]["generated_text"][-1]["content"]
            return self._extract_json_field(generated_text, "cleaned_text")
            
        except Exception as e:
            logger.error(f"Local LLM error during clean: {e}")
            raise e

    def summarize(self, full_cleaned_text: str) -> str:
        if not full_cleaned_text.strip():
            return ""
            
        system_prompt = (
            "You are an expert summarizer. Summarize the following video transcript. "
            "Output ONLY valid JSON with a single key 'summary'."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Transcript:\n{full_cleaned_text}"}
        ]
        
        try:
            outputs = self.generator(
                messages,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True
            )
            
            generated_text = outputs[0]["generated_text"][-1]["content"]
            return self._extract_json_field(generated_text, "summary")
            
        except Exception as e:
            logger.error(f"Local LLM error during summarize: {e}")
            raise e
