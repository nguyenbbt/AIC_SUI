import os
import logging
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from .base import TranscriptLLM

logger = logging.getLogger(__name__)

# Pydantic schemas for structured output
class CleanedTranscriptSchema(BaseModel):
    cleaned_text: str = Field(description="The cleaned transcript text with corrected punctuation, spelling, and grammar.")

class SummarySchema(BaseModel):
    summary: str = Field(description="A concise summary of the main events and topics in the video.")


class GeminiTranscriptLLM(TranscriptLLM):
    """
    Implementation of TranscriptLLM using Google's Gemini models via the google-genai SDK.
    Features structured outputs, automatic retries for rate limits, and token tracking.
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set. Required for GeminiTranscriptLLM.")
        
        self.client = genai.Client(api_key=api_key)
        self.total_tokens_used = 0

    def _update_usage(self, response):
        if response.usage_metadata:
            self.total_tokens_used += response.usage_metadata.total_token_count

    # Retry on exceptions, typically rate limits (429) or temporary server errors
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def clean(self, raw_text: str, context: str = "") -> str:
        if not raw_text.strip():
            return ""
            
        prompt = (
            "You are an expert Vietnamese transcriber and editor. Your task is to clean up a noisy "
            "ASR (Automatic Speech Recognition) transcript. Fix typos, add missing punctuation, "
            "and correct grammatical errors while preserving the original meaning and conversational tone.\n\n"
        )
        if context:
            prompt += f"Context from previous sentences: {context}\n\n"
            
        prompt += f"Raw ASR Text to clean:\n{raw_text}"
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CleanedTranscriptSchema,
                    temperature=0.1, # Low temperature for factual cleanup
                )
            )
            self._update_usage(response)
            
            # The SDK parses it directly if we use pydantic BaseModel in some versions,
            # or we might need to parse the JSON string.
            # Using parsed property if available, otherwise parse text.
            if hasattr(response, 'parsed') and response.parsed is not None:
                return response.parsed.cleaned_text
            else:
                import json
                data = json.loads(response.text)
                return data.get("cleaned_text", raw_text)
                
        except Exception as e:
            logger.error(f"Gemini API error during clean: {e}")
            raise e

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def summarize(self, full_cleaned_text: str) -> str:
        if not full_cleaned_text.strip():
            return ""
            
        prompt = (
            "You are an expert content summarizer. Based on the following complete video transcript, "
            "generate a concise and accurate summary of the main events and topics discussed.\n\n"
            f"Transcript:\n{full_cleaned_text}"
        )
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SummarySchema,
                    temperature=0.3,
                )
            )
            self._update_usage(response)
            
            if hasattr(response, 'parsed') and response.parsed is not None:
                return response.parsed.summary
            else:
                import json
                data = json.loads(response.text)
                return data.get("summary", "")
                
        except Exception as e:
            logger.error(f"Gemini API error during summarize: {e}")
            raise e
