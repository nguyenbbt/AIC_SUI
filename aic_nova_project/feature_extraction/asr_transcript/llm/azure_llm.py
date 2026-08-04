import os
import json
import logging
from tenacity import retry, wait_exponential, stop_after_attempt
from openai import AzureOpenAI
from .base import TranscriptLLM
from .summary_prompt import (
    SUMMARY_SYSTEM_PROMPT,
    build_summary_prompt,
    validate_summary_contract,
)

logger = logging.getLogger(__name__)

class AzureTranscriptLLM(TranscriptLLM):
    """
    Implementation of TranscriptLLM using Azure OpenAI models.
    """
    
    def __init__(self, model_name: str = "azure/gpt-4o"):
        # The user provided: MODEL=azure/gpt-4o, but the actual deployment name is likely needed.
        # Often model_name could just be the deployment name. 
        # We will strip 'azure/' if present to get the deployment name.
        self.deployment_name = model_name.replace("azure/", "") if model_name.startswith("azure/") else model_name
        
        api_key = os.environ.get("OPENAI_API_KEY")
        azure_endpoint = os.environ.get("BASE_URL")
        api_version = os.environ.get("API_VERSION", "2024-08-01-preview")
        
        if not api_key or not azure_endpoint:
            raise ValueError("OPENAI_API_KEY and BASE_URL environment variables are required for AzureTranscriptLLM.")
            
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )
        self.total_tokens_used = 0

    def _update_usage(self, response):
        if hasattr(response, 'usage') and response.usage:
            self.total_tokens_used += response.usage.total_tokens

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
            
        prompt += f"Raw ASR Text to clean:\n{raw_text}\n\n"
        prompt += "Return the result as a JSON object with a single key 'cleaned_text'."
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You clean ASR transcripts and return valid JSON "
                            "with exactly one key named cleaned_text."
                        ),
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            self._update_usage(response)
            
            result_text = response.choices[0].message.content
            data = json.loads(result_text)
            return data.get("cleaned_text", raw_text)
                
        except Exception as e:
            logger.error(f"Azure API error during clean: {e}")
            raise e

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    def summarize(self, full_cleaned_text: str) -> str:
        if not full_cleaned_text.strip():
            return ""
            
        prompt = build_summary_prompt(full_cleaned_text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            self._update_usage(response)
            
            result_text = response.choices[0].message.content
            data = json.loads(result_text)
            return validate_summary_contract(
                data.get("summary", ""),
                full_cleaned_text,
            )
                
        except Exception as e:
            logger.error(f"Azure API error during summarize: {e}")
            raise e
