import logging
import torch
import re
from transformers import pipeline
from .base import TranscriptLLM
from .cleaning_prompt import build_cleaning_prompt
from .summary_prompt import (
    SummaryContractError,
    SUMMARY_SYSTEM_PROMPT,
    build_summary_contract_repair_prompt,
    build_summary_language_repair_prompt,
    build_summary_prompt,
    validate_summary_contract,
)

logger = logging.getLogger(__name__)

_MAX_SUMMARY_CONTRACT_ATTEMPTS = 3
_MAX_SUMMARY_FORMAT_REPAIRS = 1


class LocalTranscriptLLM(TranscriptLLM):
    """
    Implementation using a local open-source model (e.g. Qwen2.5-7B-Instruct).
    Requires transformers and accelerate. Runs completely offline.
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",
    ):
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
            
        raise ValueError(f"Model output does not contain JSON field {field_name!r}")

    def clean(self, raw_text: str, context: str = "") -> str:
        if not raw_text.strip():
            return ""
            
        messages = [
            {
                "role": "system",
                "content": "Tuân thủ chính xác hợp đồng hiệu đính transcript.",
            },
            {"role": "user", "content": build_cleaning_prompt(raw_text, context)},
        ]
        
        try:
            max_tokens = max(
                128,
                min(1024, int(len(raw_text.split()) * 2.5) + 64),
            )
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
            
        messages = [
            {
                "role": "system",
                "content": SUMMARY_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_summary_prompt(full_cleaned_text),
            },
        ]
        
        try:
            contract_attempt = 0
            format_repairs = 0
            while contract_attempt < _MAX_SUMMARY_CONTRACT_ATTEMPTS:
                outputs = self.generator(
                    messages,
                    max_new_tokens=512,
                    do_sample=False,
                )

                generated_text = outputs[0]["generated_text"][-1][
                    "content"
                ]
                try:
                    summary = self._extract_json_field(
                        generated_text,
                        "summary",
                    )
                except ValueError as error:
                    output_preview = " ".join(
                        generated_text.split()
                    )[:240]
                    logger.warning(
                        "summary_json_parse_failed "
                        "format_repair=%d output_chars=%d "
                        "output_preview=%r",
                        format_repairs + 1,
                        len(generated_text),
                        output_preview,
                        extra={
                            "format_repair": format_repairs + 1,
                            "output_chars": len(generated_text),
                            "output_preview": output_preview,
                        },
                    )
                    if format_repairs == _MAX_SUMMARY_FORMAT_REPAIRS:
                        raise
                    format_repairs += 1
                    logger.warning(
                        "Local LLM returned malformed summary JSON; "
                        "requesting one targeted format repair."
                    )
                    messages = [
                        {
                            "role": "system",
                            "content": SUMMARY_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": build_summary_contract_repair_prompt(
                                generated_text,
                                full_cleaned_text,
                                str(error),
                            ),
                        },
                    ]
                    continue

                contract_attempt += 1
                try:
                    return validate_summary_contract(
                        summary,
                        full_cleaned_text,
                    )
                except SummaryContractError as error:
                    normalized_summary = summary.strip()
                    summary_preview = " ".join(
                        normalized_summary.split()
                    )[:240]
                    logger.warning(
                        "summary_contract_validation_failed "
                        "attempt=%d violation=%s summary_chars=%d "
                        "summary_preview=%r",
                        contract_attempt,
                        error.code,
                        len(normalized_summary),
                        summary_preview,
                        extra={
                            "attempt": contract_attempt,
                            "violation": error.code,
                            "summary_chars": len(normalized_summary),
                            "summary_preview": summary_preview,
                        },
                    )
                    if (
                        contract_attempt
                        == _MAX_SUMMARY_CONTRACT_ATTEMPTS
                    ):
                        raise
                    logger.warning(
                        "Local LLM returned an invalid summary; "
                        "requesting one targeted rewrite."
                    )
                    messages = [
                        {
                            "role": "system",
                            "content": SUMMARY_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": (
                                build_summary_language_repair_prompt(summary)
                                if error.code == "language"
                                else build_summary_contract_repair_prompt(
                                    summary,
                                    full_cleaned_text,
                                    str(error),
                                )
                            ),
                        },
                    ]
            
        except Exception as e:
            logger.error(f"Local LLM error during summarize: {e}")
            raise e
