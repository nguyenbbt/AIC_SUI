"""OpenAI-compatible structured adapter for optional query rewriting."""

from __future__ import annotations

import json
import math
from typing import Any

import httpx

from online.domain.errors import ResourceUnavailableError
from query_understanding.rewrite import (
    QueryRewriteProposal,
    QueryRewriteRequest,
    RewritePurpose,
)


DEFAULT_REWRITE_MODEL = "gpt-5.4-mini-2026-03-17"
DEFAULT_REWRITE_API_MODE = "responses"
PROMPT_VERSION = "aic-query-rewrite-v3-vi-q1-only"
_REWRITE_API_MODES = frozenset({"responses", "chat_completions"})
_REWRITE_JSON_SCHEMA = {
    "type": "object",
    "properties": {"primary_text": {"type": "string"}},
    "required": ["primary_text"],
    "additionalProperties": False,
}


class OpenAIQueryRewriter:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_REWRITE_MODEL,
        base_url: str = "https://api.openai.com/v1",
        api_mode: str = DEFAULT_REWRITE_API_MODE,
        timeout_sec: float = 4.5,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("OpenAI query rewrite requires a non-empty API key")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("OpenAI query rewrite requires a non-empty model")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("OpenAI query rewrite requires a non-empty base_url")
        if (
            not isinstance(api_mode, str)
            or api_mode.strip().lower() not in _REWRITE_API_MODES
        ):
            raise ValueError(
                "OpenAI query rewrite api_mode must be responses or chat_completions"
            )
        if (
            isinstance(timeout_sec, bool)
            or not isinstance(timeout_sec, (int, float))
            or not math.isfinite(timeout_sec)
            or timeout_sec <= 0
        ):
            raise ValueError("OpenAI query rewrite timeout_sec must be positive and finite")
        self._model = model.strip()
        self._client = client
        self._base_url = base_url.strip().rstrip("/")
        self._api_mode = api_mode.strip().lower()
        self._timeout_sec = float(timeout_sec)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    async def rewrite(self, request: QueryRewriteRequest) -> QueryRewriteProposal:
        instructions = (
            "You rewrite queries for a video keyframe retrieval system. Preserve every named entity, "
            "visible object, action, attribute, count, spatial/temporal relation, quoted/OCR-like text, "
            "and uncertainty from the input. Never add a color, place, object, action, count, identity, "
            "or event not supported by the input. Do not answer questions. Do not add boilerplate such "
            "as 'khung hình có', 'visual scene', 'image of', or 'scene showing'. Return JSON only. "
            + (
                "For KIS: primary_text is the single q1, a natural Vietnamese visual description "
                "organized as subject-action-attributes-relations-setting. It must preserve meaning "
                "while being lexically useful and distinct from the original q0. Do not translate it "
                "to English and do not produce a second variant."
                if request.purpose is RewritePurpose.KIS
                else "For VQA evidence retrieval: primary_text describes only the visual evidence that "
                "could answer the question, without proposing an answer. Produce only this one "
                "Vietnamese evidence query and do not produce a second variant."
            )
        )
        path, payload = self._request_payload(instructions, request.text)
        try:
            if self._client is not None:
                response = await self._client.post(path, json=payload)
            else:
                async with httpx.AsyncClient(
                    base_url=self._base_url,
                    timeout=self._timeout_sec,
                    headers=self._headers,
                ) as client:
                    response = await client.post(path, json=payload)
            response.raise_for_status()
            data = response.json()
            output_text = (
                _responses_output_text(data)
                if self._api_mode == "responses"
                else _chat_completions_output_text(data)
            )
            output = json.loads(output_text)
            if (
                not isinstance(output, dict)
                or not isinstance(output.get("primary_text"), str)
                or not output["primary_text"].strip()
            ):
                raise ValueError("OpenAI rewrite violates the structured output contract")
            return QueryRewriteProposal(
                primary_text=output["primary_text"],
                paraphrases=(),
                provider_id="openai",
                model_id=self._model,
                prompt_version=PROMPT_VERSION,
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ResourceUnavailableError(
                "OpenAI query rewrite provider failed",
                details={"resource": "query_rewrite"},
            ) from exc

    def _request_payload(
        self,
        instructions: str,
        text: str,
    ) -> tuple[str, dict[str, Any]]:
        if self._api_mode == "responses":
            return "/responses", {
                "model": self._model,
                "reasoning": {"effort": "none"},
                "instructions": instructions,
                "input": text,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "aic_query_rewrite",
                        "strict": True,
                        "schema": _REWRITE_JSON_SCHEMA,
                    }
                },
                "max_output_tokens": 256,
            }
        return "/chat/completions", {
            "model": self._model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": text},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "aic_query_rewrite",
                    "strict": True,
                    "schema": _REWRITE_JSON_SCHEMA,
                },
            },
            "max_completion_tokens": 256,
        }

    async def aclose(self) -> None:
        """Injected clients are owned and closed by their caller."""


def _responses_output_text(data: Any) -> str:
    if not isinstance(data, dict):
        raise ValueError("invalid Responses API payload")
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                return content["text"]
    raise ValueError("Responses API output text is missing")


def _chat_completions_output_text(data: Any) -> str:
    if not isinstance(data, dict):
        raise ValueError("invalid Chat Completions API payload")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Chat Completions API choices are missing")
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Chat Completions API output text is missing")
    return content


__all__ = [
    "DEFAULT_REWRITE_API_MODE",
    "DEFAULT_REWRITE_MODEL",
    "OpenAIQueryRewriter",
    "PROMPT_VERSION",
]
