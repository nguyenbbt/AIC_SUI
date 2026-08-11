"""OpenAI Responses API adapter for optional query rewriting."""

from __future__ import annotations

import json
from typing import Any

import httpx

from online.domain.errors import ResourceUnavailableError
from query_understanding.rewrite import (
    QueryRewriteProposal,
    QueryRewriteRequest,
    RewritePurpose,
)


DEFAULT_REWRITE_MODEL = "gpt-5.4-mini-2026-03-17"
PROMPT_VERSION = "aic-query-rewrite-v2"


class OpenAIQueryRewriter:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_REWRITE_MODEL,
        base_url: str = "https://api.openai.com/v1",
        timeout_sec: float = 4.5,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenAI query rewrite requires a non-empty API key")
        self._model = model
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = timeout_sec
        self._headers = {"Authorization": f"Bearer {api_key}"}

    async def rewrite(self, request: QueryRewriteRequest) -> QueryRewriteProposal:
        instructions = (
            "You rewrite queries for a video keyframe retrieval system. Preserve every named entity, "
            "visible object, action, attribute, count, spatial/temporal relation, quoted/OCR-like text, "
            "and uncertainty from the input. Never add a color, place, object, action, count, identity, "
            "or event not supported by the input. Do not answer questions. Do not add boilerplate such "
            "as 'khung hình có', 'visual scene', 'image of', or 'scene showing'. Return JSON only. "
            + (
                "For KIS: primary_text is q1, a natural Vietnamese visual description organized as "
                "subject-action-attributes-relations-setting. paraphrases must contain exactly one item: "
                "q2, a concise English caption optimized for OpenCLIP visual retrieval. q1 and q2 must "
                "both be meaning-preserving, and must be lexically distinct from the original q0 and "
                "from each other."
                if request.purpose is RewritePurpose.KIS
                else "For VQA evidence retrieval: primary_text describes only the visual evidence that "
                "could answer the question, without proposing an answer. paraphrases contains exactly "
                "one concise English visual-retrieval caption with the same evidence need."
            )
        )
        payload = {
            "model": self._model,
            "reasoning": {"effort": "none"},
            "instructions": instructions,
            "input": request.text,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "aic_query_rewrite",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "primary_text": {"type": "string"},
                            "paraphrases": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "maxItems": 1,
                            },
                        },
                        "required": ["primary_text", "paraphrases"],
                        "additionalProperties": False,
                    },
                }
            },
            "max_output_tokens": 256,
        }
        try:
            if self._client is not None:
                response = await self._client.post("/responses", json=payload)
            else:
                async with httpx.AsyncClient(
                    base_url=self._base_url,
                    timeout=self._timeout_sec,
                    headers=self._headers,
                ) as client:
                    response = await client.post("/responses", json=payload)
            response.raise_for_status()
            data = response.json()
            output = json.loads(_responses_output_text(data))
            if (
                not isinstance(output, dict)
                or not isinstance(output.get("primary_text"), str)
                or not output["primary_text"].strip()
                or not isinstance(output.get("paraphrases"), list)
                or len(output["paraphrases"]) > 2
                or any(not isinstance(value, str) for value in output["paraphrases"])
            ):
                raise ValueError("OpenAI rewrite violates the structured output contract")
            return QueryRewriteProposal(
                primary_text=output["primary_text"],
                paraphrases=tuple(output["paraphrases"]),
                provider_id="openai",
                model_id=self._model,
                prompt_version=PROMPT_VERSION,
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ResourceUnavailableError(
                "OpenAI query rewrite provider failed",
                details={"resource": "query_rewrite"},
            ) from exc

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


__all__ = ["DEFAULT_REWRITE_MODEL", "OpenAIQueryRewriter", "PROMPT_VERSION"]
