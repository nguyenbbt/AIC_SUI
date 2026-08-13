"""Local OpenAI-compatible Qwen3.5 VLM adapter."""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
import re
from typing import Any

import httpx
from PIL import Image

from online.domain.errors import ContractMismatchError, ResourceUnavailableError
from online.domain.vqa import EvidenceType, VLMRequest, VLMResponse
from online.vqa.vlm_request import EVIDENCE_ONLY_INSTRUCTION, validate_vlm_response


DEFAULT_QWEN_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_QWEN_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


class QwenVLMAdapter:
    def __init__(
        self,
        *,
        data_root: Path,
        base_url: str = "http://localhost:8001/v1",
        model: str = DEFAULT_QWEN_MODEL,
        revision: str = DEFAULT_QWEN_REVISION,
        timeout_sec: float = 15.0,
        max_image_long_edge: int = 768,
        client: httpx.Client | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Qwen model must be non-empty")
        if (
            not isinstance(revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", revision) is None
        ):
            raise ValueError("Qwen revision must be a pinned 40-character commit SHA")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("Qwen base_url must be non-empty")
        if (
            isinstance(timeout_sec, bool)
            or not isinstance(timeout_sec, (int, float))
            or not math.isfinite(timeout_sec)
            or timeout_sec <= 0
        ):
            raise ValueError("Qwen timeout_sec must be a positive finite number")
        if (
            isinstance(max_image_long_edge, bool)
            or not isinstance(max_image_long_edge, int)
            or max_image_long_edge < 64
        ):
            raise ValueError("Qwen max_image_long_edge must be an integer >= 64")
        self.model = model.strip()
        self.revision = revision
        self._data_root = Path(data_root).expanduser().resolve()
        self._max_image_long_edge = max_image_long_edge
        self._owns_client = client is None
        self._client = client or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_sec)

    def connect(self) -> None:
        self.health_check()

    def health_check(self) -> None:
        try:
            response = self._client.get("/models")
            response.raise_for_status()
            models = tuple(
                item for item in response.json().get("data", []) if isinstance(item, dict)
            )
            served = next((item for item in models if item.get("id") == self.model), None)
            if served is None:
                raise ContractMismatchError("Configured Qwen model is not served")
            served_revision = served.get("revision") or served.get("model_revision")
            if served_revision is not None and served_revision != self.revision:
                raise ContractMismatchError(
                    "Served Qwen model revision does not match configuration"
                )
        except ContractMismatchError:
            raise
        except (httpx.HTTPError, AttributeError, TypeError, ValueError) as exc:
            raise ResourceUnavailableError(
                "Qwen VLM service is unavailable", details={"resource": "qwen_vlm"}
            ) from exc

    def close(self, *, wait: bool = True) -> None:
        del wait
        if self._owns_client:
            self._client.close()

    def answer(self, request: VLMRequest) -> VLMResponse:
        content: list[dict[str, Any]] = []
        public_evidence: list[dict[str, Any]] = []
        for evidence in request.evidence:
            dumped = evidence.model_dump(mode="json", exclude={"image_reference"})
            public_evidence.append(dumped)
            if evidence.evidence_type is EvidenceType.IMAGE:
                image_reference = getattr(evidence, "image_reference")
                content.append({
                    "type": "text",
                    "text": f"Evidence image ID: {evidence.evidence_id}",
                })
                content.append({
                    "type": "image_url",
                    "image_url": {"url": self._image_data_url(image_reference)},
                })
        content.append({
            "type": "text",
            "text": json.dumps(
                {
                    "question": request.question.model_dump(mode="json"),
                    "evidence": public_evidence,
                },
                ensure_ascii=False,
            ),
        })
        schema = VLMResponse.model_json_schema()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": EVIDENCE_ONLY_INSTRUCTION},
                {"role": "user", "content": content},
            ],
            "temperature": request.temperature,
            "max_tokens": min(request.max_output_tokens, 256),
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "aic_vqa_response", "strict": True, "schema": schema},
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            response = self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
            parsed = json.loads(raw)
        except (httpx.HTTPError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ResourceUnavailableError(
                "Qwen VLM request failed", details={"resource": "qwen_vlm"}
            ) from exc
        return validate_vlm_response(parsed, request)

    def _image_data_url(self, reference: str) -> str:
        path = (self._data_root / Path(reference)).resolve()
        if not path.is_relative_to(self._data_root):
            raise ContractMismatchError("VLM image reference escapes DATA_ROOT")
        if not path.is_file():
            raise ResourceUnavailableError("VLM evidence image is unavailable")
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((self._max_image_long_edge, self._max_image_long_edge))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=85, optimize=True)
        except (OSError, ValueError) as exc:
            raise ResourceUnavailableError("VLM evidence image cannot be decoded") from exc
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"


__all__ = ["DEFAULT_QWEN_MODEL", "DEFAULT_QWEN_REVISION", "QwenVLMAdapter"]
