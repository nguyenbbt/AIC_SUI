"""Mockable multimodal model boundary for evidence-grounded VQA."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from online.domain.vqa import VLMRequest, VLMResponse


@runtime_checkable
class VLMPort(Protocol):
    def answer(self, request: VLMRequest) -> VLMResponse: ...
