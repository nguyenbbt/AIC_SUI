from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
from PIL import Image

from online.adapters.qwen_vlm import DEFAULT_QWEN_MODEL, QwenVLMAdapter
from online.domain.vqa import ImageEvidence, VQAAnswerType, VQAQuestion
from online.vqa.vlm_request import build_vlm_request
from query_understanding.openai_rewriter import DEFAULT_REWRITE_MODEL, OpenAIQueryRewriter
from query_understanding.rewrite import QueryRewriteRequest, RewritePurpose


def test_openai_rewriter_uses_structured_responses_output() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == DEFAULT_REWRITE_MODEL
        assert payload["reasoning"] == {"effort": "none"}
        return httpx.Response(200, json={"output_text": json.dumps({"primary_text": "a person beside a car", "paraphrases": ["human near vehicle"]})})

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://example.test")
        adapter = OpenAIQueryRewriter(api_key="test-key", client=client)
        result = await adapter.rewrite(QueryRewriteRequest(request_id="r1", purpose=RewritePurpose.KIS, text="người cạnh ô tô"))
        await client.aclose()
        return result

    result = asyncio.run(run())
    assert result.primary_text == "a person beside a car"
    assert result.model_id == DEFAULT_REWRITE_MODEL


def test_qwen_vlm_sends_resized_local_evidence_and_validates_grounding(tmp_path: Path) -> None:
    path = tmp_path / "keyframes/V001/001.jpg"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (1200, 600), "red").save(path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": DEFAULT_QWEN_MODEL}]})
        payload = json.loads(request.content)
        assert payload["chat_template_kwargs"]["enable_thinking"] is False
        assert payload["max_tokens"] == 256
        assert payload["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        content = json.dumps({"status": "answered", "answer": "có", "answer_type": "yes_no", "confidence": "high", "evidence_ids": ["image-1"]})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://qwen.test/v1")
    adapter = QwenVLMAdapter(data_root=tmp_path, client=client)
    adapter.health_check()
    evidence = ImageEvidence(evidence_id="image-1", video_id="V001", frame_id="V001_00000_001", shot_id=0, timestamp_sec=1, source_frame_idx=30, image_reference="keyframes/V001/001.jpg")
    request = build_vlm_request(VQAQuestion(question_id="q1", question="Có người không?", answer_type=VQAAnswerType.YES_NO), (evidence,))
    assert adapter.answer(request).answer == "có"
    client.close()
