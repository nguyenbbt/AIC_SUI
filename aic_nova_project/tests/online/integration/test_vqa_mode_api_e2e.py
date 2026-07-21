from __future__ import annotations

from fastapi.testclient import TestClient

from online.config import OnlineDataConfig
from online.domain.vqa import (
    ImageEvidence,
    VLMConfidence,
    VLMResponse,
    VLMResponseStatus,
)
from online.testing import (
    FakeEvidenceHydrator,
    FakeImageResolver,
    FakeTextEncoder,
    build_advanced_runtime_bundle,
    build_integration_fixture,
)
from retrieval_api.composition import (
    RuntimeCompositionConfig,
    attach_advanced_modes,
    build_online_runtime,
)
from retrieval_api.search_engine import create_app


class _GroundedVLM:
    def answer(self, request):
        return VLMResponse(
            status=VLMResponseStatus.ANSWERED,
            answer="Grounded fake answer",
            answer_type=request.question.answer_type,
            confidence=VLMConfidence.HIGH,
            evidence_ids=(request.evidence[0].evidence_id,),
        )


def test_vqa_internal_route_uses_real_rewrite_and_seven_branch_kis_handoff() -> None:
    kis_fixture = build_integration_fixture()
    metadata = kis_fixture.metadata()
    images = {
        frame.frame_id: ImageEvidence(
            evidence_id=f"image:{frame.frame_id}",
            video_id=frame.video_id,
            frame_id=frame.frame_id,
            shot_id=frame.shot_id,
            timestamp_sec=frame.timestamp_sec,
            image_reference=f"fixture://wave3/{frame.frame_id}",
        )
        for frame in kis_fixture.frames
    }
    bundle = build_advanced_runtime_bundle(
        metadata_reader=metadata,
        evidence_hydrator=FakeEvidenceHydrator(),
        image_resolver=FakeImageResolver(images),
        vlm=_GroundedVLM(),
    )
    runtime = build_online_runtime(
        data_config=OnlineDataConfig(),
        runtime_config=RuntimeCompositionConfig(default_top_k=5),
        milvus=kis_fixture.milvus(),
        elasticsearch=kis_fixture.elasticsearch(),
        metadata=metadata,
        object_reader=kis_fixture.object_reader(),
        visual_encoder=FakeTextEncoder(dimension=4),
        vietnamese_encoder=FakeTextEncoder(dimension=6),
    )
    attach_advanced_modes(runtime, bundle=bundle)

    try:
        question = bundle.fixture.vqa_question
        response = TestClient(
            create_app(
                orchestrator=runtime.orchestrator,
                trake_mode=runtime.trake_mode,
                vqa_mode=runtime.vqa_mode,
            )
        ).post(
            "/internal/unstable/vqa",
            json={
                "question_id": question.question_id,
                "question": question.question,
                "answer_type": question.answer_type.value,
            },
        )
    finally:
        runtime.close()

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["question_id"] == question.question_id
    assert result["response"]["status"] == "answered"
    assert set(result["response"]["evidence_ids"]).issubset(
        {item["evidence_id"] for item in result["evidence"]}
    )
    operations = {call.operation for call in bundle.calls}
    assert {"resolve_images", "answer"}.issubset(operations)
