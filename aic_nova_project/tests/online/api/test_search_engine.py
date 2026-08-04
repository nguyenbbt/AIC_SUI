from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from online.domain.candidates import CandidateDiagnostics, FusedFrameCandidate
from online.domain.diagnostics import BranchDiagnostics, QueryDiagnostics
from online.domain.enums import BranchStatus, RetrievalBranch
from online.domain.errors import ContractMismatchError, ResourceUnavailableError
from online.modes.kis import KISSearchResult
from retrieval_api.search_engine import (
    KISCompetitionRow,
    create_app,
    serialize_kis_competition_candidates,
)


def fused_frame(
    frame_id: str = "L21_V001_001",
    *,
    source_frame_idx: int = 15,
) -> FusedFrameCandidate:
    video_id, keyframe_text = frame_id.rsplit("_", 1)
    keyframe_no = int(keyframe_text)
    return FusedFrameCandidate(
        frame_id=frame_id,
        video_id=video_id,
        keyframe_no=keyframe_no,
        local_index=keyframe_no - 1,
        timestamp_sec=1.5,
        source_frame_idx=source_frame_idx,
        final_score=0.9,
        branch_scores={RetrievalBranch.VISUAL_DENSE: 0.9},
        evidence=(),
        diagnostics=CandidateDiagnostics(),
    )


def diagnostics(query_id: str) -> QueryDiagnostics:
    return QueryDiagnostics(
        query_id=query_id,
        total_latency_ms=1.0,
        stage_latencies_ms={"fusion": 0.5},
        branches={
            RetrievalBranch.VISUAL_DENSE: BranchDiagnostics(
                status=BranchStatus.SUCCESS,
                latency_ms=1.0,
                requested_top_k=10,
                raw_result_count=1,
                output_candidate_count=1,
            )
        },
        normalization_method="rrf",
        fusion_method="experimental_weighted_sum_normalized_v1",
        fusion_weights={RetrievalBranch.VISUAL_DENSE: 1.0},
    )


class FakeOrchestrator:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls = []

    async def search(self, bundle):
        self.calls.append(bundle)
        if self.error is not None:
            raise self.error
        return KISSearchResult(
            candidates=(fused_frame(),),
            diagnostics=diagnostics(bundle.query_id),
        )


class SearchEngineAPITests(unittest.TestCase):
    def test_health_live_and_ready(self) -> None:
        client = TestClient(create_app(orchestrator=FakeOrchestrator()))

        self.assertEqual(client.get("/health/live").json()["status"], "healthy")
        self.assertEqual(client.get("/health/ready").json()["status"], "ready")

    def test_ready_reports_missing_orchestrator_as_503(self) -> None:
        client = TestClient(create_app())

        response = client.get("/health/ready")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "RESOURCE_UNAVAILABLE")

    def test_search_routes_kis_query_and_hides_diagnostics_by_default(self) -> None:
        orchestrator = FakeOrchestrator()
        client = TestClient(create_app(orchestrator=orchestrator))

        response = client.post(
            "/search",
            json={
                "query": "person near a car",
                "mode": "kis_text",
                "paraphrases": ["human beside vehicle"],
                "query_id": "api-query",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query_id"], "api-query")
        self.assertEqual(payload["candidates"][0]["frame_id"], "L21_V001_001")
        self.assertEqual(payload["candidates"][0]["source_frame_idx"], 15)
        self.assertIsNone(payload["diagnostics"])
        self.assertEqual(orchestrator.calls[0].text_variants[1].text, "human beside vehicle")

    def test_search_can_include_diagnostics(self) -> None:
        client = TestClient(create_app(orchestrator=FakeOrchestrator()))

        response = client.post(
            "/search",
            json={
                "query": "query",
                "include_diagnostics": True,
                "query_id": "diag-query",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["diagnostics"]["query_id"], "diag-query")

    def test_validation_and_domain_errors_are_safe(self) -> None:
        client = TestClient(create_app(orchestrator=FakeOrchestrator()))
        validation = client.post("/search", json={"query": " "})
        self.assertEqual(validation.status_code, 422)

        unavailable = TestClient(
            create_app(orchestrator=FakeOrchestrator(error=ResourceUnavailableError("secret backend down")))
        ).post("/search", json={"query": "query"})
        self.assertEqual(unavailable.status_code, 503)
        self.assertEqual(unavailable.json()["error"]["message"], "A required search resource is unavailable")
        self.assertNotIn("secret", unavailable.json()["error"]["message"])
        self.assertNotIn("secret", str(unavailable.json()["error"]["details"]))

        mismatch = TestClient(
            create_app(orchestrator=FakeOrchestrator(error=ContractMismatchError("bad contract")))
        ).post("/search", json={"query": "query"})
        self.assertEqual(mismatch.status_code, 409)

    def test_public_error_details_are_allowlisted(self) -> None:
        error = ResourceUnavailableError(
            "TOP_SECRET raw message",
            details={
                "branch": "visual_dense",
                "reason": "token=TOP_SECRET",
                "authorization": "Bearer TOP_SECRET",
                "uri": "http://localhost/private/path?token=TOP_SECRET",
                "payload": {"token": "TOP_SECRET"},
                "items": ("TOP_SECRET",),
            },
        )
        response = TestClient(create_app(orchestrator=FakeOrchestrator(error=error))).post(
            "/search",
            json={"query": "query"},
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["details"], {"branch": "visual_dense"})
        self.assertNotIn("TOP_SECRET", response.text)
        self.assertNotIn("reason", response.text)

        malicious_allowed_value = ResourceUnavailableError(
            "safe public message only",
            details={
                "branch": "token=TOP_SECRET",
                "constraint": "TOP_SECRET",
                "frame_id": "TOP_SECRET",
                "resource": "token=TOP_SECRET",
            },
        )
        redacted = TestClient(
            create_app(orchestrator=FakeOrchestrator(error=malicious_allowed_value))
        ).post("/search", json={"query": "query"})
        self.assertEqual(redacted.json()["error"]["details"], {})
        self.assertNotIn("TOP_SECRET", redacted.text)

    def test_api_leaves_core_branch_policy_to_the_orchestrator(self) -> None:
        orchestrator = FakeOrchestrator()
        response = TestClient(create_app(orchestrator=orchestrator)).post(
            "/search",
            json={"query": "query", "enabled_branches": ["ocr_bm25"]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            orchestrator.calls[0].enabled_branches,
            (RetrievalBranch.OCR_BM25,),
        )

    def test_kis_competition_serializer_has_golden_field_names_types_and_order(self) -> None:
        adapted = serialize_kis_competition_candidates((fused_frame(),))

        self.assertEqual(
            adapted,
            (
                KISCompetitionRow(video_id="L21_V001", source_frame_idx=15),
            ),
        )
        self.assertEqual(
            [row.model_dump() for row in adapted],
            [{"video_id": "L21_V001", "source_frame_idx": 15}],
        )
        self.assertEqual(tuple(adapted[0].model_dump()), ("video_id", "source_frame_idx"))
        self.assertIsInstance(adapted[0].video_id, str)
        self.assertIsInstance(adapted[0].source_frame_idx, int)

    def test_kis_competition_serializer_reads_source_field_and_rejects_underdeduped_input(self) -> None:
        first = fused_frame(source_frame_idx=901)
        second = fused_frame("L21_V001_002", source_frame_idx=901)
        self.assertEqual(
            serialize_kis_competition_candidates((first,))[0].source_frame_idx,
            901,
        )
        with self.assertRaises(ContractMismatchError):
            serialize_kis_competition_candidates((first, second))


if __name__ == "__main__":
    unittest.main()
