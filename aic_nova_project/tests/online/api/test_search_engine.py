from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from online.domain.candidates import CandidateDiagnostics, FusedFrameCandidate
from online.domain.diagnostics import BranchDiagnostics, QueryDiagnostics
from online.domain.enums import BranchStatus, RetrievalBranch
from online.domain.errors import ContractMismatchError, ResourceUnavailableError
from online.modes.kis import KISSearchResult
from retrieval_api.search_engine import competition_candidates, create_app


def fused_frame(frame_id: str = "V001_00000_015") -> FusedFrameCandidate:
    return FusedFrameCandidate(
        frame_id=frame_id,
        video_id="V001",
        shot_id=0,
        timestamp_sec=1.5,
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
        normalization_method="rrf_query_variant_aggregation",
        fusion_method="weighted_mean_normalized",
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
        self.assertEqual(payload["candidates"][0]["frame_id"], "V001_00000_015")
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
        self.assertNotIn("secret", str(unavailable.json()["error"]["details"]))

        mismatch = TestClient(
            create_app(orchestrator=FakeOrchestrator(error=ContractMismatchError("bad contract")))
        ).post("/search", json={"query": "query"})
        self.assertEqual(mismatch.status_code, 409)

    def test_competition_candidates_adapter_is_minimal_and_stable(self) -> None:
        adapted = competition_candidates((fused_frame(),))

        self.assertEqual(
            adapted,
            (
                {
                    "frame_id": "V001_00000_015",
                    "video_id": "V001",
                    "timestamp_sec": 1.5,
                    "score": 0.9,
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
