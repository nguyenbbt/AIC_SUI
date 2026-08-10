from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

from online.domain.diagnostics import QueryDiagnostics
from online.lifecycle import InfrastructureLifecycle
from online.modes.kis import KISSearchResult
from online.testing import build_advanced_runtime_bundle
from online.testing.advanced_composition import attach_advanced_fake_modes
from retrieval_api.composition import OnlineRuntime, create_runtime_app_from_env


class _KISSearch:
    def __init__(self, candidates) -> None:
        self._candidates = tuple(candidates)
        self.closed = False

    async def search(self, bundle):
        return KISSearchResult(
            candidates=self._candidates,
            diagnostics=QueryDiagnostics(
                query_id=bundle.query_id,
                total_latency_ms=0.0,
                stage_latencies_ms={},
                branches={},
                normalization_method="fake",
                fusion_method="fake",
                fusion_weights={},
            ),
        )

    def close(self, *, wait=True):
        self.closed = True


class _Closable:
    def __init__(self) -> None:
        self.closed = False

    def close(self, *, wait=True):
        self.closed = True


class _Executor:
    def __init__(self) -> None:
        self.closed = False

    def shutdown(self, *, wait=True, cancel_futures=True):
        self.closed = True


def test_production_lifespan_waits_for_blocked_vqa_before_closing_bundle() -> None:
    bundle = build_advanced_runtime_bundle(block_vqa=True, block_timeout_sec=2.0)
    kis = _KISSearch(bundle.fixture.ranked_vqa_candidates)
    retrieval = _Closable()
    executor = _Executor()
    runtime = OnlineRuntime(
        orchestrator=kis,
        lifecycle=InfrastructureLifecycle(),
        retrieval=retrieval,
        ranking_executor=executor,
    )
    attach_advanced_fake_modes(runtime, bundle)
    client = TestClient(create_runtime_app_from_env(runtime_factory=lambda: runtime))
    client.__enter__()

    question = bundle.fixture.vqa_question
    response_holder = []
    request_thread = threading.Thread(
        target=lambda: response_holder.append(
            client.post(
                "/internal/unstable/vqa",
                json={
                    "question_id": question.question_id,
                    "question": question.question,
                    "answer_type": question.answer_type.value,
                },
            )
        )
    )
    request_thread.start()
    deadline = time.monotonic() + 1.0
    while not bundle.vqa_started_event.is_set() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert bundle.vqa_started_event.is_set()

    shutdown_thread = threading.Thread(target=lambda: client.__exit__(None, None, None))
    shutdown_thread.start()
    shutdown_thread.join(timeout=0.05)
    assert shutdown_thread.is_alive()
    assert not bundle.closed

    bundle.release_vqa()
    request_thread.join(timeout=1.0)
    shutdown_thread.join(timeout=1.0)
    assert not request_thread.is_alive() and not shutdown_thread.is_alive()
    assert response_holder[0].status_code == 200
    assert response_holder[0].json()["question_id"] == question.question_id
    assert bundle.closed
    assert not bundle.vqa_lifecycle.closed_before_release
    assert kis.closed and retrieval.closed and executor.closed
    runtime.close()
