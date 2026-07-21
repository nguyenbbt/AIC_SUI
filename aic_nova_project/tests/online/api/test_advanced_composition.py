from __future__ import annotations

import asyncio

from online.domain.diagnostics import QueryDiagnostics
from online.lifecycle import InfrastructureLifecycle
from online.modes.kis import KISSearchResult
from online.testing import build_advanced_runtime_bundle
from retrieval_api.composition import OnlineRuntime, attach_advanced_modes


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


def test_runtime_close_waits_for_blocked_vqa_before_closing_bundle() -> None:
    async def scenario() -> None:
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
        attach_advanced_modes(runtime, bundle=bundle)

        request = asyncio.create_task(
            runtime.vqa_mode.answer(bundle.fixture.vqa_question)
        )
        await asyncio.wait_for(bundle.vqa_started_event.wait(), timeout=1.0)
        close_task = asyncio.create_task(asyncio.to_thread(runtime.close))
        await asyncio.sleep(0.02)
        assert not close_task.done()
        assert not bundle.closed

        bundle.release_vqa()
        result = await asyncio.wait_for(request, timeout=1.0)
        await asyncio.wait_for(close_task, timeout=1.0)

        assert result.question_id == bundle.fixture.vqa_question.question_id
        assert bundle.closed
        assert not bundle.vqa_lifecycle.closed_before_release
        assert kis.closed and retrieval.closed and executor.closed

    asyncio.run(scenario())
