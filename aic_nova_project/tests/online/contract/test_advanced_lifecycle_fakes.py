from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from online.domain.errors import BranchTimeoutError, ResourceUnavailableError
from online.testing import (
    AdvancedRuntimeConfig,
    BlockingLifecycleFake,
    LifecycleEvent,
    build_advanced_runtime_bundle,
)


class AdvancedLifecycleFakeTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_gate_blocks_until_release_and_close_is_idempotent(self) -> None:
        fake = BlockingLifecycleFake("async-test", timeout_sec=2.0)
        task = asyncio.create_task(fake.execute("request-async"))
        await asyncio.wait_for(fake.started_event.wait(), timeout=1.0)
        self.assertFalse(task.done())
        with self.assertRaises(ResourceUnavailableError):
            fake.close()
        self.assertTrue(fake.close_attempted_while_active)
        fake.release_event.set()
        await asyncio.wait_for(task, timeout=1.0)
        fake.close()
        fake.close()
        fake.release_request()
        self.assertEqual(fake.close_count, 1)
        self.assertEqual(fake.release_count, 1)
        self.assertEqual(fake.active_count, 0)
        self.assertIsNotNone(fake.close_called_at_ns)
        self.assertIsNotNone(fake.last_completion_at_ns)

    async def test_sync_visual_port_uses_asyncio_event_bridge(self) -> None:
        bundle = build_advanced_runtime_bundle(
            config=AdvancedRuntimeConfig(block_trake=True, block_timeout_sec=2.0)
        )
        try:
            task = asyncio.create_task(
                asyncio.to_thread(
                    bundle.visual_corpus.iter_ordered_frame_embedding_batches,
                    "V001",
                    2,
                )
            )
            await asyncio.wait_for(bundle.trake_started_event.wait(), timeout=1.0)
            self.assertFalse(task.done())
            with self.assertRaises(ResourceUnavailableError):
                bundle.close()
            self.assertTrue(bundle.trake_lifecycle.close_attempted_while_active)
            bundle.trake_release_event.set()
            batches = await asyncio.wait_for(task, timeout=1.0)
            self.assertEqual(batches[0][0].video_id, "V001")
            bundle.close()
            bundle.close()
            self.assertEqual(bundle.close_count, 1)
        finally:
            if not bundle.closed:
                bundle.release_all()
                await bundle.trake_lifecycle.wait_until_idle(timeout_sec=1.0)
                await bundle.vqa_lifecycle.wait_until_idle(timeout_sec=1.0)
                bundle.close()

    async def test_parent_tracks_active_request_scoped_view(self) -> None:
        parent = build_advanced_runtime_bundle(
            config=AdvancedRuntimeConfig(block_trake=True, block_timeout_sec=2.0)
        )
        child = parent.for_request("scoped-request")
        try:
            task = asyncio.create_task(
                asyncio.to_thread(
                    child.visual_corpus.iter_ordered_frame_embedding_batches,
                    "V001",
                    2,
                )
            )
            await asyncio.wait_for(parent.trake_started_event.wait(), timeout=1.0)
            self.assertEqual(parent.trake_lifecycle.active_count, 1)
            with self.assertRaises(ResourceUnavailableError):
                parent.close()
            self.assertFalse(parent.closed)
            parent.release_trake()
            await asyncio.wait_for(task, timeout=1.0)
            child.close()
            self.assertFalse(parent.closed)
            parent.close()
            self.assertTrue(parent.closed)
        finally:
            parent.release_all()
            if not child.closed:
                child.close()
            if not parent.closed:
                await parent.wait_until_idle(timeout_sec=1.0)
                parent.close()

    async def test_root_close_invalidates_existing_request_scopes(self) -> None:
        parent = build_advanced_runtime_bundle()
        child = parent.for_request("scoped-request")
        parent.close()
        self.assertTrue(parent.closed)
        self.assertTrue(child.closed)
        with self.assertRaises(ResourceUnavailableError):
            child.health_check()
        with self.assertRaises(ResourceUnavailableError):
            child.connect()
        with self.assertRaises(ResourceUnavailableError):
            child.for_request("descendant")
        with self.assertRaises(ResourceUnavailableError):
            child.visual_corpus.list_video_ids()
        child.close()

    async def test_sync_vlm_port_blocks_and_release_is_request_scoped(self) -> None:
        bundle = build_advanced_runtime_bundle(
            config=AdvancedRuntimeConfig(block_vqa=True, block_timeout_sec=2.0)
        )
        fixture = bundle.fixture
        records = (
            tuple(fixture.images_by_frame_id.values())
            + fixture.ocr_evidence
            + fixture.asr_evidence
            + fixture.summary_evidence
        )
        by_id = {record.evidence_id: record for record in records}
        from online.domain.vqa import VLMRequest

        request = VLMRequest(
            request_id="vqa-blocked",
            question=fixture.vqa_question,
            evidence=tuple(
                by_id[evidence_id]
                for evidence_id in fixture.expected_vqa_answer_evidence_ids
            ),
        )
        try:
            task = asyncio.create_task(asyncio.to_thread(bundle.vlm.answer, request))
            await asyncio.wait_for(bundle.vqa_started_event.wait(), timeout=1.0)
            self.assertFalse(task.done())
            bundle.release_vqa()
            response = await asyncio.wait_for(task, timeout=1.0)
            self.assertEqual(response.evidence_ids, fixture.expected_vqa_answer_evidence_ids)
            bundle.close()
        finally:
            if not bundle.closed:
                bundle.release_all()
                await bundle.vqa_lifecycle.wait_until_idle(timeout_sec=1.0)
                bundle.close()

    async def test_timeout_releases_active_count_and_can_be_closed(self) -> None:
        fake = BlockingLifecycleFake("timeout-test", timeout_sec=0.05)
        with self.assertRaises(BranchTimeoutError):
            await fake.execute("request-timeout")
        self.assertEqual(fake.active_count, 0)
        fake.close()
        self.assertEqual(fake.close_count, 1)

    async def test_lifecycle_return_value_is_defensively_immutable(self) -> None:
        original = bytearray(b"fixture")
        fake = BlockingLifecycleFake(return_value=original)
        original.extend(b"-mutated")
        fake.release_request()
        result = await fake.execute("request-immutable")
        self.assertEqual(result, b"fixture")
        self.assertIsInstance(result, bytes)
        fake.close()

    async def test_concurrent_bundle_close_is_idempotent(self) -> None:
        bundle = build_advanced_runtime_bundle()
        await asyncio.gather(
            asyncio.to_thread(bundle.close),
            asyncio.to_thread(bundle.close),
        )
        self.assertTrue(bundle.closed)
        self.assertEqual(bundle.close_count, 1)
        self.assertEqual(bundle.trake_lifecycle.close_count, 1)
        self.assertEqual(bundle.vqa_lifecycle.close_count, 1)

    async def test_bundle_close_uses_one_total_timeout_budget(self) -> None:
        bundle = build_advanced_runtime_bundle()
        observed: list[float | None] = []

        def trake_wait(timeout_sec):
            observed.append(timeout_sec)
            return True

        def vqa_wait(timeout_sec):
            observed.append(timeout_sec)
            return False

        with patch(
            "online.testing.advanced_runtime.monotonic_ns",
            side_effect=(0, 60_000_000),
        ), patch.object(
            bundle.trake_lifecycle,
            "wait_until_idle_blocking",
            side_effect=trake_wait,
        ), patch.object(
            bundle.vqa_lifecycle,
            "wait_until_idle_blocking",
            side_effect=vqa_wait,
        ):
            with self.assertRaises(BranchTimeoutError):
                bundle.close(wait=True, timeout_sec=0.1)
        self.assertEqual(observed[0], 0.1)
        self.assertAlmostEqual(observed[1], 0.04)
        bundle.close()

    async def test_two_active_calls_drain_without_cross_contamination(self) -> None:
        fake = BlockingLifecycleFake("parallel-test", timeout_sec=2.0)
        tasks = [
            asyncio.create_task(fake.execute("request-one")),
            asyncio.create_task(fake.execute("request-two")),
        ]
        await asyncio.wait_for(fake.started_event.wait(), timeout=1.0)
        for _ in range(100):
            if fake.active_count == 2:
                break
            await asyncio.sleep(0)
        self.assertEqual(fake.active_count, 2)
        fake.release_request()
        await asyncio.gather(*tasks)
        self.assertEqual(fake.active_count, 0)
        fake.close()
        self.assertFalse(fake.closed_before_release)


class LifecycleEventReuseTests(unittest.TestCase):
    def test_clear_allows_reuse_on_a_new_isolated_event_loop(self) -> None:
        event = LifecycleEvent()

        async def observe_once() -> None:
            waiter = asyncio.create_task(event.wait())
            await asyncio.sleep(0)
            event.set()
            await waiter

        asyncio.run(observe_once())
        event.clear()
        asyncio.run(observe_once())


if __name__ == "__main__":
    unittest.main()
