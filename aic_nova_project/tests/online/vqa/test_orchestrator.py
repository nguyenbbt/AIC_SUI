from __future__ import annotations

import asyncio
from threading import Event

import pytest

from online.domain.candidates import CandidateDiagnostics, FusedFrameCandidate
from online.domain.errors import BranchTimeoutError, ContractMismatchError, ResourceUnavailableError
from online.domain.vqa import ImageEvidence, VLMConfidence, VLMResponse, VLMResponseStatus, VQAEvidenceBudget, VQAAnswerType, VQAQuestion
from online.ports.records import FrameMetadata
from online.vqa.evidence_selector import EvidenceSelector
from online.vqa.orchestrator import VQAOrchestrator


QUESTION = VQAQuestion(question_id="q1", question="Ai?", answer_type=VQAAnswerType.SHORT_TEXT)


def frame() -> FusedFrameCandidate:
    return FusedFrameCandidate(frame_id="V001_00000_001", video_id="V001", shot_id=0, timestamp_sec=1, final_score=1, branch_scores={}, evidence=(), diagnostics=CandidateDiagnostics())


class Retriever:
    def __init__(self, values=()) -> None:
        self.values = tuple(values)

    async def retrieve_candidates(self, question):
        return self.values


class Metadata:
    def get_frames_by_ids(self, frame_ids):
        return {}

    def get_ordered_frames_by_video(self, video_id):
        return (FrameMetadata(frame_id="V001_00000_001", video_id="V001", shot_id=0, timestamp_sec=1),)


class Images:
    def __init__(self, available: bool = True) -> None:
        self.available = available

    def resolve_images(self, frame_ids):
        if not self.available:
            return {}
        return {"V001_00000_001": ImageEvidence(evidence_id="image-1", video_id="V001", frame_id="V001_00000_001", shot_id=0, timestamp_sec=1, image_reference="fixture://image/1")}


class Hydrator:
    def get_ocr_evidence(self, frame_ids):
        return ()

    def get_asr_evidence(self, video_id, start_sec, end_sec):
        return ()

    def get_summary_evidence(self, video_ids):
        return ()


class VLM:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = 0

    def answer(self, request):
        self.calls += 1
        value = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        if isinstance(value, Exception):
            raise value
        return value


def selector(*, images: bool = True) -> EvidenceSelector:
    return EvidenceSelector(metadata_reader=Metadata(), image_resolver=Images(images), evidence_hydrator=Hydrator())


def answered() -> VLMResponse:
    return VLMResponse(status=VLMResponseStatus.ANSWERED, answer="Người đàn ông", answer_type=VQAAnswerType.SHORT_TEXT, confidence=VLMConfidence.HIGH, evidence_ids=("image-1",))


def test_answered_fake_e2e_and_determinism() -> None:
    async def scenario():
        vlm = VLM((answered(),))
        service = VQAOrchestrator(candidate_retriever=Retriever((frame(),)), evidence_selector=selector(), vlm=vlm)
        first = await service.answer(QUESTION)
        second = await service.answer(QUESTION)
        service.close()
        return first, second, vlm.calls

    first, second, calls = asyncio.run(scenario())
    assert first.response.answer == "Người đàn ông"
    assert first.evidence[0].evidence_id == "image-1"
    assert first.model_copy(update={"diagnostics": second.diagnostics}) == second
    assert calls == 2


def test_no_frames_and_no_images_do_not_call_vlm() -> None:
    async def scenario(values, images):
        vlm = VLM((answered(),))
        service = VQAOrchestrator(candidate_retriever=Retriever(values), evidence_selector=selector(images=images), vlm=vlm)
        budget = VQAEvidenceBudget(max_videos=1, max_primary_frames_per_video=1, max_primary_frames_total=1, max_images_total=1)
        result = await service.answer(QUESTION, budget)
        service.close()
        return result, vlm.calls

    no_frames, first_calls = asyncio.run(scenario((), True))
    no_images, second_calls = asyncio.run(scenario((frame(),), False))
    assert no_frames.response.status is VLMResponseStatus.INSUFFICIENT_EVIDENCE
    assert no_images.response.status is VLMResponseStatus.INSUFFICIENT_EVIDENCE
    assert first_calls == second_calls == 0


def test_malformed_response_retries_once_then_succeeds_or_fails() -> None:
    async def scenario(responses):
        vlm = VLM(responses)
        service = VQAOrchestrator(candidate_retriever=Retriever((frame(),)), evidence_selector=selector(), vlm=vlm)
        try:
            return await service.answer(QUESTION), vlm.calls
        finally:
            service.close()

    result, calls = asyncio.run(scenario(({}, answered())))
    assert result.diagnostics.vlm_retry_count == 1
    assert calls == 2
    with pytest.raises(ContractMismatchError):
        asyncio.run(scenario(({}, {})))


def test_timeout_and_resource_failure_surface_and_close_is_idempotent() -> None:
    class SlowVLM:
        def answer(self, request):
            Event().wait(0.1)
            return answered()

    async def timeout_scenario():
        service = VQAOrchestrator(
            candidate_retriever=Retriever((frame(),)),
            evidence_selector=selector(),
            vlm=SlowVLM(),
            total_timeout_sec=1,
            vlm_timeout_sec=0.01,
        )
        with pytest.raises(BranchTimeoutError):
            await service.answer(QUESTION)
        service.close()
        service.close()

    asyncio.run(timeout_scenario())

    async def unavailable_scenario():
        service = VQAOrchestrator(candidate_retriever=Retriever((frame(),)), evidence_selector=selector(), vlm=VLM((ResourceUnavailableError("vlm unavailable"),)))
        with pytest.raises(ResourceUnavailableError):
            await service.answer(QUESTION)
        service.close()

    asyncio.run(unavailable_scenario())


def test_close_rejects_active_execution_then_drains_cleanly() -> None:
    class BlockingRetriever:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def retrieve_candidates(self, question):
            self.started.set()
            await self.release.wait()
            return ()

    async def scenario() -> None:
        retriever = BlockingRetriever()
        service = VQAOrchestrator(
            candidate_retriever=retriever,
            evidence_selector=selector(),
            vlm=VLM((answered(),)),
        )
        task = asyncio.create_task(service.answer(QUESTION))
        await retriever.started.wait()
        with pytest.raises(RuntimeError, match="active execution"):
            service.close(wait=False)
        retriever.release.set()
        await task
        service.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("name", ("total_timeout_sec", "vlm_timeout_sec"))
@pytest.mark.parametrize("timeout", (True, "1", 0, -1, float("nan"), float("inf")))
def test_orchestrator_rejects_invalid_timeout(name: str, timeout: object) -> None:
    kwargs = {name: timeout}
    with pytest.raises(ValueError):
        VQAOrchestrator(
            candidate_retriever=Retriever(),
            evidence_selector=selector(),
            vlm=VLM((answered(),)),
            **kwargs,  # type: ignore[arg-type]
        )


def test_vlm_adapter_contract_error_retries_once_then_succeeds() -> None:
    async def scenario():
        vlm = VLM((ContractMismatchError("malformed"), answered()))
        service = VQAOrchestrator(
            candidate_retriever=Retriever((frame(),)),
            evidence_selector=selector(),
            vlm=vlm,
        )
        try:
            return await service.answer(QUESTION), vlm.calls
        finally:
            service.close()

    result, calls = asyncio.run(scenario())
    assert result.diagnostics.vlm_retry_count == 1
    assert calls == 2


def test_retryable_unavailable_retries_but_generic_unavailable_does_not() -> None:
    async def scenario(responses):
        vlm = VLM(responses)
        service = VQAOrchestrator(
            candidate_retriever=Retriever((frame(),)),
            evidence_selector=selector(),
            vlm=vlm,
        )
        try:
            return await service.answer(QUESTION), vlm.calls
        finally:
            service.close()

    retryable = ResourceUnavailableError("transient", details={"retryable": True})
    result, calls = asyncio.run(scenario((retryable, answered())))
    assert result.diagnostics.vlm_retry_count == 1
    assert calls == 2

    generic = ResourceUnavailableError("configuration unavailable")
    generic_vlm = VLM((generic, answered()))

    async def generic_scenario():
        service = VQAOrchestrator(
            candidate_retriever=Retriever((frame(),)),
            evidence_selector=selector(),
            vlm=generic_vlm,
        )
        try:
            with pytest.raises(ResourceUnavailableError):
                await service.answer(QUESTION)
        finally:
            service.close()

    asyncio.run(generic_scenario())
    assert generic_vlm.calls == 1


def test_vlm_timeout_does_not_retry_or_wait_for_total_deadline() -> None:
    class SlowVLM:
        def __init__(self) -> None:
            self.calls = 0

        def answer(self, request):
            self.calls += 1
            Event().wait(0.08)
            return answered()

    async def scenario():
        vlm = SlowVLM()
        service = VQAOrchestrator(
            candidate_retriever=Retriever((frame(),)),
            evidence_selector=selector(),
            vlm=vlm,
            total_timeout_sec=0.5,
            vlm_timeout_sec=0.01,
        )
        started = asyncio.get_running_loop().time()
        try:
            with pytest.raises(BranchTimeoutError):
                await service.answer(QUESTION)
            return vlm.calls, asyncio.get_running_loop().time() - started
        finally:
            service.close()

    calls, elapsed = asyncio.run(scenario())
    assert calls == 1
    assert elapsed < 0.5


def test_total_deadline_expiry_does_not_submit_retry() -> None:
    class SlowMalformedVLM:
        def __init__(self) -> None:
            self.calls = 0

        def answer(self, request):
            self.calls += 1
            Event().wait(0.05)
            raise ContractMismatchError("malformed")

    async def scenario():
        vlm = SlowMalformedVLM()
        service = VQAOrchestrator(
            candidate_retriever=Retriever((frame(),)),
            evidence_selector=selector(),
            vlm=vlm,
            total_timeout_sec=0.01,
            vlm_timeout_sec=1,
        )
        try:
            with pytest.raises(BranchTimeoutError):
                await service.answer(QUESTION)
            return vlm.calls
        finally:
            service.close()

    assert asyncio.run(scenario()) == 1


@pytest.mark.parametrize("value", (None, "frames", b"frames", 123, (object(),)))
def test_invalid_retriever_output_is_contract_mismatch(value: object) -> None:
    class RawRetriever:
        async def retrieve_candidates(self, question):
            return value

    async def scenario():
        service = VQAOrchestrator(
            candidate_retriever=RawRetriever(),
            evidence_selector=selector(),
            vlm=VLM((answered(),)),
        )
        try:
            with pytest.raises(ContractMismatchError):
                await service.answer(QUESTION)
        finally:
            service.close()

    asyncio.run(scenario())


def test_unexpected_retriever_exception_is_safely_wrapped() -> None:
    class BrokenRetriever:
        async def retrieve_candidates(self, question):
            raise KeyError("sensitive query content")

    async def scenario():
        service = VQAOrchestrator(
            candidate_retriever=BrokenRetriever(),
            evidence_selector=selector(),
            vlm=VLM((answered(),)),
        )
        try:
            with pytest.raises(ResourceUnavailableError) as captured:
                await service.answer(QUESTION)
            return captured.value
        finally:
            service.close()

    error = asyncio.run(scenario())
    assert error.details == {"stage": "candidate_retrieval", "exception_type": "KeyError"}
    assert "sensitive" not in str(error.to_safe_dict())


def test_image_resolver_unavailable_surfaces_without_calling_vlm() -> None:
    class UnavailableImages:
        def resolve_images(self, frame_ids):
            raise ResourceUnavailableError("images unavailable")

    async def scenario():
        vlm = VLM((answered(),))
        evidence_selector = EvidenceSelector(
            metadata_reader=Metadata(),
            image_resolver=UnavailableImages(),
            evidence_hydrator=Hydrator(),
        )
        service = VQAOrchestrator(
            candidate_retriever=Retriever((frame(),)),
            evidence_selector=evidence_selector,
            vlm=vlm,
        )
        try:
            with pytest.raises(ResourceUnavailableError):
                await service.answer(QUESTION)
            return vlm.calls
        finally:
            service.close()

    assert asyncio.run(scenario()) == 0


@pytest.mark.parametrize("stage", ("asr", "summary"))
def test_optional_text_failure_keeps_image_evidence_without_fabrication(stage: str) -> None:
    class DegradedHydrator(Hydrator):
        def get_asr_evidence(self, video_id, start_sec, end_sec):
            if stage == "asr":
                raise BranchTimeoutError("asr timeout")
            return ()

        def get_summary_evidence(self, video_ids):
            if stage == "summary":
                raise ResourceUnavailableError("summary unavailable")
            return ()

    async def scenario():
        vlm = VLM((answered(),))
        evidence_selector = EvidenceSelector(
            metadata_reader=Metadata(),
            image_resolver=Images(),
            evidence_hydrator=DegradedHydrator(),
        )
        service = VQAOrchestrator(
            candidate_retriever=Retriever((frame(),)),
            evidence_selector=evidence_selector,
            vlm=vlm,
        )
        try:
            return await service.answer(QUESTION), vlm.calls
        finally:
            service.close()

    result, calls = asyncio.run(scenario())
    assert calls == 1
    assert tuple(item.evidence_id for item in result.evidence) == ("image-1",)
    assert any(warning.startswith(stage.upper()) for warning in result.diagnostics.warnings)
