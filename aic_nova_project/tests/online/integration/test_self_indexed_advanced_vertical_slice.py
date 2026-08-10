from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

from online.adapters.evidence import ElasticsearchEvidenceHydrator
from online.adapters.images import FilesystemImageResolver
from online.adapters.visual_corpus import MilvusSQLiteVisualCorpusAdapter
from online.config import ElasticsearchResourceConfig, MilvusResourceConfig
from online.domain.vqa import VLMConfidence, VLMResponse, VLMResponseStatus
from online.testing import FakeMetadataReaderPort, build_advanced_modes_fixture
from online.trake import TRAKEService
from online.vqa import EvidenceSelector, VQAOrchestrator


class _RecordReader:
    def __init__(self, records: Sequence[Mapping[str, object]]) -> None:
        self._records = tuple(records)

    def iter_records(self, name, output_fields, *, filter_expression, batch_size):
        records = self._records
        if "==" in filter_expression:
            video_id = filter_expression.split("==", 1)[1].strip().strip('"')
            records = tuple(record for record in records if record["video_id"] == video_id)
        for start in range(0, len(records), batch_size):
            yield records[start : start + batch_size]


class _EvidenceBackend:
    def __init__(self, fixture) -> None:
        self._ocr = {item.frame_id: item for item in fixture.ocr_evidence}
        self._asr = tuple(fixture.asr_evidence)
        self._summaries = {item.video_id: item for item in fixture.summary_evidence}

    def find_documents(self, index, filters, source_fields, *, limit=2):
        if index == "ocr_texts":
            item = self._ocr.get(filters["frame_id"])
            return () if item is None else (
                {
                    "frame_id": item.frame_id,
                    "video_id": item.video_id,
                    "ocr_text_concat": item.text,
                },
            )
        item = self._summaries.get(filters["video_id"])
        return () if item is None else ({"video_id": item.video_id, "summary": item.text},)

    def find_documents_overlapping_interval(
        self, *, index, video_id, start_sec, end_sec, source_fields, limit
    ):
        return tuple(
            {
                "video_id": item.video_id,
                "interval_id": item.interval_id,
                "start_time_sec": item.start_time_sec,
                "end_time_sec": item.end_time_sec,
                "cleaned_text": item.text,
            }
            for item in self._asr
            if item.video_id == video_id
            and item.start_time_sec <= end_sec
            and item.end_time_sec >= start_sec
        )[:limit]


class _CandidateRetriever:
    def __init__(self, candidates) -> None:
        self._candidates = candidates

    async def retrieve_candidates(self, question):
        return self._candidates


class _GroundedVLM:
    def answer(self, request):
        return VLMResponse(
            status=VLMResponseStatus.ANSWERED,
            answer="fixture-grounded-answer",
            answer_type=request.question.answer_type,
            confidence=VLMConfidence.HIGH,
            evidence_ids=(request.evidence[0].evidence_id,),
        )


def test_real_adapter_shapes_drive_trake_and_vqa_without_sdk_objects(tmp_path) -> None:
    fixture = build_advanced_modes_fixture()
    records = tuple(
        {
            "frame_id": frame.frame_id,
            "video_id": frame.video_id,
            "shot_id": frame.shot_id,
            "embedding": frame.vector,
        }
        for video_id in sorted(fixture.visual_frames_by_video)
        for frame in fixture.visual_frames_by_video[video_id]
    )
    metadata = FakeMetadataReaderPort(fixture.frame_metadata)
    corpus = MilvusSQLiteVisualCorpusAdapter(
        MilvusResourceConfig(),
        milvus=_RecordReader(records),
        metadata_reader=metadata,
        scan_batch_size=3,
    )

    trake = TRAKEService(corpus=corpus, encoder=fixture.text_encoder())
    trake_execution = asyncio.run(trake.execute(fixture.trake_query))
    trake.close()

    assert trake_execution.results[0].video_id == fixture.expected_dante_video_id
    assert tuple(match.local_index for match in trake_execution.results[0].sequence) == (
        fixture.expected_dante_positions
    )
    assert tuple(
        (match.video_id, match.source_frame_idx)
        for match in trake_execution.results[0].sequence
    )

    metadata_by_id = {item.frame_id: item for item in fixture.frame_metadata}
    for frame_id in fixture.images_by_frame_id:
        path = tmp_path / metadata_by_id[frame_id].image_rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF-self-indexed-fixture")

    selector = EvidenceSelector(
        metadata_reader=metadata,
        image_resolver=FilesystemImageResolver(data_root=tmp_path, metadata_reader=metadata),
        evidence_hydrator=ElasticsearchEvidenceHydrator(
            ElasticsearchResourceConfig(), backend=_EvidenceBackend(fixture)
        ),
    )
    vqa = VQAOrchestrator(
        candidate_retriever=_CandidateRetriever(fixture.ranked_vqa_candidates),
        evidence_selector=selector,
        vlm=_GroundedVLM(),
    )
    result = asyncio.run(vqa.answer(fixture.vqa_question))
    vqa.close()

    assert result.response.status is VLMResponseStatus.ANSWERED
    assert set(result.response.evidence_ids).issubset(
        {item.evidence_id for item in result.evidence}
    )
    image_evidence = tuple(item for item in result.evidence if item.evidence_type.value == "image")
    assert image_evidence
    assert all(not item.image_reference.startswith(("/", "C:")) for item in image_evidence)
