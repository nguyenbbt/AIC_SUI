from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from online.adapters.evidence import ElasticsearchEvidenceHydrator
from online.adapters.images import FilesystemImageResolver
from online.adapters.visual_corpus import MilvusSQLiteVisualCorpusAdapter
from online.config import ElasticsearchResourceConfig, MilvusResourceConfig
from online.domain.errors import ContractMismatchError
from online.ports.records import FrameMetadata
from online.testing import FakeMetadataReaderPort


FRAMES = (
    FrameMetadata(
        frame_id="V001_00000_015",
        video_id="V001",
        shot_id=0,
        source_frame_idx=15,
        timestamp_sec=0.5,
        image_rel_path="keyframes/V001/a.webp",
    ),
    FrameMetadata(
        frame_id="V001_00001_050",
        video_id="V001",
        shot_id=1,
        source_frame_idx=90,
        timestamp_sec=3.0,
        image_rel_path="keyframes/V001/b.webp",
    ),
)


class RecordReader:
    def __init__(self, records: Sequence[Mapping[str, object]]) -> None:
        self.records = tuple(records)

    def iter_records(self, name, output_fields, *, filter_expression, batch_size):
        records = self.records
        if "==" in filter_expression:
            video_id = filter_expression.split("==", 1)[1].strip().strip('"')
            records = tuple(record for record in records if record["video_id"] == video_id)
        for start in range(0, len(records), batch_size):
            yield records[start : start + batch_size]


def test_visual_corpus_joins_vectors_to_sqlite_and_orders_timeline() -> None:
    reader = RecordReader(
        (
            {
                "frame_id": FRAMES[1].frame_id,
                "video_id": "V001",
                "shot_id": 1,
                "embedding": (0.0, 1.0),
            },
            {
                "frame_id": FRAMES[0].frame_id,
                "video_id": "V001",
                "shot_id": 0,
                "embedding": (1.0, 0.0),
            },
        )
    )
    adapter = MilvusSQLiteVisualCorpusAdapter(
        MilvusResourceConfig(),
        milvus=reader,
        metadata_reader=FakeMetadataReaderPort(FRAMES),
        scan_batch_size=1,
    )

    assert adapter.list_video_ids() == ("V001",)
    batches = tuple(adapter.iter_ordered_frame_embedding_batches("V001", 1))
    assert [batch[0].frame_id for batch in batches] == [frame.frame_id for frame in FRAMES]
    assert [batch[0].local_index for batch in batches] == [0, 1]
    assert batches[1][0].source_frame_idx == 90


def test_visual_corpus_rejects_missing_sqlite_join() -> None:
    adapter = MilvusSQLiteVisualCorpusAdapter(
        MilvusResourceConfig(),
        milvus=RecordReader(
            (
                {
                    "frame_id": FRAMES[0].frame_id,
                    "video_id": "V001",
                    "shot_id": 0,
                    "embedding": (1.0, 0.0),
                },
            )
        ),
        metadata_reader=FakeMetadataReaderPort(()),
    )
    with pytest.raises(ContractMismatchError):
        tuple(adapter.iter_ordered_frame_embedding_batches("V001", 10))


def test_filesystem_image_resolver_verifies_file_and_keeps_relative_reference(tmp_path) -> None:
    image = tmp_path / "keyframes" / "V001" / "a.webp"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"RIFF-fixture")
    resolver = FilesystemImageResolver(
        data_root=tmp_path,
        metadata_reader=FakeMetadataReaderPort(FRAMES),
    )

    resolver.health_check()
    result = resolver.resolve_images(tuple(frame.frame_id for frame in FRAMES))

    assert tuple(result) == (FRAMES[0].frame_id,)
    assert result[FRAMES[0].frame_id].image_reference == FRAMES[0].image_rel_path


class EvidenceBackend:
    def find_documents(self, index, filters, source_fields, *, limit=2):
        if index == "ocr_texts" and filters["frame_id"] == FRAMES[0].frame_id:
            return (
                {
                    "frame_id": FRAMES[0].frame_id,
                    "video_id": "V001",
                    "ocr_text_concat": "bien bao giao thong",
                },
            )
        if index == "video_summaries" and filters["video_id"] == "V001":
            return ({"video_id": "V001", "summary": "mot video giao thong"},)
        return ()

    def find_documents_overlapping_interval(
        self, *, index, video_id, start_sec, end_sec, source_fields, limit
    ):
        return (
            {
                "video_id": video_id,
                "interval_id": "0",
                "start_time_sec": 1.0,
                "end_time_sec": 4.0,
                "cleaned_text": "loi noi trong video",
            },
        )


def test_elasticsearch_evidence_hydrator_preserves_candidate_levels() -> None:
    adapter = ElasticsearchEvidenceHydrator(
        ElasticsearchResourceConfig(), backend=EvidenceBackend()
    )

    ocr = adapter.get_ocr_evidence((FRAMES[0].frame_id, FRAMES[1].frame_id))
    asr = adapter.get_asr_evidence("V001", 2.0, 3.0)
    summary = adapter.get_summary_evidence(("V001", "V404"))

    assert ocr[0].evidence_id == f"ocr:{FRAMES[0].frame_id}"
    assert asr[0].evidence_id == "asr:V001:0"
    assert summary[0].evidence_id == "summary:V001"
