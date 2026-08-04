from collections import defaultdict
from copy import deepcopy
from pathlib import Path


from indexing.src.indexing.clients.es_client import (
    ASR_INDEX,
    OCR_INDEX,
    SUMMARY_INDEX,
)
from indexing.src.indexing.clients.milvus_client import (
    ASR_COLLECTION,
    OCR_COLLECTION,
    SUMMARY_COLLECTION,
    VISUAL_COLLECTION,
)
from indexing.src.indexing.data_loader import detect_embedding_dim
from indexing.src.indexing.orchestrator import IndexingOrchestrator


class _MemoryMilvus:
    def __init__(self) -> None:
        self.records = defaultdict(list)

    def snapshot_by_video_id(self, collection: str, video_id: str) -> list:
        return deepcopy([
            record
            for record in self.records[collection]
            if record["video_id"] == video_id
        ])

    def delete_by_video_id(self, collection: str, video_id: str) -> None:
        self.records[collection] = [
            record
            for record in self.records[collection]
            if record["video_id"] != video_id
        ]

    def insert_batch(self, collection: str, records: list, dim: int) -> int:
        assert all(len(record["embedding"]) == dim for record in records)
        self.records[collection].extend(deepcopy(records))
        return len(records)


class _MemoryElasticsearch:
    def __init__(self) -> None:
        self.documents = defaultdict(dict)

    def snapshot_by_video_id(self, index: str, video_id: str) -> list:
        return [
            {"_id": document_id, "_source": deepcopy(source)}
            for document_id, source in self.documents[index].items()
            if source["video_id"] == video_id
        ]

    def delete_by_video_id(self, index: str, video_id: str) -> None:
        self.documents[index] = {
            document_id: source
            for document_id, source in self.documents[index].items()
            if source["video_id"] != video_id
        }

    def bulk_index(self, index: str, documents: list, id_field: str) -> int:
        for document in documents:
            self.documents[index][str(document[id_field])] = deepcopy(document)
        return len(documents)

    def restore_snapshot(self, index: str, documents: list) -> None:
        for document in documents:
            self.documents[index][document["_id"]] = deepcopy(
                document["_source"]
            )


class _MemoryTabular:
    def __init__(self) -> None:
        self.metadata = []
        self.objects = []

    def snapshot_by_video_id(self, video_id: str) -> tuple[list, list]:
        metadata = [
            record
            for record in self.metadata
            if record["video_id"] == video_id
        ]
        frame_ids = {record["frame_id"] for record in metadata}
        objects = [
            record
            for record in self.objects
            if record["frame_id"] in frame_ids
        ]
        return deepcopy(metadata), deepcopy(objects)

    def delete_by_video_id(self, video_id: str) -> None:
        metadata, _ = self.snapshot_by_video_id(video_id)
        frame_ids = {record["frame_id"] for record in metadata}
        self.metadata = [
            record
            for record in self.metadata
            if record["video_id"] != video_id
        ]
        self.objects = [
            record
            for record in self.objects
            if record["frame_id"] not in frame_ids
        ]

    def insert_metadata_batch(self, records: list) -> None:
        self.metadata.extend(deepcopy(records))

    def insert_objects_batch(self, records: list) -> None:
        self.objects.extend(deepcopy(records))

    def restore_snapshot(self, metadata: list, objects: list) -> None:
        self.metadata.extend(deepcopy(metadata))
        self.objects.extend(deepcopy(objects))


def test_one_producer_bundle_is_joinable_through_module_7(
    offline_artifact_bundle: Path,
) -> None:
    visual_dim = detect_embedding_dim(
        offline_artifact_bundle / "embeddings" / "visual"
    )
    text_dim = detect_embedding_dim(
        offline_artifact_bundle / "embeddings" / "text_asr"
    )

    milvus = _MemoryMilvus()
    elasticsearch = _MemoryElasticsearch()
    tabular = _MemoryTabular()
    orchestrator = IndexingOrchestrator(
        milvus,
        elasticsearch,
        tabular,
        batch_size=1,
    )

    assert orchestrator.process_video(
        "V001",
        offline_artifact_bundle,
        visual_dim=visual_dim,
        text_dim=text_dim,
        ocr_dim=text_dim,
    )

    frame_id = "V001_00000_015"
    assert {r["frame_id"] for r in milvus.records[VISUAL_COLLECTION]} == {
        frame_id
    }
    assert {r["frame_id"] for r in milvus.records[OCR_COLLECTION]} == {
        frame_id
    }
    assert {r["frame_id"] for r in tabular.metadata} == {frame_id}
    assert {r["frame_id"] for r in tabular.objects} == {frame_id}
    assert {
        (r["video_id"], r["interval_id"])
        for r in milvus.records[ASR_COLLECTION]
    } == {("V001", "0")}
    assert {
        (r["video_id"], r["interval_id"])
        for r in elasticsearch.documents[ASR_INDEX].values()
    } == {("V001", "0")}
    assert {r["video_id"] for r in milvus.records[SUMMARY_COLLECTION]} == {
        "V001"
    }
    assert set(elasticsearch.documents[SUMMARY_INDEX]) == {"V001"}
    assert set(elasticsearch.documents[OCR_INDEX]) == {frame_id}
