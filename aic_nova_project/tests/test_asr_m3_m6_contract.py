import json
from unittest.mock import MagicMock, patch

import pandas as pd

from feature_extraction.asr_transcript.pipeline import ASRTranscriptPipeline
from feature_extraction.visual_embedding.config import (
    EXPECTED_VISUAL_EMBEDDING_DIMENSION,
)
from feature_extraction.text_embedding.src.text_embedding.config import (
    TEXT_EMBEDDING_DIMENSION,
)
from feature_extraction.text_embedding.src.text_embedding.data_readers import (
    parse_asr_file,
)
from feature_extraction.text_embedding.src.text_embedding.embedding_writer import (
    write_embeddings_to_parquet,
)
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
from indexing.src.indexing.data_loader import (
    load_asr_transcripts,
    load_metadata_and_objects,
    load_text_asr_embeddings,
    load_text_summary_embeddings,
    load_video_summary,
    load_visual_embeddings,
)
from indexing.src.indexing.orchestrator import (
    IndexingOrchestrator,
    VideoSnapshot,
)


@patch("feature_extraction.asr_transcript.llm.gemini_llm.GeminiTranscriptLLM")
def test_module3_asr_contract_flows_through_module6_to_module7(
    mock_llm_class: MagicMock,
    tmp_path,
) -> None:
    """Module 6 and Module 7 must consume the exact artifact from Module 3."""
    video_id = "V001"
    video_dir = tmp_path / "raw_videos"
    metadata_dir = tmp_path / "metadata"
    caption_dir = tmp_path / "captions"
    module3_output_dir = tmp_path
    transcript_dir = module3_output_dir / "transcripts"

    video_dir.mkdir()
    metadata_dir.mkdir()
    caption_dir.mkdir()
    transcript_dir.mkdir(parents=True)

    frame_id = "V001_00000_015"
    visual_embedding = [1.0] + [0.0] * (
        EXPECTED_VISUAL_EMBEDDING_DIMENSION - 1
    )
    text_embedding = [1.0] + [0.0] * (TEXT_EMBEDDING_DIMENSION - 1)
    (metadata_dir / f"{video_id}.json").write_text(
        json.dumps(
            {
                "contract_version": "self-indexed-v2",
                "video_id": video_id,
                "source_path": "videos/V001.mp4",
                "source_video_rel_path": "videos/V001.mp4",
                "fps": 30.0,
                "duration_sec": 20.0,
                "frame_count": 600,
                "width": 32,
                "height": 24,
                "num_shots": 1,
                "shots": [
                    {
                        "shot_id": 0,
                        "start_frame": 0,
                        "end_frame": 599,
                        "start_time_sec": 0.0,
                        "end_time_sec": 20.0,
                        "keyframes": [
                            {
                                "position": 0.15,
                                "position_code": 15,
                                "frame_index": 4,
                                "source_frame_idx": 4,
                                "time_sec": 0.133,
                                "file_path": (
                                    "keyframes/V001/"
                                    "shot_00000_pos_015.webp"
                                ),
                                "image_rel_path": (
                                    "keyframes/V001/"
                                    "shot_00000_pos_015.webp"
                                ),
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    visual_dir = tmp_path / "embeddings" / "visual"
    visual_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "frame_id": frame_id,
                "video_id": video_id,
                "shot_id": 0,
                "embedding": visual_embedding,
            }
        ]
    ).to_parquet(visual_dir / f"{video_id}.parquet", index=False)

    mock_llm = MagicMock()
    mock_llm.clean.return_value = "Văn bản đã làm sạch."
    mock_llm.summarize.return_value = "Bản tóm tắt."
    mock_llm_class.return_value = mock_llm

    raw_transcript_path = transcript_dir / f"{video_id}_raw.json"
    with open(raw_transcript_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video_id": video_id,
                "source": "asr",
                "segments": [
                    {"timestamp": [10.2, 17.8], "text": "van ban chua lam sach"},
                ],
            },
            f,
        )

    module3_pipeline = ASRTranscriptPipeline(
        video_dir=str(video_dir),
        metadata_dir=str(metadata_dir),
        caption_dir=str(caption_dir),
        output_dir=str(module3_output_dir),
        llm_provider="gemini",
        group_size=1,
    )
    module3_pipeline.process_video(video_id)

    cleaned_path = transcript_dir / f"{video_id}_cleaned.json"
    with open(cleaned_path, "r", encoding="utf-8") as f:
        cleaned_payload = json.load(f)

    assert cleaned_payload["video_id"] == video_id
    assert cleaned_payload["intervals"][0]["interval_id"] == "0"
    assert cleaned_payload["intervals"][0]["start_time_sec"] == 10.2
    assert cleaned_payload["intervals"][0]["end_time_sec"] == 17.8

    records = parse_asr_file(cleaned_path)
    assert len(records) == 1
    assert records[0]["video_id"] == video_id
    assert records[0]["interval_id"] == "0"
    assert records[0]["start_time_sec"] == 10.2
    assert records[0]["end_time_sec"] == 17.8
    assert records[0]["text"] == "Văn bản đã làm sạch."

    records[0]["embedding"] = text_embedding
    parquet_path = tmp_path / "embeddings" / "text_asr" / f"{video_id}.parquet"
    write_embeddings_to_parquet(records, parquet_path)
    assert parquet_path.exists()

    parquet_records = pd.read_parquet(parquet_path).to_dict(orient="records")
    assert len(parquet_records) == 1
    assert parquet_records[0]["video_id"] == video_id
    assert parquet_records[0]["interval_id"] == "0"
    assert parquet_records[0]["start_time_sec"] == 10.2
    assert parquet_records[0]["end_time_sec"] == 17.8
    assert parquet_records[0]["text"] == "Văn bản đã làm sạch."

    semantic_records = load_text_asr_embeddings(tmp_path, video_id)
    lexical_records = load_asr_transcripts(tmp_path, video_id)

    assert len(semantic_records) == 1
    assert len(lexical_records) == 1
    semantic_key = (
        semantic_records[0]["video_id"],
        semantic_records[0]["interval_id"],
    )
    lexical_key = (
        lexical_records[0]["video_id"],
        lexical_records[0]["interval_id"],
    )
    assert semantic_key == lexical_key == (video_id, "0")
    assert semantic_records[0]["start_time_sec"] == lexical_records[0][
        "start_time_sec"
    ]
    assert semantic_records[0]["end_time_sec"] == lexical_records[0][
        "end_time_sec"
    ]

    summary_parquet_path = (
        tmp_path / "embeddings" / "text_summary" / f"{video_id}.parquet"
    )
    write_embeddings_to_parquet(
        [
            {
                "video_id": video_id,
                "text": "Bản tóm tắt.",
                "embedding": text_embedding,
            }
        ],
        summary_parquet_path,
    )

    visual_records = load_visual_embeddings(tmp_path, video_id)
    summary_semantic_records = load_text_summary_embeddings(
        tmp_path,
        video_id,
    )
    summary_lexical_records = load_video_summary(tmp_path, video_id)
    metadata_records, _ = load_metadata_and_objects(tmp_path, video_id)

    milvus = MagicMock()
    elasticsearch = MagicMock()
    tabular = MagicMock()
    orchestrator = IndexingOrchestrator(
        milvus_client=milvus,
        es_client=elasticsearch,
        tabular_client=tabular,
        batch_size=10,
    )
    elasticsearch.bulk_index.side_effect = (
        lambda index, documents, id_field: len(documents)
    )
    empty_snapshot = VideoSnapshot(
        milvus={
            VISUAL_COLLECTION: [],
            ASR_COLLECTION: [],
            SUMMARY_COLLECTION: [],
            OCR_COLLECTION: [],
        },
        elasticsearch={OCR_INDEX: [], ASR_INDEX: [], SUMMARY_INDEX: []},
        metadata=[],
        objects=[],
    )
    indexed_snapshot = VideoSnapshot(
        milvus={
            VISUAL_COLLECTION: visual_records,
            ASR_COLLECTION: semantic_records,
            SUMMARY_COLLECTION: summary_semantic_records,
            OCR_COLLECTION: [],
        },
        elasticsearch={
            OCR_INDEX: [],
            ASR_INDEX: [
                {
                    "_id": f"{video_id}_0",
                    "_source": {
                        **lexical_records[0],
                        "_doc_id": f"{video_id}_0",
                    },
                }
            ],
            SUMMARY_INDEX: [
                {
                    "_id": video_id,
                    "_source": summary_lexical_records[0],
                }
            ],
        },
        metadata=metadata_records,
        objects=[],
        video={
            "video_id": video_id,
            "source_video_rel_path": "videos/V001.mp4",
            "fps": 30.0,
            "duration_sec": 20.0,
            "frame_count": 600,
            "width": 32,
            "height": 24,
        },
    )
    orchestrator._capture_snapshot = MagicMock(
        side_effect=[empty_snapshot, indexed_snapshot]
    )

    assert orchestrator.process_video(
        video_id,
        tmp_path,
        visual_dim=EXPECTED_VISUAL_EMBEDDING_DIMENSION,
        text_dim=TEXT_EMBEDDING_DIMENSION,
    )

    milvus_asr_calls = [
        call
        for call in milvus.insert_batch.call_args_list
        if call.args[0] == ASR_COLLECTION
    ]
    es_asr_calls = [
        call
        for call in elasticsearch.bulk_index.call_args_list
        if call.args[0] == ASR_INDEX
    ]
    assert len(milvus_asr_calls) == 1
    assert len(es_asr_calls) == 1

    milvus_record = milvus_asr_calls[0].args[1][0]
    es_record = es_asr_calls[0].args[1][0]
    assert (milvus_record["video_id"], milvus_record["interval_id"]) == (
        es_record["video_id"],
        es_record["interval_id"],
    )
    assert es_record["_doc_id"] == f"{video_id}_0"
