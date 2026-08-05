from pathlib import Path

import numpy as np
from PIL import Image
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FixtureOCRDetector:
    def detect(self, image, **kwargs) -> list:
        del image, kwargs
        return [[[2, 2], [28, 2], [28, 16], [2, 16]]]


class _FixtureOCRRecognizer:
    def recognize_batch(self, images) -> list[tuple[str, float]]:
        return [("thanh pho ho chi minh", 0.99) for _ in images]


class _FixtureObjectDetector:
    def detect_batch(self, images) -> list[list[dict]]:
        return [
            [
                {
                    "label": "bus",
                    "confidence": 0.95,
                    "bbox": [1, 1, 24, 18],
                    "model_source": "fixture-detector",
                }
            ]
            for _ in images
        ]


@pytest.fixture
def offline_artifact_bundle(tmp_path: Path, monkeypatch) -> Path:
    """Create one canonical video bundle with Module 1-6 producers."""
    monkeypatch.syspath_prepend(
        str(PROJECT_ROOT / "feature_extraction" / "ocr" / "src")
    )
    monkeypatch.syspath_prepend(
        str(
            PROJECT_ROOT
            / "feature_extraction"
            / "object_detection"
            / "src"
        )
    )
    monkeypatch.syspath_prepend(
        str(
            PROJECT_ROOT
            / "feature_extraction"
            / "text_embedding"
            / "src"
        )
    )

    from data_pipeline.shot_keyframe.metadata_schema import VideoMetadata
    from feature_extraction.asr_transcript.artifact_writer import (
        write_cleaned_transcript,
        write_video_summary,
    )
    from feature_extraction.visual_embedding.embedding_writer import (
        write_embeddings_to_parquet as write_visual_embeddings,
    )
    from object_detection.pipeline import ObjectDetectionPipeline
    from object_detection.resume_validation import build_object_provenance
    from ocr_module.pipeline import OCRPipeline
    from text_embedding.encoders.base import BaseTextEncoder
    from text_embedding.pipeline import TextEmbeddingPipeline

    class FixtureTextEncoder(BaseTextEncoder):
        model_name = "fixture-text-encoder"
        model_revision = "offline-contract-v1"
        max_length = 32
        embedding_dim = 2

        def encode_batch(self, texts: list[str]) -> np.ndarray:
            return np.asarray(
                [[0.6, 0.8] for _ in texts],
                dtype=np.float32,
            )

        def encode_long_text(self, text: str) -> np.ndarray:
            assert text
            return np.asarray([0.6, 0.8], dtype=np.float32)

    data_dir = tmp_path / "offline_bundle"
    metadata_dir = data_dir / "metadata"
    video_keyframe_dir = data_dir / "keyframes" / "V001"
    transcript_dir = data_dir / "transcripts"
    summary_dir = data_dir / "summaries"
    ocr_dir = data_dir / "ocr"
    object_dir = data_dir / "object_detection"
    visual_dir = data_dir / "embeddings" / "visual"
    for directory in (
        metadata_dir,
        video_keyframe_dir,
        transcript_dir,
        summary_dir,
        ocr_dir,
        object_dir,
        visual_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    local_frame_name = "shot_00000_pos_015.webp"
    global_frame_id = "V001_00000_015"
    keyframe_path = video_keyframe_dir / local_frame_name
    Image.new("RGB", (32, 24), color=(240, 240, 240)).save(
        keyframe_path
    )

    # Module 1: serialize the producer's Pydantic metadata contract.
    metadata = VideoMetadata(
        video_id="V001",
        source_path="videos/V001.mp4",
        fps=30.0,
        duration_sec=1.0,
        num_shots=1,
        shots=[
            {
                "shot_id": 0,
                "start_frame": 0,
                "end_frame": 30,
                "start_time_sec": 0.0,
                "end_time_sec": 1.0,
                "keyframes": [
                    {
                        "position": 0.15,
                        "frame_index": 4,
                        "time_sec": 0.133,
                        "file_path": (
                            f"keyframes/V001/{local_frame_name}"
                        ),
                    }
                ],
            }
        ],
    )
    (metadata_dir / "V001.json").write_text(
        metadata.model_dump_json(indent=2),
        encoding="utf-8",
    )

    # Module 2: publish a real visual Parquet artifact.
    write_visual_embeddings(
        "V001",
        [
            {
                "frame_id": global_frame_id,
                "video_id": "V001",
                "shot_id": 0,
                "position": 0.15,
                "model_name": "fixture-visual-encoder",
                "embedding_dim": 2,
                "embedding": [0.6, 0.8],
            }
        ],
        str(visual_dir),
    )

    # Module 3: use the same serializers called by the ASR pipeline.
    write_cleaned_transcript(
        transcript_dir / "V001_cleaned.json",
        video_id="V001",
        source="caption",
        llm_provider="FixtureLLM",
        intervals=[
            {
                "interval_id": "0",
                "start_time_sec": 0.0,
                "end_time_sec": 1.0,
                "raw_text": "Xin chao thanh pho",
                "cleaned_text": "Xin chào thành phố",
                "segment_ids": [0],
                "cleaning_failed": False,
            }
        ],
    )
    write_video_summary(
        summary_dir / "V001.json",
        video_id="V001",
        summary="Một video ngắn về Thành phố Hồ Chí Minh.",
        llm_provider="FixtureLLM",
    )

    # Module 4: run the producer pipeline with deterministic test engines.
    ocr_pipeline = OCRPipeline.__new__(OCRPipeline)
    ocr_pipeline.detector = _FixtureOCRDetector()
    ocr_pipeline.recognizer = _FixtureOCRRecognizer()
    ocr_pipeline.backbone = "fixture-backbone"
    ocr_pipeline.confidence_threshold = 0.4
    ocr_pipeline.process_video(
        "V001",
        data_dir / "keyframes",
        metadata_dir,
        ocr_dir,
        batch_size=1,
    )

    # Module 5: run the producer pipeline with a deterministic detector.
    object_pipeline = ObjectDetectionPipeline.__new__(
        ObjectDetectionPipeline
    )
    object_pipeline.detectors = [_FixtureObjectDetector()]
    object_pipeline.nms_threshold = 0.5
    object_pipeline.provenance = build_object_provenance(
        yolo_world_model="fixture-detector",
        custom_vocab_file=None,
        co_detr_backbone=None,
        confidence_threshold=0.25,
        nms_threshold=0.5,
    )
    object_pipeline.process_video(
        "V001",
        metadata_dir / "V001.json",
        video_keyframe_dir,
        object_dir / "V001.json",
        batch_size=1,
    )

    # Module 6: embed the exact Module 3 and Module 4 producer artifacts.
    text_pipeline = TextEmbeddingPipeline(FixtureTextEncoder())
    text_pipeline.process_asr(
        transcript_dir,
        data_dir / "embeddings" / "text_asr",
    )
    text_pipeline.process_summary(
        summary_dir,
        data_dir / "embeddings" / "text_summary",
    )
    text_pipeline.process_ocr(
        ocr_dir,
        data_dir / "embeddings" / "text_ocr",
    )

    return data_dir
