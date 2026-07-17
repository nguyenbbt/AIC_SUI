"""Composition boundary between Person-A data ports and Person-B retrieval."""

from __future__ import annotations

from collections.abc import Mapping

from online.config import OnlineDataConfig
from online.domain.enums import RetrievalBranch
from online.ports import (
    ElasticsearchSearchPort,
    MetadataReaderPort,
    MilvusSearchPort,
    TextEncoderPort,
)

from .branches import (
    ASRLexicalBranch,
    ASRSemanticBranch,
    OCRLexicalBranch,
    OCRSemanticBranch,
    SummaryLexicalBranch,
    SummarySemanticBranch,
    VisualSemanticBranch,
)
from .service import RetrievalInvocationConfig, RetrievalService


def build_retrieval_service(
    *,
    data_config: OnlineDataConfig,
    milvus: MilvusSearchPort,
    elasticsearch: ElasticsearchSearchPort,
    metadata: MetadataReaderPort,
    visual_encoder: TextEncoderPort,
    vietnamese_encoder: TextEncoderPort,
    invocation_configs: Mapping[
        tuple[RetrievalBranch | str, str],
        RetrievalInvocationConfig | Mapping[str, object],
    ],
    max_workers: int,
) -> RetrievalService:
    """Wire all baseline branches without constructing or owning A adapters.

    The caller owns adapter connection lifecycle and encoder lifecycle. Resource
    names are copied from Person A's environment-loadable configuration so
    candidate provenance reflects the resources actually queried. Per-branch
    settings remain required inputs; in particular, OQ-003 has not approved
    default top-k values.
    """

    if not isinstance(data_config, OnlineDataConfig):
        raise TypeError("data_config must be a validated OnlineDataConfig")
    _require_port("milvus", milvus, MilvusSearchPort)
    _require_port("elasticsearch", elasticsearch, ElasticsearchSearchPort)
    _require_port("metadata", metadata, MetadataReaderPort)
    _require_port("visual_encoder", visual_encoder, TextEncoderPort)
    _require_port("vietnamese_encoder", vietnamese_encoder, TextEncoderPort)

    milvus_config = data_config.milvus
    elasticsearch_config = data_config.elasticsearch
    branches = {
        RetrievalBranch.VISUAL_DENSE: VisualSemanticBranch(
            encoder=visual_encoder,
            milvus=milvus,
            metadata=metadata,
            source_resource=milvus_config.visual_collection,
        ),
        RetrievalBranch.OCR_DENSE: OCRSemanticBranch(
            encoder=vietnamese_encoder,
            milvus=milvus,
            metadata=metadata,
            source_resource=milvus_config.ocr_collection,
        ),
        RetrievalBranch.OCR_BM25: OCRLexicalBranch(
            elasticsearch=elasticsearch,
            metadata=metadata,
            source_resource=elasticsearch_config.ocr_index,
        ),
        RetrievalBranch.ASR_DENSE: ASRSemanticBranch(
            encoder=vietnamese_encoder,
            milvus=milvus,
            source_resource=milvus_config.asr_collection,
        ),
        RetrievalBranch.ASR_BM25: ASRLexicalBranch(
            elasticsearch=elasticsearch,
            source_resource=elasticsearch_config.asr_index,
        ),
        RetrievalBranch.SUMMARY_DENSE: SummarySemanticBranch(
            encoder=vietnamese_encoder,
            milvus=milvus,
            source_resource=milvus_config.summary_collection,
        ),
        RetrievalBranch.SUMMARY_BM25: SummaryLexicalBranch(
            elasticsearch=elasticsearch,
            source_resource=elasticsearch_config.summary_index,
        ),
    }
    return RetrievalService(
        branches=branches,
        invocation_configs=invocation_configs,
        max_workers=max_workers,
    )


def _require_port(name: str, value: object, port: type[object]) -> None:
    if not isinstance(value, port):
        raise TypeError(f"{name} does not implement the required Online port")


__all__ = ["build_retrieval_service"]
