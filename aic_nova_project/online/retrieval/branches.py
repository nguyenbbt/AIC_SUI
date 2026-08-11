"""Independent retrieval branches that preserve raw scores and provenance."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from time import perf_counter
from typing import Literal

from online.domain.candidates import (
    ASRIntervalCandidate,
    BranchResult,
    CandidateProvenance,
    FrameCandidate,
    VideoCandidate,
)
from online.domain.enums import BranchStatus, CandidateLevel, RetrievalBranch
from online.domain.errors import (
    ContractMismatchError,
    InvalidQueryError,
    MissingMetadataError,
)
from online.domain.query import QueryBundle, TextQueryVariant
from online.ports import (
    ElasticsearchSearchPort,
    MetadataReaderPort,
    MilvusSearchPort,
    TextEncoderPort,
)
from online.ports.records import ASRSearchHit, FrameMetadata, FrameSearchHit, VideoSearchHit


VISUAL_SOURCE_RESOURCE = "visual_features"
OCR_DENSE_SOURCE_RESOURCE = "ocr_features"
OCR_LEXICAL_SOURCE_RESOURCE = "ocr_texts"
ASR_DENSE_SOURCE_RESOURCE = "asr_features"
ASR_LEXICAL_SOURCE_RESOURCE = "asr_transcripts"
SUMMARY_DENSE_SOURCE_RESOURCE = "summary_features"
SUMMARY_LEXICAL_SOURCE_RESOURCE = "video_summaries"


class _BranchBase:
    branch: RetrievalBranch
    backend: Literal["milvus", "elasticsearch"]

    def __init__(
        self,
        *,
        source_resource: str,
        clock: Callable[[], float],
    ) -> None:
        if not isinstance(source_resource, str) or not source_resource.strip():
            raise ValueError("source_resource must be non-empty")
        self.source_resource = source_resource.strip()
        self._clock = clock

    def _validate_bundle(self, query: QueryBundle, top_k: int) -> None:
        self._validate_top_k(top_k)
        if not isinstance(query, QueryBundle):
            raise InvalidQueryError("query must be a validated QueryBundle")
        if self.branch not in query.enabled_branches:
            raise InvalidQueryError(f"{self.branch.value} branch is not enabled for this query")

    @staticmethod
    def _validate_variant(variant: TextQueryVariant) -> None:
        if not isinstance(variant, TextQueryVariant):
            raise InvalidQueryError("variant must be a validated TextQueryVariant")

    @staticmethod
    def _encode_one(
        encoder: TextEncoderPort,
        variant: TextQueryVariant,
    ) -> Sequence[float]:
        vectors = tuple(encoder.encode_texts((variant.text,)))
        if len(vectors) != 1:
            raise ContractMismatchError(
                "Text encoder must return exactly one vector for one query variant",
                details={"expected": 1, "actual": len(vectors)},
            )
        return vectors[0]

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise InvalidQueryError("top_k must be a positive integer")

    def _elapsed_ms(self, started_at: float) -> float:
        elapsed = (self._clock() - started_at) * 1000.0
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ContractMismatchError("Branch clock returned an invalid duration")
        return elapsed


class _FrameBranchBase(_BranchBase):
    def __init__(
        self,
        *,
        metadata: MetadataReaderPort,
        source_resource: str,
        clock: Callable[[], float],
    ) -> None:
        self.metadata = metadata
        super().__init__(source_resource=source_resource, clock=clock)

    def _build_result(
        self,
        hits: Sequence[FrameSearchHit],
        variant: TextQueryVariant,
        *,
        top_k: int,
        started_at: float,
    ) -> BranchResult[FrameCandidate]:
        hits = tuple(hits)
        if len(hits) > top_k:
            raise ContractMismatchError(
                "Frame search port returned more hits than requested",
                details={
                    "branch": self.branch.value,
                    "requested_top_k": top_k,
                    "actual": len(hits),
                },
            )
        if any(not isinstance(hit, FrameSearchHit) for hit in hits):
            raise ContractMismatchError(
                "Frame search port returned an invalid hit record",
                details={"branch": self.branch.value},
            )
        candidates = self._hydrate(hits, variant)
        return BranchResult[FrameCandidate](
            branch=self.branch,
            candidate_level=CandidateLevel.FRAME,
            query_variant_id=variant.variant_id,
            candidates=candidates,
            requested_top_k=top_k,
            latency_ms=self._elapsed_ms(started_at),
            status=BranchStatus.SUCCESS,
        )

    def _hydrate(
        self,
        hits: Sequence[FrameSearchHit],
        variant: TextQueryVariant,
    ) -> tuple[FrameCandidate, ...]:
        if not hits:
            return ()

        frame_ids = tuple(hit.frame_id for hit in hits)
        metadata_by_id = self.metadata.get_frames_by_ids(frame_ids)
        if not isinstance(metadata_by_id, Mapping):
            raise ContractMismatchError("Metadata port returned a non-mapping response")

        missing_count = sum(frame_id not in metadata_by_id for frame_id in frame_ids)
        if missing_count:
            raise MissingMetadataError(
                "One or more frame hits could not be hydrated",
                details={
                    "branch": self.branch.value,
                    "missing_count": missing_count,
                    "hit_count": len(hits),
                },
            )

        candidates: list[FrameCandidate] = []
        for rank, hit in enumerate(hits, start=1):
            metadata = metadata_by_id[hit.frame_id]
            self._validate_join(hit, metadata)
            candidates.append(
                FrameCandidate(
                    frame_id=metadata.frame_id,
                    video_id=metadata.video_id,
                    shot_id=metadata.shot_id,
                    timestamp_sec=metadata.timestamp_sec,
                    source_frame_idx=metadata.source_frame_idx,
                    image_rel_path=metadata.image_rel_path,
                    rank=rank,
                    raw_score=hit.raw_score,
                    provenance=CandidateProvenance(
                        branch=self.branch,
                        backend=self.backend,
                        source_resource=self.source_resource,
                        query_variant_id=variant.variant_id,
                        query_text=variant.text,
                    ),
                )
            )
        return tuple(candidates)

    @staticmethod
    def _validate_join(hit: FrameSearchHit, metadata: FrameMetadata) -> None:
        if not isinstance(metadata, FrameMetadata):
            raise ContractMismatchError("Metadata port returned an invalid frame record")
        if metadata.frame_id != hit.frame_id:
            raise ContractMismatchError("Hydrated frame_id does not match frame hit")
        if metadata.video_id != hit.video_id:
            raise ContractMismatchError(
                "Hydrated video_id does not match frame hit",
                details={"frame_id": hit.frame_id},
            )
        if hit.shot_id is not None and metadata.shot_id != hit.shot_id:
            raise ContractMismatchError(
                "Hydrated shot_id does not match frame hit",
                details={"frame_id": hit.frame_id},
            )



class _ASRBranchBase(_BranchBase):
    def _build_result(
        self,
        hits: Sequence[ASRSearchHit],
        variant: TextQueryVariant,
        *,
        top_k: int,
        started_at: float,
    ) -> BranchResult[ASRIntervalCandidate]:
        hits = tuple(hits)
        if len(hits) > top_k:
            raise ContractMismatchError(
                "ASR search port returned more hits than requested",
                details={
                    "branch": self.branch.value,
                    "requested_top_k": top_k,
                    "actual": len(hits),
                },
            )
        if any(not isinstance(hit, ASRSearchHit) for hit in hits):
            raise ContractMismatchError(
                "ASR search port returned an invalid hit record",
                details={"branch": self.branch.value},
            )

        candidates = tuple(
            ASRIntervalCandidate(
                video_id=hit.video_id,
                interval_id=hit.interval_id,
                start_time_sec=hit.start_time_sec,
                end_time_sec=hit.end_time_sec,
                rank=rank,
                raw_score=hit.raw_score,
                text=hit.text,
                provenance=CandidateProvenance(
                    branch=self.branch,
                    backend=self.backend,
                    source_resource=self.source_resource,
                    query_variant_id=variant.variant_id,
                    query_text=variant.text,
                ),
            )
            for rank, hit in enumerate(hits, start=1)
        )
        return BranchResult[ASRIntervalCandidate](
            branch=self.branch,
            candidate_level=CandidateLevel.ASR_INTERVAL,
            query_variant_id=variant.variant_id,
            candidates=candidates,
            requested_top_k=top_k,
            latency_ms=self._elapsed_ms(started_at),
            status=BranchStatus.SUCCESS,
        )


class _VideoBranchBase(_BranchBase):
    def _build_result(
        self,
        hits: Sequence[VideoSearchHit],
        variant: TextQueryVariant,
        *,
        top_k: int,
        started_at: float,
    ) -> BranchResult[VideoCandidate]:
        hits = tuple(hits)
        if len(hits) > top_k:
            raise ContractMismatchError(
                "Summary search port returned more hits than requested",
                details={
                    "branch": self.branch.value,
                    "requested_top_k": top_k,
                    "actual": len(hits),
                },
            )
        if any(not isinstance(hit, VideoSearchHit) for hit in hits):
            raise ContractMismatchError(
                "Summary search port returned an invalid hit record",
                details={"branch": self.branch.value},
            )

        candidates = tuple(
            VideoCandidate(
                video_id=hit.video_id,
                rank=rank,
                raw_score=hit.raw_score,
                summary=hit.summary,
                provenance=CandidateProvenance(
                    branch=self.branch,
                    backend=self.backend,
                    source_resource=self.source_resource,
                    query_variant_id=variant.variant_id,
                    query_text=variant.text,
                ),
            )
            for rank, hit in enumerate(hits, start=1)
        )
        return BranchResult[VideoCandidate](
            branch=self.branch,
            candidate_level=CandidateLevel.VIDEO,
            query_variant_id=variant.variant_id,
            candidates=candidates,
            requested_top_k=top_k,
            latency_ms=self._elapsed_ms(started_at),
            status=BranchStatus.SUCCESS,
        )


class VisualSemanticBranch(_FrameBranchBase):
    """OpenCLIP text-to-frame retrieval with mandatory batch hydration.

    One ``BranchResult`` is produced per query variant. This class deliberately
    does not aggregate q0/q1, normalize scores, fuse branches or deduplicate
    frames. Missing-metadata behavior is still an open policy, so it surfaces as
    a typed error instead of silently dropping hits.
    """

    branch = RetrievalBranch.VISUAL_DENSE
    backend = "milvus"

    def __init__(
        self,
        *,
        encoder: TextEncoderPort,
        milvus: MilvusSearchPort,
        metadata: MetadataReaderPort,
        source_resource: str = VISUAL_SOURCE_RESOURCE,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.encoder = encoder
        self.milvus = milvus
        super().__init__(
            metadata=metadata,
            source_resource=source_resource,
            clock=clock,
        )

    def retrieve(
        self,
        query: QueryBundle,
        *,
        top_k: int,
    ) -> tuple[BranchResult[FrameCandidate], ...]:
        """Run all text variants independently without score aggregation."""

        self._validate_bundle(query, top_k)
        return tuple(
            self.retrieve_variant(variant, top_k=top_k)
            for variant in query.text_variants
        )

    def retrieve_variant(
        self,
        variant: TextQueryVariant,
        *,
        top_k: int,
    ) -> BranchResult[FrameCandidate]:
        """Encode, search and hydrate exactly one query variant."""

        self._validate_top_k(top_k)
        self._validate_variant(variant)

        started_at = self._clock()
        vector = self._encode_one(self.encoder, variant)
        hits = self.milvus.search_visual(vector, top_k)
        return self._build_result(
            hits,
            variant,
            top_k=top_k,
            started_at=started_at,
        )


class OCRSemanticBranch(_FrameBranchBase):
    """Vietnamese semantic OCR retrieval; one result per query variant."""

    branch = RetrievalBranch.OCR_DENSE
    backend = "milvus"

    def __init__(
        self,
        *,
        encoder: TextEncoderPort,
        milvus: MilvusSearchPort,
        metadata: MetadataReaderPort,
        source_resource: str = OCR_DENSE_SOURCE_RESOURCE,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.encoder = encoder
        self.milvus = milvus
        super().__init__(
            metadata=metadata,
            source_resource=source_resource,
            clock=clock,
        )

    def retrieve(
        self,
        query: QueryBundle,
        *,
        top_k: int,
    ) -> tuple[BranchResult[FrameCandidate], ...]:
        self._validate_bundle(query, top_k)
        return tuple(
            self.retrieve_variant(variant, top_k=top_k)
            for variant in query.text_variants
        )

    def retrieve_variant(
        self,
        variant: TextQueryVariant,
        *,
        top_k: int,
    ) -> BranchResult[FrameCandidate]:
        self._validate_top_k(top_k)
        self._validate_variant(variant)
        started_at = self._clock()
        vector = self._encode_one(self.encoder, variant)
        hits = self.milvus.search_ocr(vector, top_k)
        return self._build_result(
            hits,
            variant,
            top_k=top_k,
            started_at=started_at,
        )


class OCRLexicalBranch(_FrameBranchBase):
    """Elasticsearch OCR retrieval restricted to the original q0 text."""

    branch = RetrievalBranch.OCR_BM25
    backend = "elasticsearch"

    def __init__(
        self,
        *,
        elasticsearch: ElasticsearchSearchPort,
        metadata: MetadataReaderPort,
        fuzzy: bool | None = None,
        source_resource: str = OCR_LEXICAL_SOURCE_RESOURCE,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if fuzzy is not None and not isinstance(fuzzy, bool):
            raise ValueError("fuzzy must be bool or None")
        self.elasticsearch = elasticsearch
        self.fuzzy = fuzzy
        super().__init__(
            metadata=metadata,
            source_resource=source_resource,
            clock=clock,
        )

    def retrieve(
        self,
        query: QueryBundle,
        *,
        top_k: int,
    ) -> tuple[BranchResult[FrameCandidate], ...]:
        self._validate_bundle(query, top_k)
        return (self.retrieve_variant(query.text_variants[0], top_k=top_k),)

    def retrieve_variant(
        self,
        variant: TextQueryVariant,
        *,
        top_k: int,
    ) -> BranchResult[FrameCandidate]:
        self._validate_top_k(top_k)
        self._validate_variant(variant)
        if variant.variant_id != "q0":
            raise InvalidQueryError("OCR lexical branch accepts only q0")
        started_at = self._clock()
        hits = self.elasticsearch.search_ocr(
            variant.text,
            top_k,
            fuzzy=self.fuzzy,
        )
        return self._build_result(
            hits,
            variant,
            top_k=top_k,
            started_at=started_at,
        )


class ASRSemanticBranch(_ASRBranchBase):
    """Vietnamese semantic ASR retrieval; output remains interval-level."""

    branch = RetrievalBranch.ASR_DENSE
    backend = "milvus"

    def __init__(
        self,
        *,
        encoder: TextEncoderPort,
        milvus: MilvusSearchPort,
        source_resource: str = ASR_DENSE_SOURCE_RESOURCE,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.encoder = encoder
        self.milvus = milvus
        super().__init__(source_resource=source_resource, clock=clock)

    def retrieve(
        self,
        query: QueryBundle,
        *,
        top_k: int,
    ) -> tuple[BranchResult[ASRIntervalCandidate], ...]:
        self._validate_bundle(query, top_k)
        return tuple(
            self.retrieve_variant(variant, top_k=top_k)
            for variant in query.text_variants
        )

    def retrieve_variant(
        self,
        variant: TextQueryVariant,
        *,
        top_k: int,
    ) -> BranchResult[ASRIntervalCandidate]:
        self._validate_top_k(top_k)
        self._validate_variant(variant)
        started_at = self._clock()
        vector = self._encode_one(self.encoder, variant)
        hits = self.milvus.search_asr(vector, top_k)
        return self._build_result(
            hits,
            variant,
            top_k=top_k,
            started_at=started_at,
        )


class ASRLexicalBranch(_ASRBranchBase):
    """Elasticsearch ASR retrieval restricted to q0 and interval output."""

    branch = RetrievalBranch.ASR_BM25
    backend = "elasticsearch"

    def __init__(
        self,
        *,
        elasticsearch: ElasticsearchSearchPort,
        fuzzy: bool | None = None,
        source_resource: str = ASR_LEXICAL_SOURCE_RESOURCE,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if fuzzy is not None and not isinstance(fuzzy, bool):
            raise ValueError("fuzzy must be bool or None")
        self.elasticsearch = elasticsearch
        self.fuzzy = fuzzy
        super().__init__(source_resource=source_resource, clock=clock)

    def retrieve(
        self,
        query: QueryBundle,
        *,
        top_k: int,
    ) -> tuple[BranchResult[ASRIntervalCandidate], ...]:
        self._validate_bundle(query, top_k)
        return (self.retrieve_variant(query.text_variants[0], top_k=top_k),)

    def retrieve_variant(
        self,
        variant: TextQueryVariant,
        *,
        top_k: int,
    ) -> BranchResult[ASRIntervalCandidate]:
        self._validate_top_k(top_k)
        self._validate_variant(variant)
        if variant.variant_id != "q0":
            raise InvalidQueryError("ASR lexical branch accepts only q0")
        started_at = self._clock()
        hits = self.elasticsearch.search_asr(
            variant.text,
            top_k,
            fuzzy=self.fuzzy,
        )
        return self._build_result(
            hits,
            variant,
            top_k=top_k,
            started_at=started_at,
        )


class SummarySemanticBranch(_VideoBranchBase):
    """Vietnamese semantic summary retrieval with video-level output only."""

    branch = RetrievalBranch.SUMMARY_DENSE
    backend = "milvus"

    def __init__(
        self,
        *,
        encoder: TextEncoderPort,
        milvus: MilvusSearchPort,
        source_resource: str = SUMMARY_DENSE_SOURCE_RESOURCE,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.encoder = encoder
        self.milvus = milvus
        super().__init__(source_resource=source_resource, clock=clock)

    def retrieve(
        self,
        query: QueryBundle,
        *,
        top_k: int,
    ) -> tuple[BranchResult[VideoCandidate], ...]:
        self._validate_bundle(query, top_k)
        return tuple(
            self.retrieve_variant(variant, top_k=top_k)
            for variant in query.text_variants
        )

    def retrieve_variant(
        self,
        variant: TextQueryVariant,
        *,
        top_k: int,
    ) -> BranchResult[VideoCandidate]:
        self._validate_top_k(top_k)
        self._validate_variant(variant)
        started_at = self._clock()
        vector = self._encode_one(self.encoder, variant)
        hits = self.milvus.search_summary(vector, top_k)
        return self._build_result(
            hits,
            variant,
            top_k=top_k,
            started_at=started_at,
        )


class SummaryLexicalBranch(_VideoBranchBase):
    """Elasticsearch summary retrieval restricted to q0 and video output."""

    branch = RetrievalBranch.SUMMARY_BM25
    backend = "elasticsearch"

    def __init__(
        self,
        *,
        elasticsearch: ElasticsearchSearchPort,
        fuzzy: bool | None = None,
        source_resource: str = SUMMARY_LEXICAL_SOURCE_RESOURCE,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if fuzzy is not None and not isinstance(fuzzy, bool):
            raise ValueError("fuzzy must be bool or None")
        self.elasticsearch = elasticsearch
        self.fuzzy = fuzzy
        super().__init__(source_resource=source_resource, clock=clock)

    def retrieve(
        self,
        query: QueryBundle,
        *,
        top_k: int,
    ) -> tuple[BranchResult[VideoCandidate], ...]:
        self._validate_bundle(query, top_k)
        return (self.retrieve_variant(query.text_variants[0], top_k=top_k),)

    def retrieve_variant(
        self,
        variant: TextQueryVariant,
        *,
        top_k: int,
    ) -> BranchResult[VideoCandidate]:
        self._validate_top_k(top_k)
        self._validate_variant(variant)
        if variant.variant_id != "q0":
            raise InvalidQueryError("Summary lexical branch accepts only q0")
        started_at = self._clock()
        hits = self.elasticsearch.search_summary(
            variant.text,
            top_k,
            fuzzy=self.fuzzy,
        )
        return self._build_result(
            hits,
            variant,
            top_k=top_k,
            started_at=started_at,
        )


__all__ = [
    "ASR_DENSE_SOURCE_RESOURCE",
    "ASR_LEXICAL_SOURCE_RESOURCE",
    "ASRLexicalBranch",
    "ASRSemanticBranch",
    "OCR_DENSE_SOURCE_RESOURCE",
    "OCR_LEXICAL_SOURCE_RESOURCE",
    "OCRLexicalBranch",
    "OCRSemanticBranch",
    "SUMMARY_DENSE_SOURCE_RESOURCE",
    "SUMMARY_LEXICAL_SOURCE_RESOURCE",
    "SummaryLexicalBranch",
    "SummarySemanticBranch",
    "VISUAL_SOURCE_RESOURCE",
    "VisualSemanticBranch",
]
