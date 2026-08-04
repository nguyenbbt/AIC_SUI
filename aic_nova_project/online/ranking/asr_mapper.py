"""Map ASR interval evidence to frame-level evidence for fusion."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from online.domain.candidates import (
    ASRIntervalCandidate,
    BranchResult,
    CandidateProvenance,
    FrameCandidate,
)
from online.domain.enums import BranchStatus, CandidateLevel
from online.domain.errors import ContractMismatchError
from online.ports.metadata import MetadataReaderPort
from online.ports.records import FrameMetadata


@dataclass(frozen=True)
class ASRMappingConfig:
    """Configurable interval-to-frame policy.

    ``timestamp_inclusive_distributed_v1`` maps frames whose timestamp is within
    ``[start_time_sec, end_time_sec]`` and distributes the interval score across
    matched frames so long intervals do not multiply evidence without bound.
    """

    policy_name: str = "timestamp_inclusive_distributed_v1"
    max_frames_per_interval: int = 50
    interval_rrf_k: int = 60

    def __post_init__(self) -> None:
        if not isinstance(self.policy_name, str) or not self.policy_name.strip():
            raise ValueError("policy_name must be non-empty")
        if (
            isinstance(self.max_frames_per_interval, bool)
            or not isinstance(self.max_frames_per_interval, int)
            or self.max_frames_per_interval < 1
        ):
            raise ValueError("max_frames_per_interval must be a positive integer")
        if (
            isinstance(self.interval_rrf_k, bool)
            or not isinstance(self.interval_rrf_k, int)
            or self.interval_rrf_k < 1
        ):
            raise ValueError("interval_rrf_k must be a positive integer")


@dataclass(frozen=True)
class ASRMappingResult:
    branch_result: BranchResult[FrameCandidate]
    mapping_loss_count: int
    policy_name: str
    input_interval_count: int = 0
    mapped_interval_count: int = 0
    output_frame_count: int = 0
    truncated_interval_count: int = 0
    truncated_frame_count: int = 0
    max_frames_per_interval: int = 50


class ASRIntervalFrameMapper:
    """Convert ASR interval candidates into frame candidates deterministically."""

    def __init__(self, config: ASRMappingConfig | None = None) -> None:
        self.config = config or ASRMappingConfig()

    @property
    def name(self) -> str:
        return self.config.policy_name

    def map_result(
        self,
        result: BranchResult[ASRIntervalCandidate],
        metadata: MetadataReaderPort,
    ) -> ASRMappingResult:
        if not isinstance(result, BranchResult):
            raise TypeError("result must be a BranchResult")
        if result.candidate_level is not CandidateLevel.ASR_INTERVAL:
            raise ContractMismatchError("ASR mapper requires ASR interval BranchResult")
        if not isinstance(metadata, MetadataReaderPort):
            raise TypeError("metadata must implement MetadataReaderPort")

        if result.status in {BranchStatus.FAILED, BranchStatus.DISABLED}:
            return ASRMappingResult(
                branch_result=_empty_frame_result(result),
                mapping_loss_count=0,
                policy_name=self.name,
                input_interval_count=len(result.candidates),
                max_frames_per_interval=self.config.max_frames_per_interval,
            )

        mapped: list[FrameCandidate] = []
        mapping_loss_count = 0
        mapped_interval_count = 0
        truncated_interval_count = 0
        truncated_frame_count = 0
        frames_cache: dict[str, tuple[FrameMetadata, ...]] = {}
        for interval in result.candidates:
            if not isinstance(interval, ASRIntervalCandidate):
                raise ContractMismatchError("ASR BranchResult contains a non-ASR candidate")
            frames = frames_cache.get(interval.video_id)
            if frames is None:
                frames = tuple(metadata.get_ordered_frames_by_video(interval.video_id))
                self._validate_frame_metadata(interval.video_id, frames)
                frames_cache[interval.video_id] = frames
            interval_frames = self.map_interval(interval, frames)
            if not interval_frames:
                mapping_loss_count += 1
            else:
                mapped_interval_count += 1
            full_match_count = len(self._matched_frames(interval, frames))
            if full_match_count > len(interval_frames):
                truncated_interval_count += 1
                truncated_frame_count += full_match_count - len(interval_frames)
            mapped.extend(interval_frames)

        ordered = tuple(
            sorted(
                mapped,
                key=lambda item: (
                    item.rank,
                    item.timestamp_sec,
                    item.local_index,
                    item.frame_id,
                ),
            )
        )
        reranked = tuple(
            candidate.model_copy(update={"rank": rank})
            for rank, candidate in enumerate(ordered, start=1)
        )
        return ASRMappingResult(
            branch_result=BranchResult[FrameCandidate](
                branch=result.branch,
                candidate_level=CandidateLevel.FRAME,
                query_variant_id=result.query_variant_id,
                candidates=reranked,
                requested_top_k=result.requested_top_k,
                latency_ms=result.latency_ms,
                status=result.status,
                warnings=result.warnings,
                missing_metadata_count=result.missing_metadata_count,
            ),
            mapping_loss_count=mapping_loss_count,
            policy_name=self.name,
            input_interval_count=len(result.candidates),
            mapped_interval_count=mapped_interval_count,
            output_frame_count=len(reranked),
            truncated_interval_count=truncated_interval_count,
            truncated_frame_count=truncated_frame_count,
            max_frames_per_interval=self.config.max_frames_per_interval,
        )

    def map_interval(
        self,
        interval: ASRIntervalCandidate,
        ordered_frames: Sequence[FrameMetadata],
    ) -> tuple[FrameCandidate, ...]:
        if not isinstance(interval, ASRIntervalCandidate):
            raise TypeError("interval must be an ASRIntervalCandidate")
        frames = tuple(ordered_frames)
        self._validate_frame_metadata(interval.video_id, frames)
        matched = self._limit_frames(interval, self._matched_frames(interval, frames))
        if not matched:
            return ()

        raw_score = interval.raw_score / len(matched)
        interval_normalized_score = self._interval_normalized_score(interval)
        normalized_score = interval_normalized_score / len(matched)
        return tuple(
            FrameCandidate(
                frame_id=frame.frame_id,
                video_id=frame.video_id,
                keyframe_no=frame.keyframe_no,
                local_index=frame.local_index,
                timestamp_sec=frame.timestamp_sec,
                source_frame_idx=frame.source_frame_idx,
                rank=interval.rank,
                raw_score=raw_score,
                normalized_score=normalized_score,
                provenance=CandidateProvenance(
                    branch=interval.provenance.branch,
                    backend=interval.provenance.backend,
                    source_resource=interval.provenance.source_resource,
                    query_variant_id=interval.provenance.query_variant_id,
                    query_text=interval.provenance.query_text,
                    source_candidate_id=interval.interval_id,
                    source_start_time_sec=interval.start_time_sec,
                    source_end_time_sec=interval.end_time_sec,
                    source_normalized_score=interval_normalized_score,
                ),
            )
            for frame in matched
        )

    @staticmethod
    def _validate_frame_metadata(video_id: str, frames: Sequence[FrameMetadata]) -> None:
        if any(not isinstance(frame, FrameMetadata) for frame in frames):
            raise ContractMismatchError("metadata port returned invalid frame metadata")
        wrong_video = tuple(frame.video_id for frame in frames if frame.video_id != video_id)
        if wrong_video:
            raise ContractMismatchError(
                "ASR mapper received frame metadata from another video",
                details={"expected_video_id": video_id, "wrong_video_count": len(wrong_video)},
            )
        ordering = tuple(
            (frame.local_index, frame.timestamp_sec, frame.frame_id)
            for frame in frames
        )
        if ordering != tuple(sorted(ordering)):
            raise ContractMismatchError(
                "ASR mapper requires metadata ordered by local_index",
                details={"video_id": video_id},
            )
        if len({frame.local_index for frame in frames}) != len(frames):
            raise ContractMismatchError(
                "ASR mapper received duplicate local_index metadata",
                details={"video_id": video_id},
            )

    def _matched_frames(
        self,
        interval: ASRIntervalCandidate,
        ordered_frames: Sequence[FrameMetadata],
    ) -> tuple[FrameMetadata, ...]:
        return tuple(
            frame
            for frame in ordered_frames
            if interval.start_time_sec <= frame.timestamp_sec <= interval.end_time_sec
        )

    def _limit_frames(
        self,
        interval: ASRIntervalCandidate,
        matched_frames: Sequence[FrameMetadata],
    ) -> tuple[FrameMetadata, ...]:
        matched = tuple(matched_frames)
        if len(matched) <= self.config.max_frames_per_interval:
            return matched
        center = (interval.start_time_sec + interval.end_time_sec) / 2.0
        return tuple(
            sorted(
                matched,
                key=lambda frame: (
                    abs(frame.timestamp_sec - center),
                    frame.timestamp_sec,
                    frame.local_index,
                    frame.frame_id,
                ),
            )[: self.config.max_frames_per_interval]
        )

    def _interval_normalized_score(self, interval: ASRIntervalCandidate) -> float:
        score = (
            1.0 / (self.config.interval_rrf_k + interval.rank)
            if interval.normalized_score is None
            else interval.normalized_score
        )
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ContractMismatchError("ASR interval normalized score is invalid")
        return score


def _empty_frame_result(result: BranchResult[Any]) -> BranchResult[FrameCandidate]:
    return BranchResult[FrameCandidate](
        branch=result.branch,
        candidate_level=CandidateLevel.FRAME,
        query_variant_id=result.query_variant_id,
        candidates=(),
        requested_top_k=result.requested_top_k,
        latency_ms=result.latency_ms,
        status=result.status,
        warnings=result.warnings,
        missing_metadata_count=result.missing_metadata_count,
    )
