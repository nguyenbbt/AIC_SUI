"""Map ASR interval evidence to frame-level evidence for fusion."""

from __future__ import annotations

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

    def __post_init__(self) -> None:
        if not isinstance(self.policy_name, str) or not self.policy_name.strip():
            raise ValueError("policy_name must be non-empty")


@dataclass(frozen=True)
class ASRMappingResult:
    branch_result: BranchResult[FrameCandidate]
    mapping_loss_count: int
    policy_name: str


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
            )

        mapped: list[FrameCandidate] = []
        mapping_loss_count = 0
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
            mapped.extend(interval_frames)

        ordered = tuple(sorted(mapped, key=lambda item: (item.rank, item.timestamp_sec, item.frame_id)))
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
            ),
            mapping_loss_count=mapping_loss_count,
            policy_name=self.name,
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
        matched = tuple(
            frame
            for frame in frames
            if interval.start_time_sec <= frame.timestamp_sec <= interval.end_time_sec
        )
        if not matched:
            return ()

        raw_score = interval.raw_score / len(matched)
        normalized_score = (
            None
            if interval.normalized_score is None
            else interval.normalized_score / len(matched)
        )
        return tuple(
            FrameCandidate(
                frame_id=frame.frame_id,
                video_id=frame.video_id,
                shot_id=frame.shot_id,
                timestamp_sec=frame.timestamp_sec,
                rank=interval.rank,
                raw_score=raw_score,
                normalized_score=normalized_score,
                provenance=CandidateProvenance(
                    branch=interval.provenance.branch,
                    backend="derived",
                    source_resource=interval.provenance.source_resource,
                    query_variant_id=interval.provenance.query_variant_id,
                    query_text=interval.provenance.query_text,
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
    )
