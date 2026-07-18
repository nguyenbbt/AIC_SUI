"""KIS orchestration over Person-B retrieval and Person-C ranking."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

from online.domain.base import FiniteFloat, StrictFrozenModel
from online.domain.candidates import BranchResult, FusedFrameCandidate
from online.domain.diagnostics import BranchDiagnostics, QueryDiagnostics
from online.domain.enums import BranchStatus, CandidateLevel, RetrievalBranch
from online.domain.errors import ContractMismatchError
from online.domain.query import QueryBundle
from online.ports.metadata import MetadataReaderPort
from online.ports.objects import ObjectReaderPort
from online.ranking.aggregation import RRFQueryVariantAggregator
from online.ranking.asr_mapper import ASRIntervalFrameMapper
from online.ranking.dedup import ShotDeduplicator
from online.ranking.fusion import FusionConfig, WeightedFrameFusion
from online.ranking.object_filter import ObjectConstraintProcessor
from online.ranking.summary import SummaryScorePropagator
from online.retrieval.service import RetrievalServicePort


class KISSearchResult(StrictFrozenModel):
    candidates: tuple[FusedFrameCandidate, ...]
    diagnostics: QueryDiagnostics


class KISRankingService:
    """Rank already-retrieved KIS BranchResults into final frame candidates."""

    def __init__(
        self,
        *,
        metadata: MetadataReaderPort,
        object_reader: ObjectReaderPort | None = None,
        asr_mapper: ASRIntervalFrameMapper | None = None,
        aggregator: RRFQueryVariantAggregator | None = None,
        fusion: WeightedFrameFusion | None = None,
        summary: SummaryScorePropagator | None = None,
        dedup: ShotDeduplicator | None = None,
    ) -> None:
        if not isinstance(metadata, MetadataReaderPort):
            raise TypeError("metadata must implement MetadataReaderPort")
        if object_reader is not None and not isinstance(object_reader, ObjectReaderPort):
            raise TypeError("object_reader must implement ObjectReaderPort")
        self.metadata = metadata
        self.object_reader = object_reader
        self.asr_mapper = asr_mapper or ASRIntervalFrameMapper()
        self.aggregator = aggregator or RRFQueryVariantAggregator()
        self.fusion = fusion or WeightedFrameFusion()
        self.summary = summary or SummaryScorePropagator()
        self.dedup = dedup or ShotDeduplicator()

    def rank(
        self,
        bundle: QueryBundle,
        branch_results: Sequence[BranchResult[Any]],
    ) -> KISSearchResult:
        if not isinstance(bundle, QueryBundle):
            raise ContractMismatchError("bundle must be a validated QueryBundle")
        started_at = perf_counter()
        stage_latencies: dict[str, float] = {}
        mapping_loss_count = 0
        mapping_losses_by_branch: dict[RetrievalBranch, int] = defaultdict(int)
        mapped_outputs_by_branch: dict[RetrievalBranch, int] = defaultdict(int)
        raw_results = _as_branch_results(branch_results)

        stage_start = perf_counter()
        frame_results: list[BranchResult[Any]] = []
        summary_results: list[BranchResult[Any]] = []
        for result in raw_results:
            if result.candidate_level is CandidateLevel.FRAME:
                frame_results.append(result)
            elif result.candidate_level is CandidateLevel.ASR_INTERVAL:
                mapped = self.asr_mapper.map_result(result, self.metadata)
                mapping_loss_count += mapped.mapping_loss_count
                mapping_losses_by_branch[result.branch] += mapped.mapping_loss_count
                mapped_outputs_by_branch[result.branch] += mapped.branch_result.returned_count
                frame_results.append(mapped.branch_result)
            elif result.candidate_level is CandidateLevel.VIDEO:
                summary_results.append(result)
        stage_latencies["asr_mapping"] = _elapsed_ms(stage_start)

        stage_start = perf_counter()
        aggregated = self.aggregator.aggregate(tuple(frame_results))
        stage_latencies["query_variant_aggregation"] = _elapsed_ms(stage_start)

        stage_start = perf_counter()
        fused = self.fusion.fuse(aggregated)
        stage_latencies["fusion"] = _elapsed_ms(stage_start)

        stage_start = perf_counter()
        boosted = self.summary.propagate(fused, tuple(summary_results))
        stage_latencies["summary_propagation"] = _elapsed_ms(stage_start)

        stage_start = perf_counter()
        object_filtered = self._apply_objects(bundle, boosted)
        object_filter_removals = len(boosted) - len(object_filtered)
        stage_latencies["object_processing"] = _elapsed_ms(stage_start)

        stage_start = perf_counter()
        deduped = self.dedup.deduplicate(object_filtered)
        dedup_removals = len(object_filtered) - len(deduped)
        stage_latencies["dedup"] = _elapsed_ms(stage_start)

        total_latency_ms = _elapsed_ms(started_at)
        diagnostics = QueryDiagnostics(
            query_id=bundle.query_id,
            total_latency_ms=total_latency_ms,
            stage_latencies_ms=stage_latencies,
            branches=_branch_diagnostics(
                raw_results,
                mapping_losses_by_branch=mapping_losses_by_branch,
                mapped_outputs_by_branch=mapped_outputs_by_branch,
            ),
            missing_metadata_count=0,
            object_filter_removals=object_filter_removals,
            dedup_removals=dedup_removals,
            normalization_method=self.aggregator.name,
            fusion_method=self.fusion.name,
            fusion_weights=_fusion_weights(self.fusion.config),
            warnings=(),
            errors=(),
        )
        return KISSearchResult(candidates=deduped, diagnostics=diagnostics)

    def _apply_objects(
        self,
        bundle: QueryBundle,
        candidates: tuple[FusedFrameCandidate, ...],
    ) -> tuple[FusedFrameCandidate, ...]:
        if not bundle.object_constraints:
            return candidates
        if self.object_reader is None:
            raise ContractMismatchError(
                "object constraints require an ObjectReaderPort"
            )
        return ObjectConstraintProcessor(self.object_reader).process(
            candidates,
            bundle.object_constraints,
        )


class KISSearchOrchestrator:
    """Async C-07 orchestrator for the internal KIS baseline."""

    def __init__(
        self,
        *,
        retrieval: RetrievalServicePort,
        ranking: KISRankingService,
    ) -> None:
        if not isinstance(retrieval, RetrievalServicePort):
            raise TypeError("retrieval must implement RetrievalServicePort")
        if not isinstance(ranking, KISRankingService):
            raise TypeError("ranking must be a KISRankingService")
        self.retrieval = retrieval
        self.ranking = ranking

    async def search(self, bundle: QueryBundle) -> KISSearchResult:
        if not isinstance(bundle, QueryBundle):
            raise ContractMismatchError("bundle must be a validated QueryBundle")
        branch_results = await self.retrieval.retrieve(bundle)
        return self.ranking.rank(bundle, branch_results)


def _branch_diagnostics(
    results: Sequence[BranchResult[Any]],
    *,
    mapping_losses_by_branch: Mapping[RetrievalBranch, int],
    mapped_outputs_by_branch: Mapping[RetrievalBranch, int],
) -> Mapping[RetrievalBranch, BranchDiagnostics]:
    grouped: dict[RetrievalBranch, list[BranchResult[Any]]] = defaultdict(list)
    for result in results:
        grouped[result.branch].append(result)
    diagnostics: dict[RetrievalBranch, BranchDiagnostics] = {}
    for branch, values in grouped.items():
        diagnostics[branch] = BranchDiagnostics(
            status=_combined_status(tuple(result.status for result in values)),
            latency_ms=sum(result.latency_ms for result in values),
            requested_top_k=max(result.requested_top_k for result in values),
            raw_result_count=sum(result.returned_count for result in values),
            output_candidate_count=mapped_outputs_by_branch.get(
                branch,
                sum(result.returned_count for result in values),
            ),
            mapping_loss_count=mapping_losses_by_branch.get(branch, 0),
            warnings=tuple(
                dict.fromkeys(
                    warning
                    for result in values
                    for warning in result.warnings
                )
            ),
        )
    return diagnostics


def _combined_status(statuses: Sequence[BranchStatus]) -> BranchStatus:
    if any(status is BranchStatus.FAILED for status in statuses):
        return BranchStatus.FAILED
    if any(status is BranchStatus.DEGRADED for status in statuses):
        return BranchStatus.DEGRADED
    if all(status is BranchStatus.DISABLED for status in statuses):
        return BranchStatus.DISABLED
    return BranchStatus.SUCCESS


def _fusion_weights(config: FusionConfig) -> Mapping[RetrievalBranch, FiniteFloat]:
    weights = {
        branch: config.weight_for(branch)
        for branch in RetrievalBranch
        if config.weight_for(branch) > 0.0
    }
    return weights or {RetrievalBranch.VISUAL_DENSE: 1.0}


def _as_branch_results(results: Sequence[BranchResult[Any]]) -> tuple[BranchResult[Any], ...]:
    if isinstance(results, (str, bytes)):
        raise TypeError("branch_results must be a sequence")
    values = tuple(results)
    if any(not isinstance(result, BranchResult) for result in values):
        raise TypeError("branch_results must contain BranchResult objects")
    return values


def _elapsed_ms(started_at: float) -> float:
    elapsed = (perf_counter() - started_at) * 1000.0
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise RuntimeError("monotonic clock returned an invalid duration")
    return elapsed
