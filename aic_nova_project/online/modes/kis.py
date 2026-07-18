"""KIS orchestration over Person-B retrieval and Person-C ranking."""

from __future__ import annotations

import math
import asyncio
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from time import perf_counter
from typing import Any, Protocol, runtime_checkable

from online.domain.base import FiniteFloat, StrictFrozenModel
from online.domain.candidates import BranchResult, FusedFrameCandidate
from online.domain.diagnostics import BranchDiagnostics, QueryDiagnostics
from online.domain.enums import BranchStatus, CandidateLevel, RetrievalBranch
from online.domain.errors import BranchTimeoutError, ContractMismatchError, ResourceUnavailableError
from online.domain.query import QueryBundle
from online.ports.metadata import MetadataReaderPort
from online.ports.objects import ObjectReaderPort
from online.ranking.aggregation import RRFQueryVariantAggregator
from online.ranking.asr_mapper import ASRIntervalFrameMapper
from online.ranking.dedup import ShotDeduplicator
from online.ranking.fusion import FusionConfig, WeightedFrameFusion
from online.ranking.normalizers import RRFScoreNormalizer, ScoreNormalizer
from online.ranking.object_filter import ObjectConstraintProcessor
from online.ranking.summary import SummaryScorePropagator
from online.retrieval.service import RetrievalServicePort


CORE_BRANCHES = frozenset({RetrievalBranch.VISUAL_DENSE})


class KISSearchResult(StrictFrozenModel):
    candidates: tuple[FusedFrameCandidate, ...]
    diagnostics: QueryDiagnostics


@runtime_checkable
class RankingServicePort(Protocol):
    def rank(
        self,
        bundle: QueryBundle,
        branch_results: Sequence[BranchResult[Any]],
    ) -> KISSearchResult: ...


class KISRankingService:
    """Rank already-retrieved KIS BranchResults into final frame candidates."""

    def __init__(
        self,
        *,
        metadata: MetadataReaderPort,
        object_reader: ObjectReaderPort | None = None,
        asr_mapper: ASRIntervalFrameMapper | None = None,
        aggregator: RRFQueryVariantAggregator | None = None,
        normalizer: ScoreNormalizer | None = None,
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
        self.normalizer = normalizer or RRFScoreNormalizer()
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
        _raise_for_core_branch_failure(raw_results)

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
        normalized = _normalize_branch_results(tuple(frame_results), self.normalizer)
        stage_latencies["branch_normalization"] = _elapsed_ms(stage_start)

        stage_start = perf_counter()
        aggregated = self.aggregator.aggregate(normalized)
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
            normalization_method=self.normalizer.name,
            fusion_method=self.fusion.name,
            fusion_weights=_fusion_weights(self.fusion.config),
            warnings=_query_warnings(
                raw_results,
                aggregation_method=self.aggregator.name,
                summary_method=self.summary.name,
                asr_mapping_method=self.asr_mapper.name,
            ),
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
        ranking: RankingServicePort,
        ranking_executor: Executor | None = None,
    ) -> None:
        if not isinstance(retrieval, RetrievalServicePort):
            raise TypeError("retrieval must implement RetrievalServicePort")
        if not isinstance(ranking, RankingServicePort):
            raise TypeError("ranking must implement RankingServicePort")
        self.retrieval = retrieval
        self.ranking = ranking
        self._owns_executor = ranking_executor is None
        self._ranking_executor = ranking_executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="aic-ranking",
        )
        self._closed = False

    async def search(self, bundle: QueryBundle) -> KISSearchResult:
        if not isinstance(bundle, QueryBundle):
            raise ContractMismatchError("bundle must be a validated QueryBundle")
        if self._closed:
            raise ResourceUnavailableError(
                "Search orchestrator is closed",
                details={"resource": "search_orchestrator"},
            )
        branch_results = await self.retrieval.retrieve(bundle)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._ranking_executor,
            self.ranking.rank,
            bundle,
            branch_results,
        )

    def close(self, *, wait: bool = True) -> None:
        self._closed = True
        if self._owns_executor and isinstance(self._ranking_executor, ThreadPoolExecutor):
            self._ranking_executor.shutdown(wait=wait, cancel_futures=True)


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


def _raise_for_core_branch_failure(results: Sequence[BranchResult[Any]]) -> None:
    grouped: dict[RetrievalBranch, list[BranchResult[Any]]] = defaultdict(list)
    for result in results:
        grouped[result.branch].append(result)
    for branch in CORE_BRANCHES:
        values = tuple(grouped.get(branch, ()))
        if not values:
            raise ResourceUnavailableError(
                "Core visual retrieval branch did not return a result",
                details={"branch": branch.value},
            )
        failed = tuple(
            result
            for result in values
            if result.status in {BranchStatus.FAILED, BranchStatus.DISABLED}
        )
        if failed:
            warnings = tuple(warning for result in failed for warning in result.warnings)
            details = {"branch": branch.value, "status": _combined_status(tuple(result.status for result in failed)).value}
            if any("BRANCH_TIMEOUT" in warning for warning in warnings):
                raise BranchTimeoutError("Core visual retrieval timed out", details=details)
            raise ResourceUnavailableError("Core visual retrieval is unavailable", details=details)


def _normalize_branch_results(
    results: Sequence[BranchResult[Any]],
    normalizer: ScoreNormalizer,
) -> tuple[BranchResult[Any], ...]:
    normalized: list[BranchResult[Any]] = []
    for result in results:
        if result.status not in {BranchStatus.SUCCESS, BranchStatus.DEGRADED}:
            normalized.append(result)
            continue
        if all(candidate.normalized_score is not None for candidate in result.candidates):
            normalized.append(result)
            continue
        if any(candidate.normalized_score is not None for candidate in result.candidates):
            raise ContractMismatchError(
                "BranchResult mixes normalized and unnormalized candidates",
                details={"branch": result.branch.value, "query_variant_id": result.query_variant_id},
            )
        normalized.append(
            result.model_copy(update={"candidates": normalizer.normalize(result.candidates)})
        )
    return tuple(normalized)


def _query_warnings(
    results: Sequence[BranchResult[Any]],
    *,
    aggregation_method: str,
    summary_method: str,
    asr_mapping_method: str,
) -> tuple[str, ...]:
    values = [
        "experimental_policy=rrf_k_60_equal_weight_fusion_summary_cap_v1",
        f"aggregation_method={aggregation_method}",
        f"summary_method={summary_method}",
        f"asr_mapping_method={asr_mapping_method}",
    ]
    values.extend(
        f"{result.branch.value}:{warning}"
        for result in results
        if result.branch not in CORE_BRANCHES and result.status is not BranchStatus.SUCCESS
        for warning in result.warnings
    )
    return tuple(dict.fromkeys(values))


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
