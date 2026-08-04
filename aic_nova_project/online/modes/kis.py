"""KIS orchestration over Person-B retrieval and Person-C ranking."""

from __future__ import annotations

import math
import asyncio
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from threading import Condition
from time import perf_counter
from typing import Any, Protocol, runtime_checkable

from online.domain.base import FiniteFloat, StrictFrozenModel
from online.domain.candidates import BranchResult, FusedFrameCandidate
from online.domain.diagnostics import BranchDiagnostics, QueryDiagnostics
from online.domain.enums import BranchStatus, CandidateLevel, RetrievalBranch
from online.domain.errors import (
    BranchTimeoutError,
    ContractMismatchError,
    ErrorCode,
    InvalidQueryError,
    ResourceUnavailableError,
)
from online.domain.query import QueryBundle
from online.ports.metadata import MetadataReaderPort
from online.ports.objects import ObjectReaderPort
from online.ranking.aggregation import RRFQueryVariantAggregator
from online.ranking.asr_mapper import ASRIntervalFrameMapper
from online.ranking.dedup import CompetitionFrameDeduplicator
from online.ranking.fusion import FRAME_FUSION_BRANCHES, FusionConfig, WeightedFrameFusion
from online.ranking.normalizers import RRFScoreNormalizer, ScoreNormalizer
from online.ranking.object_filter import ObjectConstraintProcessor, ObjectProcessingConfig
from online.ranking.summary import SummaryScorePropagator
from online.ranking.sorting import fused_candidate_sort_key
from online.retrieval.service import RetrievalServicePort
from query_understanding.providers.objects import OBJECT_LABEL_NORMALIZER_VERSION


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
        dedup: CompetitionFrameDeduplicator | None = None,
        object_config: ObjectProcessingConfig | None = None,
        final_top_k: int = 100,
        policy_name: str = "person_c_experimental_baseline_v1",
        policy_status: str = "experimental",
        core_visual_policy: str = "q0_required",
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
        self.dedup = dedup or CompetitionFrameDeduplicator()
        self.object_config = object_config or ObjectProcessingConfig()
        if (
            isinstance(final_top_k, bool)
            or not isinstance(final_top_k, int)
            or final_top_k < 1
        ):
            raise ValueError("final_top_k must be a positive integer")
        self.final_top_k = final_top_k
        self.policy_name = policy_name
        self.policy_status = policy_status
        if core_visual_policy != "q0_required":
            raise ValueError("only q0_required core visual policy is implemented")
        self.core_visual_policy = core_visual_policy

    def validate_bundle(self, bundle: QueryBundle) -> None:
        if not isinstance(bundle, QueryBundle):
            raise ContractMismatchError("bundle must be a validated QueryBundle")
        if (
            self.core_visual_policy == "q0_required"
            and RetrievalBranch.VISUAL_DENSE not in bundle.enabled_branches
        ):
            raise InvalidQueryError(
                "KIS baseline requires visual_dense retrieval",
                details={"branch": RetrievalBranch.VISUAL_DENSE.value},
            )

    def rank(
        self,
        bundle: QueryBundle,
        branch_results: Sequence[BranchResult[Any]],
    ) -> KISSearchResult:
        self.validate_bundle(bundle)
        started_at = perf_counter()
        stage_latencies: dict[str, float] = {}
        mapping_loss_count = 0
        mapping_losses_by_branch: dict[RetrievalBranch, int] = defaultdict(int)
        mapped_outputs_by_branch: dict[RetrievalBranch, int] = defaultdict(int)
        asr_stats: dict[str, int] = defaultdict(int)
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
                asr_stats["input_interval_count"] += mapped.input_interval_count
                asr_stats["mapped_interval_count"] += mapped.mapped_interval_count
                asr_stats["output_frame_count"] += mapped.output_frame_count
                asr_stats["truncated_interval_count"] += mapped.truncated_interval_count
                asr_stats["truncated_frame_count"] += mapped.truncated_frame_count
                asr_stats["max_frames_per_interval"] = mapped.max_frames_per_interval
                frame_results.append(mapped.branch_result)
            elif result.candidate_level is CandidateLevel.VIDEO:
                summary_results.append(result)
        stage_latencies["asr_mapping"] = _elapsed_ms(stage_start)

        stage_start = perf_counter()
        aggregated = self.aggregator.aggregate(tuple(frame_results))
        stage_latencies["query_variant_aggregation"] = _elapsed_ms(stage_start)

        stage_start = perf_counter()
        normalized = _normalize_branch_results(aggregated, self.normalizer)
        weighted = self.aggregator.apply_normalized_weights(normalized)
        stage_latencies["branch_normalization"] = _elapsed_ms(stage_start)

        stage_start = perf_counter()
        fused = self.fusion.fuse(weighted)
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

        stage_start = perf_counter()
        final_candidates = tuple(
            sorted(deduped, key=fused_candidate_sort_key)
        )[: self.final_top_k]
        stage_latencies["final_sort_top_k"] = _elapsed_ms(stage_start)

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
            missing_metadata_count=sum(result.missing_metadata_count for result in raw_results),
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
                policy_name=self.policy_name,
                policy_status=self.policy_status,
                asr_stats=asr_stats,
                object_method=ObjectConstraintProcessor.name,
                dedup_method=self.dedup.name,
                normalizer=self.normalizer,
                aggregator=self.aggregator,
                summary=self.summary,
                object_config=self.object_config,
                final_top_k=self.final_top_k,
            ),
            errors=(),
        )
        return KISSearchResult(candidates=final_candidates, diagnostics=diagnostics)

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
        return ObjectConstraintProcessor(
            self.object_reader,
            config=self.object_config,
        ).process(
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
        self._state = Condition()
        self._active_requests = 0
        self._closing = False
        self._closed = False

    async def search(self, bundle: QueryBundle) -> KISSearchResult:
        if not isinstance(bundle, QueryBundle):
            raise ContractMismatchError("bundle must be a validated QueryBundle")
        with self._state:
            if self._closing or self._closed:
                raise ResourceUnavailableError(
                    "Search orchestrator is closing",
                    details={"resource": "search_orchestrator"},
                )
            self._active_requests += 1
        try:
            validate_bundle = getattr(self.ranking, "validate_bundle", None)
            if callable(validate_bundle):
                validate_bundle(bundle)
            branch_results = await self.retrieval.retrieve(bundle)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._ranking_executor,
                self.ranking.rank,
                bundle,
                branch_results,
            )
        finally:
            with self._state:
                self._active_requests -= 1
                self._state.notify_all()

    def close(self, *, wait: bool = True) -> None:
        with self._state:
            if self._closed:
                return
            self._closing = True
            if wait:
                while self._active_requests:
                    self._state.wait()
            elif self._active_requests:
                raise ResourceUnavailableError(
                    "Search orchestrator has active requests",
                    details={"resource": "search_orchestrator"},
                )
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
            status=_combined_branch_status(branch, values),
            latency_ms=max(result.latency_ms for result in values),
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


def _combined_branch_status(
    branch: RetrievalBranch,
    results: Sequence[BranchResult[Any]],
) -> BranchStatus:
    if branch not in CORE_BRANCHES:
        return _combined_status(tuple(result.status for result in results))
    q0 = tuple(result for result in results if result.query_variant_id == "q0")
    if q0 and all(result.status is BranchStatus.SUCCESS for result in q0):
        if any(result.status is not BranchStatus.SUCCESS for result in results):
            return BranchStatus.DEGRADED
        return BranchStatus.SUCCESS
    return _combined_status(tuple(result.status for result in results))


def _fusion_weights(config: FusionConfig) -> Mapping[RetrievalBranch, FiniteFloat]:
    return {
        branch: config.weight_for(branch)
        for branch in FRAME_FUSION_BRANCHES
        if config.weight_for(branch) > 0.0
    }


def _raise_for_core_branch_failure(results: Sequence[BranchResult[Any]]) -> None:
    grouped: dict[RetrievalBranch, list[BranchResult[Any]]] = defaultdict(list)
    for result in results:
        grouped[result.branch].append(result)
    for branch in CORE_BRANCHES:
        values = tuple(grouped.get(branch, ()))
        q0 = tuple(result for result in values if result.query_variant_id == "q0")
        if not q0:
            raise ResourceUnavailableError(
                "Core visual q0 retrieval did not return a result",
                details={"branch": branch.value, "query_variant_id": "q0"},
            )
        failed = tuple(
            result
            for result in q0
            if result.status in {BranchStatus.FAILED, BranchStatus.DISABLED}
        )
        if failed:
            warnings = tuple(warning for result in failed for warning in result.warnings)
            details = {
                "branch": branch.value,
                "query_variant_id": "q0",
                "status": _combined_status(tuple(result.status for result in failed)).value,
            }
            if any(warning == ErrorCode.BRANCH_TIMEOUT.value for warning in warnings):
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
    policy_name: str,
    policy_status: str,
    asr_stats: Mapping[str, int],
    object_method: str,
    dedup_method: str,
    normalizer: ScoreNormalizer,
    aggregator: RRFQueryVariantAggregator,
    summary: SummaryScorePropagator,
    object_config: ObjectProcessingConfig,
    final_top_k: int,
) -> tuple[str, ...]:
    values = [
        f"ranking_policy={policy_name}:{policy_status}",
        f"aggregation_method={aggregation_method}",
        f"summary_method={summary_method}",
        f"asr_mapping_method={asr_mapping_method}",
        f"object_method={object_method}",
        f"dedup_method={dedup_method}",
        "branch_latency_method=max_variant_latency",
        f"normalization_rrf_k={getattr(normalizer, 'k', 'n/a')}",
        "query_variant_weights="
        + ",".join(
            f"{variant_id}:{weight}"
            for variant_id, weight in sorted(aggregator.config.query_variant_weights.items())
        ),
        f"summary_weight={summary.config.weight}",
        f"summary_max_boost={summary.config.max_boost}",
        f"summary_fallback_rrf_k={summary.config.fallback_rrf_k}",
        f"object_soft_boost_per_constraint={object_config.soft_boost_per_constraint}",
        f"object_max_total_boost={object_config.max_total_boost}",
        f"object_label_normalizer={OBJECT_LABEL_NORMALIZER_VERSION}",
        f"object_position_policy={object_config.position_policy_name}",
        f"final_top_k={final_top_k}",
    ]
    values.extend(f"asr_{key}={value}" for key, value in sorted(asr_stats.items()))
    values.extend(
        f"branch={result.branch.value};query_variant_id={result.query_variant_id};code={warning}"
        for result in results
        if result.status is not BranchStatus.SUCCESS
        and not (result.branch in CORE_BRANCHES and result.query_variant_id == "q0")
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
