"""Runtime wiring for A adapters, B retrieval and C orchestration."""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from fastapi import FastAPI

from online.adapters.elasticsearch import ElasticsearchSearchAdapter
from online.adapters.evidence import ElasticsearchEvidenceHydrator
from online.adapters.images import FilesystemImageResolver
from online.adapters.manifest import DatasetManifestGate
from online.adapters.milvus import MilvusSearchAdapter
from online.adapters.sqlite import SQLiteReadAdapter
from online.adapters.qwen_vlm import DEFAULT_QWEN_MODEL, DEFAULT_QWEN_REVISION, QwenVLMAdapter
from online.adapters.visual_corpus import MilvusSQLiteVisualCorpusAdapter
from online.config import OnlineDataConfig
from online.domain.enums import RetrievalBranch
from online.lifecycle import (
    ComponentHealth,
    HealthStatus,
    InfrastructureHealth,
    InfrastructureLifecycle,
)
from online.modes.kis import KISRankingService, KISSearchOrchestrator
from online.modes.trake import TRAKEModeAdapter
from online.modes.vqa import VQAModeAdapter
from online.ports import (
    ElasticsearchSearchPort,
    EvidenceHydrationPort,
    ImageResolverPort,
    MetadataReaderPort,
    MilvusSearchPort,
    ObjectReaderPort,
    ObjectCatalogPort,
    TextEncoderPort,
    VLMPort,
    VisualCorpusPort,
)
from online.ranking.aggregation import QueryVariantAggregationConfig, RRFQueryVariantAggregator
from online.ranking.asr_mapper import ASRIntervalFrameMapper, ASRMappingConfig
from online.ranking.fusion import FRAME_FUSION_BRANCHES, FusionConfig, WeightedFrameFusion
from online.ranking.normalizers import RRFScoreNormalizer
from online.ranking.object_filter import ObjectProcessingConfig
from online.ranking.policy import RankingPolicyConfig
from online.ranking.summary import SummaryPropagationConfig, SummaryScorePropagator
from online.retrieval.encoders import OpenCLIPTextEncoder, VietnameseTextEncoder
from online.retrieval.factory import build_retrieval_service
from online.retrieval.query_builder import BASELINE_KIS_BRANCHES
from online.retrieval.service import RetrievalInvocationConfig, RetrievalService
from online.retrieval.vqa import VQACandidateRetriever
from online.trake import TRAKEService
from online.vqa import EvidenceSelector, VQAOrchestrator
from query_understanding.openai_rewriter import DEFAULT_REWRITE_MODEL, OpenAIQueryRewriter
from query_understanding.rewrite import NoOpQueryRewriter, QueryRewriteService
from retrieval_api.search_engine import HealthResponse, create_app
from retrieval_api.ui_resources import DatasetUIResources


VARIANT_IDS = ("q0", "q1")


@dataclass(frozen=True)
class RuntimeCompositionConfig:
    max_workers: int = 8
    ranking_max_workers: int = 2
    default_top_k: int = 50
    default_timeout_sec: float = 5.0
    branch_top_k: Mapping[RetrievalBranch, int] | None = None
    branch_timeout_sec: Mapping[RetrievalBranch, float] | None = None
    visual_expected_dimension: int | None = None
    vietnamese_expected_dimension: int | None = None
    ranking_policy: RankingPolicyConfig | None = None
    deployment_mode: str = "development"
    trake_enabled: bool = False
    vqa_enabled: bool = False
    query_rewrite_enabled: bool = False
    qwen_vlm_auto_configure: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.deployment_mode, str):
            raise ValueError("deployment_mode must be development, test or production")
        deployment_mode = self.deployment_mode.strip().lower()
        if deployment_mode not in {"development", "test", "production"}:
            raise ValueError("deployment_mode must be development, test or production")
        object.__setattr__(self, "deployment_mode", deployment_mode)
        if (
            not isinstance(self.trake_enabled, bool)
            or not isinstance(self.vqa_enabled, bool)
            or not isinstance(self.query_rewrite_enabled, bool)
            or not isinstance(self.qwen_vlm_auto_configure, bool)
        ):
            raise ValueError("advanced mode flags must be booleans")
        if _invalid_positive_int(self.max_workers):
            raise ValueError("max_workers must be a positive integer")
        if _invalid_positive_int(self.ranking_max_workers):
            raise ValueError("ranking_max_workers must be a positive integer")
        if _invalid_positive_int(self.default_top_k):
            raise ValueError("default_top_k must be a positive integer")
        if _invalid_positive_float(self.default_timeout_sec):
            raise ValueError("default_timeout_sec must be > 0")
        for name, mapping in (
            ("branch_top_k", self.branch_top_k or {}),
            ("branch_timeout_sec", self.branch_timeout_sec or {}),
        ):
            for raw_branch, value in mapping.items():
                RetrievalBranch(raw_branch)
                if name == "branch_top_k" and _invalid_positive_int(value):
                    raise ValueError("branch top_k values must be positive integers")
                if name == "branch_timeout_sec" and _invalid_positive_float(value):
                    raise ValueError("branch timeout values must be > 0")
        object.__setattr__(
            self,
            "branch_top_k",
            None if self.branch_top_k is None else MappingProxyType(dict(self.branch_top_k)),
        )
        object.__setattr__(
            self,
            "branch_timeout_sec",
            None if self.branch_timeout_sec is None else MappingProxyType(dict(self.branch_timeout_sec)),
        )
        for value in (self.visual_expected_dimension, self.vietnamese_expected_dimension):
            if value is not None and _invalid_positive_int(value):
                raise ValueError("expected dimensions must be positive integers")
        policy = self.ranking_policy or RankingPolicyConfig()
        if deployment_mode == "production" and policy.policy_status == "experimental":
            raise ValueError("experimental ranking policy is not allowed in production mode")
        object.__setattr__(self, "ranking_policy", policy)

    @classmethod
    def from_env(cls, prefix: str = "AIC_ONLINE_") -> "RuntimeCompositionConfig":
        return cls(
            max_workers=_env_int(prefix, "RETRIEVAL_MAX_WORKERS", 8),
            ranking_max_workers=_env_int(prefix, "RANKING_MAX_WORKERS", 2),
            default_top_k=_env_int(prefix, "RETRIEVAL_TOP_K", 50),
            default_timeout_sec=_env_float(prefix, "RETRIEVAL_TIMEOUT_SEC", 5.0),
            branch_top_k={
                branch: _env_int(prefix, f"{branch.name}_TOP_K", _env_int(prefix, "RETRIEVAL_TOP_K", 50))
                for branch in RetrievalBranch
            },
            branch_timeout_sec={
                branch: _env_float(
                    prefix,
                    f"{branch.name}_TIMEOUT_SEC",
                    _env_float(prefix, "RETRIEVAL_TIMEOUT_SEC", 5.0),
                )
                for branch in RetrievalBranch
            },
            visual_expected_dimension=_env_optional_int(prefix, "VISUAL_ENCODER_DIMENSION"),
            vietnamese_expected_dimension=_env_optional_int(prefix, "VIETNAMESE_ENCODER_DIMENSION"),
            ranking_policy=_ranking_policy_from_env(prefix),
            deployment_mode=os.getenv(f"{prefix}DEPLOYMENT_MODE", "development").strip() or "development",
            trake_enabled=_env_bool(prefix, "TRAKE_ENABLED", False),
            vqa_enabled=_env_bool(prefix, "VQA_ENABLED", False),
            query_rewrite_enabled=_env_bool(prefix, "QUERY_REWRITE_ENABLED", False),
            qwen_vlm_auto_configure=_env_bool(prefix, "QWEN_VLM_AUTO_CONFIGURE", False),
        )

    def top_k_for(self, branch: RetrievalBranch) -> int:
        return int((self.branch_top_k or {}).get(branch, self.default_top_k))

    def timeout_for(self, branch: RetrievalBranch) -> float:
        return float((self.branch_timeout_sec or {}).get(branch, self.default_timeout_sec))


@dataclass
class OnlineRuntime:
    orchestrator: KISSearchOrchestrator
    lifecycle: InfrastructureLifecycle
    retrieval: RetrievalService
    ranking_executor: ThreadPoolExecutor
    trake_mode: TRAKEModeAdapter | None = None
    vqa_mode: VQAModeAdapter | None = None
    advanced_resources: tuple[Any, ...] = ()
    readiness_probes: tuple["RuntimeReadinessProbe", ...] = ()
    readiness_components: tuple[ComponentHealth, ...] = ()
    last_health: InfrastructureHealth | None = None
    ui_resources: DatasetUIResources | None = None
    rewriter: QueryRewriteService | None = None

    def start(self) -> InfrastructureHealth:
        infrastructure = self.lifecycle.start()
        self.readiness_components = tuple(_run_readiness_probe(probe) for probe in self.readiness_probes)
        self.last_health = _merge_health(infrastructure, self.readiness_components)
        return self.last_health

    def health(self) -> InfrastructureHealth:
        self.last_health = _merge_health(self.lifecycle.health(), self.readiness_components)
        return self.last_health

    def close(self) -> None:
        try:
            if self.vqa_mode is not None:
                self.vqa_mode.close(wait=True)
        finally:
            try:
                if self.trake_mode is not None:
                    self.trake_mode.close(wait=True)
            finally:
                try:
                    self.orchestrator.close(wait=True)
                finally:
                    try:
                        self.retrieval.close(wait=True)
                    finally:
                        try:
                            self.ranking_executor.shutdown(wait=True, cancel_futures=True)
                        finally:
                            try:
                                self.lifecycle.close()
                            finally:
                                for resource in reversed(self.advanced_resources):
                                    resource.close(wait=True)


@dataclass(frozen=True)
class RuntimeReadinessProbe:
    name: str
    required: bool
    check: Callable[[], None]


class AdvancedManagedResourcePort(Protocol):
    def close(self, *, wait: bool = True) -> None: ...


@dataclass(frozen=True)
class AdvancedModeDependencies:
    """Production-only public ports required by TRAKE and VQA composition."""

    visual_corpus: VisualCorpusPort
    event_encoder: TextEncoderPort
    metadata_reader: MetadataReaderPort
    image_resolver: ImageResolverPort
    evidence_hydrator: EvidenceHydrationPort
    vlm: VLMPort
    managed_resources: tuple[AdvancedManagedResourcePort, ...] = ()


def build_invocation_configs(
    config: RuntimeCompositionConfig,
) -> Mapping[tuple[RetrievalBranch, str], RetrievalInvocationConfig]:
    return {
        (branch, variant_id): RetrievalInvocationConfig(
            top_k=config.top_k_for(branch),
            timeout_sec=config.timeout_for(branch),
        )
        for branch in BASELINE_KIS_BRANCHES
        for variant_id in VARIANT_IDS
    }


def build_online_runtime(
    *,
    data_config: OnlineDataConfig | None = None,
    runtime_config: RuntimeCompositionConfig | None = None,
    milvus: MilvusSearchPort | None = None,
    elasticsearch: ElasticsearchSearchPort | None = None,
    metadata: MetadataReaderPort | None = None,
    object_reader: ObjectReaderPort | None = None,
    visual_encoder: TextEncoderPort | None = None,
    vietnamese_encoder: TextEncoderPort | None = None,
    trake_mode: TRAKEModeAdapter | None = None,
    vqa_mode: VQAModeAdapter | None = None,
    advanced_vlm: VLMPort | None = None,
    advanced_rewriter: QueryRewriteService | None = None,
) -> OnlineRuntime:
    data_config = data_config or OnlineDataConfig.from_env()
    runtime_config = runtime_config or RuntimeCompositionConfig.from_env()
    if runtime_config.deployment_mode == "production":
        if not data_config.dataset.manifest_required:
            raise ValueError("production requires the dataset manifest startup gate")
        if data_config.dataset.expected_fingerprint is None:
            raise ValueError("production requires DATASET_EXPECTED_FINGERPRINT")
    if runtime_config.vqa_enabled and vqa_mode is not None:
        raise ValueError("vqa_mode and vqa_enabled cannot both configure VQA")
    if runtime_config.trake_enabled and trake_mode is not None:
        raise ValueError("trake_mode and trake_enabled cannot both configure TRAKE")
    if runtime_config.query_rewrite_enabled and advanced_rewriter is None:
        api_key = os.getenv("AIC_ONLINE_OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("QUERY_REWRITE_ENABLED requires AIC_ONLINE_OPENAI_API_KEY")
        advanced_rewriter = QueryRewriteService(
            OpenAIQueryRewriter(
                api_key=api_key,
                base_url=os.getenv(
                    "AIC_ONLINE_OPENAI_BASE_URL", "https://api.openai.com/v1"
                ),
                model=os.getenv("AIC_ONLINE_OPENAI_REWRITE_MODEL", DEFAULT_REWRITE_MODEL),
                timeout_sec=_env_float("AIC_ONLINE_", "OPENAI_REWRITE_TIMEOUT_SEC", 4.5),
            ),
            timeout_sec=5.0,
        )
    if (
        runtime_config.vqa_enabled
        and advanced_vlm is None
        and runtime_config.qwen_vlm_auto_configure
    ):
        advanced_vlm = QwenVLMAdapter(
            data_root=data_config.dataset.data_root,
            base_url=os.getenv("AIC_ONLINE_QWEN_VLM_BASE_URL", "http://localhost:8001/v1"),
            model=os.getenv("AIC_ONLINE_QWEN_VLM_MODEL", DEFAULT_QWEN_MODEL),
            revision=os.getenv("AIC_ONLINE_QWEN_VLM_REVISION", DEFAULT_QWEN_REVISION),
            timeout_sec=_env_float("AIC_ONLINE_", "QWEN_VLM_TIMEOUT_SEC", 15.0),
            max_image_long_edge=_env_int("AIC_ONLINE_", "QWEN_VLM_MAX_IMAGE_LONG_EDGE", 768),
        )
    if runtime_config.vqa_enabled and advanced_vlm is None:
        raise ValueError(
            "VQA_ENABLED requires an explicitly configured VLMPort or "
            "QWEN_VLM_AUTO_CONFIGURE=true"
        )
    visual_expected_dimension = runtime_config.visual_expected_dimension
    vietnamese_expected_dimension = runtime_config.vietnamese_expected_dimension
    if data_config.dataset.manifest_required:
        visual_expected_dimension = visual_expected_dimension or 512
        vietnamese_expected_dimension = vietnamese_expected_dimension or 768

    sqlite_adapter = None
    if metadata is None or object_reader is None:
        sqlite_adapter = SQLiteReadAdapter(data_config.sqlite)
        metadata = metadata or sqlite_adapter
        object_reader = object_reader or sqlite_adapter
    milvus = milvus or MilvusSearchAdapter(data_config.milvus)
    elasticsearch = elasticsearch or ElasticsearchSearchAdapter(data_config.elasticsearch)
    visual_encoder = visual_encoder or OpenCLIPTextEncoder(
        expected_dimension=visual_expected_dimension,
    )
    vietnamese_encoder = vietnamese_encoder or VietnameseTextEncoder(
        expected_dimension=vietnamese_expected_dimension,
    )

    lifecycle = InfrastructureLifecycle()
    manifest_gate: DatasetManifestGate | None = None
    if data_config.dataset.manifest_required:
        manifest_gate = DatasetManifestGate(data_config.dataset)
        lifecycle.register(
            "dataset_manifest",
            manifest_gate,
            required=True,
        )
    lifecycle.register("milvus", _as_managed(milvus), required=True)
    lifecycle.register("elasticsearch", _as_managed(elasticsearch), required=False)
    if sqlite_adapter is not None:
        lifecycle.register("sqlite", sqlite_adapter, required=True)

    retrieval = build_retrieval_service(
        data_config=data_config,
        milvus=milvus,
        elasticsearch=elasticsearch,
        metadata=metadata,
        visual_encoder=visual_encoder,
        vietnamese_encoder=vietnamese_encoder,
        invocation_configs=build_invocation_configs(runtime_config),
        max_workers=runtime_config.max_workers,
    )
    policy = runtime_config.ranking_policy or RankingPolicyConfig()
    ranking = KISRankingService(
        metadata=metadata,
        object_reader=object_reader,
        asr_mapper=ASRIntervalFrameMapper(
            ASRMappingConfig(
                policy_name=policy.asr_mapping_method,
                max_frames_per_interval=policy.asr_max_frames_per_interval,
                interval_rrf_k=policy.asr_interval_rrf_k,
            )
        ),
        aggregator=RRFQueryVariantAggregator(
            QueryVariantAggregationConfig(
                query_variant_weights=policy.query_variant_weights,
                method_name=policy.aggregation_method,
            )
        ),
        normalizer=RRFScoreNormalizer(k=policy.normalization_rrf_k),
        fusion=WeightedFrameFusion(
            FusionConfig(
                weights=policy.fusion_weights,
                default_weight=policy.fusion_default_weight,
                method_name=policy.fusion_method,
            )
        ),
        summary=SummaryScorePropagator(
            SummaryPropagationConfig(
                weight=policy.summary_weight,
                max_boost=policy.summary_max_boost,
                method_name=policy.summary_method,
                fallback_rrf_k=policy.normalization_rrf_k,
                query_variant_weights=policy.query_variant_weights,
            )
        ),
        object_config=ObjectProcessingConfig(
            soft_boost_per_constraint=policy.object_soft_boost_per_constraint,
            max_total_boost=policy.object_max_total_boost,
        ),
        policy_name=policy.policy_name,
        policy_status=policy.policy_status,
        core_visual_policy=policy.core_visual_policy,
    )
    ranking_executor = ThreadPoolExecutor(
        max_workers=runtime_config.ranking_max_workers,
        thread_name_prefix="aic-ranking",
    )
    runtime = OnlineRuntime(
        orchestrator=KISSearchOrchestrator(
            retrieval=retrieval,
            ranking=ranking,
            ranking_executor=ranking_executor,
        ),
        lifecycle=lifecycle,
        retrieval=retrieval,
        ranking_executor=ranking_executor,
        trake_mode=trake_mode,
        vqa_mode=vqa_mode,
        readiness_probes=(
            RuntimeReadinessProbe(
                name="visual_encoder",
                required=True,
                check=_encoder_readiness_probe(
                    visual_encoder,
                    expected_dimension=visual_expected_dimension,
                ),
            ),
            RuntimeReadinessProbe(
                name="vietnamese_encoder",
                required=False,
                check=_encoder_readiness_probe(
                    vietnamese_encoder,
                    expected_dimension=vietnamese_expected_dimension,
                ),
            ),
        ),
        ui_resources=DatasetUIResources(
            data_root=data_config.dataset.data_root,
            metadata_reader=metadata,
            object_catalog=(
                object_reader if isinstance(object_reader, ObjectCatalogPort) else None
            ),
            identity_provider=(
                (lambda: (
                    manifest_gate.manifest.dataset_id,
                    manifest_gate.manifest.dataset_fingerprint,
                ))
                if manifest_gate is not None
                else None
            ),
        ),
        rewriter=advanced_rewriter,
    )
    if runtime_config.trake_enabled:
        if not callable(getattr(milvus, "iter_records", None)):
            raise TypeError("TRAKE requires a Milvus adapter with full-corpus iteration")
        attach_trake_mode(
            runtime,
            visual_corpus=MilvusSQLiteVisualCorpusAdapter(
                data_config.milvus,
                milvus=milvus,  # type: ignore[arg-type]
                metadata_reader=metadata,
            ),
            event_encoder=visual_encoder,
        )
    if runtime_config.vqa_enabled:
        image_resolver = FilesystemImageResolver(
            data_root=data_config.dataset.data_root,
            metadata_reader=metadata,
        )
        attach_vqa_mode(
            runtime,
            metadata_reader=metadata,
            image_resolver=image_resolver,
            evidence_hydrator=ElasticsearchEvidenceHydrator(
                data_config.elasticsearch,
                backend=elasticsearch,  # type: ignore[arg-type]
            ),
            vlm=advanced_vlm,
            rewriter=advanced_rewriter,
            readiness=_vqa_data_readiness(image_resolver, advanced_vlm),
        )
        if isinstance(advanced_vlm, QwenVLMAdapter):
            runtime.advanced_resources = (*runtime.advanced_resources, advanced_vlm)
    return runtime


def attach_trake_mode(
    runtime: OnlineRuntime,
    *,
    visual_corpus: VisualCorpusPort,
    event_encoder: TextEncoderPort,
    readiness: Callable[[], None] | None = None,
) -> OnlineRuntime:
    if runtime.trake_mode is not None:
        raise ValueError("TRAKE mode is already configured")
    if not isinstance(visual_corpus, VisualCorpusPort):
        raise TypeError("visual_corpus must implement VisualCorpusPort")
    if not isinstance(event_encoder, TextEncoderPort):
        raise TypeError("event_encoder must implement TextEncoderPort")
    runtime.trake_mode = TRAKEModeAdapter(
        TRAKEService(corpus=visual_corpus, encoder=event_encoder)
    )
    runtime.readiness_probes = (
        *runtime.readiness_probes,
        RuntimeReadinessProbe(
            name="trake",
            required=True,
            check=readiness or _configured_dependency_readiness,
        ),
    )
    return runtime


def attach_vqa_mode(
    runtime: OnlineRuntime,
    *,
    metadata_reader: MetadataReaderPort,
    image_resolver: ImageResolverPort,
    evidence_hydrator: EvidenceHydrationPort,
    vlm: VLMPort,
    rewriter: QueryRewriteService | None = None,
    readiness: Callable[[], None] | None = None,
) -> OnlineRuntime:
    if runtime.vqa_mode is not None:
        raise ValueError("VQA mode is already configured")
    for name, value, protocol in (
        ("metadata_reader", metadata_reader, MetadataReaderPort),
        ("image_resolver", image_resolver, ImageResolverPort),
        ("evidence_hydrator", evidence_hydrator, EvidenceHydrationPort),
        ("vlm", vlm, VLMPort),
    ):
        if not isinstance(value, protocol):
            raise TypeError(f"{name} must implement {protocol.__name__}")
    active_rewriter = rewriter or QueryRewriteService(NoOpQueryRewriter())
    if not isinstance(active_rewriter, QueryRewriteService):
        raise TypeError("rewriter must be a QueryRewriteService")
    candidate_retriever = VQACandidateRetriever(
        rewriter=active_rewriter,
        kis_search=runtime.orchestrator,
    )
    selector = EvidenceSelector(
        metadata_reader=metadata_reader,
        image_resolver=image_resolver,
        evidence_hydrator=evidence_hydrator,
    )
    runtime.vqa_mode = VQAModeAdapter(
        VQAOrchestrator(
            candidate_retriever=candidate_retriever,
            evidence_selector=selector,
            vlm=vlm,
        )
    )
    runtime.readiness_probes = (
        *runtime.readiness_probes,
        RuntimeReadinessProbe(
            name="vqa",
            required=True,
            check=readiness or _configured_dependency_readiness,
        ),
    )
    return runtime


def attach_advanced_modes(
    runtime: OnlineRuntime,
    *,
    dependencies: AdvancedModeDependencies,
    rewriter: QueryRewriteService | None = None,
    trake_readiness: Callable[[], None] | None = None,
    vqa_readiness: Callable[[], None] | None = None,
) -> OnlineRuntime:
    """Attach TRAKE/VQA services using only production public ports."""

    if not isinstance(runtime, OnlineRuntime):
        raise TypeError("runtime must be an OnlineRuntime")
    if not isinstance(dependencies, AdvancedModeDependencies):
        raise TypeError("dependencies must be AdvancedModeDependencies")
    _validate_advanced_dependencies(dependencies)
    if runtime.trake_mode is not None or runtime.vqa_mode is not None:
        raise ValueError("advanced modes are already configured")
    active_rewriter = rewriter or QueryRewriteService(NoOpQueryRewriter())
    if not isinstance(active_rewriter, QueryRewriteService):
        raise TypeError("rewriter must be a QueryRewriteService")

    attach_trake_mode(
        runtime,
        visual_corpus=dependencies.visual_corpus,
        event_encoder=dependencies.event_encoder,
        readiness=trake_readiness,
    )
    attach_vqa_mode(
        runtime,
        metadata_reader=dependencies.metadata_reader,
        image_resolver=dependencies.image_resolver,
        evidence_hydrator=dependencies.evidence_hydrator,
        vlm=dependencies.vlm,
        rewriter=active_rewriter,
        readiness=vqa_readiness,
    )
    runtime.advanced_resources = (*runtime.advanced_resources, *dependencies.managed_resources)
    return runtime


def _configured_dependency_readiness() -> None:
    """The dependency container itself is the default configuration proof."""


def _vqa_data_readiness(
    image_resolver: FilesystemImageResolver,
    vlm: VLMPort,
) -> Callable[[], None]:
    def check() -> None:
        image_resolver.health_check()
        vlm_health = getattr(vlm, "health_check", None)
        if callable(vlm_health):
            vlm_health()

    return check


def _validate_advanced_dependencies(dependencies: AdvancedModeDependencies) -> None:
    ports = (
        ("visual_corpus", dependencies.visual_corpus, VisualCorpusPort),
        ("event_encoder", dependencies.event_encoder, TextEncoderPort),
        ("metadata_reader", dependencies.metadata_reader, MetadataReaderPort),
        ("image_resolver", dependencies.image_resolver, ImageResolverPort),
        ("evidence_hydrator", dependencies.evidence_hydrator, EvidenceHydrationPort),
        ("vlm", dependencies.vlm, VLMPort),
    )
    for name, value, protocol in ports:
        if not isinstance(value, protocol):
            raise TypeError(f"dependencies.{name} must implement {protocol.__name__}")
    if not isinstance(dependencies.managed_resources, tuple):
        raise TypeError("dependencies.managed_resources must be a tuple")
    if any(not callable(getattr(resource, "close", None)) for resource in dependencies.managed_resources):
        raise TypeError("advanced managed resources must provide close()")


def create_runtime_app_from_env(
    *,
    runtime_factory: Callable[[], OnlineRuntime] | None = None,
) -> FastAPI:
    runtime_factory = runtime_factory or build_online_runtime

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = runtime_factory()
        app.state.online_runtime = runtime
        app.state.orchestrator = runtime.orchestrator
        app.state.trake_mode = runtime.trake_mode
        app.state.vqa_mode = runtime.vqa_mode
        app.state.health_provider = _health_response_for_runtime(runtime)
        app.state.ui_resources = runtime.ui_resources
        app.state.rewriter = runtime.rewriter
        runtime.start()
        try:
            yield
        finally:
            await asyncio.to_thread(runtime.close)

    return create_app(lifespan=lifespan)


def _health_response_for_runtime(runtime: OnlineRuntime) -> Callable[[], HealthResponse]:
    def provider() -> HealthResponse:
        health = runtime.health()
        status_value = "ready" if health.status is HealthStatus.HEALTHY else health.status.value
        components = {component.name: component for component in health.components}
        kis_components = tuple(
            component for component in health.components if component.name not in {"trake", "vqa"}
        )

        def readiness(mode: str, *, enabled: bool) -> str:
            if not enabled:
                return "disabled"
            component = components.get(mode)
            if component is not None:
                return "ready" if component.healthy else "unavailable"
            return "ready" if all(item.healthy for item in kis_components if item.required) else "unavailable"

        trake_enabled = runtime.trake_mode is not None
        vqa_enabled = runtime.vqa_mode is not None
        return HealthResponse(
            status=status_value,
            checks={
                "kis.enabled": "true",
                "kis.readiness": readiness("kis", enabled=True),
                "trake.enabled": str(trake_enabled).lower(),
                "trake.readiness": readiness("trake", enabled=trake_enabled),
                "vqa.enabled": str(vqa_enabled).lower(),
                "vqa.readiness": readiness("vqa", enabled=vqa_enabled),
                "rewrite.enabled": str(runtime.rewriter is not None).lower(),
                "ui_resources.enabled": str(runtime.ui_resources is not None).lower(),
            },
        )

    return provider


def _as_managed(resource: Any) -> Any:
    return resource


def _env_int(prefix: str, name: str, default: int) -> int:
    return int(os.getenv(f"{prefix}{name}", str(default)))


def _env_optional_int(prefix: str, name: str) -> int | None:
    value = os.getenv(f"{prefix}{name}")
    return None if value is None or not value.strip() else int(value)


def _env_float(prefix: str, name: str, default: float) -> float:
    return float(os.getenv(f"{prefix}{name}", str(default)))


def _env_bool(prefix: str, name: str, default: bool) -> bool:
    raw = os.getenv(f"{prefix}{name}")
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{prefix}{name} must be a boolean")


def _ranking_policy_from_env(prefix: str) -> RankingPolicyConfig:
    return RankingPolicyConfig(
        policy_name=os.getenv(f"{prefix}RANKING_POLICY_NAME", "person_c_experimental_baseline_v1"),
        policy_status=os.getenv(f"{prefix}RANKING_POLICY_STATUS", "experimental"),
        normalization_rrf_k=_env_int(prefix, "RANKING_NORMALIZATION_RRF_K", 60),
        query_variant_weights={
            "q0": _env_float(prefix, "RANKING_QUERY_Q0_WEIGHT", 1.0),
            "q1": _env_float(prefix, "RANKING_QUERY_Q1_WEIGHT", 1.0),
        },
        fusion_default_weight=_env_float(prefix, "RANKING_FUSION_DEFAULT_WEIGHT", 1.0),
        fusion_weights={
            branch: _env_float(prefix, f"RANKING_FUSION_{branch.name}_WEIGHT", 1.0)
            for branch in FRAME_FUSION_BRANCHES
        },
        summary_weight=_env_float(prefix, "RANKING_SUMMARY_WEIGHT", 0.1),
        summary_max_boost=_env_float(prefix, "RANKING_SUMMARY_MAX_BOOST", 0.2),
        asr_max_frames_per_interval=_env_int(prefix, "RANKING_ASR_MAX_FRAMES_PER_INTERVAL", 50),
        asr_interval_rrf_k=_env_int(prefix, "RANKING_ASR_INTERVAL_RRF_K", 60),
        object_soft_boost_per_constraint=_env_float(prefix, "RANKING_OBJECT_SOFT_BOOST", 0.05),
        object_max_total_boost=_env_float(prefix, "RANKING_OBJECT_MAX_TOTAL_BOOST", 0.2),
    )


def _invalid_positive_int(value: object) -> bool:
    return isinstance(value, bool) or not isinstance(value, int) or value < 1


def _invalid_positive_float(value: object) -> bool:
    return (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    )


def _encoder_readiness_probe(
    encoder: TextEncoderPort,
    *,
    expected_dimension: int | None,
) -> Callable[[], None]:
    def check() -> None:
        encoded = encoder.encode_texts(("readiness probe",))
        vectors = tuple(tuple(float(value) for value in vector) for vector in encoded)
        if len(vectors) != 1 or not vectors[0]:
            raise ValueError("encoder readiness probe returned an invalid batch")
        vector = vectors[0]
        dimension = encoder.dimension
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("encoder readiness probe reported an invalid dimension")
        if len(vector) != dimension:
            raise ValueError("encoder readiness vector dimension is inconsistent")
        if expected_dimension is not None and dimension != expected_dimension:
            raise ValueError("encoder readiness dimension does not match configured dimension")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("encoder readiness vector must be finite")
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("encoder readiness vector must have a positive finite norm")

    return check


def _run_readiness_probe(probe: RuntimeReadinessProbe) -> ComponentHealth:
    try:
        probe.check()
    except Exception as exc:
        return ComponentHealth(
            name=probe.name,
            required=probe.required,
            healthy=False,
            message=f"{type(exc).__name__}: readiness probe failed",
        )
    return ComponentHealth(name=probe.name, required=probe.required, healthy=True)


def _merge_health(
    infrastructure: InfrastructureHealth,
    extra_components: tuple[ComponentHealth, ...],
) -> InfrastructureHealth:
    components = infrastructure.components + extra_components
    if any(not component.healthy and component.required for component in components):
        status = HealthStatus.UNHEALTHY
    elif any(not component.healthy for component in components):
        status = HealthStatus.DEGRADED
    else:
        status = HealthStatus.HEALTHY
    return InfrastructureHealth(status=status, components=components)


__all__ = [
    "OnlineRuntime",
    "AdvancedModeDependencies",
    "AdvancedManagedResourcePort",
    "RuntimeReadinessProbe",
    "RuntimeCompositionConfig",
    "VARIANT_IDS",
    "build_invocation_configs",
    "build_online_runtime",
    "attach_trake_mode",
    "attach_vqa_mode",
    "attach_advanced_modes",
    "create_runtime_app_from_env",
]
