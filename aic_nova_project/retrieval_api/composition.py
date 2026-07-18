"""Runtime wiring for A adapters, B retrieval and C orchestration."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from fastapi import FastAPI

from online.adapters.elasticsearch import ElasticsearchSearchAdapter
from online.adapters.milvus import MilvusSearchAdapter
from online.adapters.sqlite import SQLiteReadAdapter
from online.config import OnlineDataConfig
from online.domain.enums import RetrievalBranch
from online.lifecycle import HealthStatus, InfrastructureHealth, InfrastructureLifecycle
from online.modes.kis import KISRankingService, KISSearchOrchestrator
from online.ports import (
    ElasticsearchSearchPort,
    MetadataReaderPort,
    MilvusSearchPort,
    ObjectReaderPort,
    TextEncoderPort,
)
from online.retrieval.encoders import PECoreTextEncoder, VietnameseTextEncoder
from online.retrieval.factory import build_retrieval_service
from online.retrieval.query_builder import BASELINE_KIS_BRANCHES
from online.retrieval.service import RetrievalInvocationConfig, RetrievalService
from retrieval_api.search_engine import HealthResponse, create_app


VARIANT_IDS = ("q0", "q1", "q2")


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

    def __post_init__(self) -> None:
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
    last_health: InfrastructureHealth | None = None

    def start(self) -> InfrastructureHealth:
        self.last_health = self.lifecycle.start()
        return self.last_health

    def health(self) -> InfrastructureHealth:
        self.last_health = self.lifecycle.health()
        return self.last_health

    def close(self) -> None:
        try:
            self.retrieval.close(wait=True)
        finally:
            try:
                self.orchestrator.close(wait=True)
            finally:
                try:
                    self.ranking_executor.shutdown(wait=True, cancel_futures=True)
                finally:
                    self.lifecycle.close()


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
) -> OnlineRuntime:
    data_config = data_config or OnlineDataConfig.from_env()
    runtime_config = runtime_config or RuntimeCompositionConfig.from_env()

    sqlite_adapter = None
    if metadata is None or object_reader is None:
        sqlite_adapter = SQLiteReadAdapter(data_config.sqlite)
        metadata = metadata or sqlite_adapter
        object_reader = object_reader or sqlite_adapter
    milvus = milvus or MilvusSearchAdapter(data_config.milvus)
    elasticsearch = elasticsearch or ElasticsearchSearchAdapter(data_config.elasticsearch)
    visual_encoder = visual_encoder or PECoreTextEncoder(
        expected_dimension=runtime_config.visual_expected_dimension,
    )
    vietnamese_encoder = vietnamese_encoder or VietnameseTextEncoder(
        expected_dimension=runtime_config.vietnamese_expected_dimension,
    )

    lifecycle = InfrastructureLifecycle()
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
    ranking = KISRankingService(metadata=metadata, object_reader=object_reader)
    ranking_executor = ThreadPoolExecutor(
        max_workers=runtime_config.ranking_max_workers,
        thread_name_prefix="aic-ranking",
    )
    return OnlineRuntime(
        orchestrator=KISSearchOrchestrator(
            retrieval=retrieval,
            ranking=ranking,
            ranking_executor=ranking_executor,
        ),
        lifecycle=lifecycle,
        retrieval=retrieval,
        ranking_executor=ranking_executor,
    )


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
        app.state.health_provider = _health_response_for_runtime(runtime)
        runtime.start()
        try:
            yield
        finally:
            runtime.close()

    return create_app(lifespan=lifespan)


def _health_response_for_runtime(runtime: OnlineRuntime) -> Callable[[], HealthResponse]:
    def provider() -> HealthResponse:
        health = runtime.health()
        status_value = "ready" if health.status is HealthStatus.HEALTHY else health.status.value
        return HealthResponse(
            status=status_value,
            checks={
                component.name: "healthy" if component.healthy else "unhealthy"
                for component in health.components
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


def _invalid_positive_int(value: object) -> bool:
    return isinstance(value, bool) or not isinstance(value, int) or value < 1


def _invalid_positive_float(value: object) -> bool:
    return isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0


__all__ = [
    "OnlineRuntime",
    "RuntimeCompositionConfig",
    "VARIANT_IDS",
    "build_invocation_configs",
    "build_online_runtime",
    "create_runtime_app_from_env",
]
