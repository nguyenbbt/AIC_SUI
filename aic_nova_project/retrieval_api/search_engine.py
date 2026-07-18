"""FastAPI composition root for Person-C KIS search."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, Protocol, runtime_checkable

from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import Field

from online.domain.base import NonEmptyStr, StrictFrozenModel
from online.domain.candidates import FusedFrameCandidate
from online.domain.diagnostics import QueryDiagnostics
from online.domain.enums import QueryMode, RetrievalBranch
from online.domain.errors import (
    BranchTimeoutError,
    ContractMismatchError,
    DataInfrastructureError,
    DimensionMismatchError,
    InvalidQueryError,
    MissingMetadataError,
    ResourceUnavailableError,
)
from online.domain.query import ObjectConstraint
from online.modes.kis import KISSearchResult
from query_understanding.parser import parse_kis_query


@runtime_checkable
class SearchOrchestratorPort(Protocol):
    async def search(self, bundle: object) -> KISSearchResult: ...


class SearchRequest(StrictFrozenModel):
    query: NonEmptyStr
    mode: QueryMode = QueryMode.KIS_TEXT
    paraphrases: tuple[NonEmptyStr, ...] = ()
    object_constraints: tuple[ObjectConstraint, ...] = ()
    enabled_branches: tuple[RetrievalBranch, ...] | None = None
    include_diagnostics: bool = False
    query_id: NonEmptyStr | None = None


class SearchResponse(StrictFrozenModel):
    query_id: NonEmptyStr
    candidates: tuple[FusedFrameCandidate, ...]
    diagnostics: QueryDiagnostics | None = None


class HealthResponse(StrictFrozenModel):
    status: NonEmptyStr
    checks: Mapping[str, str] = Field(default_factory=dict)


def create_app(
    *,
    orchestrator: SearchOrchestratorPort | None = None,
    health_provider: Callable[[], HealthResponse] | None = None,
    lifespan: Any | None = None,
) -> FastAPI:
    app = FastAPI(title="AIC Online Retrieval API", lifespan=lifespan)
    app.state.orchestrator = orchestrator
    app.state.health_provider = health_provider

    @app.exception_handler(DataInfrastructureError)
    async def handle_domain_error(_: Request, exc: DataInfrastructureError) -> JSONResponse:
        return JSONResponse(
            status_code=_http_status_for_error(exc),
            content={"error": _public_error_payload(exc)},
        )

    @app.get("/health/live", response_model=HealthResponse)
    async def live() -> HealthResponse:
        return HealthResponse(status="healthy", checks={"api": "healthy"})

    @app.get("/health/ready", response_model=HealthResponse)
    async def ready(
        response: Response,
        current: SearchOrchestratorPort = Depends(_get_orchestrator),
    ) -> HealthResponse:
        provider = getattr(app.state, "health_provider", None)
        if provider is not None:
            health = provider()
            if health.status == "unhealthy":
                response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return health
        return HealthResponse(
            status="ready",
            checks={
                "api": "healthy",
                "orchestrator": "configured" if current is not None else "missing",
            },
        )

    @app.post("/search", response_model=SearchResponse)
    async def search(
        request: SearchRequest,
        current: SearchOrchestratorPort = Depends(_get_orchestrator),
    ) -> SearchResponse:
        bundle = parse_kis_query(
            request.query,
            mode=request.mode,
            paraphrases=request.paraphrases,
            object_constraints=request.object_constraints,
            enabled_branches=request.enabled_branches,
            query_id=request.query_id,
        )
        result = await current.search(bundle)
        return SearchResponse(
            query_id=result.diagnostics.query_id,
            candidates=result.candidates,
            diagnostics=result.diagnostics if request.include_diagnostics else None,
        )

    return app


def _get_orchestrator(request: Request) -> SearchOrchestratorPort:
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise ResourceUnavailableError(
            "Search orchestrator is not configured",
            details={"resource": "search_orchestrator"},
        )
    if not isinstance(orchestrator, SearchOrchestratorPort):
        raise ContractMismatchError("Configured orchestrator does not implement search")
    return orchestrator


def _http_status_for_error(exc: DataInfrastructureError) -> int:
    if isinstance(exc, InvalidQueryError):
        return status.HTTP_422_UNPROCESSABLE_ENTITY
    if isinstance(exc, ResourceUnavailableError):
        return status.HTTP_503_SERVICE_UNAVAILABLE
    if isinstance(exc, BranchTimeoutError):
        return status.HTTP_504_GATEWAY_TIMEOUT
    if isinstance(exc, (ContractMismatchError, DimensionMismatchError, MissingMetadataError)):
        return status.HTTP_409_CONFLICT
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def _public_error_payload(exc: DataInfrastructureError) -> dict[str, Any]:
    safe = exc.to_safe_dict()
    return {
        "code": safe["code"],
        "message": _PUBLIC_ERROR_MESSAGES.get(safe["code"], "The search service could not complete the request"),
        "details": safe["details"],
    }


_PUBLIC_ERROR_MESSAGES = {
    "BRANCH_TIMEOUT": "A required retrieval branch timed out",
    "CONTRACT_MISMATCH": "A retrieval contract mismatch prevented search",
    "DIMENSION_MISMATCH": "A vector dimension mismatch prevented search",
    "INVALID_QUERY": "The query is invalid for the current online contract",
    "MISSING_METADATA": "Required frame metadata is missing",
    "RESOURCE_UNAVAILABLE": "A required search resource is unavailable",
}


def competition_candidates(
    candidates: Sequence[FusedFrameCandidate],
) -> tuple[Mapping[str, Any], ...]:
    """Small stable adapter for competition-style frame submissions."""

    return tuple(
        {
            "frame_id": candidate.frame_id,
            "video_id": candidate.video_id,
            "timestamp_sec": candidate.timestamp_sec,
            "score": candidate.final_score,
        }
        for candidate in candidates
    )


__all__ = [
    "HealthResponse",
    "SearchOrchestratorPort",
    "SearchRequest",
    "SearchResponse",
    "competition_candidates",
    "create_app",
]
