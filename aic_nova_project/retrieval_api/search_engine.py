"""FastAPI composition root for Person-C KIS search."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, Protocol, runtime_checkable

from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import Field

from online.domain.base import NonEmptyStr, StrictFrozenModel
from online.domain.candidates import FusedFrameCandidate
from online.domain.diagnostics import QueryDiagnostics
from online.domain.enums import BranchStatus, QueryMode, RetrievalBranch
from online.domain.errors import (
    BranchTimeoutError,
    ContractMismatchError,
    DataInfrastructureError,
    DimensionMismatchError,
    InvalidQueryError,
    MissingMetadataError,
    ResourceUnavailableError,
)
from online.domain.identifiers import parse_canonical_frame_id
from online.domain.query import ObjectConstraint
from online.modes.kis import KISSearchResult
from online.domain.trake import TRAKEQuery
from online.domain.vqa import VQAEvidenceBudget, VQAQuestion, VQAResult
from online.trake.service import TRAKEExecution
from query_understanding.parser import parse_kis_query
from retrieval_api.advanced_models import (
    InternalTRAKERequest,
    InternalTRAKEResponse,
    InternalVQARequest,
    InternalVQAResponse,
)


@runtime_checkable
class SearchOrchestratorPort(Protocol):
    async def search(self, bundle: object) -> KISSearchResult: ...


@runtime_checkable
class TRAKEModePort(Protocol):
    async def execute(self, query: TRAKEQuery) -> TRAKEExecution: ...


@runtime_checkable
class VQAModePort(Protocol):
    async def answer(
        self, question: VQAQuestion, budget: VQAEvidenceBudget
    ) -> VQAResult: ...


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
    trake_mode: TRAKEModePort | None = None,
    vqa_mode: VQAModePort | None = None,
    health_provider: Callable[[], HealthResponse] | None = None,
    lifespan: Any | None = None,
) -> FastAPI:
    app = FastAPI(title="AIC Online Retrieval API", lifespan=lifespan)
    app.state.orchestrator = orchestrator
    app.state.trake_mode = trake_mode
    app.state.vqa_mode = vqa_mode
    app.state.health_provider = health_provider

    @app.exception_handler(DataInfrastructureError)
    async def handle_domain_error(_: Request, exc: DataInfrastructureError) -> JSONResponse:
        return JSONResponse(
            status_code=_http_status_for_error(exc),
            content={"error": _public_error_payload(exc)},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "INVALID_QUERY",
                    "message": "The query is invalid for the current online contract",
                    "details": {"validation_error_count": len(exc.errors())},
                }
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, __: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "The service could not complete the request",
                    "details": {},
                }
            },
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

    @app.post(
        "/internal/unstable/trake",
        response_model=InternalTRAKEResponse,
        summary="Unstable internal TRAKE endpoint",
        description="Internal integration contract; not a competition-ready API.",
    )
    async def trake(request: InternalTRAKERequest) -> InternalTRAKEResponse:
        current = _get_trake_mode(app)
        execution = await current.execute(request.to_domain())
        return InternalTRAKEResponse(
            query_id=execution.query_id,
            results=execution.results,
            diagnostics=execution.diagnostics,
        )

    @app.post(
        "/internal/unstable/vqa",
        response_model=InternalVQAResponse,
        summary="Unstable internal VQA endpoint",
        description="Internal integration contract; not a competition-ready API.",
    )
    async def vqa(request: InternalVQARequest) -> InternalVQAResponse:
        current = _get_vqa_mode(app)
        result = await current.answer(request.to_domain(), request.evidence_budget)
        return InternalVQAResponse(question_id=result.question_id, result=result)

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


def _get_trake_mode(app: FastAPI) -> TRAKEModePort:
    mode = getattr(app.state, "trake_mode", None)
    if mode is None:
        raise ResourceUnavailableError(
            "TRAKE mode is not configured", details={"resource": "trake_mode"}
        )
    if not isinstance(mode, TRAKEModePort):
        raise ContractMismatchError("Configured TRAKE mode is invalid")
    return mode


def _get_vqa_mode(app: FastAPI) -> VQAModePort:
    mode = getattr(app.state, "vqa_mode", None)
    if mode is None:
        raise ResourceUnavailableError(
            "VQA mode is not configured", details={"resource": "vqa_mode"}
        )
    if not isinstance(mode, VQAModePort):
        raise ContractMismatchError("Configured VQA mode is invalid")
    return mode


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
        "details": _public_details(safe["details"]),
    }


_PUBLIC_ERROR_MESSAGES = {
    "BRANCH_TIMEOUT": "A required retrieval branch timed out",
    "CONTRACT_MISMATCH": "A retrieval contract mismatch prevented search",
    "DIMENSION_MISMATCH": "A vector dimension mismatch prevented search",
    "INVALID_QUERY": "The query is invalid for the current online contract",
    "MISSING_METADATA": "Required frame metadata is missing",
    "RESOURCE_UNAVAILABLE": "A required search resource is unavailable",
}

_PUBLIC_DETAIL_KEYS = frozenset(
    {
        "actual",
        "actual_dimension",
        "branch",
        "constraint",
        "expected",
        "expected_dimension",
        "frame_id",
        "missing_count",
        "query_variant_id",
        "status",
        "validation_error_count",
    }
)

_SAFE_QUERY_VARIANT = re.compile(r"^q[0-9]{1,3}$")


def _public_details(details: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in details.items():
        if key not in _PUBLIC_DETAIL_KEYS:
            continue
        if key == "branch":
            try:
                output[key] = RetrievalBranch(value).value
            except (TypeError, ValueError):
                continue
        elif key == "status":
            try:
                output[key] = BranchStatus(value).value
            except (TypeError, ValueError):
                continue
        elif key == "query_variant_id":
            if isinstance(value, str) and _SAFE_QUERY_VARIANT.fullmatch(value):
                output[key] = value
        elif key == "frame_id":
            if isinstance(value, str):
                try:
                    parse_canonical_frame_id(value)
                except DataInfrastructureError:
                    continue
                output[key] = value
        elif key == "constraint":
            if value == "object_position":
                output[key] = value
        elif key in {
            "actual",
            "actual_dimension",
            "expected",
            "expected_dimension",
            "missing_count",
            "validation_error_count",
        }:
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                public_key = {
                    "actual": "actual_dimension",
                    "expected": "expected_dimension",
                }.get(key, key)
                output[public_key] = value
            elif isinstance(value, float) and math.isfinite(value) and value >= 0.0:
                output[key] = value
    return output


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
    "TRAKEModePort",
    "VQAModePort",
    "competition_candidates",
    "create_app",
]
