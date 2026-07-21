"""VQA candidate retrieval by reusing the complete KIS search pipeline."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, runtime_checkable

from online.domain.candidates import FusedFrameCandidate
from online.domain.diagnostics import QueryDiagnostics
from online.domain.enums import QueryMode, RetrievalBranch
from online.domain.errors import (
    ContractMismatchError,
    DataInfrastructureError,
    ResourceUnavailableError,
)
from online.domain.query import QueryBundle
from online.domain.vqa import VQAQuestion
from query_understanding.rewrite import QueryRewriteResult, QueryRewriteService

from .query_builder import KISQueryBuilder


@runtime_checkable
class KISSearchResultPort(Protocol):
    candidates: Sequence[FusedFrameCandidate]
    diagnostics: QueryDiagnostics


@runtime_checkable
class KISSearchPort(Protocol):
    async def search(self, bundle: QueryBundle) -> KISSearchResultPort: ...


@dataclass(frozen=True, slots=True)
class VQARetrievalExecution:
    question_id: str
    rewrite: QueryRewriteResult
    query_bundle: QueryBundle
    candidates: tuple[FusedFrameCandidate, ...]
    kis_diagnostics: QueryDiagnostics
    total_latency_ms: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.total_latency_ms) or self.total_latency_ms < 0:
            raise ValueError("total_latency_ms must be finite and >= 0")


class VQACandidateRetriever:
    """Adapt a VQA question to the existing KIS retrieval/ranking boundary.

    This class intentionally stops at ranked frame candidates. Evidence
    selection, hydration and VLM execution remain Person C responsibilities.
    """

    def __init__(
        self,
        *,
        rewriter: QueryRewriteService,
        kis_search: KISSearchPort,
        query_builder: KISQueryBuilder | None = None,
        enabled_branches: Sequence[RetrievalBranch | str] | None = None,
    ) -> None:
        if not isinstance(rewriter, QueryRewriteService):
            raise TypeError("rewriter must be a QueryRewriteService")
        if not isinstance(kis_search, KISSearchPort):
            raise TypeError("kis_search must implement KISSearchPort")
        if query_builder is not None and not isinstance(query_builder, KISQueryBuilder):
            raise TypeError("query_builder must be a KISQueryBuilder")
        if isinstance(enabled_branches, (str, bytes)):
            raise TypeError("enabled_branches must be a sequence")
        self._rewriter = rewriter
        self._kis_search = kis_search
        self._query_builder = query_builder or KISQueryBuilder()
        self._enabled_branches = (
            None if enabled_branches is None else tuple(enabled_branches)
        )

    async def execute(self, question: VQAQuestion) -> VQARetrievalExecution:
        if not isinstance(question, VQAQuestion):
            raise ContractMismatchError("question must be a validated VQAQuestion")
        started_at = perf_counter()
        rewrite = await self._rewriter.rewrite_vqa(question)
        build_kwargs: dict[str, object] = {
            "mode": QueryMode.KIS_TEXT,
            "paraphrases": rewrite.paraphrases,
            "query_id": _vqa_query_id(question.question_id),
        }
        if self._enabled_branches is not None:
            build_kwargs["enabled_branches"] = self._enabled_branches
        bundle = self._query_builder.build(
            rewrite.primary_text,
            **build_kwargs,
        )

        try:
            raw_result = await self._kis_search.search(bundle)
        except DataInfrastructureError:
            raise
        except Exception as exc:
            raise ResourceUnavailableError(
                "KIS search failed unexpectedly during VQA retrieval",
                details={
                    "stage": "vqa_kis_search",
                    "exception_type": type(exc).__name__,
                },
            ) from exc

        if not isinstance(raw_result, KISSearchResultPort):
            raise ContractMismatchError("KIS search returned an invalid result object")
        if isinstance(raw_result.candidates, (str, bytes)) or not isinstance(
            raw_result.candidates,
            Sequence,
        ):
            raise ContractMismatchError("KIS search candidates must be a sequence")
        candidates = tuple(raw_result.candidates)
        if any(not isinstance(item, FusedFrameCandidate) for item in candidates):
            raise ContractMismatchError("KIS search returned an invalid frame candidate")
        if not isinstance(raw_result.diagnostics, QueryDiagnostics):
            raise ContractMismatchError("KIS search returned invalid query diagnostics")
        if raw_result.diagnostics.query_id != bundle.query_id:
            raise ContractMismatchError(
                "KIS diagnostics query ID does not match the VQA retrieval bundle"
            )

        return VQARetrievalExecution(
            question_id=question.question_id,
            rewrite=rewrite,
            query_bundle=bundle,
            candidates=candidates,
            kis_diagnostics=raw_result.diagnostics,
            total_latency_ms=_elapsed_ms(started_at),
        )

    async def retrieve_candidates(
        self,
        question: VQAQuestion,
    ) -> tuple[FusedFrameCandidate, ...]:
        return (await self.execute(question)).candidates


def _vqa_query_id(question_id: str) -> str:
    return f"vqa:{question_id}"


def _elapsed_ms(started_at: float) -> float:
    return max(0.0, (perf_counter() - started_at) * 1000.0)


__all__ = [
    "KISSearchPort",
    "KISSearchResultPort",
    "VQACandidateRetriever",
    "VQARetrievalExecution",
]
