"""Structured, provider-neutral query rewriting with safe degradation."""

from __future__ import annotations

import asyncio
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from threading import Lock
from time import perf_counter
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from online.domain.errors import BranchTimeoutError, DataInfrastructureError
from online.domain.vqa import VQAQuestion


MAX_REWRITE_PARAPHRASES = 1
MAX_DIAGNOSTIC_IDENTIFIER_LENGTH = 128


class RewritePurpose(str, Enum):
    KIS = "kis"
    VQA_EVIDENCE = "vqa_evidence"


class RewriteStatus(str, Enum):
    SUCCESS = "success"
    DEGRADED = "degraded"


class RewriteProviderStatus(str, Enum):
    SUCCESS = "success"
    NOOP = "noop"


@dataclass(frozen=True, slots=True)
class QueryRewriteRequest:
    request_id: str
    purpose: RewritePurpose
    text: str
    answer_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_id",
            _required_text(self.request_id, "request_id"),
        )
        object.__setattr__(self, "text", _required_text(self.text, "text"))
        try:
            purpose = RewritePurpose(self.purpose)
        except (TypeError, ValueError) as exc:
            raise ValueError("purpose must be a supported rewrite purpose") from exc
        object.__setattr__(self, "purpose", purpose)
        if purpose is RewritePurpose.KIS:
            if self.answer_type is not None:
                raise ValueError("KIS rewrite must not contain answer_type")
        else:
            object.__setattr__(
                self,
                "answer_type",
                _required_text(self.answer_type, "answer_type"),
            )


@dataclass(frozen=True, slots=True)
class QueryRewriteProposal:
    """Structured output produced by a concrete rewrite provider.

    For KIS, ``primary_text`` is the first proposed paraphrase. For VQA it is
    the primary visual-evidence description used as q0 by the KIS pipeline.
    """

    primary_text: str
    paraphrases: tuple[str, ...] = ()
    status: RewriteProviderStatus = RewriteProviderStatus.SUCCESS
    provider_id: str | None = None
    model_id: str | None = None
    prompt_version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            RewriteProviderStatus(self.status),
        )
        if isinstance(self.paraphrases, (str, bytes)):
            raise ValueError("paraphrases must be a sequence of strings")
        try:
            paraphrases = tuple(self.paraphrases)
        except TypeError as exc:
            raise ValueError("paraphrases must be a sequence of strings") from exc
        object.__setattr__(self, "paraphrases", paraphrases)


@dataclass(frozen=True, slots=True)
class QueryRewriteResult:
    request_id: str
    purpose: RewritePurpose
    original_text: str
    primary_text: str
    paraphrases: tuple[str, ...]
    status: RewriteStatus
    warnings: tuple[str, ...]
    latency_ms: float
    provider_id: str | None = None
    model_id: str | None = None
    prompt_version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _required_text(self.request_id, "request_id"))
        object.__setattr__(
            self,
            "original_text",
            _required_text(self.original_text, "original_text"),
        )
        object.__setattr__(
            self,
            "primary_text",
            _required_text(self.primary_text, "primary_text"),
        )
        object.__setattr__(self, "purpose", RewritePurpose(self.purpose))
        object.__setattr__(self, "status", RewriteStatus(self.status))
        if isinstance(self.paraphrases, (str, bytes)):
            raise ValueError("paraphrases must be a sequence of strings")
        paraphrases = tuple(self.paraphrases)
        if any(not isinstance(value, str) or not value for value in paraphrases):
            raise ValueError("paraphrases must contain non-empty strings")
        if len(set(paraphrases)) != len(paraphrases):
            raise ValueError("paraphrases must not contain duplicates")
        object.__setattr__(self, "paraphrases", paraphrases)
        if isinstance(self.warnings, (str, bytes)):
            raise ValueError("warnings must be a sequence of strings")
        warnings = tuple(self.warnings)
        if len(warnings) > 8 or any(
            not isinstance(value, str) or not value for value in warnings
        ):
            raise ValueError("warnings must contain at most eight non-empty strings")
        if self.status is RewriteStatus.DEGRADED and not warnings:
            raise ValueError("degraded rewrite result requires a warning")
        object.__setattr__(self, "warnings", warnings)
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("latency_ms must be finite and >= 0")
        if len(paraphrases) > MAX_REWRITE_PARAPHRASES:
            raise ValueError("rewrite result contains too many paraphrases")

    @property
    def variants(self) -> tuple[str, ...]:
        return (self.primary_text, *self.paraphrases)


@runtime_checkable
class QueryRewritePort(Protocol):
    async def rewrite(self, request: QueryRewriteRequest) -> QueryRewriteProposal: ...


class NoOpQueryRewriter:
    """Provider-free baseline that deliberately requests degraded fallback."""

    async def rewrite(self, request: QueryRewriteRequest) -> QueryRewriteProposal:
        return QueryRewriteProposal(
            primary_text=request.text,
            status=RewriteProviderStatus.NOOP,
            provider_id="noop",
        )


class MappingQueryRewriter:
    """Deterministic structured rewriter for SDK-free integration tests."""

    def __init__(
        self,
        rewrites: Mapping[
            tuple[RewritePurpose | str, str],
            QueryRewriteProposal,
        ],
    ) -> None:
        normalized: dict[tuple[RewritePurpose, str], QueryRewriteProposal] = {}
        for raw_key, proposal in rewrites.items():
            if not isinstance(raw_key, tuple) or len(raw_key) != 2:
                raise ValueError("rewrite mapping keys must be (purpose, text)")
            purpose, text = raw_key
            try:
                normalized_purpose = RewritePurpose(purpose)
            except (TypeError, ValueError) as exc:
                raise ValueError("rewrite mapping contains an invalid purpose") from exc
            normalized_text = _required_text(text, "mapping text")
            if not isinstance(proposal, QueryRewriteProposal):
                raise TypeError("rewrite mapping values must be QueryRewriteProposal")
            normalized[(normalized_purpose, normalized_text)] = proposal
        self._rewrites = MappingProxyType(normalized)
        self._calls: list[QueryRewriteRequest] = []
        self._lock = Lock()

    @property
    def calls(self) -> tuple[QueryRewriteRequest, ...]:
        with self._lock:
            return tuple(self._calls)

    async def rewrite(self, request: QueryRewriteRequest) -> QueryRewriteProposal:
        with self._lock:
            self._calls.append(request)
        proposal = self._rewrites.get((request.purpose, request.text))
        if proposal is not None:
            return proposal
        return QueryRewriteProposal(
            primary_text=request.text,
            status=RewriteProviderStatus.NOOP,
            provider_id="mapping",
        )


class QueryRewriteService:
    """Normalize structured rewrites and degrade provider failures safely."""

    def __init__(
        self,
        rewriter: QueryRewritePort | None = None,
        *,
        timeout_sec: float = 5.0,
        max_paraphrases: int = MAX_REWRITE_PARAPHRASES,
    ) -> None:
        resolved_rewriter = rewriter if rewriter is not None else NoOpQueryRewriter()
        if not isinstance(resolved_rewriter, QueryRewritePort):
            raise TypeError("rewriter must implement QueryRewritePort")
        if (
            isinstance(timeout_sec, bool)
            or not isinstance(timeout_sec, (int, float))
            or not math.isfinite(timeout_sec)
            or timeout_sec <= 0
        ):
            raise ValueError("timeout_sec must be a positive finite number")
        if (
            isinstance(max_paraphrases, bool)
            or not isinstance(max_paraphrases, int)
            or not 0 <= max_paraphrases <= MAX_REWRITE_PARAPHRASES
        ):
            raise ValueError("max_paraphrases must be within [0, 1]")
        self._rewriter = resolved_rewriter
        self._timeout_sec = float(timeout_sec)
        self._max_paraphrases = max_paraphrases

    async def rewrite_kis(
        self,
        original_query: str,
        *,
        request_id: str,
    ) -> QueryRewriteResult:
        return await self.rewrite(
            QueryRewriteRequest(
                request_id=request_id,
                purpose=RewritePurpose.KIS,
                text=original_query,
            )
        )

    async def rewrite_vqa(self, question: VQAQuestion) -> QueryRewriteResult:
        if not isinstance(question, VQAQuestion):
            raise TypeError("question must be a validated VQAQuestion")
        return await self.rewrite(
            QueryRewriteRequest(
                request_id=question.question_id,
                purpose=RewritePurpose.VQA_EVIDENCE,
                text=question.question,
                answer_type=question.answer_type.value,
            )
        )

    async def rewrite(self, request: QueryRewriteRequest) -> QueryRewriteResult:
        if not isinstance(request, QueryRewriteRequest):
            raise TypeError("request must be a QueryRewriteRequest")
        started_at = perf_counter()
        try:
            proposal = await asyncio.wait_for(
                self._rewriter.rewrite(request),
                timeout=self._timeout_sec,
            )
        except TimeoutError:
            return self._fallback_result(
                request,
                started_at,
                warning="QUERY_REWRITE_TIMEOUT",
            )
        except BranchTimeoutError:
            return self._fallback_result(
                request,
                started_at,
                warning="QUERY_REWRITE_TIMEOUT",
            )
        except DataInfrastructureError:
            return self._fallback_result(
                request,
                started_at,
                warning="QUERY_REWRITE_UNAVAILABLE",
            )
        except Exception:
            return self._fallback_result(
                request,
                started_at,
                warning="QUERY_REWRITE_PROVIDER_ERROR",
            )

        if not isinstance(proposal, QueryRewriteProposal):
            return self._fallback_result(
                request,
                started_at,
                warning="QUERY_REWRITE_INVALID_OUTPUT",
            )
        if proposal.status is RewriteProviderStatus.NOOP:
            return self._fallback_result(
                request,
                started_at,
                warning="QUERY_REWRITE_NOOP",
                proposal=proposal,
            )

        try:
            if request.purpose is RewritePurpose.KIS:
                primary_text = request.text
                paraphrases = _normalize_variants(
                    (proposal.primary_text, *proposal.paraphrases),
                    exclude=(request.text,),
                    limit=self._max_paraphrases,
                )
                if not paraphrases:
                    return self._fallback_result(
                        request,
                        started_at,
                        warning="QUERY_REWRITE_NO_USABLE_VARIANTS",
                        proposal=proposal,
                    )
            else:
                normalized = _normalize_variants(
                    (proposal.primary_text, *proposal.paraphrases),
                    limit=1 + self._max_paraphrases,
                )
                if not normalized:
                    return self._fallback_result(
                        request,
                        started_at,
                        warning="QUERY_REWRITE_INVALID_OUTPUT",
                        proposal=proposal,
                    )
                primary_text = normalized[0]
                paraphrases = normalized[1:]
        except (TypeError, ValueError):
            return self._fallback_result(
                request,
                started_at,
                warning="QUERY_REWRITE_INVALID_OUTPUT",
                proposal=proposal,
            )

        identifiers, metadata_warning = _safe_proposal_identifiers(proposal)
        warnings = () if metadata_warning is None else (metadata_warning,)
        return QueryRewriteResult(
            request_id=request.request_id,
            purpose=request.purpose,
            original_text=request.text,
            primary_text=primary_text,
            paraphrases=paraphrases,
            status=RewriteStatus.SUCCESS,
            warnings=warnings,
            latency_ms=_elapsed_ms(started_at),
            **identifiers,
        )

    @staticmethod
    def _fallback_result(
        request: QueryRewriteRequest,
        started_at: float,
        *,
        warning: str,
        proposal: QueryRewriteProposal | None = None,
    ) -> QueryRewriteResult:
        identifiers: dict[str, str | None] = {
            "provider_id": None,
            "model_id": None,
            "prompt_version": None,
        }
        warnings = [warning]
        if proposal is not None:
            identifiers, metadata_warning = _safe_proposal_identifiers(proposal)
            if metadata_warning is not None:
                warnings.append(metadata_warning)
        return QueryRewriteResult(
            request_id=request.request_id,
            purpose=request.purpose,
            original_text=request.text,
            primary_text=request.text,
            paraphrases=(),
            status=RewriteStatus.DEGRADED,
            warnings=tuple(warnings),
            latency_ms=_elapsed_ms(started_at),
            **identifiers,
        )


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _normalize_variants(
    values: Sequence[object],
    *,
    exclude: Sequence[str] = (),
    limit: int,
) -> tuple[str, ...]:
    if limit == 0:
        return ()
    if isinstance(values, (str, bytes)):
        raise TypeError("rewrite variants must be a sequence")
    references = [" ".join(value.split()) for value in exclude]
    seen = set(references)
    output: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError("rewrite variants must contain strings")
        normalized = " ".join(value.split())
        if (
            not normalized
            or normalized in seen
            or any(_is_near_duplicate(normalized, reference) for reference in references)
        ):
            continue
        seen.add(normalized)
        references.append(normalized)
        output.append(normalized)
        if len(output) == limit:
            break
    return tuple(output)


_GENERIC_REWRITE_PREFIXES = (
    "khung hình có",
    "khung hình cho thấy",
    "hình ảnh có",
    "hình ảnh cho thấy",
    "cảnh có",
    "cảnh cho thấy",
    "visual scene",
    "scene showing",
    "image showing",
    "image of",
    "a photo of",
    "a video frame of",
)


def _is_near_duplicate(candidate: str, reference: str) -> bool:
    left = _similarity_text(candidate)
    right = _similarity_text(reference)
    if not left or not right:
        return left == right
    if left == right:
        return True
    left_tokens = left.split()
    right_tokens = right.split()
    if sorted(left_tokens) == sorted(right_tokens):
        return True
    return SequenceMatcher(None, left, right).ratio() >= 0.94


def _similarity_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))
    changed = True
    while changed:
        changed = False
        for prefix in _GENERIC_REWRITE_PREFIXES:
            canonical_prefix = " ".join(re.findall(r"\w+", prefix.casefold()))
            if normalized == canonical_prefix:
                return ""
            if normalized.startswith(canonical_prefix + " "):
                normalized = normalized[len(canonical_prefix) + 1 :]
                changed = True
                break
    return normalized


_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SENSITIVE_IDENTIFIER_PARTS = (
    "api_key",
    "apikey",
    "bearer",
    "credential",
    "password",
    "secret",
    "token",
)


def _safe_proposal_identifiers(
    proposal: QueryRewriteProposal,
) -> tuple[dict[str, str | None], str | None]:
    output: dict[str, str | None] = {}
    omitted = False
    for field_name in ("provider_id", "model_id", "prompt_version"):
        value = getattr(proposal, field_name)
        if value is None:
            output[field_name] = None
            continue
        if (
            not isinstance(value, str)
            or not value
            or len(value) > MAX_DIAGNOSTIC_IDENTIFIER_LENGTH
            or _SAFE_IDENTIFIER.fullmatch(value) is None
            or value.lower().startswith("sk-")
            or any(part in value.lower() for part in _SENSITIVE_IDENTIFIER_PARTS)
        ):
            output[field_name] = None
            omitted = True
        else:
            output[field_name] = value
    return output, "QUERY_REWRITE_METADATA_OMITTED" if omitted else None


def _elapsed_ms(started_at: float) -> float:
    return max(0.0, (perf_counter() - started_at) * 1000.0)


__all__ = [
    "MappingQueryRewriter",
    "MAX_REWRITE_PARAPHRASES",
    "NoOpQueryRewriter",
    "QueryRewritePort",
    "QueryRewriteProposal",
    "QueryRewriteRequest",
    "QueryRewriteResult",
    "QueryRewriteService",
    "RewriteProviderStatus",
    "RewritePurpose",
    "RewriteStatus",
]
