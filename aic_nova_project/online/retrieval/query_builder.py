"""Build strict internal query bundles for Textual KIS and Video KIS."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from online.domain.enums import QueryMode, RetrievalBranch
from online.domain.errors import InvalidQueryError
from online.domain.query import (
    ObjectConstraint,
    QueryBundle,
    QueryOptions,
    TextQueryVariant,
)


BASELINE_KIS_BRANCHES: tuple[RetrievalBranch, ...] = (
    RetrievalBranch.VISUAL_DENSE,
    RetrievalBranch.OCR_DENSE,
    RetrievalBranch.OCR_BM25,
    RetrievalBranch.ASR_DENSE,
    RetrievalBranch.ASR_BM25,
    RetrievalBranch.SUMMARY_DENSE,
    RetrievalBranch.SUMMARY_BM25,
)


class KISQueryBuilder:
    """Create one shared text-query contract for t-KIS and v-KIS.

    Paraphrase generation is intentionally outside this class. Callers provide
    zero, one or two paraphrases, which become q1 and q2. Each variant remains
    independent for retrieval; no embedding averaging or score aggregation is
    performed here.
    """

    def build(
        self,
        original_query: str,
        *,
        mode: QueryMode | str,
        paraphrases: Sequence[str] = (),
        object_constraints: Sequence[ObjectConstraint | Mapping[str, Any]] = (),
        enabled_branches: Sequence[RetrievalBranch | str] | None = None,
        options: QueryOptions | Mapping[str, Any] | None = None,
        query_id: str | None = None,
    ) -> QueryBundle:
        normalized_mode = self._validate_mode(mode)
        query_text = self._clean_text(original_query, "original_query")

        if isinstance(paraphrases, (str, bytes)):
            raise InvalidQueryError("paraphrases must be a sequence of text values")
        paraphrase_values = tuple(paraphrases)
        if len(paraphrase_values) > 2:
            raise InvalidQueryError("KIS baseline accepts at most two paraphrases")

        variants = [TextQueryVariant(variant_id="q0", text=query_text)]
        for index, text in enumerate(paraphrase_values, start=1):
            variants.append(
                TextQueryVariant(
                    variant_id=f"q{index}",
                    text=self._clean_text(text, f"q{index}"),
                )
            )

        if enabled_branches is None:
            branches: Sequence[RetrievalBranch | str] = BASELINE_KIS_BRANCHES
        elif isinstance(enabled_branches, (str, bytes)):
            raise InvalidQueryError("enabled_branches must be a sequence")
        else:
            branches = enabled_branches

        try:
            return QueryBundle(
                query_id=uuid4().hex if query_id is None else query_id,
                mode=normalized_mode,
                original_query=query_text,
                text_variants=tuple(variants),
                object_constraints=tuple(object_constraints),
                enabled_branches=tuple(branches),
                options=QueryOptions() if options is None else options,
            )
        except ValidationError as exc:
            raise InvalidQueryError(
                "Invalid KIS query bundle",
                details={"validation_error_count": exc.error_count()},
            ) from exc

    @staticmethod
    def _validate_mode(mode: QueryMode | str) -> QueryMode:
        try:
            normalized = QueryMode(mode)
        except (TypeError, ValueError) as exc:
            raise InvalidQueryError("Unknown query mode") from exc
        if normalized not in {QueryMode.KIS_TEXT, QueryMode.KIS_VIDEO}:
            raise InvalidQueryError("KIS query builder supports only KIS modes")
        return normalized

    @staticmethod
    def _clean_text(value: object, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise InvalidQueryError(f"{name} must be non-empty text")
        return value.strip()
