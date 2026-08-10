"""Public query-understanding facade for KIS text input."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from online.domain.enums import QueryMode, RetrievalBranch
from online.domain.query import ObjectConstraint, QueryBundle, QueryOptions
from online.retrieval.query_builder import KISQueryBuilder


_KIS_BUILDER = KISQueryBuilder()


def parse_kis_query(
    original_query: str,
    *,
    mode: QueryMode | str,
    paraphrases: Sequence[str] = (),
    object_constraints: Sequence[ObjectConstraint | Mapping[str, Any]] = (),
    enabled_branches: Sequence[RetrievalBranch | str] | None = None,
    options: QueryOptions | Mapping[str, Any] | None = None,
    query_id: str | None = None,
) -> QueryBundle:
    """Validate contestant-authored/task-provided text into one KIS bundle."""

    return _KIS_BUILDER.build(
        original_query,
        mode=mode,
        paraphrases=paraphrases,
        object_constraints=object_constraints,
        enabled_branches=enabled_branches,
        options=options,
        query_id=query_id,
    )


__all__ = ["KISQueryBuilder", "parse_kis_query"]
