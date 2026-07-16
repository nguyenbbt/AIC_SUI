"""Read-only Elasticsearch lexical search adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from online.config import ElasticsearchResourceConfig
from online.domain.errors import ContractMismatchError, InvalidQueryError, ResourceUnavailableError
from online.ports.records import ASRSearchHit, FrameSearchHit, VideoSearchHit

from ._errors import call_backend


class ElasticsearchSearchAdapter:
    def __init__(
        self,
        config: ElasticsearchResourceConfig,
        *,
        client: Any | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self._client = client
        self._client_factory = client_factory
        self._owns_client = client is None

    def connect(self) -> None:
        if self._client is not None:
            return
        factory = self._client_factory
        if factory is None:
            try:
                from elasticsearch import Elasticsearch
            except ImportError as exc:
                raise ResourceUnavailableError(
                    "elasticsearch client is not installed",
                    details={"resource": "elasticsearch"},
                ) from exc
            factory = Elasticsearch
        self._client = call_backend(
            "connect",
            "elasticsearch",
            lambda: factory(self.config.uri, request_timeout=self.config.timeout_sec),
        )
        call_backend("connect", "elasticsearch", lambda: self._client.info())

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            call_backend("close", "elasticsearch", self._client.close)
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            raise ResourceUnavailableError("Elasticsearch adapter is not connected")
        return self._client

    def health_check(self) -> None:
        healthy = call_backend("health_check", "elasticsearch", lambda: self._get_client().ping())
        if not healthy:
            raise ResourceUnavailableError("Elasticsearch ping failed")

    def has_icu_plugin(self) -> bool:
        response = call_backend(
            "plugins", "elasticsearch", lambda: self._get_client().nodes.info(metric="plugins")
        )
        nodes = response.get("nodes", {}) if isinstance(response, Mapping) else {}
        for node in nodes.values():
            if not isinstance(node, Mapping):
                continue
            for plugin in node.get("plugins", ()):
                if isinstance(plugin, Mapping) and plugin.get("name") == "analysis-icu":
                    return True
        return False

    @staticmethod
    def _validate_query(query: str, top_k: int) -> str:
        if not isinstance(query, str) or not query.strip():
            raise InvalidQueryError("lexical query must not be empty")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise InvalidQueryError("top_k must be >= 1")
        return query

    def _body(
        self,
        query: str,
        top_k: int,
        field: str,
        source_fields: Sequence[str],
        fuzzy: bool,
    ) -> dict[str, Any]:
        match_value: Any = {"query": query, "fuzziness": self.config.fuzziness} if fuzzy else {"query": query}
        return {
            "size": top_k,
            "_source": list(source_fields),
            "query": {"match": {field: match_value}},
        }

    def _search(
        self,
        index: str,
        query: str,
        top_k: int,
        field: str,
        source_fields: Sequence[str],
        fuzzy: bool | None,
    ) -> Sequence[Mapping[str, Any]]:
        query = self._validate_query(query, top_k)
        use_fuzzy = self.config.fuzzy_enabled if fuzzy is None else fuzzy
        body = self._body(query, top_k, field, source_fields, use_fuzzy)
        response = call_backend(
            "search",
            index,
            lambda: self._get_client().search(
                index=index, body=body, request_timeout=self.config.timeout_sec
            ),
        )
        try:
            hits = response["hits"]["hits"]
        except (KeyError, TypeError) as exc:
            raise ContractMismatchError(
                "Elasticsearch response is missing hits", details={"resource": index}
            ) from exc
        if not isinstance(hits, Sequence):
            raise ContractMismatchError("Elasticsearch hits must be a sequence")
        return tuple(hits)

    @staticmethod
    def _source_and_score(hit: Mapping[str, Any], index: str) -> tuple[Mapping[str, Any], float]:
        source = hit.get("_source")
        score = hit.get("_score")
        if not isinstance(source, Mapping):
            raise ContractMismatchError(
                "Elasticsearch hit is missing _source", details={"resource": index}
            )
        try:
            numeric_score = float(score)
        except (TypeError, ValueError) as exc:
            raise ContractMismatchError(
                "Elasticsearch hit is missing a numeric _score",
                details={"resource": index},
            ) from exc
        if not math.isfinite(numeric_score):
            raise ContractMismatchError("Elasticsearch _score is not finite")
        return source, numeric_score

    @staticmethod
    def _required(source: Mapping[str, Any], field: str) -> Any:
        value = source.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ContractMismatchError(
                "Elasticsearch hit is missing a required source field",
                details={"field": field},
            )
        return value

    def search_ocr(
        self, query: str, top_k: int, *, fuzzy: bool | None = None
    ) -> Sequence[FrameSearchHit]:
        fields = ("frame_id", "video_id", "shot_id")
        hits = self._search(
            self.config.ocr_index,
            query,
            top_k,
            "ocr_text_concat",
            fields,
            fuzzy,
        )
        output = []
        for hit in hits:
            source, score = self._source_and_score(hit, self.config.ocr_index)
            try:
                output.append(
                    FrameSearchHit(
                        frame_id=str(self._required(source, "frame_id")),
                        video_id=str(self._required(source, "video_id")),
                        shot_id=int(self._required(source, "shot_id")),
                        raw_score=score,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ContractMismatchError("Invalid OCR Elasticsearch hit") from exc
        return tuple(output)

    def search_asr(
        self, query: str, top_k: int, *, fuzzy: bool | None = None
    ) -> Sequence[ASRSearchHit]:
        fields = ("video_id", "interval_id", "start_time", "end_time", "cleaned_text")
        hits = self._search(
            self.config.asr_index,
            query,
            top_k,
            "cleaned_text",
            fields,
            fuzzy,
        )
        output = []
        for hit in hits:
            source, score = self._source_and_score(hit, self.config.asr_index)
            try:
                output.append(
                    ASRSearchHit(
                        video_id=str(self._required(source, "video_id")),
                        interval_id=str(self._required(source, "interval_id")),
                        start_time_sec=float(self._required(source, "start_time")),
                        end_time_sec=float(self._required(source, "end_time")),
                        text=(None if source.get("cleaned_text") is None else str(source["cleaned_text"])),
                        raw_score=score,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ContractMismatchError("Invalid ASR Elasticsearch hit") from exc
        return tuple(output)

    def search_summary(
        self, query: str, top_k: int, *, fuzzy: bool | None = None
    ) -> Sequence[VideoSearchHit]:
        fields = ("video_id", "summary")
        hits = self._search(
            self.config.summary_index,
            query,
            top_k,
            "summary",
            fields,
            fuzzy,
        )
        output = []
        for hit in hits:
            source, score = self._source_and_score(hit, self.config.summary_index)
            output.append(
                VideoSearchHit(
                    video_id=str(self._required(source, "video_id")),
                    summary=(None if source.get("summary") is None else str(source["summary"])),
                    raw_score=score,
                )
            )
        return tuple(output)

    def index_exists(self, index: str) -> bool:
        return bool(
            call_backend(
                "index_exists", index, lambda: self._get_client().indices.exists(index=index)
            )
        )

    def get_mapping(self, index: str) -> Mapping[str, Any]:
        response = call_backend(
            "get_mapping", index, lambda: self._get_client().indices.get_mapping(index=index)
        )
        if not isinstance(response, Mapping):
            raise ContractMismatchError("Elasticsearch mapping response is invalid")
        root = response.get(index)
        if root is None and len(response) == 1:
            root = next(iter(response.values()))
        if not isinstance(root, Mapping):
            raise ContractMismatchError("Elasticsearch index mapping is missing")
        mappings = root.get("mappings", root)
        return mappings if isinstance(mappings, Mapping) else {}

    def sample_documents(
        self, index: str, source_fields: Sequence[str], limit: int
    ) -> Sequence[Mapping[str, Any]]:
        if limit < 1:
            raise InvalidQueryError("limit must be >= 1")
        body = {
            "size": limit,
            "_source": list(source_fields),
            "query": {"match_all": {}},
        }
        response = call_backend(
            "sample_documents",
            index,
            lambda: self._get_client().search(
                index=index, body=body, request_timeout=self.config.timeout_sec
            ),
        )
        try:
            return tuple(hit["_source"] for hit in response["hits"]["hits"])
        except (KeyError, TypeError) as exc:
            raise ContractMismatchError("Elasticsearch sample response is invalid") from exc

    def find_documents(
        self,
        index: str,
        filters: Mapping[str, object],
        source_fields: Sequence[str],
        *,
        limit: int = 2,
    ) -> Sequence[Mapping[str, Any]]:
        if not filters:
            raise InvalidQueryError("at least one exact filter is required")
        body = {
            "size": limit,
            "_source": list(source_fields),
            "query": {
                "bool": {
                    "filter": [{"term": {field: value}} for field, value in filters.items()]
                }
            },
        }
        response = call_backend(
            "find_documents",
            index,
            lambda: self._get_client().search(
                index=index, body=body, request_timeout=self.config.timeout_sec
            ),
        )
        try:
            return tuple(hit["_source"] for hit in response["hits"]["hits"])
        except (KeyError, TypeError) as exc:
            raise ContractMismatchError("Elasticsearch exact lookup response is invalid") from exc
