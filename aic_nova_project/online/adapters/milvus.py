"""Read-only Milvus search adapter with SDK-neutral outputs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from online.config import MilvusResourceConfig
from online.domain.errors import ContractMismatchError, DimensionMismatchError, InvalidQueryError, ResourceUnavailableError
from online.ports.records import ASRSearchHit, FrameSearchHit, VideoSearchHit

from ._errors import call_backend


class MilvusBackend(Protocol):
    def connect(self) -> None: ...

    def close(self) -> None: ...

    def collection_exists(self, name: str) -> bool: ...

    def describe_collection(self, name: str) -> Mapping[str, Any]: ...

    def search(
        self,
        name: str,
        vector: Sequence[float],
        output_fields: Sequence[str],
        top_k: int,
        search_params: Mapping[str, Any],
        timeout_sec: float,
    ) -> Sequence[Any]: ...

    def sample_records(
        self, name: str, output_fields: Sequence[str], limit: int
    ) -> Sequence[Mapping[str, Any]]: ...


class _PymilvusBackend:
    """Small wrapper around the legacy Collection API used by Offline indexing."""

    def __init__(self, config: MilvusResourceConfig) -> None:
        self.config = config
        self._modules: tuple[Any, Any, Any] | None = None

    def _sdk(self) -> tuple[Any, Any, Any]:
        if self._modules is None:
            try:
                from pymilvus import Collection, connections, utility
            except ImportError as exc:
                raise ResourceUnavailableError(
                    "pymilvus is not installed", details={"resource": "milvus"}
                ) from exc
            self._modules = (Collection, connections, utility)
        return self._modules

    def connect(self) -> None:
        _, connections, _ = self._sdk()
        connections.connect(alias=self.config.alias, uri=self.config.uri)

    def close(self) -> None:
        _, connections, _ = self._sdk()
        connections.disconnect(alias=self.config.alias)

    def collection_exists(self, name: str) -> bool:
        _, _, utility = self._sdk()
        return bool(utility.has_collection(name, using=self.config.alias))

    def _collection(self, name: str) -> Any:
        Collection, _, _ = self._sdk()
        return Collection(name, using=self.config.alias)

    def describe_collection(self, name: str) -> Mapping[str, Any]:
        collection = self._collection(name)
        fields: dict[str, str] = {}
        dimension: int | None = None
        for field in collection.schema.fields:
            dtype = str(field.dtype).split(".")[-1].upper()
            fields[field.name] = dtype
            if field.name == "embedding":
                params = getattr(field, "params", {}) or {}
                if params.get("dim") is not None:
                    dimension = int(params["dim"])
        metric_type: str | None = None
        index_type: str | None = None
        for index in collection.indexes:
            params = getattr(index, "params", {}) or {}
            if getattr(index, "field_name", None) == "embedding" or len(collection.indexes) == 1:
                metric_type = params.get("metric_type")
                index_type = params.get("index_type")
                break
        return {
            "fields": fields,
            "dimension": dimension,
            "metric_type": metric_type,
            "index_type": index_type,
        }

    def search(
        self,
        name: str,
        vector: Sequence[float],
        output_fields: Sequence[str],
        top_k: int,
        search_params: Mapping[str, Any],
        timeout_sec: float,
    ) -> Sequence[Any]:
        collection = self._collection(name)
        collection.load()
        result = collection.search(
            data=[list(vector)],
            anns_field="embedding",
            param=dict(search_params),
            limit=top_k,
            output_fields=list(output_fields),
            timeout=timeout_sec,
        )
        return tuple(result[0]) if result else ()

    def sample_records(
        self, name: str, output_fields: Sequence[str], limit: int
    ) -> Sequence[Mapping[str, Any]]:
        collection = self._collection(name)
        collection.load()
        return tuple(collection.query(expr="pk >= 0", output_fields=list(output_fields), limit=limit))


class MilvusSearchAdapter:
    def __init__(
        self,
        config: MilvusResourceConfig,
        *,
        backend: MilvusBackend | None = None,
    ) -> None:
        self.config = config
        self._backend = backend or _PymilvusBackend(config)
        self._connected = False
        self._dimensions: dict[str, int] = {}

    def connect(self) -> None:
        if self._connected:
            return
        call_backend("connect", "milvus", self._backend.connect)
        self._connected = True

    def close(self) -> None:
        if self._connected:
            call_backend("close", "milvus", self._backend.close)
        self._connected = False
        self._dimensions.clear()

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise ResourceUnavailableError("Milvus adapter is not connected")

    def health_check(self) -> None:
        self._ensure_connected()
        exists = call_backend(
            "health_check",
            self.config.visual_collection,
            lambda: self._backend.collection_exists(self.config.visual_collection),
        )
        if not exists:
            raise ResourceUnavailableError(
                "Core Milvus visual collection is missing",
                details={"resource": self.config.visual_collection},
            )

    def describe_collection(self, name: str) -> Mapping[str, Any]:
        self._ensure_connected()
        exists = call_backend(
            "collection_exists", name, lambda: self._backend.collection_exists(name)
        )
        if not exists:
            raise ResourceUnavailableError(
                "Milvus collection is missing", details={"resource": name}
            )
        return call_backend(
            "describe_collection", name, lambda: self._backend.describe_collection(name)
        )

    def collection_exists(self, name: str) -> bool:
        self._ensure_connected()
        return call_backend(
            "collection_exists", name, lambda: self._backend.collection_exists(name)
        )

    def _dimension(self, name: str) -> int:
        if name not in self._dimensions:
            description = self.describe_collection(name)
            dimension = description.get("dimension")
            if not isinstance(dimension, int) or dimension < 1:
                raise ContractMismatchError(
                    "Milvus embedding dimension is missing or invalid",
                    details={"resource": name},
                )
            self._dimensions[name] = dimension
        return self._dimensions[name]

    def _validate_vector(self, name: str, vector: Sequence[float]) -> tuple[float, ...]:
        if isinstance(vector, (str, bytes)):
            raise InvalidQueryError("query vector must be a numeric sequence")
        try:
            values = tuple(float(value) for value in vector)
        except (TypeError, ValueError) as exc:
            raise InvalidQueryError("query vector must contain numeric values") from exc
        expected = self._dimension(name)
        if len(values) != expected:
            raise DimensionMismatchError(
                "Query vector dimension does not match collection",
                details={"resource": name, "expected": expected, "actual": len(values)},
            )
        if not all(math.isfinite(value) for value in values):
            raise InvalidQueryError("query vector contains NaN or Infinity")
        norm = math.sqrt(sum(value * value for value in values))
        if abs(norm - 1.0) > self.config.norm_tolerance:
            raise InvalidQueryError(
                "query vector must be L2-normalized",
                details={"resource": name, "norm": norm},
            )
        return values

    def _search(
        self,
        name: str,
        vector: Sequence[float],
        output_fields: Sequence[str],
        top_k: int,
    ) -> Sequence[Any]:
        self._ensure_connected()
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise InvalidQueryError("top_k must be >= 1")
        values = self._validate_vector(name, vector)
        params = {"metric_type": "IP", "params": {"ef": self.config.search_ef}}
        return call_backend(
            "search",
            name,
            lambda: self._backend.search(
                name,
                values,
                output_fields,
                top_k,
                params,
                self.config.timeout_sec,
            ),
        )

    @staticmethod
    def _hit_values(hit: Any) -> tuple[Mapping[str, Any], float]:
        entity: Any
        if isinstance(hit, Mapping):
            entity = hit.get("entity", hit)
            score = hit.get("distance", hit.get("score"))
        else:
            entity = getattr(hit, "entity", hit)
            score = getattr(hit, "distance", getattr(hit, "score", None))
        if not isinstance(entity, Mapping):
            getter = getattr(entity, "get", None)
            if getter is None:
                raise ContractMismatchError("Milvus hit does not expose output fields")
            entity = _GetterMapping(getter)
        if score is None:
            raise ContractMismatchError("Milvus hit is missing similarity score")
        try:
            numeric_score = float(score)
        except (TypeError, ValueError) as exc:
            raise ContractMismatchError("Milvus hit has invalid similarity score") from exc
        if not math.isfinite(numeric_score):
            raise ContractMismatchError("Milvus hit similarity score is not finite")
        return entity, numeric_score

    @staticmethod
    def _required(entity: Mapping[str, Any], field: str) -> Any:
        value = entity.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ContractMismatchError(
                "Milvus hit is missing a required output field", details={"field": field}
            )
        return value

    def search_visual(self, vector: Sequence[float], top_k: int) -> Sequence[FrameSearchHit]:
        hits = self._search(
            self.config.visual_collection,
            vector,
            ("frame_id", "video_id", "shot_id"),
            top_k,
        )
        output: list[FrameSearchHit] = []
        for hit in hits:
            entity, score = self._hit_values(hit)
            try:
                output.append(
                    FrameSearchHit(
                        frame_id=str(self._required(entity, "frame_id")),
                        video_id=str(self._required(entity, "video_id")),
                        shot_id=int(self._required(entity, "shot_id")),
                        raw_score=score,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ContractMismatchError("Invalid visual Milvus hit") from exc
        return tuple(output)

    def search_ocr(self, vector: Sequence[float], top_k: int) -> Sequence[FrameSearchHit]:
        hits = self._search(
            self.config.ocr_collection, vector, ("frame_id", "video_id"), top_k
        )
        output = []
        for hit in hits:
            entity, score = self._hit_values(hit)
            output.append(
                FrameSearchHit(
                    frame_id=str(self._required(entity, "frame_id")),
                    video_id=str(self._required(entity, "video_id")),
                    shot_id=None,
                    raw_score=score,
                )
            )
        return tuple(output)

    def search_asr(self, vector: Sequence[float], top_k: int) -> Sequence[ASRSearchHit]:
        fields = ("video_id", "interval_id", "start_time_sec", "end_time_sec")
        hits = self._search(self.config.asr_collection, vector, fields, top_k)
        output = []
        for hit in hits:
            entity, score = self._hit_values(hit)
            try:
                output.append(
                    ASRSearchHit(
                        video_id=str(self._required(entity, "video_id")),
                        interval_id=str(self._required(entity, "interval_id")),
                        start_time_sec=float(self._required(entity, "start_time_sec")),
                        end_time_sec=float(self._required(entity, "end_time_sec")),
                        raw_score=score,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ContractMismatchError("Invalid ASR Milvus hit") from exc
        return tuple(output)

    def search_summary(self, vector: Sequence[float], top_k: int) -> Sequence[VideoSearchHit]:
        hits = self._search(self.config.summary_collection, vector, ("video_id",), top_k)
        output = []
        for hit in hits:
            entity, score = self._hit_values(hit)
            output.append(
                VideoSearchHit(
                    video_id=str(self._required(entity, "video_id")), raw_score=score
                )
            )
        return tuple(output)

    def sample_records(
        self, name: str, output_fields: Sequence[str], limit: int
    ) -> Sequence[Mapping[str, Any]]:
        self._ensure_connected()
        if limit < 1:
            raise InvalidQueryError("limit must be >= 1")
        return call_backend(
            "sample_records",
            name,
            lambda: self._backend.sample_records(name, output_fields, limit),
        )


class _GetterMapping(Mapping[str, Any]):
    """Mapping facade for pymilvus entity objects that only expose get()."""

    def __init__(self, getter: Any) -> None:
        self._getter = getter

    def __getitem__(self, key: str) -> Any:
        value = self._getter(key)
        if value is None:
            raise KeyError(key)
        return value

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def get(self, key: str, default: Any = None) -> Any:
        value = self._getter(key)
        return default if value is None else value
