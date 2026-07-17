"""Read-only Milvus search adapter with SDK-neutral outputs."""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from online.config import MilvusResourceConfig
from online.domain.errors import ContractMismatchError, DimensionMismatchError, InvalidQueryError, ResourceUnavailableError
from online.ports.records import ASRSearchHit, FrameSearchHit, VideoSearchHit

from ._errors import call_backend
from ._concurrency import ConcurrentReadGuard


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
        self._state_lock = threading.RLock()
        self._read_guard = ConcurrentReadGuard("milvus")

    def connect(self) -> None:
        with self._state_lock:
            if self._connected:
                return
            call_backend("connect", "milvus", self._backend.connect)
            self._connected = True

    def close(self) -> None:
        self._read_guard.begin_close()
        try:
            with self._state_lock:
                try:
                    if self._connected:
                        call_backend("close", "milvus", self._backend.close)
                finally:
                    self._connected = False
                    self._dimensions.clear()
        finally:
            self._read_guard.end_close()

    def _ensure_connected(self) -> None:
        with self._state_lock:
            if not self._connected:
                raise ResourceUnavailableError("Milvus adapter is not connected")

    def health_check(self) -> None:
        with self._read_guard.read():
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
        with self._read_guard.read():
            self._ensure_connected()
            exists = call_backend(
                "collection_exists", name, lambda: self._backend.collection_exists(name)
            )
            if not exists:
                raise ResourceUnavailableError(
                    "Milvus collection is missing", details={"resource": name}
                )
            description = call_backend(
                "describe_collection",
                name,
                lambda: self._backend.describe_collection(name),
            )
            if not isinstance(description, Mapping):
                raise ContractMismatchError(
                    "Milvus collection description is invalid",
                    details={"resource": name},
                )
            return description

    def collection_exists(self, name: str) -> bool:
        with self._read_guard.read():
            self._ensure_connected()
            return call_backend(
                "collection_exists", name, lambda: self._backend.collection_exists(name)
            )

    def _dimension(self, name: str) -> int:
        with self._state_lock:
            cached = self._dimensions.get(name)
        if cached is not None:
            return cached
        description = self.describe_collection(name)
        dimension = description.get("dimension")
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 1
        ):
            raise ContractMismatchError(
                "Milvus embedding dimension is missing or invalid",
                details={"resource": name},
            )
        with self._state_lock:
            self._dimensions[name] = dimension
        return dimension

    def _validate_vector(self, name: str, vector: Sequence[float]) -> tuple[float, ...]:
        if isinstance(vector, (str, bytes)):
            raise InvalidQueryError("query vector must be a numeric sequence")
        try:
            raw_values = tuple(vector)
        except (TypeError, ValueError) as exc:
            raise InvalidQueryError("query vector must contain numeric values") from exc
        if any(isinstance(value, bool) for value in raw_values):
            raise InvalidQueryError("query vector must contain numeric values")
        try:
            values = tuple(float(value) for value in raw_values)
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
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise InvalidQueryError("top_k must be >= 1")
        with self._read_guard.read():
            self._ensure_connected()
            values = self._validate_vector(name, vector)
            params = {"metric_type": "IP", "params": {"ef": self.config.search_ef}}
            hits = call_backend(
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
        if not isinstance(hits, Sequence) or isinstance(hits, (str, bytes)):
            raise ContractMismatchError(
                "Milvus search response is not a hit sequence",
                details={"resource": name},
            )
        return tuple(hits)

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
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ContractMismatchError("Milvus hit has invalid similarity score")
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

    @classmethod
    def _required_str(cls, entity: Mapping[str, Any], field: str) -> str:
        value = cls._required(entity, field)
        if not isinstance(value, str) or value != value.strip():
            raise ContractMismatchError(
                "Milvus output field must be a canonical string",
                details={"field": field},
            )
        return value

    @classmethod
    def _required_int(cls, entity: Mapping[str, Any], field: str) -> int:
        value = cls._required(entity, field)
        if isinstance(value, bool):
            raise ContractMismatchError(
                "Milvus output field must be an integer", details={"field": field}
            )
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ContractMismatchError(
                "Milvus output field must be an integer", details={"field": field}
            ) from exc
        try:
            if float(value) != result:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ContractMismatchError(
                "Milvus output field must be an integer", details={"field": field}
            ) from exc
        return result

    @classmethod
    def _required_float(cls, entity: Mapping[str, Any], field: str) -> float:
        value = cls._required(entity, field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractMismatchError(
                "Milvus output field must be numeric", details={"field": field}
            )
        result = float(value)
        if not math.isfinite(result):
            raise ContractMismatchError(
                "Milvus output field must be finite", details={"field": field}
            )
        return result

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
                        frame_id=self._required_str(entity, "frame_id"),
                        video_id=self._required_str(entity, "video_id"),
                        shot_id=self._required_int(entity, "shot_id"),
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
            try:
                output.append(
                    FrameSearchHit(
                        frame_id=self._required_str(entity, "frame_id"),
                        video_id=self._required_str(entity, "video_id"),
                        shot_id=None,
                        raw_score=score,
                    )
                )
            except Exception as exc:
                if isinstance(exc, ContractMismatchError):
                    raise
                raise ContractMismatchError("Invalid OCR Milvus hit") from exc
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
                        video_id=self._required_str(entity, "video_id"),
                        interval_id=self._required_str(entity, "interval_id"),
                        start_time_sec=self._required_float(entity, "start_time_sec"),
                        end_time_sec=self._required_float(entity, "end_time_sec"),
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
            try:
                output.append(
                    VideoSearchHit(
                        video_id=self._required_str(entity, "video_id"),
                        raw_score=score,
                    )
                )
            except Exception as exc:
                if isinstance(exc, ContractMismatchError):
                    raise
                raise ContractMismatchError("Invalid summary Milvus hit") from exc
        return tuple(output)

    def sample_records(
        self, name: str, output_fields: Sequence[str], limit: int
    ) -> Sequence[Mapping[str, Any]]:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise InvalidQueryError("limit must be >= 1")
        if isinstance(output_fields, (str, bytes)) or not output_fields or any(
            not isinstance(field, str)
            or not field.strip()
            or field != field.strip()
            for field in output_fields
        ):
            raise InvalidQueryError("output_fields must contain non-empty field names")
        with self._read_guard.read():
            self._ensure_connected()
            records = call_backend(
                "sample_records",
                name,
                lambda: self._backend.sample_records(name, output_fields, limit),
            )
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise ContractMismatchError("Milvus sample response is invalid")
        if any(not isinstance(record, Mapping) for record in records):
            raise ContractMismatchError("Milvus sample record is invalid")
        return tuple(records)


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
