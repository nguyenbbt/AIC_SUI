"""Authenticated Modal function backend for Online text encoders."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from threading import RLock
from typing import Any, Protocol

from online.domain.errors import ContractMismatchError, ResourceUnavailableError


MODAL_ENCODER_SCHEMA_VERSION = "aic-online-encoder-v1"
DEFAULT_MODAL_ENCODER_APP = "aic-nova-online-encoders"
DEFAULT_MODAL_ENCODER_FUNCTION = "encode"


class _RemoteFunction(Protocol):
    def remote(self, **kwargs: object) -> object: ...


ModalFunctionLookup = Callable[[str, str, str | None], _RemoteFunction]


def _lookup_modal_function(
    app_name: str,
    function_name: str,
    environment_name: str | None,
) -> _RemoteFunction:
    try:
        import modal
    except ImportError as exc:
        raise ResourceUnavailableError(
            "Modal SDK is not installed",
            details={"resource": "modal_encoder", "stage": "lookup"},
        ) from exc

    kwargs = {"environment_name": environment_name} if environment_name else {}
    return modal.Function.from_name(app_name, function_name, **kwargs)


class ModalTextEmbeddingBackend:
    """Call a private deployed Modal function and cache embeddings by text."""

    def __init__(
        self,
        *,
        model_kind: str,
        model_id: str,
        model_revision: str | None,
        expected_dimension: int | None = None,
        app_name: str = DEFAULT_MODAL_ENCODER_APP,
        function_name: str = DEFAULT_MODAL_ENCODER_FUNCTION,
        environment_name: str | None = None,
        cache_size: int = 256,
        function_lookup: ModalFunctionLookup | None = None,
    ) -> None:
        if model_kind not in {"visual", "vietnamese"}:
            raise ValueError("model_kind must be visual or vietnamese")
        for name, value in (
            ("model_id", model_id),
            ("app_name", app_name),
            ("function_name", function_name),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if model_revision is not None and (
            not isinstance(model_revision, str) or not model_revision.strip()
        ):
            raise ValueError("model_revision must be non-empty when provided")
        if expected_dimension is not None and (
            isinstance(expected_dimension, bool)
            or not isinstance(expected_dimension, int)
            or expected_dimension < 1
        ):
            raise ValueError("expected_dimension must be a positive integer")
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 1:
            raise ValueError("cache_size must be a positive integer")

        self.model_kind = model_kind
        self.model_id = model_id.strip()
        self.model_revision = model_revision.strip() if model_revision else None
        self.expected_dimension = expected_dimension
        self.app_name = app_name.strip()
        self.function_name = function_name.strip()
        self.environment_name = environment_name.strip() if environment_name else None
        self.cache_size = cache_size
        self._function_lookup = function_lookup or _lookup_modal_function
        self._function: _RemoteFunction | None = None
        self._dimension: int | None = None
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._lock = RLock()

    @property
    def dimension(self) -> int:
        with self._lock:
            if self._dimension is None:
                self._fetch_missing(("dimension probe",))
            assert self._dimension is not None
            return self._dimension

    def encode_texts(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        values = tuple(texts)
        if not values:
            return ()
        with self._lock:
            missing = tuple(dict.fromkeys(text for text in values if text not in self._cache))
            if missing:
                self._fetch_missing(missing)
            output: list[tuple[float, ...]] = []
            for text in values:
                vector = self._cache[text]
                self._cache.move_to_end(text)
                output.append(vector)
            return tuple(output)

    def _fetch_missing(self, texts: tuple[str, ...]) -> None:
        function = self._get_function()
        try:
            payload = function.remote(model_kind=self.model_kind, texts=list(texts))
        except ResourceUnavailableError:
            raise
        except Exception as exc:
            raise ResourceUnavailableError(
                "Modal encoder request failed",
                details={
                    "resource": "modal_encoder",
                    "stage": "inference",
                    "cause_type": type(exc).__name__,
                },
            ) from exc

        vectors = self._validate_response(payload, expected_rows=len(texts))
        for text, vector in zip(texts, vectors, strict=True):
            self._cache[text] = vector
            self._cache.move_to_end(text)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def _get_function(self) -> _RemoteFunction:
        if self._function is not None:
            return self._function
        try:
            self._function = self._function_lookup(
                self.app_name,
                self.function_name,
                self.environment_name,
            )
        except ResourceUnavailableError:
            raise
        except Exception as exc:
            raise ResourceUnavailableError(
                "Modal encoder deployment could not be resolved",
                details={
                    "resource": "modal_encoder",
                    "stage": "lookup",
                    "cause_type": type(exc).__name__,
                },
            ) from exc
        return self._function

    def _validate_response(
        self,
        payload: object,
        *,
        expected_rows: int,
    ) -> tuple[tuple[float, ...], ...]:
        expected_fields = {
            "schema_version",
            "model_kind",
            "model_id",
            "model_revision",
            "dimension",
            "vectors",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected_fields:
            raise self._contract_error("response_shape")
        if payload["schema_version"] != MODAL_ENCODER_SCHEMA_VERSION:
            raise self._contract_error("schema_version")
        if payload["model_kind"] != self.model_kind:
            raise self._contract_error("model_kind")
        if payload["model_id"] != self.model_id:
            raise self._contract_error("model_id")
        if payload["model_revision"] != self.model_revision:
            raise self._contract_error("model_revision")

        dimension = payload["dimension"]
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise self._contract_error("dimension")
        if self.expected_dimension is not None and dimension != self.expected_dimension:
            raise self._contract_error("dimension")
        if self._dimension is not None and dimension != self._dimension:
            raise self._contract_error("dimension")

        raw_vectors = payload["vectors"]
        if isinstance(raw_vectors, (str, bytes)):
            raise self._contract_error("vectors")
        try:
            rows = tuple(raw_vectors)  # type: ignore[arg-type]
        except TypeError as exc:
            raise self._contract_error("vectors") from exc
        if len(rows) != expected_rows:
            raise self._contract_error("row_count")

        vectors: list[tuple[float, ...]] = []
        for row in rows:
            if isinstance(row, (str, bytes)):
                raise self._contract_error("vectors")
            try:
                vector = tuple(float(value) for value in row)
            except (TypeError, ValueError) as exc:
                raise self._contract_error("vectors") from exc
            if len(vector) != dimension or not all(math.isfinite(value) for value in vector):
                raise self._contract_error("vectors")
            vectors.append(vector)

        self._dimension = dimension
        return tuple(vectors)

    @staticmethod
    def _contract_error(field: str) -> ContractMismatchError:
        return ContractMismatchError(
            "Modal encoder response violates the Online contract",
            details={"resource": "modal_encoder", "field": field},
        )


__all__ = [
    "DEFAULT_MODAL_ENCODER_APP",
    "DEFAULT_MODAL_ENCODER_FUNCTION",
    "MODAL_ENCODER_SCHEMA_VERSION",
    "ModalFunctionLookup",
    "ModalTextEmbeddingBackend",
]
