"""Reusable Online text encoders for the two Offline embedding spaces."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from threading import Lock
from typing import Protocol

from online.domain.errors import (
    ContractMismatchError,
    DataInfrastructureError,
    DimensionMismatchError,
    InvalidQueryError,
    ResourceUnavailableError,
)


PE_CORE_MODEL_ID = "hf-hub:timm/PE-Core-bigG-14-448"
VIETNAMESE_MODEL_NAME = "dangvantuan/vietnamese-embedding"


class _TextEmbeddingBackend(Protocol):
    @property
    def dimension(self) -> int: ...

    def encode_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class _ValidatedTextEncoder:
    """Lazy, thread-safe adapter that validates a model backend once loaded."""

    def __init__(
        self,
        *,
        backend_factory: Callable[[], _TextEmbeddingBackend],
        model_identifier: str,
        expected_dimension: int | None,
    ) -> None:
        if expected_dimension is not None and (
            isinstance(expected_dimension, bool)
            or not isinstance(expected_dimension, int)
            or expected_dimension < 1
        ):
            raise ValueError("expected_dimension must be a positive integer")
        self._backend_factory = backend_factory
        self._model_identifier = model_identifier
        self._expected_dimension = expected_dimension
        self._backend: _TextEmbeddingBackend | None = None
        self._dimension: int | None = None
        self._load_lock = Lock()
        # Torch/SentenceTransformer backends are shared by all semantic
        # branches. Serialize inference per model instance to avoid concurrent
        # mutation/CUDA execution assumptions inside third-party model code.
        self._inference_lock = Lock()

    @property
    def dimension(self) -> int:
        self._ensure_backend()
        assert self._dimension is not None
        return self._dimension

    def encode_texts(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        cleaned = self._validate_texts(texts)
        if not cleaned:
            return ()

        backend = self._ensure_backend()
        try:
            with self._inference_lock:
                raw_vectors = backend.encode_texts(cleaned)
        except DataInfrastructureError:
            raise
        except Exception as exc:
            raise ResourceUnavailableError(
                "Text encoder inference failed",
                details={
                    "model": self._model_identifier,
                    "stage": "inference",
                    "cause_type": type(exc).__name__,
                },
            ) from exc
        return self._validate_and_normalize(raw_vectors, expected_rows=len(cleaned))

    def _ensure_backend(self) -> _TextEmbeddingBackend:
        if self._backend is not None:
            return self._backend
        with self._load_lock:
            if self._backend is not None:
                return self._backend
            try:
                backend = self._backend_factory()
                raw_dimension = backend.dimension
            except DataInfrastructureError:
                raise
            except Exception as exc:
                raise ResourceUnavailableError(
                    "Text encoder model could not be loaded",
                    details={
                        "model": self._model_identifier,
                        "stage": "load",
                        "cause_type": type(exc).__name__,
                    },
                ) from exc

            if isinstance(raw_dimension, bool) or not isinstance(raw_dimension, int) or raw_dimension < 1:
                raise DimensionMismatchError(
                    "Text encoder reported an invalid dimension",
                    details={"model": self._model_identifier, "actual": raw_dimension},
                )
            if self._expected_dimension is not None and raw_dimension != self._expected_dimension:
                raise DimensionMismatchError(
                    "Text encoder dimension does not match collection schema",
                    details={
                        "model": self._model_identifier,
                        "expected": self._expected_dimension,
                        "actual": raw_dimension,
                    },
                )
            self._backend = backend
            self._dimension = raw_dimension
            return backend

    @staticmethod
    def _validate_texts(texts: Sequence[str]) -> tuple[str, ...]:
        if isinstance(texts, (str, bytes)):
            raise InvalidQueryError("texts must be a sequence of strings")
        try:
            values = tuple(texts)
        except TypeError as exc:
            raise InvalidQueryError("texts must be a sequence of strings") from exc

        cleaned: list[str] = []
        for index, value in enumerate(values):
            if not isinstance(value, str) or not value.strip():
                raise InvalidQueryError(
                    "encoder input must contain non-empty text",
                    details={"index": index},
                )
            cleaned.append(value.strip())
        return tuple(cleaned)

    def _validate_and_normalize(
        self,
        raw_vectors: Sequence[Sequence[float]],
        *,
        expected_rows: int,
    ) -> tuple[tuple[float, ...], ...]:
        if isinstance(raw_vectors, (str, bytes)):
            raise ContractMismatchError("Text encoder returned a non-matrix output")
        try:
            rows = tuple(raw_vectors)
        except TypeError as exc:
            raise ContractMismatchError("Text encoder returned a non-matrix output") from exc
        if len(rows) != expected_rows:
            raise ContractMismatchError(
                "Text encoder output row count does not match input batch",
                details={"expected": expected_rows, "actual": len(rows)},
            )

        dimension = self.dimension
        output: list[tuple[float, ...]] = []
        for index, row in enumerate(rows):
            if isinstance(row, (str, bytes)):
                raise ContractMismatchError(
                    "Text encoder returned a non-vector row",
                    details={"index": index},
                )
            try:
                values = tuple(float(value) for value in row)
            except (TypeError, ValueError) as exc:
                raise ContractMismatchError(
                    "Text encoder returned a non-numeric vector",
                    details={"index": index},
                ) from exc
            if len(values) != dimension:
                raise DimensionMismatchError(
                    "Text encoder output dimension is inconsistent",
                    details={"index": index, "expected": dimension, "actual": len(values)},
                )
            if not all(math.isfinite(value) for value in values):
                raise ContractMismatchError(
                    "Text encoder returned a non-finite vector",
                    details={"index": index},
                )
            norm = math.sqrt(sum(value * value for value in values))
            if not math.isfinite(norm) or norm <= 0.0:
                raise ContractMismatchError(
                    "Text encoder returned a zero-norm vector",
                    details={"index": index},
                )
            output.append(tuple(value / norm for value in values))
        return tuple(output)


class PECoreTextEncoder(_ValidatedTextEncoder):
    """Text tower paired with Offline PE-Core-bigG-14-448 image vectors."""

    def __init__(
        self,
        *,
        model_id: str = PE_CORE_MODEL_ID,
        device: str = "auto",
        precision: str = "fp16",
        expected_dimension: int | None = None,
        backend_factory: Callable[[], _TextEmbeddingBackend] | None = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be non-empty")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be non-empty")
        if precision not in {"fp16", "bf16", "fp32"}:
            raise ValueError("precision must be fp16, bf16 or fp32")

        self.model_id = model_id.strip()
        self.device = device.strip()
        self.precision = precision
        factory = backend_factory or (
            lambda: _PECoreBackend(
                model_id=self.model_id,
                device=self.device,
                precision=self.precision,
            )
        )
        super().__init__(
            backend_factory=factory,
            model_identifier=self.model_id,
            expected_dimension=expected_dimension,
        )


class VietnameseTextEncoder(_ValidatedTextEncoder):
    """SentenceTransformer query encoder paired with Offline text vectors."""

    def __init__(
        self,
        *,
        model_name: str = VIETNAMESE_MODEL_NAME,
        device: str | None = None,
        max_length: int = 256,
        batch_size: int = 128,
        cache_dir: str | None = None,
        expected_dimension: int | None = None,
        backend_factory: Callable[[], _TextEmbeddingBackend] | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be non-empty")
        if device is not None and (not isinstance(device, str) or not device.strip()):
            raise ValueError("device must be non-empty when provided")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
            raise ValueError("max_length must be a positive integer")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

        self.model_name = model_name.strip()
        self.device = device.strip() if device is not None else None
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        factory = backend_factory or (
            lambda: _VietnameseSentenceTransformerBackend(
                model_name=self.model_name,
                device=self.device,
                max_length=self.max_length,
                batch_size=self.batch_size,
                cache_dir=self.cache_dir,
            )
        )
        super().__init__(
            backend_factory=factory,
            model_identifier=self.model_name,
            expected_dimension=expected_dimension,
        )


class _PECoreBackend:
    def __init__(self, *, model_id: str, device: str, precision: str) -> None:
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError("PE-Core runtime dependencies are unavailable") from exc

        self._torch = torch
        self._device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if device == "auto" and not torch.cuda.is_available():
            self._device = "cpu"
        self._precision = "fp32" if self._device.startswith("cpu") else precision
        dtype_by_precision = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        self._dtype = dtype_by_precision[self._precision]

        tokenizer_model_id = model_id
        if "::" in model_id:
            model_name, pretrained = model_id.split("::", 1)
            self._model, _, _ = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
            )
            tokenizer_model_id = model_name
        else:
            self._model, _, _ = open_clip.create_model_and_transforms(model_id)
        self._tokenizer = open_clip.get_tokenizer(tokenizer_model_id)
        self._model = self._model.to(self._device, dtype=self._dtype)
        self._model.eval()
        probe = self.encode_texts(("dimension probe",))
        self._dimension = len(probe[0])

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        tokens = self._tokenizer(list(texts)).to(self._device)
        use_autocast = self._device.startswith("cuda") and self._precision in {"fp16", "bf16"}
        autocast = (
            self._torch.autocast(device_type="cuda", dtype=self._dtype)
            if use_autocast
            else nullcontext()
        )
        with self._torch.no_grad():
            with autocast:
                features = self._model.encode_text(tokens)
        return features.detach().to(dtype=self._torch.float32).cpu().tolist()


class _VietnameseSentenceTransformerBackend:
    def __init__(
        self,
        *,
        model_name: str,
        device: str | None,
        max_length: int,
        batch_size: int,
        cache_dir: str | None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("SentenceTransformer runtime dependency is unavailable") from exc

        kwargs: dict[str, object] = {}
        if device is not None:
            kwargs["device"] = device
        if cache_dir is not None:
            kwargs["cache_folder"] = cache_dir
        self._model = SentenceTransformer(model_name, **kwargs)
        self._model.max_seq_length = max_length
        self._batch_size = batch_size

        dimension = self._model.get_sentence_embedding_dimension()
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            dimension = len(self.encode_texts(("dimension probe",))[0])
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        embeddings = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings


__all__ = [
    "PE_CORE_MODEL_ID",
    "VIETNAMESE_MODEL_NAME",
    "PECoreTextEncoder",
    "VietnameseTextEncoder",
]
