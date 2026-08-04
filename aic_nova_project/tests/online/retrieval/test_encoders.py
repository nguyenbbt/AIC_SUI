from __future__ import annotations

import math
import threading
import time
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from online.domain.errors import (
    ContractMismatchError,
    DimensionMismatchError,
    InvalidQueryError,
    ResourceUnavailableError,
)
from online.ports import TextEncoderPort
from online.retrieval.encoders import (
    OPEN_CLIP_DIMENSION,
    OPEN_CLIP_MODEL_ID,
    VIETNAMESE_MODEL_NAME,
    OpenCLIPTextEncoder,
    VietnameseTextEncoder,
)
from online.testing import FakeTextEncoder


class StaticBackend:
    def __init__(
        self,
        *,
        dimension: int = 3,
        rows: list[list[float]] | None = None,
        failure: Exception | None = None,
    ) -> None:
        self._dimension = dimension
        self.rows = rows
        self.failure = failure
        self.calls: list[tuple[str, ...]] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_texts(self, texts: tuple[str, ...]) -> list[list[float]]:
        self.calls.append(tuple(texts))
        if self.failure is not None:
            raise self.failure
        if self.rows is not None:
            return self.rows
        return [[3.0, 4.0, 0.0] for _ in texts]


class ConcurrentBackend(StaticBackend):
    def __init__(self) -> None:
        super().__init__()
        self._state_lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def encode_texts(self, texts: tuple[str, ...]) -> list[list[float]]:
        with self._state_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.02)
            return super().encode_texts(texts)
        finally:
            with self._state_lock:
                self.active -= 1


class TextEncoderTests(unittest.TestCase):
    def test_open_clip_uses_exact_model_pretrained_tokenizer_and_float32_output(
        self,
    ) -> None:
        calls: list[tuple[str, str]] = []

        class Tensor:
            def __init__(self, values=None) -> None:
                self.values = values

            def to(self, *args, **kwargs):
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self.values

        class Model:
            def to(self, *args, **kwargs):
                return self

            def eval(self):
                return None

            def encode_text(self, tokens):
                return Tensor([[1.0] + [0.0] * (OPEN_CLIP_DIMENSION - 1)])

        def create_model_and_transforms(model_name, *, pretrained):
            calls.append((model_name, pretrained))
            return Model(), None, None

        def get_tokenizer(model_name):
            calls.append(("tokenizer", model_name))
            return lambda texts: Tensor()

        fake_open_clip = SimpleNamespace(
            create_model_and_transforms=create_model_and_transforms,
            get_tokenizer=get_tokenizer,
        )
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            float16="float16",
            bfloat16="bfloat16",
            float32="float32",
            no_grad=lambda: nullcontext(),
            autocast=lambda **kwargs: nullcontext(),
        )
        with patch.dict(
            "sys.modules", {"open_clip": fake_open_clip, "torch": fake_torch}
        ):
            vector = OpenCLIPTextEncoder().encode_texts(("query",))[0]

        self.assertEqual(calls[:2], [("ViT-B-32", "openai"), ("tokenizer", "ViT-B-32")])
        self.assertEqual(len(vector), OPEN_CLIP_DIMENSION)
        self.assertTrue(all(isinstance(value, float) for value in vector))

    def test_open_clip_is_lazy_cached_normalized_and_protocol_conformant(self) -> None:
        backend = StaticBackend()
        factory_calls = 0

        def factory() -> StaticBackend:
            nonlocal factory_calls
            factory_calls += 1
            return backend

        encoder = OpenCLIPTextEncoder(expected_dimension=3, backend_factory=factory)
        self.assertEqual(encoder.model_id, OPEN_CLIP_MODEL_ID)
        self.assertEqual(factory_calls, 0)
        self.assertIsInstance(encoder, TextEncoderPort)

        vectors = encoder.encode_texts(("first", "second"))
        self.assertEqual(factory_calls, 1)
        self.assertEqual(encoder.dimension, 3)
        self.assertEqual(factory_calls, 1)
        self.assertEqual(len(vectors), 2)
        self.assertTrue(all(len(vector) == 3 for vector in vectors))
        self.assertTrue(
            all(isinstance(value, float) for vector in vectors for value in vector)
        )
        for vector in vectors:
            self.assertAlmostEqual(
                math.sqrt(sum(value * value for value in vector)), 1.0
            )

        encoder.encode_texts(("third",))
        self.assertEqual(factory_calls, 1)
        self.assertEqual(backend.calls, [("first", "second"), ("third",)])

    def test_vietnamese_encoder_uses_offline_defaults_and_same_validation(self) -> None:
        backend = StaticBackend(rows=[[1.0, 1.0, 1.0]])
        encoder = VietnameseTextEncoder(
            expected_dimension=3, backend_factory=lambda: backend
        )

        self.assertEqual(encoder.model_name, VIETNAMESE_MODEL_NAME)
        self.assertEqual(encoder.max_length, 256)
        self.assertEqual(encoder.batch_size, 128)
        vector = encoder.encode_texts(("nguoi di xe dap",))[0]
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in vector)), 1.0)

    def test_empty_batch_does_not_load_model(self) -> None:
        calls = 0

        def factory() -> StaticBackend:
            nonlocal calls
            calls += 1
            return StaticBackend()

        encoder = OpenCLIPTextEncoder(backend_factory=factory)
        self.assertEqual(encoder.encode_texts(()), ())
        self.assertEqual(calls, 0)

    def test_invalid_text_inputs_are_rejected_before_inference(self) -> None:
        backend = StaticBackend()
        encoder = VietnameseTextEncoder(backend_factory=lambda: backend)

        for invalid in ("one string", ("",), ("   ",), (1,)):
            with self.subTest(invalid=invalid):
                with self.assertRaises(InvalidQueryError):
                    encoder.encode_texts(invalid)  # type: ignore[arg-type]
        self.assertEqual(backend.calls, [])

    def test_collection_dimension_mismatch_is_explicit(self) -> None:
        encoder = OpenCLIPTextEncoder(
            expected_dimension=4,
            backend_factory=lambda: StaticBackend(dimension=3),
        )
        with self.assertRaises(DimensionMismatchError) as raised:
            _ = encoder.dimension
        self.assertEqual(
            raised.exception.details,
            {"model": OPEN_CLIP_MODEL_ID, "expected": 4, "actual": 3},
        )

    def test_invalid_encoder_configuration_is_rejected_without_loading(self) -> None:
        with self.assertRaises(ValueError):
            OpenCLIPTextEncoder(expected_dimension="3")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            OpenCLIPTextEncoder(precision="int8")
        with self.assertRaises(ValueError):
            OpenCLIPTextEncoder(model_id="ViT-L-14::openai")
        with self.assertRaises(ValueError):
            VietnameseTextEncoder(max_length=0)
        with self.assertRaises(ValueError):
            VietnameseTextEncoder(batch_size=0)

    def test_malformed_backend_outputs_are_rejected(self) -> None:
        cases = (
            (StaticBackend(rows=[]), ContractMismatchError),
            (StaticBackend(rows=[[1.0, 2.0]]), DimensionMismatchError),
            (StaticBackend(rows=[[0.0, 0.0, 0.0]]), ContractMismatchError),
            (StaticBackend(rows=[[float("nan"), 1.0, 2.0]]), ContractMismatchError),
        )
        for backend, error_type in cases:
            with self.subTest(error_type=error_type.__name__):
                encoder = VietnameseTextEncoder(
                    backend_factory=lambda backend=backend: backend
                )
                with self.assertRaises(error_type):
                    encoder.encode_texts(("query",))

    def test_loading_and_inference_failures_are_safe_resource_errors(self) -> None:
        def broken_factory() -> StaticBackend:
            raise OSError("secret local model path")

        loading_encoder = OpenCLIPTextEncoder(backend_factory=broken_factory)
        with self.assertRaises(ResourceUnavailableError) as loading:
            _ = loading_encoder.dimension
        self.assertEqual(loading.exception.details["stage"], "load")
        self.assertNotIn("secret local model path", str(loading.exception))

        inference_encoder = VietnameseTextEncoder(
            backend_factory=lambda: StaticBackend(failure=RuntimeError("GPU details"))
        )
        with self.assertRaises(ResourceUnavailableError) as inference:
            inference_encoder.encode_texts(("query",))
        self.assertEqual(inference.exception.details["stage"], "inference")
        self.assertNotIn("GPU details", str(inference.exception))

    def test_fake_encoder_is_deterministic_normalized_and_records_calls(self) -> None:
        encoder = FakeTextEncoder(dimension=5)
        first = encoder.encode_texts(("same query",))
        second = encoder.encode_texts(("same query",))

        self.assertIsInstance(encoder, TextEncoderPort)
        self.assertEqual(first, second)
        self.assertNotEqual(first, encoder.encode_texts(("different query",)))
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in first[0])), 1.0)
        self.assertEqual(encoder.calls[0], ("same query",))
        self.assertEqual(
            encoder.encode_texts(("  same query  ",)),
            encoder.encode_texts(("same query",)),
        )

    def test_shared_model_inference_is_serialized_across_branch_threads(self) -> None:
        backend = ConcurrentBackend()
        encoder = VietnameseTextEncoder(backend_factory=lambda: backend)
        errors: list[BaseException] = []

        def encode(text: str) -> None:
            try:
                encoder.encode_texts((text,))
            except BaseException as exc:  # pragma: no cover - assertion below
                errors.append(exc)

        threads = [
            threading.Thread(target=encode, args=(f"query-{i}",)) for i in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=1.0)

        self.assertEqual(errors, [])
        self.assertEqual(backend.max_active, 1)


if __name__ == "__main__":
    unittest.main()
