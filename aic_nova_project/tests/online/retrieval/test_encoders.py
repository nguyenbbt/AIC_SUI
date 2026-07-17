from __future__ import annotations

import math
import unittest

from online.domain.errors import (
    ContractMismatchError,
    DimensionMismatchError,
    InvalidQueryError,
    ResourceUnavailableError,
)
from online.ports import TextEncoderPort
from online.retrieval.encoders import (
    PE_CORE_MODEL_ID,
    VIETNAMESE_MODEL_NAME,
    PECoreTextEncoder,
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


class TextEncoderTests(unittest.TestCase):
    def test_pe_core_is_lazy_cached_normalized_and_protocol_conformant(self) -> None:
        backend = StaticBackend()
        factory_calls = 0

        def factory() -> StaticBackend:
            nonlocal factory_calls
            factory_calls += 1
            return backend

        encoder = PECoreTextEncoder(expected_dimension=3, backend_factory=factory)
        self.assertEqual(encoder.model_id, PE_CORE_MODEL_ID)
        self.assertEqual(factory_calls, 0)
        self.assertIsInstance(encoder, TextEncoderPort)

        vectors = encoder.encode_texts(("first", "second"))
        self.assertEqual(factory_calls, 1)
        self.assertEqual(encoder.dimension, 3)
        self.assertEqual(factory_calls, 1)
        self.assertEqual(len(vectors), 2)
        self.assertTrue(all(len(vector) == 3 for vector in vectors))
        self.assertTrue(all(isinstance(value, float) for vector in vectors for value in vector))
        for vector in vectors:
            self.assertAlmostEqual(math.sqrt(sum(value * value for value in vector)), 1.0)

        encoder.encode_texts(("third",))
        self.assertEqual(factory_calls, 1)
        self.assertEqual(backend.calls, [("first", "second"), ("third",)])

    def test_vietnamese_encoder_uses_offline_defaults_and_same_validation(self) -> None:
        backend = StaticBackend(rows=[[1.0, 1.0, 1.0]])
        encoder = VietnameseTextEncoder(expected_dimension=3, backend_factory=lambda: backend)

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

        encoder = PECoreTextEncoder(backend_factory=factory)
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
        encoder = PECoreTextEncoder(
            expected_dimension=4,
            backend_factory=lambda: StaticBackend(dimension=3),
        )
        with self.assertRaises(DimensionMismatchError) as raised:
            _ = encoder.dimension
        self.assertEqual(raised.exception.details, {"model": PE_CORE_MODEL_ID, "expected": 4, "actual": 3})

    def test_invalid_encoder_configuration_is_rejected_without_loading(self) -> None:
        with self.assertRaises(ValueError):
            PECoreTextEncoder(expected_dimension="3")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            PECoreTextEncoder(precision="int8")
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
                encoder = VietnameseTextEncoder(backend_factory=lambda backend=backend: backend)
                with self.assertRaises(error_type):
                    encoder.encode_texts(("query",))

    def test_loading_and_inference_failures_are_safe_resource_errors(self) -> None:
        def broken_factory() -> StaticBackend:
            raise OSError("secret local model path")

        loading_encoder = PECoreTextEncoder(backend_factory=broken_factory)
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


if __name__ == "__main__":
    unittest.main()
