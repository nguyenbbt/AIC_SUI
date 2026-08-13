from __future__ import annotations

import math
import unittest

from online.domain.errors import ContractMismatchError, ResourceUnavailableError
from online.retrieval.modal_encoders import (
    MODAL_ENCODER_SCHEMA_VERSION,
    ModalTextEmbeddingBackend,
)


class FakeRemoteFunction:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def remote(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def response(*, texts: tuple[str, ...], dimension: int = 3) -> dict[str, object]:
    return {
        "schema_version": MODAL_ENCODER_SCHEMA_VERSION,
        "model_kind": "visual",
        "model_id": "ViT-B-32::openai",
        "model_revision": None,
        "dimension": dimension,
        "vectors": [[3.0, 4.0, 0.0] for _ in texts],
    }


class ModalTextEmbeddingBackendTests(unittest.TestCase):
    def test_lookup_is_lazy_and_repeated_texts_are_cached(self) -> None:
        remote = FakeRemoteFunction(
            [
                response(texts=("dimension probe",)),
                response(texts=("first", "second")),
            ]
        )
        lookup_calls: list[tuple[str, str, str | None]] = []

        def lookup(app_name: str, function_name: str, environment_name: str | None):
            lookup_calls.append((app_name, function_name, environment_name))
            return remote

        backend = ModalTextEmbeddingBackend(
            model_kind="visual",
            model_id="ViT-B-32::openai",
            model_revision=None,
            expected_dimension=3,
            app_name="aic-nova-online-encoders",
            function_name="encode",
            environment_name="main",
            cache_size=4,
            function_lookup=lookup,
        )

        self.assertEqual(lookup_calls, [])
        self.assertEqual(backend.dimension, 3)
        first = backend.encode_texts(("first", "second", "first"))
        second = backend.encode_texts(("second", "first"))

        self.assertEqual(lookup_calls, [("aic-nova-online-encoders", "encode", "main")])
        self.assertEqual(len(remote.calls), 2)
        self.assertEqual(remote.calls[1]["texts"], ["first", "second"])
        self.assertEqual(first[0], first[2])
        self.assertEqual(second, (first[1], first[0]))
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in first[0])), 5.0)

    def test_remote_identity_and_dimension_are_strict_contracts(self) -> None:
        wrong_identity = response(texts=("dimension probe",))
        wrong_identity["model_id"] = "wrong-model"
        wrong_dimension = response(texts=("dimension probe",), dimension=4)

        for payload in (wrong_identity, wrong_dimension):
            with self.subTest(payload=payload):
                backend = ModalTextEmbeddingBackend(
                    model_kind="visual",
                    model_id="ViT-B-32::openai",
                    model_revision=None,
                    expected_dimension=3,
                    function_lookup=lambda *_: FakeRemoteFunction([payload]),
                )
                with self.assertRaises(ContractMismatchError):
                    _ = backend.dimension

    def test_remote_failure_is_sanitized(self) -> None:
        backend = ModalTextEmbeddingBackend(
            model_kind="visual",
            model_id="ViT-B-32::openai",
            model_revision=None,
            expected_dimension=3,
            function_lookup=lambda *_: FakeRemoteFunction(
                [RuntimeError("secret Modal infrastructure detail")]
            ),
        )

        with self.assertRaises(ResourceUnavailableError) as raised:
            _ = backend.dimension
        self.assertNotIn("secret Modal infrastructure detail", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
