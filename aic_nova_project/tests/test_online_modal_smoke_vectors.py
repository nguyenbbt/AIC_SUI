from __future__ import annotations

import unittest

from scripts.generate_online_modal_smoke_vectors import build_smoke_payload


class FakeEncoder:
    def __init__(self, vector: tuple[float, ...]) -> None:
        self.vector = vector
        self.calls: list[tuple[str, ...]] = []

    @property
    def dimension(self) -> int:
        return len(self.vector)

    def encode_texts(self, texts):
        self.calls.append(tuple(texts))
        return tuple(self.vector for _ in texts)


class ModalSmokeVectorTests(unittest.TestCase):
    def test_one_visual_and_one_shared_text_vector_cover_four_collections(self) -> None:
        visual = FakeEncoder((1.0, 0.0))
        vietnamese = FakeEncoder((0.0, 1.0, 0.0))

        payload = build_smoke_payload(visual, vietnamese)

        self.assertEqual(
            payload,
            {
                "visual_features": [1.0, 0.0],
                "ocr_features": [0.0, 1.0, 0.0],
                "asr_features": [0.0, 1.0, 0.0],
                "summary_features": [0.0, 1.0, 0.0],
            },
        )
        self.assertEqual(visual.calls, [("readiness probe",)])
        self.assertEqual(vietnamese.calls, [("readiness probe",)])


if __name__ == "__main__":
    unittest.main()
