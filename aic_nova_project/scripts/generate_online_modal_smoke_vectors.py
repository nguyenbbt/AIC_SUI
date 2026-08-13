"""Generate normalized Modal vectors for the read-only Offline validator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from online.adapters.manifest import DatasetManifestGate
from online.config import OnlineDataConfig
from online.ports import TextEncoderPort
from online.retrieval.encoders import (
    OPENCLIP_MODEL_ID,
    VIETNAMESE_MODEL_NAME,
    VIETNAMESE_MODEL_REVISION,
    OpenCLIPTextEncoder,
    VietnameseTextEncoder,
)
from online.retrieval.modal_encoders import ModalTextEmbeddingBackend
from retrieval_api.composition import RuntimeCompositionConfig


def build_smoke_payload(
    visual_encoder: TextEncoderPort,
    vietnamese_encoder: TextEncoderPort,
) -> dict[str, list[float]]:
    """Create the four validator vectors using the two shared model spaces."""

    visual = list(visual_encoder.encode_texts(("readiness probe",))[0])
    text = list(vietnamese_encoder.encode_texts(("readiness probe",))[0])
    return {
        "visual_features": visual,
        "ocr_features": text,
        "asr_features": text,
        "summary_features": text,
    }


def build_modal_encoders() -> tuple[TextEncoderPort, TextEncoderPort]:
    """Build authenticated Modal encoders pinned to the active READY manifest."""

    data_config = OnlineDataConfig.from_env()
    runtime_config = RuntimeCompositionConfig.from_env()
    manifest_gate = DatasetManifestGate(data_config.dataset)
    manifest_gate.connect()
    try:
        manifest = manifest_gate.manifest
        visual_dimension = manifest.visual_dimension
        text_dimension = manifest.text_dimension
    finally:
        manifest_gate.close()

    visual = OpenCLIPTextEncoder(
        expected_dimension=visual_dimension,
        backend_factory=lambda: ModalTextEmbeddingBackend(
            model_kind="visual",
            model_id=OPENCLIP_MODEL_ID,
            model_revision=None,
            expected_dimension=visual_dimension,
            app_name=runtime_config.modal_encoder_app,
            function_name=runtime_config.modal_encoder_function,
            environment_name=runtime_config.modal_environment,
            cache_size=runtime_config.modal_encoder_cache_size,
        ),
    )
    vietnamese = VietnameseTextEncoder(
        expected_dimension=text_dimension,
        backend_factory=lambda: ModalTextEmbeddingBackend(
            model_kind="vietnamese",
            model_id=VIETNAMESE_MODEL_NAME,
            model_revision=VIETNAMESE_MODEL_REVISION,
            expected_dimension=text_dimension,
            app_name=runtime_config.modal_encoder_app,
            function_name=runtime_config.modal_encoder_function,
            environment_name=runtime_config.modal_environment,
            cache_size=runtime_config.modal_encoder_cache_size,
        ),
    )
    return visual, vietnamese


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Modal encoder smoke vectors for online.validate_contract"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if not output.parent.is_dir():
        parser.error("output parent directory does not exist")

    visual, vietnamese = build_modal_encoders()
    payload = build_smoke_payload(visual, vietnamese)
    output.write_text(json.dumps(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "dimensions": {name: len(vector) for name, vector in payload.items()},
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
