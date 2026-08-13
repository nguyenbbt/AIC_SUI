"""Cost-bounded private Modal GPU service for Online query embeddings."""

from __future__ import annotations

import math
from typing import Any

import modal


MODAL_ENCODER_SCHEMA_VERSION = "aic-online-encoder-v1"
OPENCLIP_MODEL_ID = "ViT-B-32::openai"
VIETNAMESE_MODEL_NAME = "dangvantuan/vietnamese-embedding"
VIETNAMESE_MODEL_REVISION = "4ab46e46ba5902328ba0742e489e75f787932f2b"
MAX_BATCH_SIZE = 64
MAX_TEXT_LENGTH = 4096
CACHE_ROOT = "/model-cache"


app = modal.App("aic-nova-online-encoders")
model_cache = modal.Volume.from_name(
    "aic-nova-online-model-cache",
    create_if_missing=True,
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.13.0",
        "torchvision==0.28.0",
        "open_clip_torch==3.3.0",
        "sentence-transformers==5.6.0",
        "transformers==5.13.1",
        "tokenizers==0.22.2",
    )
    .env(
        {
            "HF_HOME": f"{CACHE_ROOT}/huggingface",
            "TORCH_HOME": f"{CACHE_ROOT}/torch",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

_visual_runtime: tuple[Any, Any, Any] | None = None
_vietnamese_runtime: Any | None = None


def _validated_texts(texts: object) -> tuple[str, ...]:
    if not isinstance(texts, list) or not 1 <= len(texts) <= MAX_BATCH_SIZE:
        raise ValueError(f"texts must contain 1..{MAX_BATCH_SIZE} strings")
    output: list[str] = []
    for value in texts:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("each encoder input must be a non-empty string")
        cleaned = value.strip()
        if len(cleaned) > MAX_TEXT_LENGTH:
            raise ValueError(f"encoder input exceeds {MAX_TEXT_LENGTH} characters")
        output.append(cleaned)
    return tuple(output)


def _load_visual() -> tuple[Any, Any, Any]:
    global _visual_runtime
    if _visual_runtime is None:
        import open_clip
        import torch

        model_name, pretrained = OPENCLIP_MODEL_ID.split("::", 1)
        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        model = model.to("cuda", dtype=torch.float16)
        model.eval()
        _visual_runtime = (model, open_clip.get_tokenizer(model_name), torch)
    return _visual_runtime


def _load_vietnamese() -> Any:
    global _vietnamese_runtime
    if _vietnamese_runtime is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            VIETNAMESE_MODEL_NAME,
            revision=VIETNAMESE_MODEL_REVISION,
            device="cuda",
            cache_folder=f"{CACHE_ROOT}/huggingface",
        )
        model.max_seq_length = 256
        _vietnamese_runtime = model
    return _vietnamese_runtime


def _encode_visual(texts: tuple[str, ...]) -> list[list[float]]:
    model, tokenizer, torch = _load_visual()
    tokens = tokenizer(list(texts)).to("cuda")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        features = model.encode_text(tokens).float()
    features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return features.cpu().tolist()


def _encode_vietnamese(texts: tuple[str, ...]) -> list[list[float]]:
    model = _load_vietnamese()
    embeddings = model.encode(
        list(texts),
        batch_size=MAX_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16_384,
    volumes={CACHE_ROOT: model_cache},
    max_containers=1,
    scaledown_window=300,
    timeout=600,
    startup_timeout=900,
)
def encode(model_kind: str, texts: list[str]) -> dict[str, object]:
    """Encode one bounded text batch; callable only with Modal authentication."""

    cleaned = _validated_texts(texts)
    if model_kind == "visual":
        model_id = OPENCLIP_MODEL_ID
        model_revision = None
        vectors = _encode_visual(cleaned)
    elif model_kind == "vietnamese":
        model_id = VIETNAMESE_MODEL_NAME
        model_revision = VIETNAMESE_MODEL_REVISION
        vectors = _encode_vietnamese(cleaned)
    else:
        raise ValueError("model_kind must be visual or vietnamese")

    if len(vectors) != len(cleaned) or not vectors:
        raise RuntimeError("encoder returned an invalid row count")
    dimension = len(vectors[0])
    if dimension < 1 or any(
        len(vector) != dimension or not all(math.isfinite(float(value)) for value in vector)
        for vector in vectors
    ):
        raise RuntimeError("encoder returned invalid vectors")
    return {
        "schema_version": MODAL_ENCODER_SCHEMA_VERSION,
        "model_kind": model_kind,
        "model_id": model_id,
        "model_revision": model_revision,
        "dimension": dimension,
        "vectors": vectors,
    }


@app.local_entrypoint()
def smoke(model_kind: str = "visual") -> None:
    """Run a paid one-vector smoke check with ``modal run``."""

    result = encode.remote(model_kind=model_kind, texts=["readiness probe"])
    print(
        {
            "schema_version": result["schema_version"],
            "model_kind": result["model_kind"],
            "model_id": result["model_id"],
            "dimension": result["dimension"],
            "rows": len(result["vectors"]),
        }
    )
