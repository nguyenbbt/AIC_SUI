"""Deterministic M-GPUX-compatible Qwen VLM deployment for Online VQA.

The stock ``m-gpux serve deploy`` command is an interactive wizard. This
repository-owned app keeps the same M-GPUX app and cache volume identities,
while allowing the one-click Online runner to deploy and stop a pinned
configuration non-interactively through the Modal CLI.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "m-gpux-llm-api"
FUNCTION_NAME = "serve"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
VLLM_VERSION = "0.25.1"
API_KEY_ENV = "AIC_MGPUX_QWEN_API_KEY"
UPSTREAM_ENV = "AIC_MGPUX_QWEN_UPSTREAM"
INTERNAL_VLLM_PORT = "8001"

app = modal.App("m-gpux-llm-api")

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .pip_install("vllm==0.25.1")
    .add_local_file(
        str(Path(__file__).with_name("mgpux_qwen_proxy.py")),
        remote_path="/root/mgpux_qwen_proxy.py",
        copy=True,
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            UPSTREAM_ENV: f"http://127.0.0.1:{INTERNAL_VLLM_PORT}",
        }
    )
)

hf_cache = modal.Volume.from_name("m-gpux-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("m-gpux-vllm-cache", create_if_missing=True)

_local_api_key = os.getenv(API_KEY_ENV, "").strip()
_deployment_secrets = (
    [modal.Secret.from_dict({API_KEY_ENV: _local_api_key})]
    if _local_api_key
    else []
)


def build_vllm_command() -> list[str]:
    """Return the pinned internal vLLM command with no embedded secrets."""

    return [
        "vllm",
        "serve",
        MODEL_ID,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_ID,
        "--host",
        "0.0.0.0",
        "--port",
        INTERNAL_VLLM_PORT,
        "--seed",
        "1024",
        "--max-model-len",
        "16384",
        "--gpu-memory-utilization",
        "0.90",
        "--enable-chunked-prefill",
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "8192",
        "--limit-mm-per-prompt",
        '{"image":12}',
    ]


@app.function(
    image=vllm_image,
    gpu="L4",
    timeout=24 * 60 * 60,
    scaledown_window=5 * 60,
    min_containers=0,
    max_containers=1,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    secrets=_deployment_secrets,
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def serve() -> None:
    """Start the private proxy and the internal multimodal vLLM server."""

    if not os.getenv(API_KEY_ENV, "").strip():
        raise RuntimeError(f"{API_KEY_ENV} must be configured before deployment")
    print(
        f"[AIC Nova/M-GPUX] Starting {MODEL_ID}@{MODEL_REVISION} "
        f"with vLLM {VLLM_VERSION}"
    )
    subprocess.Popen(
        [
            "uvicorn",
            "mgpux_qwen_proxy:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        cwd="/root",
    )
    subprocess.Popen(build_vllm_command())


def resolve_deployed_base_url() -> str:
    """Resolve the deployed Modal web URL and normalize it for OpenAI clients."""

    function = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    web_url = function.get_web_url()
    if not web_url:
        raise RuntimeError(f"{APP_NAME}.{FUNCTION_NAME} has no deployed web URL")
    return f"{web_url.rstrip('/')}/v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--get-url",
        action="store_true",
        help="Print the deployed OpenAI-compatible base URL.",
    )
    arguments = parser.parse_args()
    if arguments.get_url:
        print(resolve_deployed_base_url())
        return 0
    parser.error("use 'modal deploy scripts/mgpux_qwen_vlm.py' or pass --get-url")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
