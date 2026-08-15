"""Private bearer-auth proxy for the Modal-hosted Qwen vLLM process."""

from __future__ import annotations

import hmac
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response


API_KEY_ENV = "AIC_MGPUX_QWEN_API_KEY"
UPSTREAM_ENV = "AIC_MGPUX_QWEN_UPSTREAM"
DEFAULT_UPSTREAM = "http://127.0.0.1:8001"
_FORWARDED_RESPONSE_HEADERS = {"content-type", "x-request-id"}

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


def is_authorized(authorization: str | None, expected_key: str) -> bool:
    """Return whether an Authorization header contains the exact bearer key."""

    if not expected_key or not authorization:
        return False
    scheme, separator, token = authorization.partition(" ")
    return (
        bool(separator)
        and scheme.lower() == "bearer"
        and hmac.compare_digest(token, expected_key)
    )


@app.get("/health")
async def health() -> JSONResponse:
    """Expose proxy liveness without revealing model or secret information."""

    return JSONResponse({"status": "alive"})


@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_vllm(path: str, request: Request) -> Response:
    """Authenticate and forward an OpenAI-compatible request to local vLLM."""

    expected_key = os.getenv(API_KEY_ENV, "").strip()
    if not is_authorized(request.headers.get("authorization"), expected_key):
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    upstream = os.getenv(UPSTREAM_ENV, DEFAULT_UPSTREAM).rstrip("/")
    headers = {
        name: value
        for name, value in request.headers.items()
        if name.lower() in {"accept", "content-type"}
    }
    try:
        async with httpx.AsyncClient(timeout=900.0) as client:
            upstream_response = await client.request(
                request.method,
                f"{upstream}/v1/{path}",
                params=request.query_params,
                headers=headers,
                content=await request.body(),
            )
    except httpx.RequestError:
        return JSONResponse(
            {"detail": "Qwen VLM is starting or unavailable"},
            status_code=503,
        )

    response_headers = {
        name: value
        for name, value in upstream_response.headers.items()
        if name.lower() in _FORWARDED_RESPONSE_HEADERS
    }
    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )
