"""Development-only API for reviewing the operator UI without Offline data.

Run with ``python -m uvicorn retrieval_api.demo:app --port 8000``. This module
is intentionally separate from ``retrieval_api.main`` and never opens Milvus,
Elasticsearch, SQLite, model providers, or dataset files.
"""

from __future__ import annotations

import html
from hashlib import sha256
from typing import Any

from fastapi import FastAPI, Query, Response
from retrieval_api.advanced_models import InternalTRAKERequest, InternalVQARequest
from retrieval_api.search_engine import RewriteRequest, SearchRequest
from retrieval_api.submission import SubmissionPackageRequest, build_submission_zip


app = FastAPI(title="AIC Nova UI Demo", version="demo-v1")


@app.post("/submission/package", response_class=Response)
def submission_package(request: SubmissionPackageRequest) -> Response:
    return Response(
        content=build_submission_zip(request),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{request.download_filename}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/health/live")
def live() -> dict[str, Any]:
    return {"status": "healthy", "checks": {"api": "demo"}}


@app.get("/health/ready")
def ready() -> dict[str, Any]:
    return {
        "status": "ready",
        "checks": {
            "kis.enabled": "true",
            "kis.readiness": "demo",
            "trake.enabled": "true",
            "trake.readiness": "demo",
            "vqa.enabled": "true",
            "vqa.readiness": "demo",
            "rewrite.enabled": "true",
            "ui_resources.enabled": "true",
            "demo": "true",
        },
    }


@app.get("/catalog/object-labels")
def catalog() -> dict[str, Any]:
    counts = {"person": 1240, "car": 830, "motorcycle": 412, "dog": 182, "backpack": 96, "cell phone": 75, "bicycle": 211, "bus": 137}
    return {
        "dataset_id": "ui-demo",
        "dataset_fingerprint": "sha256:" + "d" * 64,
        "source": "demo_fixture",
        "labels": [{"label": label, "detection_count": count} for label, count in counts.items()],
    }


@app.post("/query/rewrite")
def rewrite(request: RewriteRequest) -> dict[str, Any]:
    query = request.query
    q1 = _demo_rewrite_variant(query)
    degraded = q1 is None
    return {
        "request_id": request.request_id,
        "original_text": query,
        "primary_text": query,
        "paraphrases": [] if degraded else [q1],
        "status": "degraded" if degraded else "success",
        "warnings": ["DEMO_REWRITE_UNSUPPORTED"] if degraded else [],
        "latency_ms": 18.4,
        "model_id": "demo-rule-rewriter",
    }


@app.post("/search")
def search(request: SearchRequest) -> dict[str, Any]:
    query_id = request.query_id or "demo-kis"
    candidates = tuple(_candidate(index) for index in range(1, 13))
    return {
        "query_id": query_id,
        "candidates": candidates,
        "diagnostics": {
            "query_id": query_id,
            "total_latency_ms": 84.2,
            "stage_latencies_ms": {"retrieval": 55.1, "ranking": 29.1},
            "branches": {
                branch: {"status": "success", "latency_ms": 12 + i, "requested_top_k": 50, "raw_result_count": 50, "output_candidate_count": 12, "warnings": []}
                for i, branch in enumerate(("visual_dense", "ocr_dense", "ocr_bm25", "asr_dense", "asr_bm25", "summary_dense", "summary_bm25"))
            },
            "normalization_method": "rrf",
            "fusion_method": "demo_weighted_fusion",
            "fusion_weights": {"visual_dense": 1.0},
            "warnings": ["DEMO_DATA_ONLY"],
        },
    }


@app.post("/trake")
def trake(request: InternalTRAKERequest) -> dict[str, Any]:
    texts = request.event_texts
    results = []
    for video_no in range(1, 4):
        video_id = f"L21_V{video_no:03d}"
        sequence = []
        for index, _ in enumerate(texts, start=1):
            sequence.append(_frame_match(video_id, index * 2, f"event-{index}"))
        results.append({"video_id": video_id, "score": 0.92 - video_no * 0.07, "event_ids": [x["event_id"] for x in sequence], "sequence": sequence})
    return {
        "schema_version": "demo-v1",
        "query_id": request.query_id,
        "results": results,
        "diagnostics": {"policy_version": "dante-index-gap-v1", "lambda_penalty": 0.001, "event_count": len(texts), "video_count": 3, "frame_count": len(texts) * 3, "similarity_latency_ms": 21.5, "dp_latency_ms": 3.2, "invalid_sequence_count": 0, "warnings": ["DEMO_DATA_ONLY"]},
    }


@app.post("/vqa")
def vqa(request: InternalVQARequest) -> dict[str, Any]:
    question_id = request.question_id
    answer_type = request.answer_type.value
    evidence = []
    for index in range(1, 4):
        candidate = _candidate(index)
        evidence.append({"evidence_id": f"image-{index}", "evidence_type": "image", "video_id": candidate["video_id"], "frame_id": candidate["frame_id"], "shot_id": candidate["shot_id"], "timestamp_sec": candidate["timestamp_sec"], "source_frame_idx": candidate["source_frame_idx"], "image_reference": candidate["image_rel_path"]})
    response = {"status": "answered", "answer": "Có một người mặc áo đỏ.", "answer_type": answer_type, "confidence": "high", "evidence_ids": ["image-1", "image-2"]}
    return {
        "schema_version": "demo-v1",
        "question_id": question_id,
        "result": {"question_id": question_id, "response": response, "evidence": evidence, "diagnostics": {"retrieved_frame_count": 12, "selected_image_count": 3, "selected_text_evidence_count": 0, "dropped_evidence_count": 9, "missing_evidence_count": 0, "vlm_latency_ms": 242.1, "vlm_retry_count": 0, "warnings": ["DEMO_DATA_ONLY"]}},
    }


@app.get("/media/keyframes/{frame_id}")
def keyframe(frame_id: str) -> Response:
    digest = sha256(frame_id.encode("utf-8")).hexdigest()
    first, second = f"#{digest[:6]}", f"#{digest[6:12]}"
    safe_id = html.escape(frame_id)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop stop-color="{first}"/><stop offset="1" stop-color="{second}"/></linearGradient></defs><rect width="960" height="540" fill="url(#g)"/><circle cx="480" cy="235" r="105" fill="#ffffff22"/><path d="M420 330h120l45 125H375z" fill="#ffffff33"/><text x="48" y="465" fill="white" font-family="system-ui" font-size="34" font-weight="700">AIC NOVA · DEMO KEYFRAME</text><text x="48" y="510" fill="#ffffffbb" font-family="monospace" font-size="22">{safe_id}</text></svg>'''
    return Response(content=svg, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/media/keyframes/{frame_id}/neighbors")
def neighbors(frame_id: str, before: int = Query(2, ge=0, le=20), after: int = Query(2, ge=0, le=20)) -> dict[str, Any]:
    video_id, shot, position = _parse_frame_id(frame_id)
    values = []
    for current in range(max(1, position - before), position + after + 1):
        current_id = f"{video_id}_{shot:05d}_{current:03d}"
        values.append({"frame_id": current_id, "video_id": video_id, "source_frame_idx": current * 90, "timestamp_sec": current * 3.0, "image_url": f"/media/keyframes/{current_id}"})
    return {"center_frame_id": frame_id, "frames": values}


@app.get("/media/videos/{video_id}")
def video(video_id: str) -> Response:
    del video_id
    return Response(status_code=204, headers={"X-AIC-Demo": "video-unavailable"})


def _candidate(index: int) -> dict[str, Any]:
    video_id = f"L21_V{((index - 1) % 4) + 1:03d}"
    frame_id = f"{video_id}_{(index - 1) // 4:05d}_{index:03d}"
    return {
        "frame_id": frame_id,
        "video_id": video_id,
        "shot_id": (index - 1) // 4,
        "timestamp_sec": index * 3.0,
        "source_frame_idx": index * 90,
        "image_rel_path": f"keyframes/{video_id}/{index:03d}.jpg",
        "final_score": round(0.97 - index * 0.045, 4),
        "branch_scores": {"visual_dense": round(0.98 - index * 0.04, 4), "ocr_bm25": round(0.64 - index * 0.02, 4)},
        "evidence": [],
        "near_frames": [],
        "objects": [{"label": "person", "confidence": 0.91, "x_min": 0.25, "y_min": 0.12, "x_max": 0.68, "y_max": 0.95, "model_source": "demo"}],
        "diagnostics": {"summary_boost": 0.03, "object_boost": 0.05, "object_constraints_satisfied": 1},
    }


def _frame_match(video_id: str, position: int, event_id: str) -> dict[str, Any]:
    frame_id = f"{video_id}_00000_{position:03d}"
    return {"event_id": event_id, "frame_id": frame_id, "video_id": video_id, "shot_id": 0, "local_index": position, "timestamp_sec": position * 3.0, "source_frame_idx": position * 90, "image_rel_path": f"keyframes/{video_id}/{position:03d}.jpg", "similarity_score": 0.9 - position * 0.02}


def _parse_frame_id(frame_id: str) -> tuple[str, int, int]:
    video_id, shot, position = frame_id.rsplit("_", 2)
    return video_id, int(shot), int(position)


def _demo_rewrite_variant(query: str) -> str | None:
    """Return one transparent Vietnamese q1; production uses the OpenAI adapter."""

    normalized = " ".join(query.strip().split())
    vietnamese = normalized
    vietnamese_replacements = (
        ("mặc áo đỏ", "mặc một chiếc áo màu đỏ"),
        ("đứng cạnh ô tô", "đang đứng bên cạnh một chiếc xe ô tô"),
        ("đứng cạnh xe ô tô", "đang đứng bên cạnh một chiếc xe ô tô"),
        ("đi xe đạp", "đang điều khiển một chiếc xe đạp"),
        ("cầm điện thoại", "đang cầm một chiếc điện thoại"),
        ("đang chạy", "đang thực hiện hành động chạy bộ"),
    )
    for source, target in vietnamese_replacements:
        vietnamese = vietnamese.replace(source, target).replace(source.capitalize(), target)
    return None if vietnamese == normalized else " ".join(vietnamese.split())


__all__ = ["app"]
