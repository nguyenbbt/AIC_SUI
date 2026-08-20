# 25 - ONLINE UI AND MODEL RUNTIME DECISIONS

## 1. Status and precedence

**Status:** CONFIRMED_DESIGN (updated 2026-08-11: q1-only)

This document records the user's 2026-08-11 decisions for the competition UI,
query rewriting and evidence-grounded VQA runtime. It supersedes only the
provider/model-specific parts of DD-031 and OQ-018. The SDK-neutral
`QueryRewritePort`, `VLMPort`, VQA evidence contract and DD-030 evidence budget
remain unchanged.

The q0/q1-only rule in section 3 supersedes every earlier planning-document
reference to q2. Older wave/task documents remain historical records and must
not be used as the active rewrite contract.

## 2. Object-constraint UI

The UI obtains the canonical vocabulary from the active SQLite `objects`
table, grouped by exact stored label. The response is tied to the active
`dataset_fingerprint`. COCO80 is a degraded fallback only when the catalog
cannot be read.

The baseline control contains:

- canonical-label autocomplete;
- minimum count presets `1+`, `2+`, `3+`;
- `soft` (`Prefer`) or `hard` (`Required`) behavior.

The UI sends the exact canonical label. It may display a Vietnamese alias but
must not submit that alias. Baseline requests use `count_operator=gte`,
`min_confidence=0.5` and `position=null`. Bounding-box drawing, region
selection and confidence tuning are deliberately omitted from the baseline UI.

## 3. Query rewrite runtime

Primary provider/model:

```text
provider_id: openai
model_id: gpt-5.4-mini-2026-03-17
API: Responses API
reasoning_effort: none
prompt_version: aic-query-rewrite-v3-vi-q1-only
timeout: 5 seconds
maximum paraphrases: 1
```

KIS always preserves the original query as `q0`. The provider creates at most
one rewrite: `q1`, a normalized Vietnamese visual description. There is no
English `q2`; the UI, API contract, query builder and ranking configuration all
enforce the same q0/q1-only decision. Prefix-only and near-duplicate variants
are rejected before retrieval.
Failure, timeout or invalid output degrades to `q0` without failing search.
VQA rewriting produces a visual-evidence retrieval description and must never
answer the question.

KIS exposes rewriting as an optional user action, off by default. VQA enables
the retrieval-oriented rewrite automatically when the provider is configured.
No web-search or external-image-search tool is enabled in the baseline.

Reference:
<https://developers.openai.com/api/docs/models/gpt-5.4-mini>

## 4. VQA VLM runtime

Primary model/runtime:

```text
model_id: Qwen/Qwen3.5-4B
revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
provider_id: local-vllm
prompt_version: aic-vqa-evidence-v1
thinking: disabled
image_long_edge: 768 pixels
request_timeout: 15 seconds per attempt
maximum_output_tokens: 256
```

The model is served as a separate OpenAI-compatible vLLM service. The Online
application sends only bounded DD-030 evidence, never the full dataset. Images
are loaded only from validated dataset-relative references, resized in memory,
and sent with their stable evidence IDs. OCR, ASR and summary evidence retain
their source IDs.

The adapter requests the existing structured `VLMResponse` schema. Free-form
fallback parsing is forbidden. The orchestrator may retry one malformed or
explicitly retryable response while the total deadline remains. Thinking is
disabled through `chat_template_kwargs.enable_thinking=false`.

References:

- <https://huggingface.co/Qwen/Qwen3.5-4B>
- <https://huggingface.co/Qwen/Qwen3.5-4B/commit/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a>

## 5. Competition UI

The React/Vite UI has three primary modes: KIS, TRAKE and VQA.

KIS provides shared Textual-KIS and Video-KIS text search, optional AI
rewriting, simplified object constraints, advanced seven-branch controls,
result thumbnails, video preview, provenance and diagnostics.

TRAKE provides ordered event editing, event reordering, per-video aligned
sequences, timeline preview and selection.

VQA provides question/answer-type input, automatic evidence retrieval,
evidence gallery, grounded answer status/confidence/evidence IDs, and an
editable submission answer that remains separate from the original model
response. The operator must explicitly choose one VLM-cited image as the VQA
submission frame; the UI does not silently submit the first evidence anymore.

The shared selection tray deduplicates KIS/VQA by
`(video_id, source_frame_idx)`, preserves TRAKE event order, limits KIS/TRAKE
answers to 100, and exports the current logical submission rows. Organizer
transport remains an OPEN_QUESTION until an official endpoint or file schema
is published.

## 6. Runtime verification gate

Model selection does not prove runtime readiness. The system owner must still:

1. pin the exact model revision and serving-image version;
2. validate GPU memory with the maximum evidence budget;
3. benchmark Vietnamese q1 rewrites and Vietnamese/English VQA inputs;
4. verify structured-output conformance and deadline behavior;
5. run a real-data vertical slice before competition readiness is claimed.

## 7. Local fixes versus integrator runtime work

The codebase now handles q0/q1-only contracts, stale rewrite invalidation,
strict demo request validation, canonical object-label validation, explicit VQA
frame selection, periodic readiness polling, stable public TRAKE/VQA response
versions, and evidence-ID/image ordering for Qwen requests.

The system owner who runs Docker must still configure and verify the actual
OpenAI key, Qwen/vLLM container and pinned revision, network routing, GPU memory,
real Offline indexes/data and production health checks. The preliminary CSV/ZIP
transport is implemented, but one real browser download must still be opened
and inspected before the first scored upload.
