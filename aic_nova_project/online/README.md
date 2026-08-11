# Online Data & Infrastructure setup

The active Offline boundary is `self-indexed-v2`: the team extracts keyframes
from raw videos and indexes OpenCLIP `ViT-B-32::openai` visual vectors. See
`docs/22-OFFLINE-TO-ONLINE-DATA-CONTRACT-SELF-INDEXED-V2.md` before connecting
real resources.

The Online adapter boundary supports Python 3.11+; the database SDKs must be
verified in the target deployment image before runtime integration is claimed.
The contract and adapter unit tests do not require Milvus, Elasticsearch,
encoder checkpoints, GPU packages, or running services.

Run these commands from the application root: the directory that directly
contains `online/`, `query_understanding/` and `tests/`. In the current Git
layout, first enter the nested project directory:

```powershell
cd aic_nova_project
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r online\requirements-runtime.txt
python -m pip install -r online\requirements-test.txt
```

The supported interpreter is Python 3.11 or newer. Using `python -m venv`
avoids depending on the optional Windows `py` launcher.

Install the model stack only on machines that will run the real B2 encoders:

```powershell
python -m pip install -r online\requirements-encoders.txt
```

Run the SDK-free tests:

```powershell
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online -q
```

The optional runtime profile requires configured, non-production resources:

```powershell
$env:AIC_ONLINE_MILVUS_URI = "http://localhost:19530"
$env:AIC_ONLINE_ES_URI = "http://localhost:9200"
$env:AIC_ONLINE_SQLITE_PATH = "data\metadata.db"
$env:AIC_ONLINE_SQLITE_VIDEOS_TABLE = "videos"
$env:AIC_ONLINE_DATASET_MANIFEST_PATH = "data\processed\dataset-manifest.json"
$env:AIC_ONLINE_DATA_ROOT = "data\processed"
$env:AIC_ONLINE_DATASET_EXPECTED_FINGERPRINT = "sha256:<fingerprint-from-offline>"
$env:AIC_ONLINE_DATASET_AUDIT_BATCH_SIZE = "500"
python -m online.validate_contract
python -m online.validate_contract --fail-on-partial
```

The validator is read-only. It performs bounded full scans of SQLite, Milvus
and Elasticsearch, verifies every indexed vector/key/path, detects duplicate
domain keys with a temporary disk-backed exact set, compares complete key-set
digests, and reconciles all ten actual counts with the READY manifest. Sampled
checks remain diagnostics only and cannot produce a full `PASS`. It reports
`FAIL` when the required visual/metadata/manifest contract or visual encoder
smoke check is missing, `PARTIAL` when optional resources/checks are
unavailable, and `PASS` only when `audit_scope=FULL`. Encoder smoke factories
can be injected through the Python
`OfflineContractValidator(..., encoder_smoke_vectors={...})` API; they are not
loaded implicitly by the CLI.

Runtime status remains `NEED_RUNTIME_VERIFICATION` until Milvus,
Elasticsearch, SQLite, encoder dimensions/norms, complete canonical JOINs, and a
real visual-to-frame vertical slice have all been checked.

Unit tests intentionally do not prove SDK/service/model compatibility. Runtime
validation must be performed against a disposable or read-only environment;
production indexes are never used as test fixtures.

## Self-indexed production adapters

The production-side adapters that do not depend on a selected LLM/VLM provider
are implemented:

- `DatasetManifestGate` accepts only a strict READY `self-indexed-v2` manifest,
  pins its identity for the process lifetime, and detects a dataset switch.
- `MilvusSQLiteVisualCorpusAdapter` reads full visual vectors, exact-JOINs them
  to SQLite metadata and emits deterministic per-video timelines for DANTE.
- `FilesystemImageResolver` checks that keyframe files remain below the
  configured data root and exposes only relative references.
- `ElasticsearchEvidenceHydrator` reads OCR by `frame_id`, ASR by closed-window
  overlap and summary by `video_id`.

Data-backed TRAKE can be enabled with:

```powershell
$env:AIC_ONLINE_TRAKE_ENABLED = "true"
```

VQA data adapters are wired by `build_online_runtime`. The approved default
adapter targets a local OpenAI-compatible `Qwen/Qwen3.5-4B` service pinned to
revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Enable explicit
environment composition with `AIC_ONLINE_QWEN_VLM_AUTO_CONFIGURE=true`, or
inject another validated `VLMPort`. Startup still fails closed when VQA is
enabled but neither choice is present.

Optional KIS/VQA rewrite uses OpenAI Responses structured output with
`gpt-5.4-mini-2026-03-17`. Set `AIC_ONLINE_QUERY_REWRITE_ENABLED=true` and
`AIC_ONLINE_OPENAI_API_KEY`; provider failure safely retains the original q0.
The key is read from environment only and is never returned by diagnostics.

Production mode requires the manifest gate and a pinned
`AIC_ONLINE_DATASET_EXPECTED_FINGERPRINT`. Conversion of relative keyframe
references into provider-specific image uploads belongs to the selected VLM
adapter.

## Person-C Ranking Policy Status

The current KIS ranking policy is an experimental benchmark baseline, not an
approved production ranking policy. Runtime deployments must review and pin
these values before claiming production readiness:

- Branch-local score normalizer: RRF with `k=60`.
- Query-variant aggregation: `weighted_sum_query_variant_v1` with q0/q1/q2
  weights.
- Fusion: `experimental_weighted_sum_normalized_v1` with equal default branch
  weights.
- Summary propagation: `summary_video_score_cap_v1`, `weight=0.1`,
  `max_boost=0.2`; it uses the same q0/q1/q2 weights and normalization RRF
  fallback configured for the frame branches.
- ASR interval mapping: `timestamp_inclusive_distributed_v1`,
  `max_frames_per_interval=50`, interval-level RRF `k=60` when upstream ASR
  results are not already normalized.
- Ranking executor: bounded thread pool from `AIC_ONLINE_RANKING_MAX_WORKERS`
  with default `2`.

The runtime composition reads the same policy from environment variables:

```powershell
$env:AIC_ONLINE_RANKING_POLICY_NAME = "person_c_experimental_baseline_v1"
$env:AIC_ONLINE_RANKING_POLICY_STATUS = "experimental"
$env:AIC_ONLINE_RANKING_NORMALIZATION_RRF_K = "60"
$env:AIC_ONLINE_RANKING_QUERY_Q0_WEIGHT = "1.0"
$env:AIC_ONLINE_RANKING_QUERY_Q1_WEIGHT = "1.0"
$env:AIC_ONLINE_RANKING_QUERY_Q2_WEIGHT = "1.0"
$env:AIC_ONLINE_RANKING_FUSION_DEFAULT_WEIGHT = "1.0"
$env:AIC_ONLINE_RANKING_SUMMARY_WEIGHT = "0.1"
$env:AIC_ONLINE_RANKING_SUMMARY_MAX_BOOST = "0.2"
$env:AIC_ONLINE_RANKING_ASR_MAX_FRAMES_PER_INTERVAL = "50"
$env:AIC_ONLINE_RANKING_ASR_INTERVAL_RRF_K = "60"
$env:AIC_ONLINE_RANKING_OBJECT_SOFT_BOOST = "0.05"
$env:AIC_ONLINE_RANKING_OBJECT_MAX_TOTAL_BOOST = "0.2"
```

If `AIC_ONLINE_DEPLOYMENT_MODE=production`, startup rejects an experimental
policy. Mark the policy `approved` only after the team closes the relevant open
questions.

Open questions for benchmark tuning use the canonical IDs from
`docs/08-OPEN-QUESTIONS.md`: OQ-004 query aggregation, OQ-005 ASR mapping,
OQ-006 normalization, OQ-007 fusion/weights, OQ-008 summary boost, OQ-009
object defaults, OQ-010/OQ-011 object position contracts, OQ-021 lifecycle, and
OQ-022 missing metadata.

The A/B/C contracts are now merged in the shared domain models. The public API
is still internal/unstable until OQ-002 is closed, so external clients must not
assume that the response schema is final.

Current evidence and diagnostics behavior:

- ASR evidence preserves backend/resource, interval ID, interval start/end and
  the interval-level normalized score as separate fields. The mapped-frame
  contribution remains in `normalized_score`; transcript text is intentionally
  not copied into every final frame.
- Summary evidence stores the original normalized/RRF score in
  `source_normalized_score` and the applied, weighted, capped contribution in
  `normalized_score`.
- `QueryDiagnostics.missing_metadata_count` is propagated from B. Branch
  latency uses the maximum variant latency as the bounded wall-clock estimate;
  per-variant outcomes and ASR truncation counters remain bounded warning tags
  because the current diagnostics schema is branch-level.
- The B query builder remains policy-neutral. C validates the configured
  `q0_required` visual policy before retrieval begins.

Run the API locally after installing runtime dependencies:

```powershell
python -m uvicorn retrieval_api.main:app --host 127.0.0.1 --port 8000
```

Minimal t-KIS request:

```json
{
  "query": "person in red near a bicycle",
  "mode": "kis_text",
  "paraphrases": ["human wearing red beside a bike"],
  "enabled_branches": ["visual_dense"],
  "include_diagnostics": true
}
```

Minimal v-KIS request:

```json
{
  "query": "find the scene matching this video clue",
  "mode": "kis_video",
  "enabled_branches": ["visual_dense"],
  "include_diagnostics": true
}
```

Operator endpoints are `/health/live`, `/health/ready`, `/search`, `/trake`,
`/vqa`, `/query/rewrite`, `/catalog/object-labels`, and read-only `/media/*`.
The old `/internal/unstable/trake` and `/internal/unstable/vqa` aliases remain
for backward compatibility. Runtime startup probes the required
visual encoder and optional Vietnamese encoder for batch shape, dimension,
finite values and positive vector norm; these checks do not replace the full
Offline contract validator or a real database vertical slice.

Concurrency contract: search ports are synchronous. SQLite serializes each
connection's calls with a re-entrant lock and uses one read-only connection per
adapter instance. Milvus and Elasticsearch allow concurrent reads through a
long-lived adapter instance; `close()` is rejected while a read is active.
Callers must stop scheduling work, drain their controlled retrieval/ranking
executors, then close the lifecycle. SDK thread-safety and alias behavior remain
`NEED_RUNTIME_VERIFICATION` until tested with the deployed SDK/services.
