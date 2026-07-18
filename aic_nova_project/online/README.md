# Online Data & Infrastructure setup

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
python -m online.validate_contract
python -m online.validate_contract --fail-on-partial
```

The validator is read-only. It reports `FAIL` when the required visual/metadata
contract or visual encoder smoke check is missing, `PARTIAL` when optional
resources/checks are unavailable, and `PASS` only after every executed required
check succeeds. Encoder smoke factories can be injected through the Python
`OfflineContractValidator(..., encoder_smoke_vectors={...})` API; they are not
loaded implicitly by the CLI.

Runtime status remains `NEED_RUNTIME_VERIFICATION` until Milvus,
Elasticsearch, SQLite, encoder dimensions/norms, canonical JOIN samples, and a
real visual-to-frame vertical slice have all been checked.

Unit tests intentionally do not prove SDK/service/model compatibility. Runtime
validation must be performed against a disposable or read-only environment;
production indexes are never used as test fixtures.

## Person-C Ranking Policy Status

The current KIS ranking policy is an experimental benchmark baseline, not an
approved production ranking policy. Runtime deployments must review and pin
these values before claiming production readiness:

- Branch-local score normalizer: RRF with `k=60`.
- Query-variant aggregation: RRF contribution aggregation with `k=60`.
- Fusion: `experimental_weighted_sum_normalized_v1` with equal default branch
  weights.
- Summary propagation: `summary_video_score_cap_v1`, `weight=0.1`,
  `max_boost=0.2`.
- ASR interval mapping: `timestamp_inclusive_distributed_v1`,
  `max_frames_per_interval=50`.
- Ranking executor: bounded thread pool from `AIC_ONLINE_RANKING_MAX_WORKERS`
  with default `2`.

Open questions for benchmark tuning: OQ-C-01 choose RRF vs min-max per branch;
OQ-C-02 approve branch weights per modality; OQ-C-03 approve summary/object
boost limits; OQ-C-04 define ASR interval provenance fields in the shared
domain; OQ-C-05 define summary evidence IDs, cap and weight provenance in the
shared domain; OQ-C-06 decide whether visual-dense can ever be disabled.

The API response schema and diagnostics fields are still unstable while A/B/C
contracts are being merged. The current implementation preserves only fields
already present in the shared domain models.

Concurrency contract: search ports are synchronous. SQLite serializes each
connection's calls with a re-entrant lock and uses one read-only connection per
adapter instance. Milvus and Elasticsearch allow concurrent reads through a
long-lived adapter instance; `close()` is rejected while a read is active.
Callers must stop scheduling work, drain their controlled retrieval/ranking
executors, then close the lifecycle. SDK thread-safety and alias behavior remain
`NEED_RUNTIME_VERIFICATION` until tested with the deployed SDK/services.
