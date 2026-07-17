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

Concurrency contract: search ports are synchronous. SQLite serializes each
connection's calls with a re-entrant lock and uses one read-only connection per
adapter instance. Milvus and Elasticsearch allow concurrent reads through a
long-lived adapter instance; `close()` is rejected while a read is active.
Callers must stop scheduling work, drain their controlled executor, then close
the lifecycle. SDK thread-safety and alias behavior remain
`NEED_RUNTIME_VERIFICATION` until tested with the deployed SDK/services.
