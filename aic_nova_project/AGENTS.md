# AGENTS.md

## 1. Project identity and current scope

This repository contains the AIC Nova multimedia video retrieval system for the
HCM AI Challenge.

The system has two phases:

```text
OFFLINE PHASE
Raw videos
→ shot/keyframe extraction
→ visual, OCR, ASR, summary and object features
→ Milvus + Elasticsearch + SQLite indexing

ONLINE PHASE
User query
→ query construction
→ retrieval branches
→ candidate conversion/hydration
→ mapping, normalization and fusion
→ deduplication
→ mode-specific output
```

The repository currently contains both Offline source and the beginning of the
Online source under `online/`.

This file is shared by all three Online developers. The three roles from
`docs/11-ONLINE-TEAM-TASK-ASSIGNMENT.md` are:

- **Person A — Data & Infrastructure**.
- **Person B — Query & Retrieval**.
- **Person C — Ranking, Orchestration & API**.

At the start of each task, identify which role owns the requested component.
Work inside that boundary unless the user explicitly authorizes a cross-layer
change. A defect in another layer must be reported with evidence and handled in
a separate focused change rather than hidden by a downstream workaround.

Supported modes:

- Textual Known-Item Search (`KIS_TEXT`).
- Video Known-Item Search (`v-KIS`; current internal enum `KIS_VIDEO`).
- Temporal Retrieval and Alignment of Key Events (`TRAKE`).
- Visual Question Answering (`VQA`).

`v-KIS` is not image-to-image retrieval. The organizer displays a clip, the
contestant watches it and manually writes a text description, then the system
reuses the same text-to-keyframe pipeline as Textual KIS.

Older documents may still contain the legacy working label `KIS_VISUAL`.
New internal Online code must use the currently merged `QueryMode.KIS_VIDEO`
until the team explicitly changes the internal contract. The exact public API
mode names remain open under OQ-002.

---

## 2. Mandatory reading order

Before analyzing, planning or modifying source code, read these files in order:

1. `AGENTS.md`
2. `docs/00-READ-ME-FIRST.md`
3. `docs/01-SYSTEM-OVERVIEW.md`
4. `docs/02-OFFLINE-PIPELINE-ACTUAL.md`
5. `docs/03-DATABASE-SCHEMA-CURRENT.md`
6. `docs/04-OFFLINE-ONLINE-CONTRACT.md`
7. `docs/05-ONLINE-PIPELINE-TARGET.md`
8. `docs/06-DESIGN-DECISIONS.md`
9. `docs/07-OUT-OF-SCOPE.md`
10. `docs/08-OPEN-QUESTIONS.md`
11. `docs/09-IMPLEMENTATION-PLAN.md`
12. `docs/10-OFFLINE-CODE-ISSUES.md`
13. `docs/11-ONLINE-TEAM-TASK-ASSIGNMENT.md`

`docs/10-OFFLINE-CODE-ISSUES.md` is an audit snapshot. Re-verify an issue against
the checked-out code before claiming that it is still present.

`docs/11-ONLINE-TEAM-TASK-ASSIGNMENT.md` defines team boundaries and the approved
internal Online model/port conventions. It does not silently close algorithmic
questions listed in `docs/08-OPEN-QUESTIONS.md`.

Do not begin a coding milestone before reading the documents relevant to that
milestone and inspecting its source dependencies.

---

## 3. Source-of-truth precedence

When information differs, use this precedence:

1. The user's latest explicit instruction.
2. `docs/06-DESIGN-DECISIONS.md` for approved architecture/behavior.
3. `docs/04-OFFLINE-ONLINE-CONTRACT.md` for the phase boundary.
4. `docs/03-DATABASE-SCHEMA-CURRENT.md` for database resources and fields.
5. `docs/11-ONLINE-TEAM-TASK-ASSIGNMENT.md` for approved team boundaries and
   internal model/port conventions.
6. `docs/05-ONLINE-PIPELINE-TARGET.md` for target Online data flow.
7. Current source code for what is implemented now.
8. `docs/09-IMPLEMENTATION-PLAN.md` for milestone order.
9. Module-level README files.
10. `docs/10-OFFLINE-CODE-ISSUES.md`, historical discussions and old notes.

The explicit internal enum clarification in Section 1 of this file supersedes
legacy `KIS_VISUAL` wording in older documents; it does not finalize the public
API schema.

If source and intended design disagree:

- Do not silently choose or add a compatibility workaround.
- Report `CONTRACT_MISMATCH`.
- Identify the files, symbols, schema and affected milestone.
- Propose the smallest safe correction.
- Do not change architecture or data contracts without user approval.

---

## 4. Required evidence labels

Use these labels in technical analysis:

- `CONFIRMED_CODE`: verified directly in current source or execution.
- `CONFIRMED_DESIGN`: explicitly approved in design documents.
- `NEED_RUNTIME_VERIFICATION`: needs a real database, model, artifact or service.
- `OPEN_QUESTION`: not decided by the team yet.
- `OPTIONAL`: planned but not required for baseline.
- `OUT_OF_SCOPE`: outside the current milestone.
- `CONTRACT_MISMATCH`: code/schema/documentation do not agree.

Never present an assumption, mock-only result or unexecuted path as confirmed.

---

## 5. Authorization and safe working behavior

For analysis, review, diagnosis or status requests, remain read-only unless the
user explicitly requests an implementation or file change.

Read-only work may include:

- Reading source, tests, docs and configuration.
- Searching symbols and tracing data flow.
- Running non-destructive tests.
- Comparing adapters with Offline schemas.
- Proposing runtime checks.

Do not without explicit authorization:

- Change source or schemas.
- Install dependencies or download checkpoints.
- Start/reset databases or run destructive flags.
- Delete artifacts.
- Run `git commit`, `git push`, `git reset`, `git rebase` or delete branches.
- Modify another person's owned layer as an unrelated refactor.

When implementation is authorized, make the smallest focused change for one
milestone, add tests and preserve unrelated user changes.

---

## 6. Offline Pipeline contract

The logical Offline Pipeline is:

1. Shot detection and keyframe extraction.
2. PE-Core visual embedding.
3. ASR transcription, cleaning and video summarization.
4. OCR extraction.
5. Object detection.
6. Vietnamese semantic text embedding.
7. Multi-database indexing.

Online business logic reads the databases, not intermediate Offline JSON or
Parquet. Intermediate artifacts may only be read by approved validation,
migration, test or debugging tools.

### Milvus

Expected collections:

```text
visual_features
ocr_features
asr_features
summary_features
```

Expected behavior:

```text
index = HNSW
metric = IP
search ef = 128
stored vectors = L2-normalized
query vectors = L2-normalized
```

Do not hardcode visual or Vietnamese text dimensions. Read the actual collection
schema and verify the Online encoder output. A value such as 768 may be common
for the current text model, but it is not a safe hardcoded contract.

### Elasticsearch

Expected indexes:

```text
ocr_texts
asr_transcripts
video_summaries
```

Text fields use `vietnamese_analyzer` with ICU tokenization/folding and lowercase.

### SQLite

Expected database and tables:

```text
data/metadata.db
metadata
objects
```

Online access must be read-only.

All URIs, paths and resource names must remain configurable.

---

## 7. Identifier and JOIN rules

Canonical keys:

```text
Video:        video_id
Keyframe:     frame_id
ASR interval: video_id + interval_id
Summary:      video_id
```

Target `frame_id` format:

```text
{video_id}_{shot_id:05d}_{position_3_digits}
```

Example:

```text
V001_00000_015
```

The same `frame_id` must JOIN across:

- Milvus `visual_features`.
- Milvus `ocr_features`.
- Elasticsearch `ocr_texts`.
- SQLite `metadata`.
- SQLite `objects`.

Online must not rewrite or guess `frame_id`. A shared but malformed ID is still
a contract violation even if equality JOIN succeeds.

Do not use Milvus internal `pk` as a domain or cross-database identifier.

`interval_id` is only unique within `video_id`. Do not treat an ASR interval as
a frame.

---

## 8. Online domain and port boundary

Person A owns the merged infrastructure boundary under:

```text
online/domain/
online/ports/
online/adapters/
online/config.py
online/lifecycle.py
online/testing/
```

Persons B and C must consume ports rather than database SDKs directly.

Three retrieval result levels must remain separate:

### Frame level

```text
FrameSearchHit
→ SQLite hydration
→ FrameCandidate
```

Required hydrated fields:

```text
frame_id
video_id
shot_id
timestamp_sec
```

### ASR interval level

```text
ASRSearchHit
→ ASRIntervalCandidate
```

Person B returns interval candidates. Person C owns interval-to-frame mapping
after OQ-005 is approved.

### Video level

```text
VideoSearchHit
→ VideoCandidate
```

Summary results remain video-level support signals. They do not create frames or
hard-prefilter videos.

Each retrieval branch/query variant returns a homogeneous `BranchResult` and
preserves:

- Branch.
- Query variant ID and text.
- Rank.
- Raw score.
- Backend/resource provenance.
- Empty-success versus failure distinction.

Database SDK objects must not escape adapters.

---

## 9. Team roles and ownership

The split is by stable system layer, not by query mode. Do not create separate
database/retrieval/ranking stacks for t-KIS, v-KIS, TRAKE and VQA.

### 9.1 Person A — Data & Infrastructure

Person A owns:

```text
online/domain/
online/ports/
online/adapters/
online/config.py
online/lifecycle.py
online/testing/ infrastructure fakes and fixtures
online/validate_contract.py
tests/online/contract/
tests/online/adapters/
```

Milestones:

1. `A0`: shared domain models, enums, errors and ports.
2. `A1`: configurable resources and connection lifecycle.
3. `A2`: read-only SQLite metadata/object adapter.
4. `A3`: Milvus adapter for all four collections.
5. `A4`: Elasticsearch adapter for all three indexes.
6. `A5`: read-only Offline contract validator.
7. `A6`: protocol-conformant fakes and integration fixture.
8. `A7`: runtime integration, health and performance support.

Person A must:

- Keep domain/ports independent of database SDK objects.
- Preserve canonical IDs and candidate levels.
- Validate schema, dimension, vector norm and real cross-database JOINs.
- Keep SQLite read-only.
- Distinguish empty results, contract mismatches, timeouts and unavailable backends.
- Make endpoints, paths, resource names and search parameters configurable.

Person A does not own query expansion, encoders, retrieval orchestration,
ASR mapping, fusion, deduplication or public API behavior.

### 9.2 Person B — Query & Retrieval

Person B primarily owns:

```text
query_understanding/
online/retrieval/
Online encoder implementations
tests/online/retrieval/
```

Milestones:

1. `B0`: confirm domain/port contract and create B-side fakes.
2. `B1`: `QueryBundle`, validation and query parser/builder.
3. `B2`: PE-Core text encoder and Vietnamese text encoder.
4. `B3`: visual semantic branch.
5. `B4`: OCR lexical and semantic branches.
6. `B5`: ASR lexical and semantic branches.
7. `B6`: summary lexical and semantic branches.
8. `B7`: retrieval service, concurrency, timeout and branch diagnostics.
9. `B8`: shared t-KIS/v-KIS behavior.
10. `B9`: integration with Person A adapters and handoff to Person C.

Person B may later own TRAKE event encoding/similarity/DANTE after the related
open questions are approved, and supports VQA retrieval reuse after KIS baseline.

Person B does not own:

- SQL, Milvus or Elasticsearch SDK access.
- ASR interval-to-frame mapping.
- Branch normalization or final fusion.
- Summary boost policy.
- Object hard/soft ranking behavior.
- Deduplication.
- Final API response.

#### Person B branch rules

Baseline retrieval branches:

```text
visual_dense
ocr_dense
ocr_bm25
asr_dense
asr_bm25
summary_dense
summary_bm25
```

Query expansion baseline:

```text
q0 = original query
q1 = paraphrase 1
q2 = paraphrase 2
```

Retrieve query variants independently. Do not average embeddings before
retrieval unless a later approved decision changes the baseline.

Direct frame hits must be batch-hydrated through `MetadataReaderPort` before
becoming `FrameCandidate` values.

ASR branches return `BranchResult[ASRIntervalCandidate]`.

Summary branches return `BranchResult[VideoCandidate]`.

Person B must not normalize or fuse raw scores.

The merged search ports are synchronous. If `RetrievalService` is asynchronous,
do not call blocking SDK methods directly on the event loop. Use a controlled
thread/executor strategy or another explicitly approved boundary, and test actual
parallel behavior and timeout handling.

### 9.3 Person C — Ranking, Orchestration & API

Person C primarily owns:

```text
online/ranking/
online/modes/
retrieval_api/
UI/backend request contract after approval
tests/online/ranking/
tests/online/integration/ orchestration cases
```

Milestones:

1. `C0`: deterministic ASR interval-to-frame mapper after OQ-005.
2. `C1`: query-variant aggregation after OQ-004.
3. `C2`: branch normalization after OQ-006.
4. `C3`: frame fusion and provenance after OQ-007.
5. `C4`: controlled summary propagation after OQ-008.
6. `C5`: structured object hard/soft processing.
7. `C6`: deterministic deduplication and near-frame grouping.
8. `C7`: shared search orchestrator.
9. `C8`: API, mode routing, health and error mapping after OQ-002.
10. `C9`: VQA evidence orchestration after OQ-012/OQ-017/OQ-018.
11. `C10`: UI/backend object-constraint contract.

Person C must consume `RetrievalService`/`BranchResult` instead of running new
Milvus or Elasticsearch retrieval queries. Person C owns mapping and ranking,
but does not silently change raw adapter records, encoder checkpoints or branch
retrieval semantics.

### 9.4 Required handoff interfaces

```text
Person A
database SDKs → SDK-neutral hits/metadata through ports

Person B
QueryBundle + A ports → homogeneous BranchResult values

Person C
BranchResult values + metadata/object ports
→ mapping/normalization/fusion/dedup/API response
```

Boundary rules:

- B may mock A only with protocol-conformant fakes.
- C may mock B only with valid `BranchResult` fixtures.
- Shared domain/port changes require review from all affected roles.
- A must not place ranking policy in adapters.
- B must not place SQL or fusion logic in retrieval branches.
- C must not bypass B by querying search backends directly.

---

## 10. Mode-specific invariants

### Textual KIS and Video KIS

```text
Textual KIS:
task-provided text
→ shared text retrieval pipeline

Video KIS:
organizer-displayed clip
→ contestant observes clip
→ contestant manually writes text
→ same shared text retrieval pipeline
```

Do not create separate retrieval algorithms for these two modes. `v-KIS` does
not receive a video file, frame or image query from the organizer in baseline.

### TRAKE

Baseline TRAKE uses PE-Core visual-semantic event scores and DANTE per video.
DANTE must never transition between videos. Do not add OCR, ASR, summary, Stable
Diffusion or QUEST to the baseline DANTE matrix without approval.

### VQA

VQA retrieves evidence first and calls a VLM afterward. The VLM must answer from
retrieved evidence rather than process the entire dataset.

### Optional branches

Stable Diffusion and QUEST are optional and must not block KIS baseline.

---

## 11. Score, hydration and failure rules

Raw scores from different branches are not directly comparable.

Required order:

```text
retrieve
→ map/hydrate
→ normalize each branch independently
→ fuse
→ deduplicate/group
```

Do not add raw Milvus IP and Elasticsearch BM25 scores.

Every final candidate must retain branch/query provenance for debugging and
tuning.

Summary must not hard-prefilter videos and must not generate arbitrary frames.

Optional OCR/ASR/summary branch failures may degrade the query and must appear
in diagnostics. Core visual retrieval, vector compatibility and SQLite metadata
failures must surface clearly.

Empty successful results and backend failures are different states.

---

## 12. Open-question policy

Read `docs/08-OPEN-QUESTIONS.md` before implementing a dependent milestone.

Do not silently decide:

- Public API request/response schema.
- Per-branch top-k.
- Query-variant aggregation.
- ASR interval-to-frame mapping.
- Normalization/fusion method and weights.
- Summary boost.
- Object hard/soft default and position calculation.
- Image path resolution.
- DANTE candidate scope, distance, lambda or output granularity.
- VQA evidence budget/model/prompt.
- Stable Diffusion or QUEST activation.
- Database retry/pooling/circuit-breaker lifecycle.
- Missing metadata policy.

Internal model shapes already approved in `docs/11` may be implemented, but they
do not determine these algorithms or production parameters.

---

## 13. Current verified readiness gates

The Person A boundary and Person B B1-B9 implementation are code-ready against
the SDK-free Online suite:

1. Standard pytest collection imports the source `online` package without test
   package shadowing.
2. Canonical `frame_id` syntax and semantic fields are validated without
   rewriting IDs, including rejection of surrounding whitespace.
3. Missing encoder smoke checks are explicit `NOT_RUN` values and prevent a
   false integration `PASS`.
4. Adapter caller errors, backend contract errors and safe diagnostics use the
   shared domain error boundary; fakes cover the same caller validation cases.
5. Retrieval branches preserve candidate level, raw score and provenance, and
   the async service bounds synchronous work with deterministic ordering,
   timeouts and lifecycle guards.

`NEED_RUNTIME_VERIFICATION` still applies to installed encoder/database SDKs,
running services, actual schemas, stored vectors and Offline-produced records.
Do not claim real-database readiness until:

- The standard test command collects and passes.
- Canonical ID validation passes.
- The relevant encoder smoke test passes dimension and norm checks.
- Cross-database joins pass on real records.
- One real vertical slice runs end-to-end.

---

## 14. Implementation procedure

For each milestone:

1. Identify the owning role and milestone.
2. Read `AGENTS.md`, `docs/11` and relevant design/contract docs.
3. Inspect current source dependencies and upstream/downstream interfaces.
4. State input, output, invariants and open questions.
5. List expected files, owner boundaries and tests.
6. Make the smallest focused change.
7. Do not modify another layer unnecessarily.
8. Add success, empty, invalid and failure tests.
9. Run the smallest relevant tests, then the Online suite when possible.
10. Report exact commands/results and affected diff.
11. Report contract impact and remaining runtime limitations honestly.

Do not combine multiple major milestones into one implementation task unless the
user explicitly requests it.

Preferred test target:

```text
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online -q
```

Run this command from the application root containing `online/` and `tests/`
(`aic_nova_project/` in the current nested Git checkout). `unittest discovery`
may be used only as an additional diagnostic; pytest remains the required gate.

Do not install missing dependencies merely to make a test run unless the user
authorizes installation.

### Team Git and review rules

- Merge the shared contract before dependent implementation.
- Use one focused branch/PR per milestone or small task.
- Do not rename shared models/enums in an unrelated feature PR.
- Update model, fixture and contract tests together when a shared field changes.
- Assign one primary owner per file/component and at least one downstream reviewer.
- Pull/fetch shared changes before integration; do not copy another person's code
  manually between branches.
- Resolve boundary disagreements at the port/model level, not with duplicate
  implementations in downstream layers.
- A task is not complete until its tests, contract handoff and limitations are
  documented.

---

## 15. Reporting format

For analysis/review:

```text
1. Scope inspected
2. Confirmed architecture/code
3. Findings ordered by severity
4. Contract mismatches
5. Tests executed and results
6. Runtime verification still required
7. Readiness decision and next gate
```

For implementation:

```text
1. Milestone scope
2. Files changed
3. Behavior implemented
4. Tests added
5. Tests executed/results
6. Contract impact
7. Remaining limitations
```

Never claim that code, a database path, a model or an integration works unless it
was verified through source inspection or execution.
