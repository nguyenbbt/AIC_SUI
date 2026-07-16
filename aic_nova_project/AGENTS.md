# AGENTS.md

## 1. Project identity

This repository contains the AIC Nova multimedia video retrieval system for the
HCM AI Challenge.

The system is divided into two major phases:

```text
OFFLINE PHASE
Raw videos
→ preprocessing
→ feature extraction
→ multi-database indexing

ONLINE PHASE
User query
→ query construction
→ retrieval
→ candidate conversion
→ hydration
→ normalization
→ fusion/reranking
→ mode-specific output
```

The system supports four principal query modes:

- Textual Known-Item Search (`KIS_TEXT`)
- Video Known-Item Search (`v-KIS`; working/legacy enum `KIS_VISUAL`)
- Temporal Retrieval and Alignment of Key Events (`TRAKE`)
- Visual Question Answering (`VQA`)

The current development task focuses on the Online Phase, but every Online
implementation decision must respect the actual outputs and contracts of the
Offline Phase.

---

## 2. Mandatory reading order

Before analyzing, planning, or modifying source code, read the following files
in this exact order:

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

Do not begin implementation before reading all of these files.

---

## 3. Source-of-truth precedence

When information differs between sources, use the following precedence:

1. The user’s latest explicit instruction.
2. `docs/06-DESIGN-DECISIONS.md`.
3. `docs/04-OFFLINE-ONLINE-CONTRACT.md`.
4. `docs/03-DATABASE-SCHEMA-CURRENT.md`.
5. Current source code on the checked-out branch.
6. Module-level README files.
7. Research papers.
8. Historical discussions or old design notes.

The source code describes what is currently implemented.

The design documents describe what the system is intended to implement.

If source code and design documents disagree:

- Do not silently choose one.
- Do not immediately change the code.
- Report the mismatch as `CONTRACT_MISMATCH`.
- Identify the files, symbols, schemas, and affected components.
- Wait for user approval before changing the contract or architecture.

---

## 4. Required evidence labels

Every technical conclusion in an analysis report must use one of these labels:

- `CONFIRMED_CODE`
  - Verified directly from current source code.
- `CONFIRMED_DESIGN`
  - Explicitly approved in the design documents.
- `NEED_RUNTIME_VERIFICATION`
  - Requires a running database, generated artifact, model, or external service.
- `OPEN_QUESTION`
  - Not decided yet.
- `OPTIONAL`
  - Supported or planned, but not required for the baseline.
- `OUT_OF_SCOPE`
  - Not part of the current implementation stage.
- `CONTRACT_MISMATCH`
  - Source code, schema, or documentation do not agree.

Never present an assumption as a confirmed fact.

---

## 5. Default operating mode: understanding only

Until the user explicitly states that implementation may begin, operate in
read-only analysis mode.

In this mode, do not:

- Modify files.
- Create source code.
- Refactor.
- Install dependencies.
- Download model checkpoints.
- Run GPU models.
- Start or reset databases.
- Execute destructive CLI flags.
- Delete generated artifacts.
- Change schemas.
- Change models.
- Rename collections, indexes, tables, or fields.
- Run `git commit`, `git push`, `git reset`, `git rebase`, or branch deletion.
- Apply formatting changes unrelated to the analysis.

You may:

- Read files.
- Search symbols and references.
- Inspect tests.
- Inspect Docker and configuration files.
- Trace data flow.
- Compare code with documentation.
- Propose non-destructive runtime checks.
- Report missing or inconsistent components.

---

## 6. Offline Pipeline that must be understood

The logical Offline Pipeline contains:

1. Video processing, shot detection, and keyframe extraction.
2. PE-Core visual embedding.
3. ASR transcription, transcript cleaning, and video summarization.
4. OCR extraction.
5. Object detection.
6. Vietnamese semantic text embedding.
7. Multi-database indexing.

For every Offline module, identify and understand:

- Directory.
- Entry point.
- Main classes and functions.
- Configuration.
- Input artifacts.
- Output artifacts.
- Output schema.
- Models and libraries.
- Downstream consumer.
- Database destination.
- Error handling.
- Resume or overwrite behavior.
- Existing tests.

Do not rely only on README descriptions. Verify important behavior in source
code.

---

## 7. Current database contract

The current architecture uses three database systems.

### 7.1 Milvus

Expected collections:

```text
visual_features
ocr_features
asr_features
summary_features
```

Roles:

- `visual_features`: frame-level visual embeddings.
- `ocr_features`: frame-level OCR semantic embeddings.
- `asr_features`: ASR interval-level semantic embeddings.
- `summary_features`: video-level summary semantic embeddings.

Expected vector behavior:

```text
index_type = HNSW
metric_type = IP
search ef = 128
stored vectors = L2-normalized
query vectors = L2-normalized
```

Do not hardcode visual embedding dimensions.

Do not expose or use Milvus internal `pk` as a cross-database identifier.

### 7.2 Elasticsearch

Expected indexes:

```text
ocr_texts
asr_transcripts
video_summaries
```

Roles:

- `ocr_texts`: frame-level OCR lexical search.
- `asr_transcripts`: ASR interval-level lexical search.
- `video_summaries`: video-level lexical search.

Expected analyzer:

```text
vietnamese_analyzer
├── icu_tokenizer
├── icu_folding
└── lowercase
```

### 7.3 SQLite

Expected database:

```text
data/metadata.db
```

Expected tables:

```text
metadata
objects
```

Roles:

- `metadata`: frame, video, shot, and timestamp mapping.
- `objects`: object label, confidence, and bounding-box detections.

The Online Pipeline should access this database as read-only.

---

## 8. Identifier rules

### 8.1 Video key

```text
video_id
```

### 8.2 Keyframe key

```text
frame_id
```

Target canonical format:

```text
{video_id}_{shot_id:05d}_{position_3_digits}
```

Example:

```text
V001_00000_015
```

`frame_id` must match across:

- Milvus `visual_features`.
- Milvus `ocr_features`.
- Elasticsearch `ocr_texts`.
- SQLite `metadata`.
- SQLite `objects`.

A local filename stem such as:

```text
shot_00000_pos_015
```

must not remain as the final cross-database key.

Before Online integration is considered valid, inspect real records and verify
the equality of `frame_id` across databases.

### 8.3 ASR interval key

```text
video_id + interval_id
```

Used across:

- Milvus `asr_features`.
- Elasticsearch `asr_transcripts`.

Do not treat an ASR interval as a keyframe.

### 8.4 Summary key

```text
video_id
```

Used across:

- Milvus `summary_features`.
- Elasticsearch `video_summaries`.

---

## 9. Candidate levels

The Online Pipeline must preserve three result levels.

### 9.1 Frame-level candidate

Contains:

```text
frame_id
video_id
shot_id
score
```

Sources include:

- Visual semantic search.
- OCR semantic search.
- OCR lexical search.
- Video KIS visual-semantic text retrieval.
- Stable Diffusion image search.
- QUEST external image search.

### 9.2 ASR interval-level candidate

Contains:

```text
video_id
interval_id
start_time_sec
end_time_sec
score
```

Sources include:

- ASR semantic search.
- ASR lexical search.

It must be mapped to keyframes using SQLite timestamps before frame-level
fusion.

### 9.3 Video-level candidate

Contains:

```text
video_id
score
```

Sources include:

- Summary semantic search.
- Summary lexical search.

It is supporting evidence, not direct frame evidence.

---

## 10. Online Pipeline decisions that must not be changed

### 10.1 Textual KIS

Baseline branches run in parallel:

- Visual semantic search.
- OCR lexical search.
- OCR semantic search.
- ASR lexical search.
- ASR semantic search.
- Summary lexical search.
- Summary semantic search.

Query expansion baseline:

```text
original query
+ paraphrase 1
+ paraphrase 2
```

Each query variant must be retrieved independently unless another strategy is
explicitly approved.

### 10.2 Summary behavior

Summary search must not be used as a hard prefilter.

A wrong or incomplete summary must never remove a potentially correct video
before frame-level retrieval.

Summary results are video-level support signals used during late fusion.

Summary scores may boost existing frame candidates belonging to the same video.

Summary search must not create arbitrary frame candidates by itself.

### 10.3 Object constraints

Baseline object constraints come from explicit UI controls.

The user may select:

- Object label.
- Count.
- Count operator.
- Position.
- Minimum confidence.
- Hard filter or soft boost.

Do not require an LLM to infer object constraints from the query in the
baseline.

Object filtering should be applied to the retrieved candidate set rather than
performing an unnecessary full-database scan.

### 10.4 Video KIS (`v-KIS`)

Baseline:

```text
organizer plays a video clip on the shared screen
→ contestant watches the clip
→ contestant manually writes a textual description
→ reuse the same text-to-keyframe retrieval pipeline as Textual KIS
→ ranked keyframes
```

The baseline system does not receive a video file, frame, or query image from
the organizer for `v-KIS`. The distinction from Textual KIS is the origin of
the textual query, not the retrieval mechanism:

- Textual KIS: use the textual description supplied by the task.
- Video KIS: the contestant observes the displayed clip and authors the text.

Keep `KIS_VISUAL` only as the current working/legacy enum until the public API
schema is explicitly finalized. Do not interpret it as image-to-image search.

### 10.5 TRAKE

Baseline TRAKE uses DANTE with visual-semantic event scores only.

```text
ordered events
→ PE-Core text encoder
→ event-keyframe similarity
→ DANTE per video
→ backtracking
→ ordered frame sequence
```

Do not automatically fuse OCR, ASR, summary, Stable Diffusion, or QUEST into
the DANTE similarity matrix.

DANTE must never transition between different videos.

### 10.6 VQA

Baseline:

```text
question
→ retrieval-oriented rewrite
→ multimodal retrieval
→ evidence collection
→ VLM
→ text answer
```

The VLM must answer from retrieved evidence rather than processing the entire
dataset.

### 10.7 Optional branches

The following are optional and must not block the baseline:

- Stable Diffusion.
- QUEST query rewrite.
- QUEST external image retrieval.

---

## 11. Score rules

Raw scores from different branches are not directly comparable.

Examples:

- Milvus Inner Product.
- Elasticsearch BM25.
- Summary video score.
- Object soft boost.
- DANTE sequence score.

Required order:

```text
retrieve
→ normalize each branch independently
→ fuse normalized scores
```

Do not directly add raw Milvus and Elasticsearch scores.

The exact normalization method, fusion method, and weights remain open until
explicitly approved.

Every final candidate must retain branch provenance and individual branch
scores for debugging and weight tuning.

---

## 12. Hydration and grouping rules

Frame candidates must be hydrated through SQLite `metadata`.

Required hydrated fields:

```text
frame_id
video_id
shot_id
timestamp_sec
```

Baseline deduplication:

1. Group by `video_id + shot_id`.
2. Keep the highest-scoring frame as representative.
3. Store other frames in the same shot as `near_frames`.
4. Use a temporal fallback only when shot information is unavailable.

Do not assume `image_path` exists in SQLite. Verify or define a separate path
resolver.

---

## 13. Required comprehension gates

Before implementation, complete these gates.

### Gate A — Documentation understanding

Explain:

- Project purpose.
- Offline and Online responsibilities.
- Query modes.
- Database roles.
- Design invariants.
- Optional and out-of-scope components.

### Gate B — Repository mapping

Identify:

- Directories.
- Entry points.
- Classes and functions.
- Config.
- CLI.
- Docker.
- Tests.
- Generated artifacts.
- Implemented, partial, scaffolded, and missing modules.

### Gate C — Offline trace

Trace one video through all Offline modules and into all databases.

### Gate D — Database contract audit

Verify:

- Four Milvus collections.
- Three Elasticsearch indexes.
- Two SQLite tables.
- OCR semantic loader/indexing.
- `frame_id` normalization.
- Vector dimension and normalization.
- Reset and rollback behavior.
- Cross-database JOIN keys.

### Gate E — Online understanding

Explain input, processing, branch output, mapping, hydration, filtering,
normalization, fusion, grouping, and final output for all four modes.

Implementation may begin only after the user explicitly approves the
comprehension reports.

---

## 14. Open-question policy

Read `docs/08-OPEN-QUESTIONS.md`.

Do not silently decide:

- Branch top-k.
- Query-variant aggregation.
- ASR interval-to-frame mapping.
- Normalization.
- Fusion method.
- Fusion weights.
- Summary boost weight.
- Object position calculation.
- Image path resolution.
- DANTE candidate scope.
- DANTE distance unit.
- DANTE lambda.
- VQA evidence budget.
- VQA model and prompt.
- Stable Diffusion activation.
- QUEST activation.
- Final API request/response schema.

Report which open questions block the requested milestone.

---

## 15. Rules after implementation is authorized

When the user explicitly authorizes a coding milestone:

1. Re-read the relevant documentation.
2. Inspect all source dependencies.
3. Summarize the task’s input, output, invariants, and open questions.
4. List files expected to change.
5. Make the smallest focused change.
6. Preserve Offline contracts.
7. Avoid unrelated refactoring.
8. Add or update tests.
9. Run the smallest relevant test set.
10. Report exact test commands and results.
11. Display the affected diff.
12. Report remaining limitations honestly.

Do not modify multiple large milestones in one task unless explicitly requested.

---

## 16. Reporting format

For system-analysis tasks, use:

```text
1. Scope inspected
2. Confirmed architecture
3. Repository map
4. Offline data flow
5. Database contract
6. Online target flow
7. Code versus documentation mismatches
8. Runtime verification still required
9. Open questions
10. Recommended next gate
```

For implementation tasks, use:

```text
1. Task scope
2. Files changed
3. Behavior implemented
4. Tests added
5. Tests executed
6. Test results
7. Contract impact
8. Remaining limitations
```

Never claim that code or a database path works unless it was verified through
source inspection or execution.
