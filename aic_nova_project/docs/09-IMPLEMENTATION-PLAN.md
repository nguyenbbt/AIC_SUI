# 09 — IMPLEMENTATION PLAN

## 1. Nguyên tắc

- Không code trước khi hoàn thành comprehension gates.
- Mỗi milestone nhỏ, testable, reviewable.
- Không đổi Offline contract.
- Adapters trước orchestration.
- Domain models không phụ thuộc SDK database.
- External services phải mock được.
- Optional branches làm sau baseline.

---

# Milestone 0 — System understanding

## Deliverables

- Repository map.
- Offline trace.
- Database schema audit.
- Online target explanation.
- List of mismatches.
- Runtime validation plan.

## Exit criteria

Người dùng xác nhận Codex hiểu đúng toàn hệ thống.

Không có source change.

---

# Milestone 1 — Domain models và config

## Scope

Tạo model trung lập cho:

- `FrameCandidate`
- `ASRIntervalCandidate`
- `VideoCandidate`
- `BranchResult`
- `FusedFrameCandidate`
- `ObjectConstraint`
- Query bundles
- Diagnostics

## Tests

- Validation.
- Serialization.
- Invalid modes.
- Score/provenance preservation.

## Exit criteria

Không import Milvus/ES/SQLite SDK trong domain layer.

---

# Milestone 2 — SQLite adapter

## Scope

Read-only adapter cho:

- Hydrate frame metadata.
- Get ordered frames by video.
- Get objects by candidate IDs.
- Count/filter object constraints.

## Tests

- Batch hydration.
- Missing frame.
- Ordering by timestamp.
- Exact/min/max count.
- Confidence filtering.
- Multiple constraints.

Position filtering chỉ code sau OQ-010/OQ-011 được chốt.

---

# Milestone 3 — Milvus adapter

## Scope

Search:

- `visual_features`
- `ocr_features`
- `asr_features`
- `summary_features`

## Requirements

- Read collection schema.
- Validate dimension.
- Normalize query vector or reject.
- Configurable top-k/search params.
- Return domain candidates.
- Không expose `pk`.

## Tests

- Mocked SDK.
- Dimension mismatch.
- Empty result.
- Output-field mapping.
- Connection errors.

---

# Milestone 4 — Elasticsearch adapter

## Scope

Search:

- `ocr_texts`
- `asr_transcripts`
- `video_summaries`

## Requirements

- Vietnamese match/fuzzy config.
- Filter/source fields.
- Domain result conversion.
- Timeout/error diagnostics.

## Tests

- Query body.
- Empty text.
- Mapping.
- ES failure.

---

# Milestone 5 — Contract validator

## Scope

Read-only command/service kiểm tra:

- Resources exist.
- Schema fields.
- Dimensions.
- Vector norm sample.
- `frame_id` JOIN.
- ASR JOIN.
- Summary JOIN.

## Exit criteria

Có report PASS/PARTIAL/FAIL.

Không reset DB.

---

# Milestone 6 — Encoders

## Scope

Interfaces:

- PE-Core text encoder.
- PE-Core image encoder [chỉ cho optional SD/QUEST; không phải baseline v-KIS].
- Vietnamese text encoder.

## Requirements

- Same checkpoint/config as Offline.
- L2 normalization.
- Dimension validation.
- Batch support.
- Mockable.

## Tests

- Norm.
- Shape.
- Empty input.
- Model loading errors.

---

# Milestone 7 — Candidate conversion và hydration

## Scope

- Frame result normalization.
- ASR interval mapping.
- Summary video score storage.
- SQLite hydration.
- Provenance.

ASR mapping implementation chỉ sau OQ-005 được chốt.

---

# Milestone 8 — Object filter

## Scope

- Structured constraints từ UI.
- Presence/count/co-occurrence.
- Hard filter.
- Soft boost.
- Position sau khi contract được chốt.

## Tests

- Exact count.
- GTE/LTE.
- Multiple labels.
- Hard vs soft.
- Detector confidence.

---

# Milestone 9 — Normalization và fusion

## Scope

- Branch normalizer interface.
- Multi-query aggregation.
- Fusion.
- Summary propagation.
- Branch score diagnostics.

Chỉ implement method đã được chốt từ OQ-004/OQ-006/OQ-007/OQ-008.

---

# Milestone 10 — Dedup/grouping

## Scope

- Group by `video_id + shot_id`.
- Best frame representative.
- `near_frames`.
- Temporal fallback.

## Tests

- Same shot.
- Different video.
- Missing shot ID.
- Tie scores.

---

# Milestone 11 — Textual KIS orchestration

## Scope

Baseline branches:

- Visual semantic.
- OCR lexical.
- OCR semantic.
- ASR lexical.
- ASR semantic.
- Summary lexical.
- Summary semantic.

## Pipeline

```text
Query bundle
→ parallel branches
→ mapping/hydration
→ object constraints
→ normalize/fusion
→ dedup
→ result
```

## Tests

- All branches.
- Partial branch failure.
- Empty optional sources.
- Summary does not prefilter.
- Provenance retained.

---

# Milestone 12 — Video KIS (`v-KIS`)

## Scope

```text
Clip do BTC trình chiếu
→ thí sinh xem và tự viết mô tả text
→ reuse Textual KIS text-to-keyframe pipeline
→ ranked keyframes
```

## Tests

- Manual text query validation.
- Same retrieval contract as Textual KIS.
- Empty results.
- Near-frame grouping.
- Không yêu cầu file video/frame/query image từ BTC.

---

# Milestone 13 — DANTE / TRAKE

## Scope

- Ordered event models.
- Event encoding.
- Per-video similarity.
- DP.
- Backtracking.
- Sequence ranking.

## Tests

- Ordering.
- No cross-video transition.
- Two/three events.
- No valid sequence.
- Temporal penalty.
- Backtracking.

Chỉ implement performance strategy sau OQ-013–OQ-016 được chốt.

---

# Milestone 14 — VQA evidence pipeline

## Scope

- Retrieval rewrite.
- Reuse Textual KIS branches.
- Evidence selection.
- OCR/ASR/summary hydration.
- VLM interface.
- Answer output.

## Tests

- Evidence budget.
- Missing evidence.
- VLM failure.
- Answer type.
- Evidence-only prompt.

---

# Milestone 15 — Optional Stable Diffusion

## Scope

Chỉ làm sau core baseline ổn định.

Manual toggle trước khi có routing policy.

---

# Milestone 16 — Optional QUEST

## Scope

- Rewrite.
- External exemplar.
- Image retrieval.

Chỉ làm sau khi OQ-020 được chốt.

---

# Milestone 17 — API layer

## Scope

- Request validation.
- Mode routing.
- Async orchestration.
- Error mapping.
- Health endpoints.
- Diagnostics.
- Competition adapters.

Không thiết kế API cuối trước OQ-002.

---

# Milestone 18 — UI object controls

## Scope

- Label selector.
- Count operator/value.
- Position region.
- Confidence.
- Hard/soft mode.
- Request serialization.

UI phải gửi structured object constraints.

---

# Milestone 19 — Integration and performance

## Tests

- Small end-to-end fixture.
- All database branches.
- Cross-DB JOIN.
- Query latency.
- Parallel execution.
- Timeout/degradation.
- Determinism.
- DANTE scaling.

## Deliverables

- Benchmark report.
- Failure-mode report.
- Tunable config list.
- Reproducible commands.

---

# 2. Coding prompt rule cho từng milestone

Mỗi milestone bắt đầu bằng:

1. Đọc `AGENTS.md`.
2. Đọc các docs liên quan.
3. Đọc source dependencies.
4. Tóm tắt input/output/constraints.
5. Liệt kê file sẽ sửa.
6. Liệt kê tests sẽ viết.
7. Chờ hoặc tuân theo approval mode.
8. Code nhỏ nhất.
9. Chạy test liên quan.
10. Hiển thị diff và limitations.

Không được gộp nhiều milestone lớn vào một prompt.
