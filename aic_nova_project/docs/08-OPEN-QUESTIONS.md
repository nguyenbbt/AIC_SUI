# 08 — OPEN QUESTIONS

## 1. Quy tắc

Codex không được tự đóng câu hỏi trong file này.

Mỗi câu hỏi phải được người dùng hoặc nhóm xác nhận rồi chuyển sang `06-DESIGN-DECISIONS.md`.

---

## OQ-001 — Runtime `frame_id` consistency

**Blocker for:** Cross-DB adapter integration.

Cần kiểm tra record thật:

```text
visual_features.frame_id
ocr_features.frame_id
ocr_texts.frame_id
metadata.frame_id
objects.frame_id
```

Lưu ý: Module 2 document output đã dùng global ID; Module 7 visual loader cần được audit xem có normalize hoặc hoàn toàn tin artifact.

---

## OQ-002 — Exact API request/response schema

**Blocker for:** Public API layer.

Cần chốt:

- Mode enum.
- Query bundle fields.
- Object constraint JSON.
- Pagination.
- Top-k.
- Diagnostics.
- Competition output adapter.

---

## OQ-003 — Top-k theo branch

**Blocker for:** Retrieval orchestration/performance tuning.

Cần chốt riêng:

- Visual q0/q1/q2.
- OCR lexical.
- OCR semantic.
- ASR lexical.
- ASR semantic.
- Summary lexical.
- Summary semantic.
- Optional image branches.

---

## OQ-004 — Multi-query aggregation

**Blocker for:** Textual KIS ranking.

Kết quả q0, q1, q2 sẽ được gộp bằng:

- Max score?
- Mean?
- Weighted fusion?
- RRF?
- Xem mỗi query variant như branch riêng?

---

## OQ-005 — ASR interval-to-frame mapping

**Blocker for:** ASR branches.

Cần chốt:

1. Tất cả keyframe nằm trong interval?
2. Một keyframe gần midpoint?
3. Nearest keyframe nếu interval không chứa frame?
4. Temporal neighborhood size?
5. Cách chia score nếu map một interval sang nhiều frame?
6. Cách dedup overlapping intervals?

---

## OQ-006 — Branch normalization

**Blocker for:** Fusion.

Ứng viên:

- Per-list min-max.
- Robust min-max.
- Rank normalization.
- RRF-only.
- Z-score.

Cần xử lý list ngắn và equal scores.

---

## OQ-007 — Fusion method và weights

**Blocker for:** Final ranking.

Cần chốt:

- Weighted score fusion hay RRF.
- Initial weights.
- Query-variant weights.
- Missing branch behavior.
- Weight normalization.
- Object soft boost scale.

---

## OQ-008 — Summary boost

**Blocker for:** Summary integration.

Đã chốt không prefilter.

Còn cần chốt:

- Lexical và semantic summary gộp thế nào?
- Weight bao nhiêu?
- Có cap boost không?
- Chỉ boost frames có evidence từ branch nào?
- Summary score absent xử lý ra sao?

---

## OQ-009 — Object hard/soft default

**Blocker for:** UI/backend object filter.

Cần chốt default:

- Hard filter.
- Soft boost.
- User bắt buộc chọn mode.
- Có fallback khi detector bỏ sót không?

---

## OQ-010 — Object position representation

**Blocker for:** Position filter.

SQLite lưu bbox pixel nhưng metadata không có width/height.

Cần chọn:

1. Resolve image path và đọc size.
2. Bổ sung width/height vào metadata contract.
3. Offline normalize bbox.
4. Dùng known fixed resolution nếu dataset bảo đảm.

Không được giả định fixed resolution nếu chưa kiểm chứng.

---

## OQ-011 — Position predicate

**Blocker for:** Position filter.

Object được xem là trong vùng nếu:

- Center point nằm trong region?
- IoU với region vượt threshold?
- Tỷ lệ diện tích bbox trong region vượt threshold?

Baseline đề xuất center point, nhưng chưa chốt.

---

## OQ-012 — Image path resolution

**Blocker for:** UI thumbnails và VQA evidence.

SQLite không lưu `file_path`.

Cần chốt:

- Deterministic path convention.
- Separate artifact manifest.
- Additional metadata store.
- Configurable resolver.

---

## OQ-013 — DANTE candidate scope

**Blocker for:** TRAKE performance.

Cần chốt:

- Chạy trên toàn bộ keyframe từng video?
- Lấy top-M per event rồi union videos?
- Candidate-video generation bằng visual branch?
- Có threshold không?

Không dùng summary hard prefilter.

---

## OQ-014 — DANTE temporal distance

**Blocker for:** DANTE implementation.

Distance dùng:

- Keyframe index gap.
- Timestamp seconds.
- Shot gap.

Cần chốt cùng đơn vị của `λ`.

---

## OQ-015 — DANTE λ

**Blocker for:** TRAKE ranking.

Cần:

- Default.
- Config range.
- Validation dataset.
- Có query-dependent λ không?

---

## OQ-016 — DANTE output granularity

**Blocker for:** TRAKE API.

Mỗi event trả:

- Một best frame?
- Best frame + near frames?
- Top-k sequences?
- Video score normalization theo số event?

---

## OQ-017 — VQA evidence budget

**Blocker for:** VQA generation.

Cần chốt:

- Số video.
- Số keyframe.
- Near-frame window.
- OCR text amount.
- ASR intervals.
- Summary length.
- Token/image budget.

---

## OQ-018 — VQA model and prompt contract

**Blocker for:** VQA final stage.

Cần chốt:

- VLM/model.
- Answer types.
- Evidence-only policy.
- Confidence.
- Fallback/no-answer.
- Output language.

---

## OQ-019 — Stable Diffusion activation

**Blocker for:** Optional branch only.

- Manual toggle?
- Always run?
- LLM decision?
- Timeout/budget?

---

## OQ-020 — QUEST activation

**Blocker for:** Optional branch only.

- Manual toggle?
- OOK detection?
- External search provider?
- Image selection?
- Safety/latency?

---

## OQ-021 — Database connection lifecycle

**Blocker for:** Serving architecture.

Cần chốt:

- Startup connection.
- Pooling.
- Retry policy.
- Circuit breaker.
- Health checks.
- Timeouts.

---

## OQ-022 — Missing metadata policy

**Blocker for:** Hydration.

Frame candidate không có SQLite metadata:

- Drop?
- Return partial?
- Fail query?
- Log contract violation?

---

## OQ-023 — Test fixtures

**Blocker for:** Integration tests.

Cần có:

- Small Milvus fixture.
- ES fixture.
- SQLite fixture.
- IDs khớp.
- One visual, OCR, ASR, summary example.
- Object count/position cases.
