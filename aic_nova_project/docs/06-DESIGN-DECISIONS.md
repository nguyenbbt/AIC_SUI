# 06 — DESIGN DECISIONS

## DD-001 — Tách Offline và Online

**Status:** CONFIRMED_DESIGN

Offline xử lý dataset và xây index.

Online xử lý query và không chạy lại full-dataset preprocessing.

---

## DD-002 — Polyglot persistence

**Status:** CONFIRMED_DESIGN

Sử dụng:

```text
Milvus
Elasticsearch
SQLite
```

Không thay bằng một database duy nhất nếu chưa có phê duyệt.

---

## DD-003 — Bốn Milvus collections

**Status:** CONFIRMED_CODE / CONFIRMED_DESIGN

```text
visual_features
ocr_features
asr_features
summary_features
```

`ocr_features` là phần hiện hành và phải được Online hỗ trợ.

---

## DD-004 — Canonical frame key

**Status:** CONFIRMED_DESIGN

```text
frame_id
```

Target format:

```text
{video_id}_{shot_id:05d}_{position_3_digits}
```

Không dùng Milvus `pk`.

Runtime validation vẫn bắt buộc.

---

## DD-005 — L2 + Inner Product

**Status:** CONFIRMED_CODE / CONFIRMED_DESIGN

- Stored vector L2-normalized.
- Query vector L2-normalized.
- Milvus metric `IP`.
- HNSW search.

---

## DD-006 — Visual dimension dynamic

**Status:** CONFIRMED_DESIGN

Không hardcode visual dimension.

Read from collection schema/model output.

---

## DD-007 — Explicit mode trong retrieval core

**Status:** CONFIRMED_DESIGN

Retrieval core hỗ trợ:

```text
KIS_TEXT
KIS_VISUAL
TRAKE
VQA
```

Agent router có thể chọn mode trong tương lai, nhưng core vẫn phải nhận explicit mode.

---

## DD-008 — Textual KIS có nhiều branch song song

**Status:** CONFIRMED_DESIGN

Baseline target branches:

- Visual semantic.
- OCR lexical.
- OCR semantic.
- ASR lexical.
- ASR semantic.
- Summary lexical.
- Summary semantic.

SD và QUEST là optional.

---

## DD-009 — Summary không prefilter

**Status:** CONFIRMED_DESIGN

Summary chỉ là video-level support signal.

Không dùng summary để loại video trước frame retrieval.

Reason:

```text
summary error
→ false video elimination
→ correct frame can never be recovered
```

---

## DD-010 — Object constraints do UI cung cấp

**Status:** CONFIRMED_DESIGN

Baseline không dùng LLM tự động trích object constraints.

UI cho phép chọn:

- Label.
- Count.
- Position.
- Confidence.
- Hard/soft mode.

---

## DD-011 — Late fusion

**Status:** CONFIRMED_DESIGN

Mỗi branch retrieval độc lập.

Normalize trước fusion.

Không cộng raw Milvus score và raw BM25.

---

## DD-012 — Summary propagation chỉ áp vào frame candidates đã tồn tại

**Status:** CONFIRMED_DESIGN

Summary không tự sinh một frame candidate.

Nó chỉ boost candidate thuộc cùng `video_id`.

---

## DD-013 — Object filter trên candidate set

**Status:** CONFIRMED_DESIGN

Không full-scan toàn SQLite trước retrieval nếu không cần.

Chạy structured filter sau union/mapping/hydration candidates.

---

## DD-014 — TRAKE baseline dùng DANTE visual-only

**Status:** CONFIRMED_DESIGN

DANTE matrix dùng event-to-visual-keyframe similarity.

Không trộn OCR, ASR, summary, SD hoặc QUEST vào baseline matrix.

---

## DD-015 — DANTE chạy per video

**Status:** CONFIRMED_DESIGN

Không được chuyển trạng thái giữa hai video.

Keyframe order lấy từ timestamp.

---

## DD-016 — VQA retrieval trước, generation sau

**Status:** CONFIRMED_DESIGN

VQA không gọi VLM trên toàn dataset.

Nó retrieval evidence trước, sau đó mới answer generation.

---

## DD-017 — Stable Diffusion optional

**Status:** OPTIONAL

Không chặn baseline.

---

## DD-018 — QUEST optional

**Status:** OPTIONAL

Rewrite và external image branch được bổ sung sau core retrieval.

---

## DD-019 — Conversational clarification chưa thuộc baseline

**Status:** OUT_OF_SCOPE

KISC/AI hỏi lại người dùng chưa được code trong milestone đầu.

---

## DD-020 — Online đọc database, không phụ thuộc artifact trung gian

**Status:** CONFIRMED_DESIGN

Business logic Online không đọc trực tiếp Offline JSON/Parquet.

Ngoại lệ:

- Migration.
- Validation tools.
- Tests.
- Debug tooling được duyệt.

---

## DD-021 — SQLite Online read-only

**Status:** CONFIRMED_DESIGN

Feedback/logging không được ghi vào `metadata.db`.

Dùng storage khác nếu cần write data Online.

---

## DD-022 — Giữ provenance

**Status:** CONFIRMED_DESIGN

Final candidate phải giữ score/branch provenance để:

- Debug.
- Tune weights.
- Phân tích lỗi.
- So sánh ablation.

---

## DD-023 — Optional failure có graceful degradation

**Status:** CONFIRMED_DESIGN

OCR/ASR/summary/SD/QUEST branch lỗi không nhất thiết làm fail toàn query.

Core dependency lỗi phải surfaced rõ.

---

## DD-024 — Không thay model hoặc schema âm thầm

**Status:** CONFIRMED_DESIGN

Mọi thay đổi:

- Model.
- Embedding space.
- Collection.
- Field.
- ID format.
- Metric.

phải có migration và phê duyệt.

---

## DD-025 — Video KIS dùng text query do thí sinh tự viết

**Status:** CONFIRMED_DESIGN

Trong Video KIS (`v-KIS`), BTC trình chiếu clip trên màn hình để thí sinh
quan sát. Baseline không nhận file video, frame hoặc query image từ BTC.

Thí sinh tự viết mô tả text từ clip đã xem, sau đó hệ thống dùng
chung pipeline text-to-keyframe với Textual KIS.

```text
Textual KIS: task-provided text → text-to-keyframe retrieval
Video KIS: displayed clip → human-authored text → same retrieval
```

Khác biệt là nguồn của query, không phải retrieval mechanism. `KIS_VISUAL`
chỉ là working/legacy enum cho đến khi OQ-002 chốt public API schema; không
được diễn giải enum này thành image-to-image retrieval.
