# 06 — DESIGN DECISIONS

> **Current contract override (2026-08-05):** Decisions that assume PE-Core or
> organizer-provided keyframes/features are superseded by
> `docs/22-OFFLINE-TO-ONLINE-DATA-CONTRACT-SELF-INDEXED-V2.md`. The active visual
> space is OpenCLIP `ViT-B-32::openai` over team-extracted keyframes.

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

---

## DD-026 — DANTE dùng toàn bộ ordered keyframes của từng video

**Status:** CONFIRMED_DESIGN

**Resolves:** OQ-013

Nguồn chuẩn là Section 4.3, Algorithm 1 và Equations (1)–(5) của
`references/AIO_DANTE+QUEST.pdf`.

Với `N` event queries và toàn bộ `T` keyframes đã index:

- Encode từng event thành `u_i` bằng visual-language text encoder tương thích
  với visual embeddings Offline.
- Tính `S[i,t] = cosine_similarity(u_i, E[t])` cho toàn bộ keyframe đã index.
- Chia keyframes theo từng `video_id` và chạy một DANTE DP độc lập trên ordered
  range của video đó.
- Không dùng top-M union, threshold hoặc summary hard prefilter trong baseline.
- Không cho transition giữa hai video.

Paper dùng continuous global keyframe index và `[s_v,e_v]`. Implementation của
AIC Nova không được phụ thuộc Milvus internal `pk`; nó tạo local contiguous
position `t = 0..T_v-1` từ ordered frames của đúng video, giữ canonical
`frame_id` để trả kết quả.

Paper dùng BEiT-3, còn Offline index hiện tại của AIC Nova dùng PE-Core. Đây là
adaptation bắt buộc duy nhất: event text phải được encode bằng PE-Core để nằm
trong cùng embedding space với stored visual vectors. Dùng BEiT-3 text query
trên PE-Core visual index là contract mismatch; chuyển toàn hệ thống sang
BEiT-3 chỉ được phép sau một model migration và re-index riêng theo DD-024.

---

## DD-027 — DANTE temporal cost dùng keyframe-index gap

**Status:** CONFIRMED_DESIGN

**Resolves:** OQ-014

Transition bắt buộc giữ strict narrative order `tau < t` và dùng đúng linear
penalty của paper:

```text
DP[i,t] = S[i,t] + max_tau<t(DP[i-1,tau] - lambda * (t - tau))
```

Khoảng cách là chênh lệch local ordered-keyframe index `t - tau`, không phải
timestamp seconds hoặc shot gap. Running-max optimization theo Equations (3)
và (4) được dùng để đạt `O(N*T_v)` cho mỗi video.

Tie-break bổ sung để implementation deterministic: nếu nhiều predecessor có
cùng score, chọn `tau` nhỏ hơn; nếu nhiều end cell có cùng score, chọn `t` nhỏ
hơn.

---

## DD-028 — DANTE lambda mặc định 0.001, configurable đến 0.01

**Status:** CONFIRMED_DESIGN

**Resolves:** OQ-015

Paper báo cáo đã tune `lambda` trong `[0.001, 0.01]`: `0.001` phù hợp index gap
3–15 và `0.01` phù hợp alignment chặt với gap 1–3.

AIC Nova chọn:

- Default `lambda = 0.001` để baseline ít phạt các event cách nhau xa.
- Valid range đóng `[0.001, 0.01]`.
- Cho phép request/config override trong range.
- Không tự chọn lambda bằng LLM hoặc dựa trên query trong baseline.
- Reject boolean, NaN, Infinity và giá trị ngoài range.

Khi có dữ liệu thật, nhóm có thể tune giá trị trong range mà không đổi contract.
Thay đổi range hoặc penalty formula cần design decision mới.

---

## DD-029 — DANTE trả top-k video, mỗi video một best sequence

**Status:** CONFIRMED_DESIGN

**Resolves:** OQ-016

Với mỗi video:

```text
DANTE[v] = max_t DP[N,t]
```

Backtracking từ best final cell trả đúng `N` keyframes, một keyframe cho mỗi
event. Baseline trả top-k videos theo `DANTE[v]`, mỗi video chỉ có một best
sequence; không trả top-k sequences bên trong cùng video và không cộng near
frames vào DANTE score.

Không normalize score theo số event vì mọi video trong cùng request dùng cùng
`N`. Sort deterministic theo `score DESC`, sau đó `video_id ASC`.

---

## DD-030 — VQA evidence budget cân bằng quality và interactive latency

**Status:** CONFIRMED_DESIGN

**Resolves:** OQ-017

Evidence selector dùng ranking hiện có và budget mặc định:

- Tối đa 3 candidate videos.
- Tối đa 3 primary frames cho mỗi video.
- Tối đa 8 primary frames toàn request.
- Thêm tối đa một ordered neighbor trước và một neighbor sau mỗi primary frame
  khi còn budget; tổng số ảnh gửi VLM không vượt 12.
- Chọn ít nhất một primary frame từ mỗi selected video trước, sau đó fill theo
  final score để tránh một video chiếm hết budget.
- OCR chỉ lấy từ selected frames, tối đa 2,000 ký tự toàn request.
- ASR chỉ lấy intervals giao với cửa sổ ±5 giây quanh selected frames, tối đa
  4,000 ký tự toàn request.
- Summary tối đa 800 ký tự mỗi selected video và 2,400 ký tự toàn request.
- Tổng OCR + ASR + summary không vượt 8,000 ký tự.
- Summary không được đưa một video mới vào evidence nếu video đó không có frame
  retrieval evidence.
- Dedup theo stable evidence ID và giữ deterministic order.

Mọi giới hạn là frozen/configurable policy values. Khi thiếu evidence, selector
trả explicit diagnostics thay vì bịa nội dung hoặc âm thầm vượt budget.

---

## DD-031 — VQA dùng Gemini 3.5 Flash với structured evidence-only output

**Status:** CONFIRMED_DESIGN

**Resolves:** OQ-018

Primary VLM là stable model `gemini-3.5-flash` qua một mockable `VLMPort`.
Lựa chọn này dùng một stable multimodal model có image understanding và
structured output; model ID vẫn nằm trong config để có thể migrate có kiểm soát.

Prompt contract:

- Chỉ trả lời từ images và text evidence được cung cấp.
- Không dùng external/world knowledge để điền phần thiếu.
- Mỗi answer phải trả `evidence_ids` là subset của evidence request.
- Nếu evidence không đủ, trả `status=insufficient_evidence`.
- Trả lời ngắn gọn bằng cùng ngôn ngữ với câu hỏi.
- Không expose chain-of-thought, prompt nội bộ, secret hoặc local path.

Structured response:

```text
status: answered | insufficient_evidence
answer: string
answer_type: short_text | yes_no | number | list
confidence: low | medium | high
evidence_ids: list[EvidenceId]
```

Runtime defaults:

- `temperature = 0.1`.
- `max_output_tokens = 512`.
- Request timeout 15 giây.
- Tối đa một retry cho transient 429/5xx nếu vẫn còn trong total deadline.
- Malformed schema được xem là VLM failure, không parse tự do.
- Adapter/model unavailable trả explicit error; không tạo answer giả.

Google API key chỉ được đọc từ environment/secret manager. Tests và local fake
integration dùng `FakeVLMPort`, không cần gọi network.
