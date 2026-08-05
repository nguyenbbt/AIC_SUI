# 14 — Phân công code TRAKE/DANTE và VQA cho A, B, C

> **Contract migration notice (2026-08-05):** Keep the DANTE/VQA task split,
> but use the OpenCLIP visual space and self-indexed frame metadata defined by
> `docs/22-OFFLINE-TO-ONLINE-DATA-CONTRACT-SELF-INDEXED-V2.md`.

## 1. Trạng thái và phạm vi

Đây là kế hoạch triển khai chính thức sau khi:

- A+B+C KIS đã merge tại commit `5d536b5`.
- Kế hoạch advanced modes đã được ghi tại commit `8f224c2`.
- Người dùng cho phép Codex chốt OQ-013–018.
- OQ-013–016 được chốt theo `references/AIO_DANTE+QUEST.pdf`.
- OQ-017–018 được chốt tại DD-030 và DD-031.
- Chưa có dữ liệu thật; mọi milestone trước phần Real Integration dùng strict
  contracts, fake ports và shared fixture.

Mục tiêu của ba người:

```text
Ordered TRAKE events
→ full per-video visual similarity
→ DANTE DP/backtracking
→ ranked video sequences

VQA question
→ retrieval rewrite
→ KIS evidence retrieval
→ bounded evidence selection
→ Gemini VLM through a port
→ evidence-grounded answer
```

QUEST/Stable Diffusion vẫn optional và không nằm trong Definition of Done của
TRAKE/VQA core. Query rewrite có thể học cách tổ chức từ QUEST Branch 1, nhưng
không được tự bật external image search trước khi OQ-020 được chốt.

---

## 2. Các quyết định đã chốt — không cần họp lại trước khi code

## 2.1 DANTE candidate scope — DD-026

- Dùng toàn bộ ordered keyframes đã index của từng video.
- Không dùng top-M union, threshold hoặc summary prefilter.
- Tính cosine similarity giữa từng event embedding và từng frame embedding.
- Chạy DP độc lập theo `video_id`; tuyệt đối không cross-video transition.
- Paper dùng global contiguous index; AIC Nova dùng local contiguous position
  trong từng video và giữ canonical `frame_id` làm domain ID.
- Paper dùng BEiT-3 nhưng AIC Nova giữ PE-Core vì Offline visual index hiện tại
  là PE-Core. Thuật toán DANTE giữ nguyên; chỉ thay encoder bằng model cùng
  embedding space với index. Dùng BEiT-3 query trên PE-Core vectors là sai
  contract và chỉ đổi được sau một migration/re-index riêng.

## 2.2 DANTE temporal distance — DD-027

```text
DP[i,t] = S[i,t] + max_tau<t(DP[i-1,tau] - lambda * (t - tau))
```

- `t - tau` là ordered-keyframe index gap.
- Không dùng timestamp seconds hoặc shot gap trong baseline.
- Dùng running-max optimization để đạt `O(N*T_v)` mỗi video.
- Equal-score predecessor chọn index nhỏ hơn.

## 2.3 DANTE lambda — DD-028

- Default `0.001`.
- Valid range `[0.001, 0.01]`.
- Request/config được override trong range.
- Không query-dependent và không do LLM chọn.
- Giá trị khác range, boolean, NaN hoặc Infinity phải bị reject.

## 2.4 DANTE output — DD-029

- Mỗi video có một DANTE score: `max_t DP[N,t]`.
- Backtracking trả đúng một frame cho mỗi ordered event.
- Mỗi video chỉ trả một best sequence.
- Response xếp top-k videos theo score giảm dần, rồi `video_id` tăng dần.
- Không trả top-k sequences bên trong cùng video.
- Không thêm near frames vào DANTE score.

## 2.5 VQA evidence budget — DD-030

- Tối đa 3 videos.
- Tối đa 3 primary frames/video.
- Tối đa 8 primary frames/request.
- Tối đa 12 images sau khi thêm ordered neighbors.
- OCR tối đa 2,000 ký tự.
- ASR tối đa 4,000 ký tự và chỉ lấy interval giao cửa sổ ±5 giây.
- Summary tối đa 800 ký tự/video, 2,400 ký tự/request.
- Tổng OCR+ASR+summary tối đa 8,000 ký tự.
- Summary không được tự đưa video mới vào evidence.

## 2.6 VQA model/prompt — DD-031

- Primary model: stable `gemini-3.5-flash`.
- Structured output, không parse free-form answer.
- Evidence-only; evidence thiếu thì `insufficient_evidence`.
- Trả lời cùng ngôn ngữ câu hỏi.
- Confidence dùng `low | medium | high`, không dùng số giả chính xác.
- Temperature `0.1`, max output 512 tokens, timeout 15 giây.
- Tối đa một transient retry trong total deadline.
- API key chỉ từ environment/secret manager.

Nguồn model: [Gemini models](https://ai.google.dev/gemini-api/docs/models) và
[Gemini image understanding](https://ai.google.dev/gemini-api/docs/image-understanding).

---

## 3. Ownership cố định

| Khu vực | Primary | Reviewer bắt buộc |
|---|---|---|
| Shared domain models/enums/config serialization | A | B + C |
| Advanced read ports và fakes | A | B + C |
| TRAKE parser/event encoding/DANTE/service | B | C; A review port/ID |
| VQA retrieval rewrite | B | C |
| VQA evidence selector/orchestrator | C | B; A review evidence ports |
| TRAKE/VQA mode routing và API | C | A + B |
| Real data validation/adapters | A | B + C |

Không sửa chéo owner boundary trong feature PR. Shared model thay đổi phải tách
thành contract PR và được cả ba approve.

---

## 4. Cấu trúc source đề xuất

Đây là ranh giới file để giảm merge conflict:

```text
online/
  domain/
    trake.py                 # A
    vqa.py                   # A
  ports/
    visual_corpus.py         # A
    evidence.py              # A
    images.py                # A
    vlm.py                   # A theo contract do C đề xuất
  trake/
    config.py                # B
    similarity.py            # B
    dante.py                 # B
    service.py               # B
  vqa/
    budget.py                # C
    evidence_selector.py     # C
    orchestrator.py          # C
  modes/
    trake.py                 # C, gọi service của B
    vqa.py                   # C
  testing/
    advanced_fakes.py        # A

query_understanding/
  trake_parser.py            # B
  vqa_rewrite.py             # B

retrieval_api/
  advanced_routes.py         # C
  composition.py             # C; A/B review wiring

tests/online/
  contract/test_advanced_models.py
  trake/test_similarity.py
  trake/test_dante.py
  trake/test_service.py
  vqa/test_budget.py
  vqa/test_evidence_selector.py
  vqa/test_orchestrator.py
  integration/test_trake_fake_e2e.py
  integration/test_vqa_fake_e2e.py
  api/test_advanced_routes.py
```

Tên file có thể điều chỉnh một lần trong contract PR. Sau đó không đổi tên trong
PR thuật toán.

---

## 5. Merge order tổng quát

```text
M0 — A: shared contracts/ports
→ M1 — A: fake fixture
→ M2-B — B: DANTE core                 ┐
→ M2-C — C: VQA selector/orchestrator  ├ chạy song song
→ M2-A — A: conformance support        ┘
→ M3-B — B: TRAKE service + VQA rewrite
→ M3-C — C: mode routing/API
→ M4 — A+B+C fake end-to-end
→ M5 — real data adapters/validation khi data có
→ M6 — benchmark/tuning/rehearsal
```

B và C không cần chờ M1 để viết test cases/thuật toán pure. Hai người chỉ chờ
M0 để import model chính thức.

---

# 6. Người A — Data, contracts, ports và fixture

## A-ADV-01 — Shared TRAKE domain contracts

### Mục tiêu

Tạo input/output ổn định để B code DANTE và C serialize kết quả mà không dùng
temporary dict.

### Models tối thiểu

- `TRAKEEvent`
  - `event_id`.
  - `text`.
  - `order` hoặc tuple order được xác định bởi container.
- `TRAKEQuery`
  - `query_id`.
  - ordered events.
  - `top_k_videos`.
  - validated DANTE config reference.
- `TRAKEFrameMatch`
  - event ID.
  - canonical `frame_id`.
  - `video_id`.
  - local ordered index.
  - timestamp.
  - event-frame cosine similarity.
- `TRAKEVideoResult`
  - `video_id`.
  - DANTE score.
  - sequence chứa đúng N matches.
- `TRAKEDiagnostics`
  - policy/version.
  - lambda.
  - event count.
  - video/frame counts.
  - similarity/DP latency.
  - invalid-sequence count.

### Validation bắt buộc

- Tối thiểu hai events cho TRAKE baseline.
- Event IDs non-empty và không trùng.
- `top_k_videos >= 1`.
- Sequence length bằng event count.
- Mọi match trong một result có cùng `video_id`.
- Local indices tăng strictly theo event order.
- Scores finite; không nhận boolean làm number.
- Models strict/frozen và serializable.

### Tests

- Success serialization.
- Empty/one-event query reject.
- Duplicate event ID reject.
- Cross-video sequence reject.
- Non-increasing local indices reject.
- NaN/Infinity reject.
- Unknown fields reject.
- Frozen behavior.

### Branch/PR

```text
feature/a-advanced-trake-contracts
```

Reviewer: B và C.

---

## A-ADV-02 — Shared VQA domain contracts

### Models tối thiểu

- `VQAQuestion`.
- `VQAEvidenceBudget` đúng DD-030.
- `EvidenceReference` có stable evidence ID/type/source IDs.
- `ImageEvidence`.
- `OCREvidence`.
- `ASREvidence` giữ interval ID/start/end.
- `SummaryEvidence` giữ video ID.
- `VLMRequest`.
- `VLMResponse` đúng structured schema DD-031.
- `VQAResult` và diagnostics.

### Validation bắt buộc

- `evidence_ids` trong VLM response phải là subset của request; validation ở C
  orchestration nhưng model phải biểu diễn được.
- `insufficient_evidence` cho phép answer rỗng/chuẩn hóa; `answered` yêu cầu
  answer và evidence IDs.
- Evidence ID không chứa local absolute path.
- ASR start/end hợp lệ.
- Summary không giả làm frame evidence.

### Tests

- Từng evidence type serialize đúng.
- Malformed VLM response reject.
- Invalid answer type/status/confidence reject.
- Extra field reject.
- Frozen behavior.

### Branch/PR

```text
feature/a-advanced-vqa-contracts
```

Reviewer: B và C.

---

## A-ADV-03 — Advanced read ports

### Port cho DANTE visual corpus

Port phải cho B đọc toàn bộ ordered frame embeddings theo từng video mà không
biết Milvus SDK:

```text
list_video_ids()
iter_ordered_frame_embedding_batches(video_id, batch_size)
```

Mỗi record có:

- canonical frame/video ID.
- local order position.
- timestamp/shot metadata.
- finite L2-normalized visual vector.

Port không expose Milvus `pk`, collection row hoặc SDK hit.

### Ports cho VQA

```text
resolve_images(frame_ids)
get_ocr_evidence(frame_ids)
get_asr_evidence(video_id, start_sec, end_sec)
get_summary_evidence(video_ids)
VLMPort.answer(validated_request)
```

Real image resolver chưa code cho đến khi OQ-012 được giải quyết; fake resolver
được phép trả fixture references.

### Tests

- Protocol conformance cho fake ports.
- Ordered batches nối lại đúng thứ tự.
- Duplicate/missing/wrong-video records surfaced.
- Vector dimension/norm validation.
- Missing evidence khác backend unavailable.

### Branch/PR

```text
feature/a-advanced-ports
```

Reviewer: B và C.

---

## A-ADV-04 — Shared advanced fixture

Fixture bắt buộc có:

- Hai videos, mỗi video ít nhất sáu ordered frames.
- Ba events có best sequence rõ ràng ở video 1.
- Video 2 có high individual similarities nhưng wrong temporal order để DANTE
  phải xếp thấp hơn.
- Một equal-score case cho tie-break.
- Một video có `T_v < N` để test no-valid-sequence.
- OCR, ASR intervals, summary và fake image references.
- Missing-image và missing-ASR cases.
- Fake VLM success, insufficient, malformed, timeout modes.

Không cần vectors từ model thật. Vectors nhỏ, finite, L2-normalized và thiết kế
để expected cosine/DP score tính tay được.

### Branch/PR

```text
feature/a-advanced-fixture
```

Reviewer: B và C.

---

## A-ADV-05 — Việc A làm khi có data thật

Chưa làm ở giai đoạn fake:

- Actual Milvus full-corpus reader.
- Actual image resolver.
- Actual OCR/ASR/summary hydration implementation nếu ports hiện có chưa đủ.
- Cross-DB real record validation.
- Actual VLM client lifecycle/health nếu C không sở hữu adapter.

Khi data có, A phải chạy read-only validator trước; không rewrite `frame_id` để
che contract mismatch.

---

# 7. Người B — TRAKE/DANTE và retrieval rewrite

## B-ADV-01 — TRAKE parser và event encoding

### Việc cần làm

- Parse ordered event descriptions thành shared `TRAKEQuery`.
- Không dùng KIS q0/q1/q2 model cho event order.
- Encode từng event bằng PE-Core-compatible text encoder hiện có.
- Validate batch size/dimension/norm.
- Giữ mapping `event_id → embedding row`.

### Tests

- Hai/ba events.
- Whitespace/duplicate/empty event.
- Encoder returns wrong rows/dimension/zero norm/non-finite.
- Order giữ nguyên qua encoding.

### Branch/PR

```text
feature/b-trake-query-encoding
```

Reviewer: A cho encoder contract, C cho query handoff.

---

## B-ADV-02 — Similarity matrix đúng paper

### Việc cần làm

Cho mỗi video `v`:

```text
U: [N, D]
E_v: [T_v, D]
S_v[i,t] = cosine(U[i], E_v[t])
```

- Dùng toàn bộ frames từ port A.
- Không top-M/threshold.
- Không mix OCR/ASR/summary.
- Không phụ thuộc global Milvus pk/index.
- Có batching để không yêu cầu load toàn corpus toàn collection cùng lúc.
- Output order theo local frame position.

### Tests

- Matrix shape.
- Cosine values tính tay.
- Two-video isolation.
- Empty video.
- Dimension mismatch.
- Non-normalized/non-finite vector surfaced.
- Batch size không làm đổi output.

### Branch/PR

```text
feature/b-trake-similarity
```

Reviewer: A.

---

## B-ADV-03 — DANTE dynamic programming

### Base/recurrence

```text
DP[0,t] = S[0,t]

running_max = max(
  running_max,
  DP[i-1,t-1] + lambda*(t-1)
)

DP[i,t] = S[i,t] + running_max - lambda*t
```

Mỗi DP cell giữ predecessor argmax để backtrack.

### Edge behavior tự quyết đã chốt

- `T_v < N` → no valid sequence.
- Cell không có predecessor → negative infinity/unreachable, không dùng zero.
- Equal predecessor score → chọn predecessor index nhỏ hơn.
- Equal final score → chọn end index nhỏ hơn.
- Final video sort: score giảm, `video_id` tăng.
- Không clamp negative cosine/DP score.

### Complexity

- Time `O(N*T_v)` mỗi video.
- Backtracking `O(N)` mỗi returned video.
- Không implement naive `O(N*T_v^2)` làm production path.

### Tests

1. Recurrence khớp exhaustive reference trên matrix nhỏ.
2. Two-event sequence.
3. Three-event sequence.
4. Strict `tau < t`.
5. No valid sequence.
6. Negative similarities.
7. Lambda 0.001 và 0.01 đổi winner theo expected fixture.
8. Equal-score predecessor/end tie-break.
9. Backtracking trả đúng N frames.
10. Không cross-video.
11. Config boolean/NaN/Infinity/out-of-range reject.
12. Optimized score/path khớp naive test oracle trên random small matrices.

### Branch/PR

```text
feature/b-dante-core
```

Reviewer: C; A review ID/order.

---

## B-ADV-04 — TRAKE service

### Pipeline

```text
validate TRAKE query
→ encode ordered events
→ enumerate video corpus
→ compute per-video similarity
→ DANTE per video
→ filter unreachable videos
→ sort/top-k
→ return domain result + diagnostics
```

### Concurrency/deadline

- Per-video work có thể parallel trong bounded executor.
- Output order deterministic bất kể completion order.
- Có total deadline và cancellation/drain contract giống KIS.
- Không nuốt one-video failure thành success nếu port contract bị sai.
- Nếu một optional video record malformed, behavior phải theo explicit policy và
  diagnostics; không âm thầm đổi ID.

### Diagnostics

- Event/video/frame counts.
- Similarity latency.
- DP latency.
- Lambda/policy version.
- Unreachable video count.
- Returned video/sequence count.

### Branch/PR

```text
feature/b-trake-service
```

Reviewer: C.

---

## B-ADV-05 — VQA retrieval rewrite

### Việc cần làm

- Nhận VQA question + answer type.
- Rewrite thành mô tả evidence thị giác cần tìm, không trả lời câu hỏi.
- Tạo q0 retrieval rewrite; q1/q2 optional paraphrases nếu rewriter có.
- Reuse KIS seven branches và `RetrievalService`.
- Trả ranked candidates cho C, không chọn evidence budget và không gọi VLM.

### Fallback

- Rewriter unavailable/timeout → deterministic rule/no-op rewrite được ghi
  degraded; không fabricate answer.
- Actual LLM provider nằm sau `QueryRewritePort`.
- Provider secrets không vào query bundle/diagnostics.

### Tests

- Question → retrieval-oriented text.
- Không chứa guessed answer.
- q0 always present.
- Rewrite timeout/failure.
- KIS optional branch degradation.
- Handoff giữ provenance.

### Branch/PR

```text
feature/b-vqa-retrieval-rewrite
```

Reviewer: C.

---

## B-ADV-06 — QUEST chỉ khi được kích hoạt sau

Không thuộc core PR. Nếu OQ-020 được chốt, B triển khai đúng hai nhánh paper:

1. LLM visually-grounded query rewrite → standard semantic retrieval.
2. External representative image → compatible image encoder → visual retrieval.

Không merge external search vào DANTE matrix.

---

# 8. Người C — VQA, orchestration và API

## C-ADV-01 — Evidence budget implementation

### Việc cần làm

Implement frozen/configurable DD-030 policy, không hardcode rải rác.

Selection order:

1. Chọn tối đa 3 videos theo ranked frame evidence.
2. Chọn top primary frame của mỗi video trước để có diversity.
3. Fill remaining primary slots theo final score, tối đa 3/video và 8 total.
4. Thêm previous/next ordered neighbors khi còn image budget, tổng <= 12.
5. Dedup theo canonical `frame_id`.
6. Hydrate OCR/ASR/summary qua ports A.
7. Truncate deterministically theo evidence rank và source order.

### Tests

- Một video không chiếm hết budget.
- 3-video/8-primary/12-image caps.
- Same frame từ nhiều branches chỉ gửi một image.
- Neighbor before/after boundary.
- OCR/ASR/summary individual và total caps.
- ASR ±5 second overlap.
- Summary-only video không được thêm.
- Missing optional evidence có diagnostics.
- Same input cho same evidence order/IDs.

### Branch/PR

```text
feature/c-vqa-evidence-budget
```

Reviewer: B; A review hydration calls.

---

## C-ADV-02 — VLM request builder và response validator

### Request

- Question và answer type.
- Images có stable evidence IDs, không local path trong serialized prompt.
- OCR/ASR/summary blocks có source IDs.
- Evidence-only system instruction.
- Structured JSON schema đúng DD-031.

### Response validation

- Parse structured output bằng schema, không regex/free-form fallback.
- `evidence_ids` phải là subset của request.
- `answered` yêu cầu non-empty answer và evidence IDs.
- `insufficient_evidence` không được biến thành HTTP success có answer bịa.
- Output language phải theo question; nếu model vi phạm, trả validation failure
  hoặc bounded retry theo policy, không tự dịch âm thầm.

### Tests

- Valid answer types.
- Unsupported status/type/confidence.
- Unknown evidence ID.
- Empty answered result.
- Malformed JSON/schema.
- Prompt không chứa secret/path.

### Branch/PR

```text
feature/c-vqa-vlm-contract
```

Reviewer: A + B.

---

## C-ADV-03 — VQA orchestrator

### Pipeline

```text
validate VQA request
→ call B retrieval rewrite/service
→ select evidence
→ resolve/hydrate through A ports
→ build VLM request
→ call VLMPort
→ validate grounding/evidence references
→ return VQAResult + diagnostics
```

### Failure policy

- Retrieval core failure → explicit domain failure.
- Optional OCR/ASR/summary failure → degraded if visual evidence still usable.
- Image resolver failure for all selected images → fail/no-answer; không gọi VLM
  với evidence giả.
- VLM timeout/unavailable → explicit VLM error.
- Insufficient evidence → valid `insufficient_evidence` result.
- Malformed model output → one bounded retry only if deadline allows, sau đó fail.

### Diagnostics

- Retrieval latency/status.
- Candidate video/frame count.
- Selected image/text evidence count.
- Dropped/truncated evidence count.
- VLM model/version, latency, retry count.
- Final status; không ghi prompt/answer secret.

### Branch/PR

```text
feature/c-vqa-orchestrator
```

Reviewer: B; A review lifecycle/ports.

---

## C-ADV-04 — TRAKE mode adapter

C không implement DANTE lại. Adapter chỉ:

- Validate mode/request handoff.
- Gọi `TRAKE service` của B.
- Convert domain result sang internal API schema.
- Giữ event-frame provenance và diagnostics.
- Map timeout/contract/resource errors an toàn.

Tests:

- Top-k video serialization.
- Sequence đúng event order.
- Empty/unreachable result.
- Timeout/error mapping.
- Không expose embeddings/SDK rows/local paths.

### Branch/PR

```text
feature/c-trake-mode-adapter
```

Reviewer: B.

---

## C-ADV-05 — Advanced API routes và composition

API vẫn internal/unstable đến khi OQ-002 đóng.

### Việc cần làm

- Route explicit `trake` và `vqa`; không dùng LLM đoán mode.
- Request/response models dùng shared contracts.
- Wire A ports, B services và C orchestrators trong composition root.
- Readiness check cho VLM adapter khi VQA enabled.
- Drain requests/executors trước khi close resources.
- Public error allowlist không lộ model key, prompt, path hoặc raw exception.

### Tests

- Mode routing.
- TRAKE fake end-to-end.
- VQA fake end-to-end.
- VLM readiness healthy/unhealthy.
- Graceful shutdown với active request.
- Error sanitization.

### Branch/PR

```text
feature/c-advanced-api-routing
```

Reviewer: A + B.

---

# 9. Milestone tích hợp chung

## INT-ADV-01 — TRAKE fake end-to-end

Primary: B. Support: A + C.

Trace bắt buộc:

```text
3 event texts
→ fake compatible text embeddings
→ full ordered embeddings của 2 videos
→ similarity matrices
→ DANTE per video
→ backtracking
→ top-k domain/API result
```

Acceptance:

- Video có best individual frames nhưng wrong order không thắng.
- Không cross-video path.
- Score/path khớp hand calculation.
- Lambda và policy version có trong diagnostics.
- Same run cho byte-equivalent serialized result ngoài latency fields.

---

## INT-ADV-02 — VQA fake end-to-end

Primary: C. Support: A + B.

Trace bắt buộc:

```text
question
→ fake retrieval rewrite
→ KIS fake branch results/ranking
→ bounded evidence
→ fake image/OCR/ASR/summary hydration
→ FakeVLMPort
→ grounded structured answer
```

Acceptance:

- Không vượt budget.
- Evidence IDs trong answer hợp lệ.
- Missing optional text degrade rõ.
- Insufficient evidence không thành fabricated answer.
- VLM timeout/malformed output surfaced.

---

## INT-ADV-03 — Full Online regression

Primary: cả ba.

```powershell
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online -q
```

Ngoài ra phải chạy riêng:

- DANTE optimized vs naive oracle tests.
- TRAKE/VQA fake E2E.
- Existing KIS parity tests để chứng minh advanced modes không làm lệch t-KIS/v-KIS.
- `git diff --check`.

---

# 10. Real-data phase — để cuối

## Người A

- Implement actual full ordered visual corpus port từ DB schema thật.
- Validate vector space/checkpoint/dimension/norm.
- Chốt OQ-012 và actual image resolver.
- Actual evidence hydration.
- Cross-DB joins và read-only health checks.

## Người B

- Chạy full DANTE trên corpus thật; đo memory/latency.
- Benchmark lambda trong range paper `[0.001, 0.01]` nhưng giữ default cho đến
  khi có số liệu đủ đổi design decision.
- Kiểm tra event text encoder khớp visual embedding space.
- Đánh giá retrieval rewrite trên real query set.

## Người C

- Chạy actual Gemini adapter theo DD-031.
- Đo VQA latency/cost/answer grounding.
- Tune evidence budget chỉ qua config/benchmark; thay contract cần decision mới.
- Đóng OQ-002 và competition output schema.

Không gọi advanced modes runtime-ready trước khi ít nhất một real TRAKE query và
một real VQA query chạy end-to-end.

---

# 11. Task bắt đầu ngay

## Người A

```text
A-ADV-01 TRAKE contracts
A-ADV-02 VQA contracts
A-ADV-03 advanced ports
```

PR đầu tiên phải là shared contracts; không gom fixture/actual adapters vào cùng PR.

## Người B

```text
B-ADV-01 TRAKE parser/event encoding test cases
B-ADV-03 DANTE pure DP test oracle và recurrence
```

B có thể viết pure test oracle ngay, sau đó đổi import sang models của A khi PR
contract merge.

## Người C

```text
C-ADV-01 VQA evidence budget selection tests
C-ADV-02 structured VLM request/response test cases
```

C dùng temporary local test builders, không tạo competing shared domain models.

---

# 12. Quy tắc review và tránh lệch pha

Mỗi PR mô tả đúng bốn phần:

```text
Input contract:
Output contract:
Policy/version dùng:
Tests đã chạy:
```

Daily sync của mỗi người:

```text
Đã hoàn thành:
Đang làm:
Contract/input đang cần:
Blocker:
```

Dừng merge và sửa boundary nếu xuất hiện:

- B dùng Milvus SDK trực tiếp trong DANTE.
- C tự tính lại DANTE recurrence.
- A thêm ranking vào adapter.
- B/C tạo hai VQA evidence schemas khác nhau.
- DANTE dùng top-M dù DD-026 yêu cầu full keyframes.
- DANTE dùng timestamp penalty dù DD-027 yêu cầu index gap.
- VQA gửi quá 12 images hoặc dùng summary-only video.
- VLM trả evidence ID không tồn tại nhưng API vẫn accept.
- Một PR advanced mode làm KIS regression fail.

---

# 13. Definition of Done

## TRAKE/DANTE code-complete trên fakes

- Full-keyframe per-video scope đúng DD-026.
- Recurrence/running-max/backtracking đúng paper.
- Optimized implementation khớp naive oracle.
- Lambda validation đúng DD-028.
- Top-k video/one-sequence output đúng DD-029.
- Không cross-video transition.
- Fake E2E và full Online regression pass.

## VQA code-complete trên fakes

- Retrieval rewrite không trả lời câu hỏi.
- Budget đúng DD-030.
- Evidence selection/hydration deterministic.
- `FakeVLMPort` success/insufficient/failure tests pass.
- Structured grounding validation đúng DD-031.
- Fake E2E và full Online regression pass.

## Runtime-ready sau này

- Actual ports/model adapters healthy.
- Real TRAKE/VQA vertical slices pass.
- Latency/cost measured.
- API schema được chốt.
- Không có contract mismatch với Offline data.
