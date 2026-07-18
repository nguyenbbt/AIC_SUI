# 15 — Kế hoạch ba người code song song TRAKE/DANTE và VQA

## 1. Kết luận ngắn

Ba người **có thể code song song ngay**, dù chưa có dữ liệu thật.

Điều kiện để không lệch pha:

- Cả ba tạo branch từ cùng một commit nền.
- Wave đầu tiên không sửa chung file.
- A xây public contracts/ports.
- B xây DANTE core thuần toán học, chưa phụ thuộc public contracts mới.
- C xây VQA budget/selection core dựa trên KIS models đã tồn tại, chưa phụ thuộc
  evidence models mới.
- Khi Wave 1 xong mới ghép ba nhánh theo thứ tự A → B → C.
- Dữ liệu thật, actual model và database adapters để ở phase cuối.

```text
Wave 1 — ba lõi độc lập
A: contracts/ports ┐
B: DANTE core      ├ chạy đồng thời
C: VQA budget core ┘

Wave 2 — ba service độc lập trên contract đã merge
A: shared fakes/fixture ┐
B: TRAKE service        ├ chạy đồng thời
C: VQA orchestrator     ┘

Wave 3 — integration
A: conformance/readiness support
B: VQA retrieval rewrite
C: TRAKE/VQA routing/API

Wave 4 — dữ liệu thật và tuning
```

---

## 2. Commit nền và Git setup

Branch tích hợp hiện tại:

```text
feature/online-phase-Knguyen
```

Commit source-of-truth tối thiểu:

```text
499fae4 docs(online): resolve DANTE and VQA design gates
```

Commit mới hơn trên cùng branch vẫn hợp lệ nếu là descendant và không thay đổi
DD-026–DD-031.

Người B push branch tích hợp:

```powershell
git push origin feature/online-phase-Knguyen
```

Mỗi người tạo branch riêng:

### Người A

```powershell
git fetch origin
git switch -c feature/a-advanced-contracts origin/feature/online-phase-Knguyen
```

### Người B

```powershell
git fetch origin
git switch -c feature/b-dante-core origin/feature/online-phase-Knguyen
```

### Người C

```powershell
git fetch origin
git switch -c feature/c-vqa-budget-core origin/feature/online-phase-Knguyen
```

Cả ba phải ghi lại:

```powershell
git rev-parse --short HEAD
git status --short
```

Không bắt đầu nếu working tree có thay đổi không rõ nguồn.

---

## 3. Tài liệu cả ba bắt buộc đọc

1. `AGENTS.md`.
2. `docs/06-DESIGN-DECISIONS.md`, đặc biệt DD-026–DD-031.
3. `docs/08-OPEN-QUESTIONS.md`.
4. `docs/14-TRAKE-DANTE-VQA-CODING-ASSIGNMENT.md`.
5. `references/AIO_DANTE+QUEST.pdf` cho Người B; A/C đọc phần liên quan nếu
   cần review contract/output.

Các quyết định không được tự thay đổi:

- DANTE dùng toàn bộ ordered keyframes mỗi video.
- DANTE temporal distance là ordered-keyframe index gap.
- Lambda mặc định `0.001`, range `[0.001, 0.01]`.
- Mỗi video trả một best sequence; response là top-k videos.
- AIC Nova giữ PE-Core vì Offline visual index hiện tại là PE-Core.
- VQA budget đúng DD-030.
- VQA structured evidence-only output đúng DD-031.
- OQ-012 image path vẫn open; chỉ fake resolver được làm trước dữ liệu thật.

---

## 4. Quy tắc chống merge conflict

## 4.1 File ownership Wave 1

| Người | Được sửa | Không được sửa |
|---|---|---|
| A | `online/domain/trake.py`, `online/domain/vqa.py`, advanced files trong `online/ports/`, contract tests | `online/trake/`, `online/vqa/`, `retrieval_api/` |
| B | `online/trake/`, `tests/online/trake/` | `online/domain/`, `online/ports/`, `online/vqa/`, `retrieval_api/` |
| C | `online/vqa/`, `tests/online/vqa/` | `online/domain/`, `online/ports/`, `online/trake/`, `retrieval_api/` |

`__init__.py` có nguy cơ conflict. Trong Wave 1:

- A được sửa `online/domain/__init__.py` và `online/ports/__init__.py`.
- B chỉ sửa/tạo `online/trake/__init__.py`.
- C chỉ sửa/tạo `online/vqa/__init__.py`.
- Không ai sửa `online/__init__.py` hoặc composition root.

## 4.2 Không tạo public model cạnh tranh

- A tạo public TRAKE/VQA models.
- B được tạo private/internal result như `_DANTEPath` chỉ chứa score và index
  positions phục vụ algorithm core.
- C được tạo C-owned policy types như `EvidenceBudgetPolicy` và private text
  chunks phục vụ selection core.
- B/C không tạo class có ý định thay thế public `TRAKEVideoResult`,
  `VQAEvidence` hoặc `VLMResponse` của A.

## 4.3 Không merge giữa Wave

Ba người làm xong và push branch riêng. Người B chỉ ghép khi cả ba đã báo:

- Commit hash.
- Exact tests.
- Files changed.
- Known limitations.

---

# 5. WAVE 1 — Ba người code song song hoàn toàn

# 5A. Người A — Advanced public contracts và ports

## Mục tiêu

Tạo public boundary ổn định cho TRAKE/VQA, không code thuật toán hoặc actual
database/model adapter.

## A-P1.1 — TRAKE public models

Tạo `online/domain/trake.py` với models tối thiểu:

- `TRAKEEvent`.
- `TRAKEQuery`.
- `DANTEPolicy` hoặc tên tương đương chứa policy version/lambda.
- `TRAKEFrameMatch`.
- `TRAKEVideoResult`.
- `TRAKEDiagnostics`.

Validation:

- Tối thiểu hai ordered events.
- Event IDs non-empty và unique.
- `top_k_videos >= 1`.
- Lambda finite và trong `[0.001, 0.01]`.
- Sequence có đúng một match mỗi event.
- Mọi match cùng `video_id`.
- Local ordered indices strictly increasing.
- Score finite, không nhận boolean.
- Models strict/frozen/serializable.

Không đưa embedding vectors vào API result.

## A-P1.2 — VQA public models

Tạo `online/domain/vqa.py`:

- `VQAQuestion`.
- Answer type enum: `short_text | yes_no | number | list`.
- `VQAEvidenceBudget` đúng DD-030.
- Stable `EvidenceId`/evidence reference.
- Image/OCR/ASR/summary evidence.
- `VLMRequest`.
- Structured `VLMResponse` đúng DD-031.
- `VQAResult` và diagnostics.

Validation:

- Giữ đúng candidate level.
- ASR giữ interval ID/start/end.
- Summary giữ video ID, không giả thành frame.
- `answered` yêu cầu answer và evidence IDs.
- `insufficient_evidence` biểu diễn explicit.
- Không chứa SDK object, API key hoặc local absolute path.

## A-P1.3 — Advanced ports

Tạo files riêng trong `online/ports/`:

- Full ordered visual corpus port theo `video_id` và batch.
- Evidence hydration port cho OCR/ASR/summary.
- Image resolver port, interface-only.
- Mockable VLM port.

Port cho DANTE phải trả:

- canonical `frame_id`/`video_id`.
- local ordered position.
- timestamp/shot metadata.
- finite visual vector trong PE-Core space.

Không expose Milvus `pk`, SDK hits/rows hoặc client objects.

## Files của A

```text
online/domain/trake.py
online/domain/vqa.py
online/domain/__init__.py
online/ports/visual_corpus.py
online/ports/evidence.py
online/ports/images.py
online/ports/vlm.py
online/ports/__init__.py
tests/online/contract/test_advanced_models.py
tests/online/contract/test_advanced_ports.py
```

## Tests A

- Success serialization.
- Empty/duplicate events.
- Cross-video/non-increasing sequence.
- Lambda boundary/NaN/Infinity/boolean.
- VQA answered/insufficient rules.
- ASR interval validation.
- Extra fields/frozen behavior.
- Protocol conformance với minimal test doubles.

Commands:

```powershell
python -m pytest tests/online/contract -q
python -m pytest tests/online/contract tests/online/adapters tests/online/retrieval tests/online/integration tests/online/ranking tests/online/modes -q
git diff --check
```

## A không làm Wave 1

- Shared fixture/fakes lớn.
- Actual database/model/image resolver.
- DANTE recurrence.
- VQA selector/orchestrator.
- API/composition.

---

# 5B. Người B — DANTE pure algorithm core

## Mục tiêu

Code và chứng minh đúng DANTE math theo paper mà không cần public models/ports
của A. Module chỉ nhận similarity matrix và trả private score/path indices.

## B-P1.1 — Lambda validation

Trong `online/trake/` tạo config/validator do B sở hữu:

- Default `0.001`.
- Range `[0.001, 0.01]`.
- Reject boolean, NaN, Infinity và out-of-range.

Public wrapper sẽ map từ model A ở Wave 2; Wave 1 không import file chưa tồn tại.

## B-P1.2 — Naive reference oracle

Implement một reference function chỉ dùng cho tests:

```text
DP[i,t] = S[i,t] + max_tau<t(DP[i-1,tau] - lambda*(t-tau))
```

Oracle có thể `O(N*T^2)` vì chỉ chạy matrix nhỏ. Nó là nguồn kiểm tra optimized
algorithm, không phải production path.

## B-P1.3 — Optimized DANTE recurrence

Production core dùng running max:

```text
DP[0,t] = S[0,t]

running_max = max(
  running_max,
  DP[i-1,t-1] + lambda*(t-1)
)

DP[i,t] = S[i,t] + running_max - lambda*t
```

Giữ backpointer để trả path positions.

Edge behavior:

- `T < N` → no valid path.
- Không có predecessor → unreachable/negative infinity, không dùng zero.
- Equal predecessor score → index nhỏ hơn.
- Equal final score → end index nhỏ hơn.
- Không clamp negative scores.
- Matrix empty/ragged/non-finite bị reject.

## B-P1.4 — Pure result

Private/internal output chỉ cần:

```text
score: float
positions: tuple[int, ...]
```

Không chứa public frame/video models. Wave 2 wrapper sẽ hydrate positions thành
`TRAKEFrameMatch`.

## Files của B

```text
online/trake/__init__.py
online/trake/config.py
online/trake/dante.py
tests/online/trake/__init__.py
tests/online/trake/test_dante.py
```

## Tests B

1. Two-event sequence.
2. Three-event sequence.
3. Strict `tau < t`.
4. `T < N`.
5. Negative similarities.
6. Lambda 0.001/0.01.
7. Equal predecessor/end tie-break.
8. Backtracked positions strictly increasing.
9. Matrix invalid/ragged/non-finite.
10. Optimized score/path khớp naive oracle trên nhiều random matrices nhỏ.
11. Complexity test bảo đảm production path không gọi nested predecessor scan.

Commands:

```powershell
python -m pytest tests/online/trake -q
python -m pytest tests/online/contract tests/online/adapters tests/online/retrieval tests/online/integration tests/online/ranking tests/online/modes tests/online/trake -q
git diff --check
```

## B không làm Wave 1

- Public domain models/ports.
- PE-Core encoder calls.
- Milvus/full corpus reader.
- Video iteration/service/concurrency.
- VQA rewrite.
- API/composition.

---

# 5C. Người C — VQA budget và selection core

## Mục tiêu

Code phần policy/selection thuần, dùng các KIS models đã tồn tại. Chưa cần public
VQA evidence models hoặc ports mới của A.

## C-P1.1 — Evidence budget policy

Tạo C-owned frozen config trong `online/vqa/budget.py`:

- max videos = 3.
- max primary/video = 3.
- max primary total = 8.
- max images total = 12.
- OCR chars = 2,000.
- ASR chars = 4,000.
- summary chars/video = 800.
- summary chars total = 2,400.
- text chars total = 8,000.
- ASR window = ±5 seconds.

Reject boolean, negative, zero ở field bắt buộc dương, NaN/Infinity và inconsistent
caps. Defaults phải nằm một nơi duy nhất.

## C-P1.2 — Primary frame selection

Input dùng `FusedFrameCandidate` hiện có.

Algorithm:

1. Sort deterministic theo final score giảm, frame ID tăng.
2. Xác định tối đa 3 videos có frame evidence.
3. Chọn một primary frame/video trước.
4. Fill remaining slots theo score.
5. Không quá 3 primary/video và 8 total.
6. Dedup canonical `frame_id`.
7. Summary không tham gia tạo video list.

Output Wave 1 có thể là tuple `FusedFrameCandidate`; không tạo competing public
VQA evidence model.

## C-P1.3 — Neighbor selection core

Function thuần nhận:

- selected primary frames.
- ordered `FrameMetadata` sequences do test truyền trực tiếp.
- image cap.

Chọn previous/next local neighbors khi còn budget, dedup và không vượt 12 ảnh.
Không gọi MetadataReaderPort/database trong pure core.

## C-P1.4 — Deterministic text budget utilities

Private `_TextEvidenceChunk` hoặc tên có dấu `_` được phép dùng nội bộ với:

- stable ID.
- evidence type.
- source rank/order.
- text.

Utilities:

- Lọc ASR chunks theo interval giao cửa sổ ±5 giây.
- Truncate OCR/ASR/summary theo individual caps.
- Enforce summary per-video/total cap.
- Enforce combined 8,000-character cap.
- Same input cho same chunk order/output.

Private chunk không phải public VQA model; Wave 2 adapter map từ models A.

## Files của C

```text
online/vqa/__init__.py
online/vqa/budget.py
online/vqa/selection.py
tests/online/vqa/__init__.py
tests/online/vqa/test_budget.py
tests/online/vqa/test_selection.py
```

## Tests C

1. Video diversity.
2. 3-video/3-per-video/8-primary caps.
3. Duplicate frame.
4. Equal-score tie-break.
5. Neighbor before/after boundary.
6. Total image cap 12.
7. OCR/ASR/summary individual caps.
8. ASR ±5-second overlap.
9. Summary-only video không được thêm.
10. Combined text cap.
11. Invalid/non-finite/inconsistent config.
12. Deterministic output.

Commands:

```powershell
python -m pytest tests/online/vqa -q
python -m pytest tests/online/contract tests/online/adapters tests/online/retrieval tests/online/integration tests/online/ranking tests/online/modes tests/online/vqa -q
git diff --check
```

## C không làm Wave 1

- Public VQA domain models/ports.
- Evidence hydration/database calls.
- VLM request/response/network.
- Full VQA orchestrator.
- TRAKE mode/API.
- `retrieval_api/composition.py`.

---

# 6. Cách ghép Wave 1

Người B/integration owner ghép theo thứ tự:

```text
1. Merge A contracts/ports.
2. Merge B DANTE core.
3. Merge C VQA budget core.
```

Lý do:

- A chỉ tạo shared files.
- B chỉ tạo `online/trake/`.
- C chỉ tạo `online/vqa/`.
- Gần như không có content conflict.

Sau merge:

```powershell
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online -q
git diff --check
```

Nếu môi trường thiếu FastAPI/runtime dependency, chạy toàn bộ unaffected suite và
ghi rõ collection blocker; không cài dependency âm thầm.

Integration owner kiểm tra:

- Public `DANTEPolicy` của A và internal config của B có cùng defaults/range.
- Public VQA budget của A và C policy có cùng DD-030 values.
- Nếu khác, sửa ở wrapper/config mapping PR; không đổi algorithm âm thầm.

---

# 7. WAVE 2 — Tiếp tục ba người code song song

Wave 2 bắt đầu sau khi Wave 1 integration commit được push. Cả ba tạo branch mới
từ commit đó.

# 7A. Người A — Shared fixture và fakes

Task:

- `A-ADV-04` trong docs/14.
- Hai videos, >=6 ordered frames/video.
- Ba TRAKE events.
- Correct-order winner và wrong-order high-similarity distractor.
- Equal-score tie case.
- `T_v < N` case.
- OCR/ASR/summary/image evidence.
- Missing evidence cases.
- FakeVLM success/insufficient/malformed/timeout.

Files:

```text
online/testing/advanced_fakes.py
tests/online/fixtures/advanced_modes.*
tests/online/contract/test_advanced_fakes.py
```

A không sửa B/C service files.

# 7B. Người B — TRAKE parser, similarity và service

Task:

- Parse/encode ordered events.
- Full per-video PE-Core similarity matrix qua A port.
- Map DANTE path positions sang public `TRAKEFrameMatch`.
- Per-video DANTE, top-k result, diagnostics.
- Bounded concurrency/deadline/deterministic output.

Files:

```text
query_understanding/trake_parser.py
online/trake/similarity.py
online/trake/service.py
tests/online/trake/test_similarity.py
tests/online/trake/test_service.py
```

B không sửa public contracts, VQA core hoặc API.

# 7C. Người C — VQA VLM contract adapter và orchestrator

Task:

- Map selection core sang public evidence models A.
- Build structured evidence-only VLM request.
- Validate VLM response evidence IDs.
- Full VQA orchestrator với fake retrieval/VLM/ports.
- Failure/degradation/diagnostics.

Files:

```text
online/vqa/evidence_selector.py
online/vqa/vlm_request.py
online/vqa/orchestrator.py
tests/online/vqa/test_evidence_selector.py
tests/online/vqa/test_vlm_contract.py
tests/online/vqa/test_orchestrator.py
```

C không sửa TRAKE service hoặc API composition trong Wave 2.

---

# 8. Cách ghép Wave 2

Thứ tự:

```text
1. Merge A shared fakes/fixture.
2. Merge B TRAKE service.
3. Merge C VQA orchestrator.
4. Chạy TRAKE fake E2E.
5. Chạy VQA fake E2E.
6. Chạy full KIS regression.
```

Vì B/C có thể dùng local test builders khi A chưa merge fixture, cả ba vẫn code
song song. Sau khi A fixture merge, B/C chỉ bổ sung một integration test dùng
shared fixture, không viết lại algorithm.

---

# 9. WAVE 3 — Song song lần cuối trước integration API

# Người A

- Conformance tests cho fake/real future ports.
- Lifecycle/readiness support cho advanced resources.
- Image resolver fake/config boundary; actual path vẫn chờ OQ-012/data thật.

# Người B

- VQA retrieval rewrite.
- Reuse KIS branches.
- q0 luôn tồn tại; rewrite failure → degraded fallback.
- Không trả answer và không gọi VLM.

# Người C

- TRAKE mode adapter gọi service B.
- VQA/TRAKE internal routes.
- Composition wiring.
- Safe error mapping, readiness và graceful shutdown.

Wave 3 có thể code song song nhưng composition PR của C merge cuối cùng.

---

# 10. Definition of Done từng Wave

## Wave 1

- A contracts/ports tests pass.
- B optimized DANTE khớp naive oracle.
- C VQA budget caps/determinism tests pass.
- Ba branch không sửa chung implementation file.
- Merged full unaffected Online suite pass.

## Wave 2

- Shared advanced fixture pass.
- TRAKE service fake E2E pass.
- VQA orchestrator fake E2E pass.
- Provenance/candidate levels giữ đúng.
- KIS regression pass.

## Wave 3

- Explicit mode routing.
- Safe API errors.
- VLM/retrieval readiness rõ.
- Lifecycles drain trước close.
- Full fake TRAKE/VQA API E2E pass.

## Real-data phase sau này

- Actual DB/model/image ports healthy.
- Một real TRAKE query pass.
- Một real VQA query pass.
- Latency/cost measured.
- OQ-002/OQ-012 và production policies được đóng.

---

# 11. Mẫu báo cáo chung

Mỗi người khi xong task gửi:

```text
Người/role:
Wave/task:
Branch:
Base commit:
Commit hash:
Files changed:
Public contract impact:
Tests run + exact result:
Known limitations:
Blocker/contract request:
```

Daily status:

```text
Đã hoàn thành:
Đang làm:
Commit hiện tại:
Tests hiện tại:
Blocker:
```

---

# 12. Checklist integration owner

Trước khi merge mỗi branch:

- [ ] Đúng base commit/wave.
- [ ] Chỉ sửa files của owner.
- [ ] Không đổi DD-026–DD-031.
- [ ] Không có public model cạnh tranh.
- [ ] Không có SDK object xuyên layer.
- [ ] Không hardcode secret/path/model key.
- [ ] Success/empty/invalid/failure tests đủ.
- [ ] `git diff --check` sạch.
- [ ] Exact test result được báo.

Sau khi merge ba branch:

- [ ] Full Online tests.
- [ ] TRAKE/VQA fake E2E khi wave yêu cầu.
- [ ] KIS t-KIS/v-KIS parity không regression.
- [ ] Diagnostics/provenance vẫn đầy đủ.

---

# 13. Task gửi ngay hôm nay

## Gửi Người A

```text
Đọc AGENTS.md và docs/14, docs/15.
Tạo branch feature/a-advanced-contracts từ branch tích hợp mới nhất.
Làm toàn bộ mục 5A: public TRAKE/VQA models + advanced ports + contract tests.
Không làm fixture, algorithm, orchestrator hay actual adapters.
```

## Người B bắt đầu

```text
Tạo branch feature/b-dante-core.
Làm toàn bộ mục 5B: naive oracle + optimized DANTE + backtracking + tests.
Không phụ thuộc files mới của A.
```

## Gửi Người C

```text
Đọc AGENTS.md và docs/14, docs/15.
Tạo branch feature/c-vqa-budget-core từ branch tích hợp mới nhất.
Làm toàn bộ mục 5C: frozen budget policy + primary/neighbor/text selection core + tests.
Không tạo public VQA models, không làm VLM/orchestrator/API ở Wave 1.
```

Ba task trên không đụng cùng implementation files và có thể bắt đầu cùng lúc.
