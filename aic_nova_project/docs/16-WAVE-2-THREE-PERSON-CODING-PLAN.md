# 16 — Kế hoạch chi tiết ba người code song song Wave 2

## 1. Mục tiêu của Wave 2

Wave 1 đã tạo xong ba lõi độc lập:

- A: public TRAKE/VQA contracts và advanced ports.
- B: DANTE pure dynamic programming core.
- C: VQA evidence-budget và selection core.

Wave 2 ghép các lõi này thành hai đường chạy hoàn chỉnh bằng fake data:

```text
TRAKE query
→ encode ordered events
→ đọc ordered visual corpus theo video
→ similarity matrix
→ DANTE từng video
→ top-k TRAKE results + diagnostics

VQA question
→ nhận ranked frame candidates giả lập
→ chọn primary/neighbor evidence
→ hydrate image/OCR/ASR/summary bằng fake ports
→ build evidence-only VLM request
→ fake VLM
→ validate grounded response
→ VQA result + diagnostics
```

Wave 2 chỉ cần chạy end-to-end với deterministic fakes. Không kết nối database,
checkpoint, image storage, LLM/VLM provider hoặc dữ liệu thật.

## 2. Trạng thái nền bắt buộc

Commit tích hợp Wave 1 đã được xác nhận:

```text
12c4a54 merge: integrate Person C VQA Wave 1 core
```

Trước khi bắt đầu, Người B/integration owner phải commit tài liệu này và push
`feature/online-phase-Knguyen`. Cả ba người phải tạo branch Wave 2 từ cùng commit
mới nhất của branch đó; không tiếp tục trực tiếp từ branch Wave 1 cũ của A/C.

Mỗi người chạy:

```powershell
git fetch origin
git switch feature/online-phase-Knguyen
git pull --ff-only origin feature/online-phase-Knguyen
git rev-parse HEAD
```

Ba người gửi lại cùng một `HEAD` hash trong nhóm trước khi code. Sau đó tạo
branch:

```powershell
# Người A
git switch -c feature/online-wave2-Qluan

# Người B
git switch -c feature/online-wave2-Knguyen

# Người C
git switch -c feature/online-wave2-Tngoc
```

Không copy file thủ công giữa ba máy. Mọi trao đổi code phải đi qua Git commit.

## 3. Tài liệu bắt buộc phải đọc

Đọc theo `AGENTS.md`, sau đó tập trung vào:

1. `docs/04-OFFLINE-ONLINE-CONTRACT.md`.
2. `docs/06-DESIGN-DECISIONS.md`, DD-026 đến DD-031.
3. `docs/11-ONLINE-TEAM-TASK-ASSIGNMENT.md`.
4. `docs/14-TRAKE-DANTE-VQA-CODING-ASSIGNMENT.md`.
5. `docs/15-THREE-PERSON-PARALLEL-CODING-PLAN.md`.
6. Tài liệu này.

Nguồn code Wave 1 cần đọc:

```text
online/domain/trake.py
online/domain/vqa.py
online/ports/visual_corpus.py
online/ports/evidence.py
online/ports/images.py
online/ports/vlm.py
online/trake/config.py
online/trake/dante.py
online/vqa/budget.py
online/vqa/selection.py
```

## 4. Kết quả bắt buộc của Wave 2

Wave 2 chỉ được chốt khi có đủ:

- Shared advanced fake fixture của A.
- TRAKE service fake end-to-end của B.
- VQA orchestrator fake end-to-end của C.
- DANTE optimized vẫn khớp naive oracle.
- Public policy A và internal policy B tiếp tục khớp.
- Public VQA budget A và internal budget C tiếp tục khớp.
- KIS regression không bị thay đổi.
- Output deterministic qua nhiều lần chạy.
- Không có SDK object, secret hoặc local absolute path trong public models.
- `git diff --check` pass.

Wave 2 chưa tạo public API route cho TRAKE/VQA. Routing/API thuộc Wave 3.

---

# 5A. Người A — Shared advanced fixture và protocol-conformant fakes

## 5A.1 Mục tiêu

Cung cấp một bộ dữ liệu giả duy nhất để B và C có thể kiểm tra cùng một kịch bản,
không tự tạo hai bộ fixture lệch nhau. A không code DANTE, evidence selection,
orchestrator hoặc API.

## 5A.2 Files thuộc quyền sở hữu của A

```text
online/testing/advanced_fakes.py
online/testing/__init__.py
tests/online/fixtures/__init__.py
tests/online/fixtures/advanced_modes.py
tests/online/contract/test_advanced_fakes.py
```

A không sửa:

```text
query_understanding/trake_parser.py
online/trake/
online/vqa/
online/modes/
retrieval_api/
```

## 5A.3 Advanced fixture chuẩn

Tạo một `AdvancedModesFixture` frozen dataclass hoặc tên tương đương, được trả
bởi một builder duy nhất như:

```text
build_advanced_modes_fixture()
```

Fixture cần chứa tối thiểu:

- Ba ordered TRAKE events với stable event IDs.
- Một validated `TRAKEQuery`.
- Một event-text encoder fake có mapping text → normalized vector xác định.
- Ordered visual frames/vectors cho từng video.
- Expected DANTE winner, sequence positions và score tính tay được.
- Ranked `FusedFrameCandidate` values cho VQA.
- Ordered `FrameMetadata` theo từng video.
- OCR, ASR và summary evidence records.
- Image evidence/reference mapping dùng `fixture://`, không dùng path máy thật.
- Một `VQAQuestion` và expected evidence IDs.
- Factories cho fake VLM behaviors.

Khuyến nghị dùng ít nhất bốn video để mỗi edge case độc lập:

```text
V001: >= 6 frames, correct narrative order, expected TRAKE winner.
V002: >= 6 frames, high individual similarities nhưng wrong event order.
V003: >= 6 frames, equal-score/tie-break case.
V004: T_v < N, no valid DANTE sequence.
```

Mọi `frame_id` phải canonical:

```text
{video_id}_{shot_id:05d}_{position_3_digits}
```

Mọi visual vector phải:

- finite;
- cùng dimension;
- L2-normalized;
- đủ đơn giản để expected cosine/DP score tính tay được.

## 5A.4 Fakes bắt buộc

### Fake visual corpus

Implement `VisualCorpusPort`:

- `list_video_ids()` trả deterministic order.
- `iter_ordered_frame_embedding_batches(video_id, batch_size)` tôn trọng batch
  size và giữ nguyên local order.
- Có call log thread-safe.
- Reject invalid `video_id`/`batch_size`.
- Có injectable behaviors cho unavailable, timeout và contract mismatch.
- Output đi qua `validate_ordered_visual_stream` trong test contract.

### Fake evidence hydrator

Implement `EvidenceHydrationPort`:

- OCR chỉ trả evidence của frame IDs được yêu cầu.
- ASR chỉ trả intervals của đúng video và giao requested time range.
- Summary chỉ trả selected video IDs.
- Missing evidence được biểu diễn bằng record vắng mặt, không fabricate.
- Backend error được biểu diễn bằng shared domain exception, không giả thành
  empty success.
- Có call log deterministic.

### Fake image resolver

Implement `ImageResolverPort`:

- Trả mapping `frame_id → ImageEvidence`.
- Dùng safe opaque fixture reference.
- Hỗ trợ missing image và resolver failure riêng biệt.
- Không trả local absolute path, `file://`, credential hoặc signed secret URL.

### Fake VLM

Implement `VLMPort` với các behavior modes:

- answered + grounded evidence IDs;
- insufficient evidence;
- timeout;
- unavailable;
- malformed/protocol-violating response dùng riêng để test defensive validation.

Fake không gọi network và không chứa Gemini SDK.

## 5A.5 Tests của A

Tối thiểu phải test:

1. Tất cả fake objects conform runtime Protocol.
2. Fixture deterministic qua hai lần build.
3. Canonical IDs và vector norms/dimensions hợp lệ.
4. Visual batches ghép lại đúng local order.
5. Correct-order, wrong-order, tie và `T_v < N` fixture thực sự khác nhau.
6. Evidence hydration không trả record ngoài request.
7. Empty-success khác backend failure.
8. Missing image khác resolver unavailable.
9. Fake VLM success/insufficient/timeout/malformed modes.
10. Fake call logs không chứa vector, secret hoặc local path trong public
    diagnostics.

Commands:

```powershell
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/contract/test_advanced_fakes.py -q
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/contract tests/online/adapters -q
git diff --check
```

## 5A.6 Definition of Done của A

- Không có TODO/pass placeholder.
- B và C có thể import fixture/fakes mà không import database SDK.
- Test của A pass độc lập.
- Commit chỉ chứa files A sở hữu.
- Push branch và gửi commit hash, test command/result cho B.

---

# 5B. Người B — TRAKE parser, similarity và service

## 5B.1 Mục tiêu

Ghép public TRAKE contracts của A với DANTE core của B để tạo một TRAKE service
hoàn chỉnh bằng fake corpus. Không làm VQA retrieval rewrite trong Wave 2.

## 5B.2 Files thuộc quyền sở hữu của B

```text
query_understanding/trake_parser.py
online/trake/similarity.py
online/trake/service.py
online/trake/__init__.py
tests/online/trake/test_parser.py
tests/online/trake/test_similarity.py
tests/online/trake/test_service.py
tests/online/integration/test_trake_fake_e2e.py   # thêm sau khi A merge
```

B không sửa:

```text
online/domain/
online/ports/
online/testing/advanced_fakes.py
online/vqa/
online/modes/
retrieval_api/
```

## 5B.3 TRAKE parser

Tạo builder/parser nhận:

```text
query_id
ordered event descriptions
top_k_videos
DANTE policy/config
```

Output là public `TRAKEQuery`.

Rules:

- Tối thiểu hai events.
- Strip input tại parser boundary, nhưng reject empty/whitespace.
- Reject duplicate event descriptions sau normalization.
- Sinh event IDs deterministic theo order nếu caller chưa cung cấp ID.
- Không reorder events.
- Không dùng KIS q0/q1/q2 expansion cho từng event.
- Không dùng LLM để đoán hoặc ghép event.
- Không parse một paragraph tự do thành nhiều event bằng heuristic mơ hồ.

## 5B.4 Event encoding

Dùng `TextEncoderPort`/PE-Core-compatible encoder hiện có:

```text
event texts [N]
→ encode_texts once
→ normalized event matrix U [N,D]
```

Validation bắt buộc:

- Encoder output đúng N rows và đúng input order.
- Dimension bằng `encoder.dimension`.
- Values numeric, finite, không nhận boolean.
- Mỗi row non-zero và L2-normalized trong tolerance rõ ràng.
- Không average event embeddings.
- Giữ mapping row `i` với `query.events[i].event_id`.

Nếu encoder output sai, raise `DimensionMismatchError` hoặc
`ContractMismatchError`; không pad/truncate/re-normalize âm thầm.

## 5B.5 Similarity matrix

Cho mỗi video:

```text
U:   [N,D]
E_v: [T_v,D]
S_v[i,t] = cosine(U[i], E_v[t])
```

Rules bắt buộc:

- Dùng toàn bộ ordered frames từ `VisualCorpusPort`.
- Gọi/tuân thủ `validate_ordered_visual_stream`.
- Không top-M union, threshold hoặc summary hard prefilter.
- Không đưa OCR/ASR/summary/object vào DANTE matrix.
- Không dùng Milvus internal `pk` hoặc global row index.
- Giữ ordered frames cùng matrix để backtracking hydrate đúng position.
- Dimension event/frame phải bằng nhau.
- Batch size chỉ thay cách đọc, không thay matrix/output.
- Negative cosine values được giữ nguyên, không clamp.

Tạo internal frozen result, ví dụ:

```text
VideoSimilarityMatrix
  video_id
  event_ids
  frames
  similarities
```

Đây là internal result, không thay thế public `TRAKEVideoResult`.

## 5B.6 TRAKE service

Pipeline:

```text
validate TRAKEQuery
→ encode ordered events một lần
→ enumerate video IDs
→ đọc/validate từng ordered video corpus
→ compute similarity matrix
→ solve_dante từng video
→ bỏ video T_v < N/no-valid-path
→ hydrate positions thành TRAKEFrameMatch
→ sort score DESC, video_id ASC
→ top_k_videos
→ TRAKE results + diagnostics
```

Mỗi `TRAKEFrameMatch` phải lấy từ frame ở đúng DANTE position và chứa:

- correct `event_id`;
- canonical `frame_id`/`video_id`;
- `shot_id`, `local_index`, `timestamp_sec`;
- raw event-frame cosine similarity từ matrix cell.

Service có thể trả internal `TRAKEExecution` frozen dataclass chứa:

```text
query_id
results: tuple[TRAKEVideoResult, ...]
diagnostics: TRAKEDiagnostics
```

Không tạo competing public TRAKE domain models.

## 5B.7 Concurrency, timeout và lifecycle

- Encode events đúng một lần trước khi fan-out per video.
- Per-video work chạy qua bounded executor, không tạo unbounded threads.
- Có validated `max_workers`, `batch_size`, `total_timeout_sec`.
- Output không phụ thuộc completion order.
- Timeout toàn corpus không được trả partial top-k như complete success.
- Thread đã chạy không thể bị kill; service phải có explicit `close()`/drain
  contract tương tự existing `RetrievalService`.
- Không close executor khi execution còn active.
- Contract/dimension/ID/norm error từ bất kỳ video nào là query failure; không
  silently skip để làm kết quả đẹp hơn.
- `T_v < N` là valid unreachable video, không phải infrastructure failure.

## 5B.8 Diagnostics

Điền public `TRAKEDiagnostics`:

- policy version/lambda từ request;
- event count;
- enumerated video count;
- total frame count;
- similarity latency;
- DP latency;
- invalid/unreachable sequence count;
- bounded warnings không chứa vector, query secret hoặc local path.

## 5B.9 Tests của B

### Parser/encoding

1. Hai và ba events giữ đúng order.
2. Empty/one/duplicate event reject.
3. Stable deterministic event IDs.
4. Encoder wrong row count/dimension.
5. Encoder zero norm/non-finite/boolean reject.

### Similarity

6. Matrix shape và cosine values tính tay.
7. Hai video hoàn toàn isolated.
8. Empty/short video.
9. Dimension/norm/ordered-stream mismatch surfaced.
10. Batch size không đổi output.

### Service

11. Correct-order video thắng wrong-order distractor.
12. `T_v < N` không tạo result và tăng diagnostics count.
13. Equal-score video/path tie deterministic.
14. Top-k sort score DESC, video ID ASC.
15. Match hydration đúng event/frame/local index/similarity.
16. Bounded concurrency thực sự parallel bằng event/release fake.
17. Total timeout không trả false success.
18. Executor close/drain/idempotence.
19. Port contract failure surfaced.
20. Existing naive-vs-optimized DANTE tests vẫn pass.

Commands:

```powershell
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/trake -q
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/contract tests/online/adapters tests/online/retrieval tests/online/integration tests/online/ranking tests/online/modes tests/online/trake -q
git diff --check
```

## 5B.10 Definition of Done của B

- TRAKE fake E2E pass với local builders.
- Sau khi A merge fixture, thêm đúng một shared-fixture E2E test.
- Không import database SDK.
- Không có VQA rewrite, API hoặc actual model/data call.
- Commit chỉ chứa files B sở hữu.
- Push branch và gửi commit hash, test command/result.

---

# 5C. Người C — VQA evidence adapter, VLM contract và orchestrator

## 5C.1 Mục tiêu

Ghép VQA selection core với public VQA models/ports để tạo evidence-grounded VQA
orchestrator chạy bằng ranked candidate fake và fake VLM. Không code public API
hoặc B-side retrieval rewrite trong Wave 2.

## 5C.2 Files thuộc quyền sở hữu của C

```text
online/vqa/evidence_selector.py
online/vqa/vlm_request.py
online/vqa/orchestrator.py
online/vqa/__init__.py
tests/online/vqa/test_evidence_selector.py
tests/online/vqa/test_vlm_contract.py
tests/online/vqa/test_orchestrator.py
tests/online/integration/test_vqa_fake_e2e.py   # thêm sau khi A merge
```

C không sửa:

```text
online/domain/
online/ports/
online/testing/advanced_fakes.py
query_understanding/
online/trake/
online/modes/
retrieval_api/
```

## 5C.3 Public/internal budget mapping

Tạo một explicit mapper:

```text
VQAEvidenceBudget (A public)
→ EvidenceBudgetPolicy (C internal)
```

Map từng field bằng tên rõ ràng, không dùng positional magic. Test tất cả defaults
và overrides. Không tạo thêm bộ default thứ ba.

## 5C.4 Evidence selector adapter

Input:

- validated `VQAQuestion`;
- ranked `FusedFrameCandidate` values;
- public `VQAEvidenceBudget`;
- `MetadataReaderPort`;
- `ImageResolverPort`;
- `EvidenceHydrationPort`.

Pipeline:

```text
select primary frames
→ lấy ordered metadata cho selected videos
→ select previous/next neighbors
→ resolve images
→ hydrate OCR for selected image frame IDs
→ hydrate ASR intervals quanh primary frame timestamps ±window
→ hydrate summaries for selected primary videos
→ map internal chunks sang public evidence models
→ apply deterministic caps/dedup/order
```

Output nên là internal frozen `EvidenceSelectionResult` chứa:

```text
evidence: tuple[VQAEvidence, ...]
retrieved_frame_count
selected_primary_count
selected_image_count
selected_text_count
dropped_count
missing_count
warnings
```

Rules:

- Image IDs deterministic từ canonical frame IDs.
- OCR evidence chỉ đến từ frame IDs đã chọn.
- ASR giữ interval ID/start/end; không biến thành frame.
- Summary giữ video level; không tạo video mới.
- Missing mapping là missing evidence, backend exception là failure/degradation.
- Dedup bằng stable evidence ID.
- Không vượt bất kỳ DD-030 cap nào.
- Không tạo local path hoặc `file://` reference.

## 5C.5 VLM request builder và response validator

Request builder trả validated public `VLMRequest`:

- question/answer type giữ nguyên;
- evidence order deterministic;
- evidence IDs unique;
- temperature/max tokens validated;
- không chứa answer guess;
- không chứa world knowledge, secret hoặc local path.

Giữ một evidence-only instruction constant dùng cho future real adapter:

```text
Chỉ trả lời từ evidence được cung cấp.
Không dùng external/world knowledge để điền phần thiếu.
Nếu evidence không đủ, trả insufficient_evidence.
Evidence IDs phải là subset của request.
Trả lời ngắn gọn cùng ngôn ngữ với câu hỏi.
Không trả chain-of-thought, secret hoặc local path.
```

Response validator:

- Parse/validate bằng public `VLMResponse`, không regex/free-form fallback.
- Reject non-`VLMResponse` protocol output sau bounded parse attempt.
- `answered` cần answer và evidence IDs.
- Response evidence IDs phải là subset của request evidence IDs.
- `insufficient_evidence` không được chứa fabricated answer.
- Không tự dịch hoặc tự sửa answer âm thầm.

## 5C.6 Internal candidate retrieval boundary

B VQA retrieval rewrite thuộc Wave 3. Để Wave 2 vẫn test full orchestrator, C được
tạo một narrow internal Protocol trong `orchestrator.py`, ví dụ:

```text
VQACandidateRetrievalPort
  async retrieve_candidates(question)
  → tuple[FusedFrameCandidate, ...]
```

Protocol này chỉ là seam nội bộ cho fake E2E, không phải public API/domain model.
Wave 3 sẽ có adapter nối B retrieval/ranking output vào seam này. Không duplicate
KIS retrieval/ranking algorithm trong C.

## 5C.7 VQA orchestrator

Pipeline:

```text
validate question/budget
→ candidate retrieval seam
→ evidence selector
→ build VLM request
→ call VLMPort
→ validate response grounding
→ VQAResult + diagnostics
```

Orchestrator nên async vì retrieval/VLM có thể chậm. Sync A ports/VLM fake không
được block event loop; dùng bounded executor hoặc explicit adapter boundary.

Cho phép internal `VQAExecution` chứa public `VQAResult` cùng stage latency/status
chi tiết. Không tạo competing public `VQAResult`.

Failure policy:

- Retrieval core failure: raise explicit domain error.
- Không có ranked frame: trả valid `insufficient_evidence`, không gọi VLM.
- Missing một số OCR/ASR/summary: degraded warning nếu image evidence còn đủ.
- Optional evidence backend failure: degraded có diagnostics; không fabricate.
- Resolver trả thiếu một số image: count missing, tiếp tục nếu còn image.
- Resolver exception/toàn bộ selected images unavailable: explicit no-answer hoặc
  resource failure; không gọi VLM với fake image.
- VLM timeout/unavailable: explicit shared domain error.
- Malformed/protocol-invalid VLM response: tối đa một bounded retry nếu deadline
  còn đủ, sau đó fail.
- `insufficient_evidence` từ VLM là valid result, không phải exception.

Lifecycle:

- Validated request timeout.
- Bounded executor.
- Deterministic output bất kể completion timing.
- Explicit close/drain, không close khi request còn active.

## 5C.8 Diagnostics

Public `VQADiagnostics` phải phản ánh:

- retrieved frame count;
- selected image/text counts;
- dropped/missing evidence counts;
- VLM latency và retry count;
- bounded warnings.

Internal execution diagnostics có thể thêm stage latency/retrieval status nhưng
không chứa prompt đầy đủ, answer, vector, secret hoặc local path.

## 5C.9 Tests của C

### Evidence adapter

1. Public/internal budget mapping khớp mọi field.
2. Primary diversity và neighbor selection giữ caps.
3. OCR chỉ selected frames.
4. ASR đúng video/window và giữ interval level.
5. Summary chỉ selected videos.
6. Stable IDs/dedup/order deterministic.
7. Missing optional evidence vs backend failure.
8. Không path/secret leak.

### VLM contract

9. Valid answer types/status/confidence.
10. Unknown/duplicate evidence ID reject.
11. Empty answered response reject.
12. Malformed mapping/object reject.
13. Insufficient response hợp lệ.
14. Evidence-only instruction không chứa guessed answer/secret/path.

### Orchestrator

15. Answered fake E2E.
16. Insufficient before VLM khi không có frames/evidence.
17. Optional text failure degrades nhưng không fabricate.
18. All-image resolver failure không gọi VLM.
19. VLM timeout/unavailable.
20. Malformed response retry tối đa một lần.
21. Grounding subset validation.
22. Determinism.
23. Executor close/drain/idempotence.

Commands:

```powershell
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/vqa -q
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/contract tests/online/adapters tests/online/retrieval tests/online/integration tests/online/ranking tests/online/modes tests/online/vqa -q
git diff --check
```

## 5C.10 Definition of Done của C

- VQA fake E2E pass với local builders.
- Sau khi A merge fixture, thêm đúng một shared-fixture E2E test.
- Không gọi actual LLM/VLM/network/filesystem/database.
- Không implement B retrieval rewrite hoặc public API routing.
- Commit chỉ chứa files C sở hữu.
- Push branch và gửi commit hash, test command/result.

---

# 6. Cách ba người code song song mà không lệch pha

## 6.1 Giai đoạn đầu

Cả ba code đồng thời từ cùng base commit:

- A xây shared fixture/fakes.
- B dùng local test builders để code TRAKE service.
- C dùng local test builders để code VQA orchestrator.

B/C không chờ A mới bắt đầu. Tuy nhiên B/C không tự tạo permanent shared fixture
trong source; local builders chỉ nằm trong test file của chính mình.

## 6.2 Khi A hoàn thành

1. A push branch và gửi commit hash.
2. B review A fixture/ports behavior.
3. Merge A vào integration branch trước.
4. B và C cập nhật branch từ integration branch.
5. B thêm `test_trake_fake_e2e.py` dùng shared fixture.
6. C thêm `test_vqa_fake_e2e.py` dùng shared fixture.

Không rewrite thuật toán B/C chỉ để khớp fixture sai. Nếu expected result không
khớp, tính tay lại vectors/scores và sửa đúng owner.

## 6.3 Thứ tự merge cuối Wave 2

```text
1. A shared fixture/fakes
2. B TRAKE parser/similarity/service
3. C VQA evidence adapter/VLM/orchestrator
```

Integration owner review từng branch trước merge. Không merge ba branch cùng lúc.

## 6.4 Shared-file rule

- `online/domain/*` và `online/ports/*`: frozen trong Wave 2.
- Nếu phát hiện contract thật sự thiếu, dừng task đó và báo
  `CONTRACT_MISMATCH`; không tự sửa shared contract.
- `online/testing/__init__.py`: A sở hữu.
- `online/trake/__init__.py`: B sở hữu.
- `online/vqa/__init__.py`: C sở hữu.
- Không ai sửa `online/modes/*` hoặc `retrieval_api/*` trong Wave 2.

## 6.5 Handoff message bắt buộc

Mỗi người khi push phải gửi:

```text
Role/milestone:
Base commit:
Branch:
Head commit:
Files changed:
Input contract:
Output contract:
Tests run/results:
Known limitations:
Runtime items not verified:
```

Không chỉ nhắn “đã xong” hoặc “test pass”.

---

# 7. Integration gate cuối Wave 2

Sau khi merge A → B → C, chạy từ `aic_nova_project/`:

```powershell
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/trake tests/online/vqa -q
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/contract tests/online/adapters tests/online/retrieval tests/online/integration tests/online/ranking tests/online/modes tests/online/trake tests/online/vqa -q
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online -q
git diff --check
```

Nếu full suite bị chặn vì môi trường thiếu FastAPI/runtime dependency:

- không cài dependency âm thầm;
- ghi rõ collection error;
- unaffected suite bắt buộc phải pass;
- blocker môi trường không được trình bày thành code pass.

## 7.1 TRAKE gate

- Same three ordered events giữ đúng order từ parser đến result.
- Full ordered keyframes từng video được dùng.
- Correct-order video thắng wrong-order distractor.
- `T_v < N` không có fabricated sequence.
- Tie-break và top-k deterministic.
- Không cross-video transition.
- Timeout/contract mismatch không trả false success.

## 7.2 VQA gate

- Ranked frames → budgeted images/text → VLM request → grounded response.
- Không vượt 3 videos/3 primary per video/8 primary/12 images.
- OCR/ASR/summary và combined text caps đúng.
- Evidence IDs unique và response IDs là subset request.
- Missing/degraded/insufficient/timeout phân biệt rõ.
- Không secret/path/world-knowledge fallback.

## 7.3 Regression gate

- Existing KIS tests pass không cần đổi expected behavior.
- Không sửa t-KIS/v-KIS shared pipeline.
- Không thêm actual database/model SDK dependency vào test collection.
- Working tree sạch sau commit/merge.

## 7.4 Wave 2 completion statement

Chỉ được tuyên bố Wave 2 hoàn tất theo wording:

```text
CONFIRMED_CODE: TRAKE và VQA chạy fake end-to-end, deterministic tests pass.
NEED_RUNTIME_VERIFICATION: database, image resolver, PE-Core checkpoint,
Gemini/VLM provider và dữ liệu thật chưa được kiểm tra.
```

Không dùng từ “production-ready” hoặc “competition-ready” sau Wave 2.

---

# 8. Những việc tuyệt đối không làm trong Wave 2

- Không actual Milvus/Elasticsearch/SQLite adapter cho advanced modes.
- Không download PE-Core/Gemini/Stable Diffusion model.
- Không gọi external LLM/VLM/search API.
- Không code VQA retrieval rewrite; thuộc B Wave 3.
- Không code TRAKE/VQA mode routing hoặc API; thuộc C Wave 3.
- Không thêm QUEST hoặc Stable Diffusion vào DANTE matrix.
- Không dùng summary hard-prefilter cho TRAKE.
- Không đổi DD-026 đến DD-031.
- Không hardcode vector dimension sản xuất.
- Không sửa/rewrite canonical IDs để che fixture hoặc contract lỗi.
- Không normalize/fuse DANTE với KIS branch scores.
- Không push secret, local absolute path, checkpoint hoặc data artifact lớn.

---

# 9. Đoạn giao việc có thể gửi thẳng cho Codex của từng người

## 9.1 Prompt cho Người A

```text
Bạn là Người A trong Wave 2. Đọc AGENTS.md và
docs/16-WAVE-2-THREE-PERSON-CODING-PLAN.md. Làm toàn bộ mục 5A trên branch
feature/online-wave2-Qluan từ đúng shared base commit. Chỉ tạo shared advanced
fixture và protocol-conformant fakes; không sửa TRAKE/VQA algorithms, modes hay
API. Chạy toàn bộ tests/commands của mục 5A, review diff, commit và push. Cuối
cùng gửi handoff message đúng mục 6.5.
```

## 9.2 Prompt cho Người B

```text
Bạn là Người B trong Wave 2. Đọc AGENTS.md và
docs/16-WAVE-2-THREE-PERSON-CODING-PLAN.md. Làm toàn bộ mục 5B trên branch
feature/online-wave2-Knguyen từ đúng shared base commit: TRAKE parser, event
encoding, similarity matrix và bounded TRAKE service. Dùng local test builders
trước; sau khi A fixture merge, thêm shared fake E2E. Không sửa public contracts,
VQA, modes hoặc API. Chạy tests/commands mục 5B, commit và push, rồi gửi handoff
message đúng mục 6.5.
```

## 9.3 Prompt cho Người C

```text
Bạn là Người C trong Wave 2. Đọc AGENTS.md và
docs/16-WAVE-2-THREE-PERSON-CODING-PLAN.md. Làm toàn bộ mục 5C trên branch
feature/online-wave2-Tngoc từ đúng shared base commit: public/internal budget
mapping, evidence selector adapter, VLM request/response validation và VQA
orchestrator với fake candidate retrieval/VLM. Sau khi A fixture merge, thêm
shared fake E2E. Không sửa public contracts, TRAKE, modes hoặc API. Chạy
tests/commands mục 5C, commit và push, rồi gửi handoff message đúng mục 6.5.
```
