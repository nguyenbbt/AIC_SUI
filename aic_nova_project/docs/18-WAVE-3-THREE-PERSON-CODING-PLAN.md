# Wave 3 — Kế hoạch code song song cho A, B, C

## 1. Mục tiêu và phạm vi

Wave 3 là wave code cuối của giai đoạn **fake/integration-ready** cho phase Online. Wave này nối các khối TRAKE và VQA đã hoàn thành ở Wave 2 vào query rewrite, KIS retrieval, mode routing, composition root và API nội bộ.

Sau Wave 3, hệ thống phải chạy được hai luồng fake end-to-end sau:

```text
TRAKE request
  -> TRAKE mode adapter
  -> TRAKEService/DANTE
  -> ranked VideoCandidate
  -> internal API response

VQA question
  -> evidence-query rewrite
  -> KIS text-to-keyframe retrieval/ranking
  -> evidence selection/hydration
  -> VLM port
  -> VQAResult
  -> internal API response
```

Wave 3 **không phải** bước xác nhận production với dữ liệu thật. Sau Wave 3 vẫn còn real adapters, model/index thật, benchmark hiệu năng và contract chính thức của ban tổ chức.

### Tránh nhầm số wave

`docs/13-ONLINE-NEXT-STEPS-ADVANCED-MODES.md` là kế hoạch lịch sử. Một số đầu việc trong đó từng được gọi là Wave 3 nhưng đã được đưa lên Wave 2 và hoàn thành. Từ thời điểm tài liệu này được chấp nhận, cách đánh số trong file này là nguồn phân công hiện hành.

## 2. Trạng thái nền bắt buộc trước khi code

Wave 2 code-complete tại commit:

```text
fdf7efd test(online): add Wave 2 shared fixture end-to-end gates
```

Kết quả kiểm tra trên máy tích hợp:

- `194 passed`: toàn bộ TRAKE và VQA.
- `381 passed`: contract, adapters, retrieval, integration, ranking, modes, TRAKE và VQA.
- `4 passed`: shared-fixture E2E mới của TRAKE/VQA.
- `compileall`: pass.
- Hai test API chưa collection được trên máy tích hợp vì môi trường chưa cài `fastapi`; đây là thiếu dependency của môi trường, chưa phải bằng chứng code API bị lỗi.

Tất cả thành viên phải bắt đầu Wave 3 từ **cùng một HEAD của nhánh tích hợp**, là commit chứa cả Wave 2 và tài liệu này. Không tiếp tục Wave 3 trực tiếp trên các branch Wave 1/Wave 2 cũ.

## 3. Push Wave 2 và chuẩn bị branch Wave 3

### 3.1 Người B — đẩy nhánh tích hợp

Từ thư mục `aic_nova_project`:

```powershell
git status --short --branch
git push origin feature/online-phase-Knguyen
git rev-parse HEAD
```

Người B gửi giá trị `git rev-parse HEAD` cho A và C. Đó là `WAVE3_BASE_COMMIT` duy nhất mà cả nhóm dùng.

### 3.2 Người A và C — lấy đúng code nền

Không merge nhánh tích hợp vào branch cũ. Tạo branch Wave 3 mới từ đúng remote HEAD:

```powershell
git fetch origin
git switch --detach origin/feature/online-phase-Knguyen
git rev-parse HEAD
```

Đối chiếu hash vừa in với `WAVE3_BASE_COMMIT`, sau đó tạo branch cá nhân:

Người A:

```powershell
git switch -c feature/online-wave3-Qluan
```

Người C:

```powershell
git switch -c feature/online-wave3-Tngoc
```

### 3.3 Người B — branch làm việc

Để nhánh tích hợp luôn sạch, người B cũng nên tạo branch riêng:

```powershell
git switch -c feature/online-wave3-Knguyen
```

Nếu nhóm quyết định B tiếp tục code ngay trên `feature/online-phase-Knguyen`, A và C vẫn phải lấy đúng `WAVE3_BASE_COMMIT`; tuy nhiên branch riêng cho B an toàn hơn khi review/merge.

## 4. Các quyết định đã khóa — không tự đổi contract

Các thành viên không cần tự lựa chọn lại các điểm sau:

1. TRAKE bám theo AIO_DANTE + QUEST đã mô tả trong references và các quyết định DD-026 đến DD-031.
2. KIS, v-KIS và t-KIS dùng chung text-to-keyframe retrieval pipeline. v-KIS chỉ khác ở chỗ người dùng xem video rồi tự viết query text.
3. VQA tìm evidence bằng chính KIS seven-branch retrieval/ranking; không tạo retrieval engine thứ hai.
4. VQA rewrite tạo **mô tả evidence cần tìm**, tuyệt đối không tự đoán câu trả lời.
5. DANTE chỉ tồn tại trong `TRAKEService`; mode/API không được cài lại thuật toán DANTE.
6. API Wave 3 chỉ là API nội bộ, chưa phải competition API. OQ-002 vẫn mở.
7. Route nội bộ Wave 3 được khóa là:
   - `POST /internal/unstable/trake`
   - `POST /internal/unstable/vqa`
8. Route KIS `/search` hiện tại phải giữ nguyên hành vi.
9. Không thêm LLM/VLM provider thật, database thật, image resolver thật hoặc network call vào fake test.
10. Stable Diffusion và QUEST enhancement không nằm trong Wave 3 baseline; chúng không được làm block merge.
11. Không đưa secret, prompt đầy đủ, stack trace hay raw provider error ra API response/log diagnostics.
12. Không sửa `online/domain/*` hoặc `online/ports/*` trong Wave 3. Nếu thật sự thiếu contract, báo `CONTRACT_MISMATCH` cho B trước khi sửa.

## 5. Ownership để ba người code không đè nhau

| Khu vực | Owner | Người khác |
|---|---|---|
| `online/testing/*`, advanced fake/conformance | A | Không sửa |
| `query_understanding/rewrite.py` và rewrite support | B | Không sửa |
| `online/retrieval/vqa.py` và VQA-to-KIS adapter | B | Không sửa |
| `online/modes/trake.py`, `online/modes/vqa.py` | C | Không sửa |
| `retrieval_api/advanced_models.py` | C | Không sửa |
| `retrieval_api/search_engine.py`, `retrieval_api/composition.py` | C | Không sửa |
| `online/domain/*`, `online/ports/*` | Frozen | Không ai tự sửa |
| Tài liệu phân công/decision | B tích hợp | A/C chỉ đề xuất |

Mỗi người chỉ sửa `__init__.py` trong khu vực mình sở hữu. Không chạy formatter trên toàn repository.

## 6. Contract bàn giao giữa A, B, C

### 6.1 A → C: advanced fake bundle

A cung cấp một factory/bundle duy nhất để C dùng trong composition và API tests. Bundle dùng các fake ports Wave 2 hiện có, tối thiểu gồm:

- visual corpus/event-vector fake;
- metadata/evidence hydrator fake;
- image resolver fake;
- VLM fake;
- cấu hình deterministic và safe call log;
- biến thể blocking/release để kiểm tra active-request shutdown;
- biến thể timeout/unavailable/invalid reference.

Bundle không import `retrieval_api` và không tự tạo app FastAPI. C là owner của composition root.

### 6.2 B → C: VQA candidate retriever

B cung cấp object có đúng method hiện tại:

```python
async def retrieve_candidates(
    self,
    question: VQAQuestion,
) -> tuple[FusedFrameCandidate, ...]:
    ...
```

Object phải thỏa `VQACandidateRetrievalPort` bằng structural typing. C không gọi private method, không đọc thuộc tính nội bộ và không cài lại rewrite/KIS search.

B có thể cung cấp thêm `execute(...) -> VQARetrievalExecution` để unit test diagnostics của rewrite và KIS, nhưng C chỉ phụ thuộc vào `retrieve_candidates(...)`.

### 6.3 B → C: TRAKE service

C chỉ gọi public API của `TRAKEService` đã hoàn thành trong Wave 2. C không truy cập trực tiếp DANTE optimizer, event encoder hoặc visual corpus port.

### 6.4 C → API consumer: internal-only schema

C đặt request/response model trong `retrieval_api/advanced_models.py`, không đưa model API vào `online/domain`.

TRAKE request nội bộ tối thiểu gồm:

- request/query ID;
- danh sách event text có thứ tự;
- optional event IDs;
- `top_k_videos`;
- DANTE policy/lambda chỉ khi public service hiện tại đã hỗ trợ.

VQA request nội bộ tối thiểu gồm:

- question ID;
- question text;
- answer type;
- optional evidence budget theo contract hiện có.

Response phải dùng các domain result hiện có, giữ provenance/diagnostics và mang dấu hiệu rõ ràng rằng schema là `unstable`. Không gọi đây là competition contract.

## 7. Người A — Advanced fake, conformance và lifecycle support

### A3-1. Chuẩn hóa advanced fake runtime bundle

File dự kiến:

- `online/testing/advanced_runtime.py`
- `online/testing/__init__.py`

Yêu cầu:

- Dùng lại fake ports Wave 2, không nhân bản interface/domain model.
- Factory nhận dữ liệu cấu hình rõ ràng và luôn cho kết quả deterministic.
- Mỗi fake lưu call log an toàn để test thứ tự và số lần gọi.
- Object/sequence trả về phải immutable hoặc copy defensively.
- Cho phép cấu hình success, empty, timeout, unavailable và invalid-reference.
- Không đọc file thật, không mở DB, không gọi network/model.

### A3-2. Blocking/release fakes cho lifecycle

Thêm fake có thể:

1. báo hiệu request đã bắt đầu;
2. chặn request ở giữa bằng `asyncio.Event`;
3. cho test ra lệnh release;
4. ghi nhận resource đã bị close hay chưa.

Mục tiêu là C có thể chứng minh shutdown không close resource trong lúc request còn hoạt động.

### A3-3. Reusable conformance tests

File dự kiến:

- `tests/online/contract/test_advanced_runtime_conformance.py`
- `tests/online/contract/test_advanced_lifecycle_fakes.py`

Test tối thiểu:

- fake thỏa runtime-checkable protocol tương ứng;
- ID, provenance và reference không bị đổi qua port boundary;
- success/empty/timeout/unavailable là các trạng thái phân biệt;
- call log không chứa secret/raw payload nhạy cảm;
- cùng input cho cùng output;
- blocking fake không bị close trước release;
- invalid evidence/image reference fail-safe, không đọc tùy tiện từ filesystem.

### A3-4. Bàn giao cho C

A ghi trong commit message hoặc handoff:

- factory import path;
- tên bundle/class;
- cách cấu hình happy path và timeout;
- cách dùng start/release event;
- test command đã chạy.

### A3 Definition of Done

- Không có real adapter/network call.
- Không sửa domain/ports hoặc composition/API.
- Tất cả test A3 pass.
- Các test Wave 1/Wave 2 liên quan vẫn pass.
- Commit nhỏ, rõ ràng, push lên `feature/online-wave3-Qluan`.

## 8. Người B — Query rewrite và VQA retrieval adapter

### B3-1. Query rewrite core

File dự kiến:

- `query_understanding/rewrite.py`
- `query_understanding/__init__.py`
- `tests/online/retrieval/test_query_rewrite.py`

Thiết kế bằng frozen dataclass/protocol nội bộ, tối thiểu có:

- rewrite purpose: KIS hoặc VQA evidence;
- original text;
- primary rewrite;
- optional paraphrases q1/q2;
- status: success hoặc degraded/no-op;
- bounded warnings/diagnostics;
- optional provider/model/prompt version dạng định danh an toàn.

Quy tắc KIS rewrite:

- q0 luôn là query gốc;
- q1/q2 là structured paraphrase;
- trim whitespace;
- bỏ chuỗi rỗng;
- deduplicate nhưng giữ thứ tự;
- không để q1/q2 trùng q0;
- timeout/provider failure trả q0-only và đánh dấu degraded.

Quy tắc VQA rewrite:

- primary output là mô tả evidence hình ảnh cần tìm;
- không sinh đáp án;
- không thêm fact không có trong question;
- q1/q2 chỉ là biến thể retrieval;
- khi rewriter không sẵn sàng, dùng question gốc làm fallback retrieval text và đánh dấu degraded.

Trong Wave 3 chỉ cần `NoOpQueryRewriter` và implementation deterministic/mapping cho test. Không tích hợp LLM provider thật.

### B3-2. VQA → KIS retrieval adapter

File dự kiến:

- `online/retrieval/vqa.py`
- `online/retrieval/__init__.py`
- `tests/online/retrieval/test_vqa_retrieval.py`

Luồng bắt buộc:

```text
VQAQuestion
  -> evidence rewrite
  -> QueryBundle mode KIS_TEXT
  -> KISSearchOrchestrator.search(...)
  -> KISSearchResult.candidates
  -> tuple[FusedFrameCandidate, ...]
```

Yêu cầu:

- Reuse `KISQueryBuilder`/KIS seven-branch retrieval hiện có.
- Không tự gọi từng OCR/ASR/summary/visual branch.
- Không tự fuse/rank lần thứ hai.
- Không hydrate evidence và không gọi VLM; đó là trách nhiệm C orchestrator.
- Query ID phải deterministic và truy vết được từ VQA question ID.
- Timeout/failure rewrite không làm mất baseline retrieval.
- Timeout/failure KIS phải tuân theo shared error policy hiện có.
- Method `retrieve_candidates` phải đúng contract ở mục 6.2.

Có thể thêm `VQARetrievalExecution` nội bộ để unit test:

- rewrite status/warnings;
- query variants đã tạo;
- KIS diagnostics;
- ranked candidates.

Không lưu diagnostics trong mutable `last_result`, vì object có thể phục vụ nhiều request đồng thời.

### B3-3. Integration test của phần B

File dự kiến:

- `tests/online/integration/test_vqa_retrieval_handoff.py`

Test tối thiểu:

- question → fake rewrite → real KIS orchestration với fake indexes → fused candidates;
- no-op/degraded rewrite vẫn chạy q0;
- q1/q2 được trim/dedup đúng;
- empty candidates hợp lệ;
- concurrent requests không lẫn query/diagnostics;
- method thỏa `VQACandidateRetrievalPort` hiện có;
- không gọi evidence hydrator/VLM.

### B3 Definition of Done

- KIS q0 invariant được giữ nguyên.
- VQA chỉ rewrite để tìm evidence, không answer.
- Adapter trả đúng ranked `FusedFrameCandidate` cho C.
- Không sửa VQA orchestrator, mode/API, domain/ports hoặc A fakes.
- Tất cả test B3 và regression Wave 1/Wave 2 pass.
- Push lên `feature/online-wave3-Knguyen`.

## 9. Người C — Mode adapters, internal API và composition

### C3-1. TRAKE mode adapter

File dự kiến:

- `online/modes/trake.py`
- `tests/online/modes/test_trake.py`

Yêu cầu:

- Validate request ở mode boundary.
- Giữ nguyên thứ tự event.
- Gọi đúng một lần vào `TRAKEService`.
- Không cài DANTE/QUEST optimizer lần nữa.
- Giữ video IDs, sequence, score breakdown, provenance và diagnostics.
- Map timeout/unavailable/invalid input sang lỗi typed, không rò raw exception.
- Cho phép concurrent requests, không mutable request-global state.

### C3-2. VQA mode adapter

File dự kiến:

- `online/modes/vqa.py`
- `tests/online/modes/test_vqa.py`

Yêu cầu:

- Validate VQA question/request.
- Gọi `VQAOrchestrator`; không tự retrieval, selection hoặc VLM call.
- Orchestrator được inject candidate retriever của B qua `VQACandidateRetrievalPort`.
- Giữ answer, evidence refs, confidence và diagnostics.
- `INSUFFICIENT_EVIDENCE` là kết quả domain hợp lệ, không biến thành HTTP 500.
- VLM unavailable/timeout được xử lý theo policy hiện có.

Trong khi B chưa merge, C dùng local test double thỏa method ở mục 6.2. Không tạo implementation retrieval thứ hai.

### C3-3. Internal unstable API models và routes

File dự kiến:

- `retrieval_api/advanced_models.py`
- `retrieval_api/search_engine.py`
- `tests/online/api/test_advanced_routes.py`

Routes khóa cứng:

```text
POST /internal/unstable/trake
POST /internal/unstable/vqa
```

Yêu cầu:

- Existing `/search` không đổi schema/hành vi.
- Route selection explicit; không dùng LLM để đoán mode.
- Request/response models là internal-only và nằm ngoài `online/domain`.
- Invalid request: 422.
- Service unavailable/disabled: 503.
- Timeout: 504.
- Unexpected internal failure: sanitized 500.
- Không trả stack trace, filesystem path, provider payload hoặc secret.
- Response giữ request ID/query ID để trace.
- OpenAPI/description phải ghi `unstable internal API`, không ghi competition-ready.

### C3-4. Composition root và readiness/lifecycle

File dự kiến:

- `retrieval_api/composition.py`
- `tests/online/api/test_advanced_composition.py`

Yêu cầu:

- Wire A advanced fake bundle, B `TRAKEService`, B VQA candidate retriever và C orchestrators/modes.
- Advanced routes chỉ enabled khi dependency tương ứng được inject đầy đủ.
- Không âm thầm tạo DB/model/image resolver thật.
- Readiness phân biệt KIS, TRAKE và VQA; VQA enabled thì phải có VLM readiness probe phù hợp.
- Không health-check VLM bằng cách gửi dummy evidence/provider request.
- Shutdown ngừng nhận request mới, chờ request đang chạy, sau đó close theo dependency order.
- VQA/mode resources đóng trước KIS retrieval; service đóng trước shared executor/index resource.
- Close idempotent.
- Dùng blocking/release fake của A để test không close resource giữa active request.

### C3-5. Shared fake E2E

File dự kiến:

- `tests/online/integration/test_trake_mode_api_e2e.py`
- `tests/online/integration/test_vqa_rewrite_api_e2e.py`

Test TRAKE:

```text
internal request -> route -> mode -> TRAKEService/DANTE -> response
```

Test VQA sau khi B merge:

```text
internal question
  -> route/mode
  -> VQAOrchestrator
  -> B rewrite + KIS retrieval/ranking
  -> evidence hydration/selection
  -> fake VLM
  -> internal response
```

Test thêm:

- rewrite degraded nhưng baseline VQA retrieval vẫn chạy;
- empty/insufficient evidence;
- VLM unavailable/timeout;
- timeout mapping 504;
- sanitized unexpected error;
- concurrent requests không lẫn ID/diagnostics;
- shutdown trong lúc một request đang bị block.

### C3 Definition of Done

- Hai internal routes chạy fake E2E.
- `/search` regression pass.
- Không duplicate DANTE hoặc KIS retrieval.
- Không real provider/database/image access.
- Readiness, error mapping và shutdown có test.
- Sau khi A/B đã merge vào nhánh tích hợp, C rebase/merge nền mới và chạy lại shared E2E trước khi bàn giao final commit.
- Push lên `feature/online-wave3-Tngoc`.

## 10. Cách làm song song và điểm đồng bộ

### Giai đoạn W3.1 — làm song song hoàn toàn

- A làm advanced fake/conformance.
- B làm rewrite và VQA-to-KIS adapter.
- C làm mode adapters, internal models/routes và unit test bằng local test doubles.

Trong giai đoạn này không ai chờ ai, miễn là tuân thủ contract mục 6.

### Giai đoạn W3.2 — integration ngắn

Thứ tự merge đề nghị:

1. Review và merge A.
2. Review và merge B.
3. C lấy nhánh tích hợp mới, thay local test double bằng dependency thật của A/B trong composition/E2E.
4. Review và merge C.
5. B chạy full gate trên nhánh tích hợp.

C chỉ phải làm integration patch nhỏ; không được viết lại phần A/B.

### Quy tắc báo lỗi khi handoff

- Sai hành vi trong file owner: `IMPLEMENTATION_BUG`.
- Contract tài liệu khác code frozen: `CONTRACT_MISMATCH`.
- Thiếu dependency như `fastapi`: `ENVIRONMENT_BLOCKER`.
- Thiếu data/model/image thật: `NEED_RUNTIME_VERIFICATION`.

Không sửa lén domain/ports để làm test pass.

## 11. Test gate bắt buộc trước khi công bố Wave 3 hoàn tất

Từ root `aic_nova_project`:

```powershell
python -m compileall online query_understanding retrieval_api tests/online
python -m pytest -q tests/online/contract
python -m pytest -q tests/online/retrieval
python -m pytest -q tests/online/modes
python -m pytest -q tests/online/trake tests/online/vqa
python -m pytest -q tests/online/integration
python -m pytest -q tests/online/api
```

Sau đó chạy regression tổng:

```powershell
python -m pytest -q tests/online
```

Điều kiện pass:

- Không collection error.
- Không skipped test cho critical fake E2E.
- Không warning do coroutine/resource chưa close.
- Không network/model/database access.
- KIS `/search` vẫn pass.
- TRAKE và VQA internal routes pass.
- Concurrent/lifecycle/error-sanitization tests pass.

Máy chạy final gate phải cài đủ test dependencies, bao gồm FastAPI. Không được kết luận Wave 3 hoàn tất nếu API tests chưa collection được. Nếu dependency thiếu, ghi `ENVIRONMENT_BLOCKER`, cài theo requirements của repository trong môi trường phù hợp rồi chạy lại; không sửa test để né import.

## 12. Những gì có sau Wave 3

Nếu toàn bộ gate trên pass, nhóm có:

- KIS text/v-KIS/t-KIS dùng chung retrieval pipeline;
- optional KIS query rewrite với q0-safe degradation;
- VQA evidence-query rewrite;
- VQA reuse KIS seven-branch retrieval/ranking;
- TRAKE/DANTE service nối vào mode và internal route;
- VQA retrieval/evidence/VLM orchestration nối vào internal route;
- fake composition, readiness, typed/sanitized errors và graceful shutdown;
- deterministic fake E2E đủ để thay adapters thật về sau mà không viết lại core orchestration.

## 13. Những gì vẫn còn sau Wave 3

Các mục này không được báo là đã hoàn thành chỉ vì fake E2E pass:

- competition/public API contract — OQ-002;
- mapping database/index thật của Offline;
- checkpoint/encoder/index compatibility với artifact thật;
- actual image-path resolution — OQ-012;
- LLM rewrite provider thật và prompt/runtime calibration;
- VLM/Gemini provider thật;
- dữ liệu, ảnh, video, metadata và evidence thật;
- benchmark latency, memory, concurrency và timeout trên máy thi;
- retrieval quality evaluation và tuning;
- Stable Diffusion/QUEST enhancement nếu nhóm quyết định bật;
- submission/competition integration test.

Các mục trên phải được đánh dấu `NEED_RUNTIME_VERIFICATION` hoặc open question tương ứng, không được dùng fake test làm bằng chứng production-ready.

## 14. Checklist báo cáo của mỗi người khi push

Mỗi người gửi cho B đúng mẫu sau:

```text
Branch:
Base commit:
Final commit:
Files changed:
Public/internal interfaces added:
Commands run:
Test results:
Known limitations:
CONTRACT_MISMATCH (nếu có):
ENVIRONMENT_BLOCKER (nếu có):
NEED_RUNTIME_VERIFICATION:
```

Không chỉ nhắn “đã xong”. B chỉ review/merge khi có commit hash và test evidence.
