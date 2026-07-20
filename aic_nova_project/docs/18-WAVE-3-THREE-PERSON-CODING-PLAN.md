# Wave 3 — Phân công chi tiết cho A, B, C

## 1. Mục tiêu chung

Wave 3 là wave code cuối của giai đoạn **fake/integration-ready** cho phase Online. Mục tiêu là nối các khối TRAKE và VQA đã hoàn thành trong Wave 2 vào query rewrite, KIS retrieval, mode routing, composition root và API nội bộ.

Khi Wave 3 hoàn tất, hệ thống phải chạy được hai luồng fake end-to-end:

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

Wave 3 chưa bao gồm database/index/model/provider thật. Các phần đó chỉ được kiểm chứng khi có artifact và dữ liệu runtime thật.

## 2. Các quyết định kỹ thuật đã khóa

Ba người cùng tuân thủ các quyết định sau, không tự tạo contract hoặc pipeline khác:

1. TRAKE bám theo AIO_DANTE + QUEST và các quyết định DD-026 đến DD-031.
2. DANTE chỉ nằm trong `TRAKEService`; mode, API và composition không cài lại thuật toán DANTE.
3. VQA sử dụng lại KIS seven-branch retrieval/ranking để tìm evidence; không tạo retrieval engine riêng.
4. VQA rewrite chỉ tạo mô tả evidence hình ảnh cần tìm, không được tự đoán câu trả lời.
5. KIS, t-KIS và v-KIS dùng chung text-to-keyframe retrieval pipeline. v-KIS chỉ khác ở việc người dùng xem video rồi tự viết query text.
6. KIS rewrite luôn giữ q0 là query gốc. q1/q2 chỉ là structured paraphrases.
7. Rewrite timeout hoặc provider failure phải degraded về q0-only, không làm hỏng baseline retrieval.
8. API trong Wave 3 chỉ là API nội bộ, vì competition/public schema OQ-002 vẫn chưa được chốt.
9. Hai route nội bộ được thống nhất là:
   - `POST /internal/unstable/trake`
   - `POST /internal/unstable/vqa`
10. Route KIS `/search` hiện tại phải giữ nguyên schema và hành vi.
11. Không tích hợp LLM/VLM provider thật, DB thật, image resolver thật hoặc network call vào fake tests.
12. Stable Diffusion và QUEST enhancement không thuộc Wave 3 baseline và không được làm block hoàn thành Wave 3.
13. Không sửa `online/domain/*` hoặc `online/ports/*`. Nếu contract frozen thật sự không đủ, báo `CONTRACT_MISMATCH` trước khi thay đổi.
14. Không trả secret, raw provider payload, filesystem path, stack trace hoặc prompt đầy đủ qua diagnostics/API.

## 3. Ranh giới ownership

| Khu vực | Owner | Quy tắc |
|---|---|---|
| `online/testing/*` | A | B và C không sửa |
| Advanced fake/conformance tests | A | B và C chỉ sử dụng |
| `query_understanding/rewrite.py` | B | A và C không sửa |
| `online/retrieval/vqa.py` | B | A và C không sửa |
| Query rewrite/VQA retrieval tests | B | A và C không sửa |
| `online/modes/trake.py` | C | A và B không sửa |
| `online/modes/vqa.py` | C | A và B không sửa |
| `retrieval_api/advanced_models.py` | C | A và B không sửa |
| `retrieval_api/search_engine.py` | C | A và B không sửa |
| `retrieval_api/composition.py` | C | A và B không sửa |
| Advanced mode/API/composition tests | C | A và B không sửa |
| `online/domain/*`, `online/ports/*` | Frozen | Không ai tự sửa |

Không chạy formatter hoặc mechanical rewrite trên toàn repository. Mỗi người chỉ sửa `__init__.py` trong package mình sở hữu.

---

# 4. Người A — Advanced fakes, conformance và lifecycle support

## A3-1. Tạo advanced fake runtime bundle

File dự kiến:

- `online/testing/advanced_runtime.py`
- `online/testing/__init__.py`

A phải tạo một factory/bundle thống nhất để C dùng trong composition và API tests. Bundle dùng lại fake ports đã có từ Wave 2, tối thiểu gồm:

- visual corpus/event-vector fake cho TRAKE;
- metadata/evidence hydrator fake;
- image resolver fake;
- VLM fake;
- deterministic configuration;
- safe call logs;
- timeout/unavailable/empty/invalid-reference behavior.

Yêu cầu chi tiết:

- Không nhân bản domain model hoặc port interface đã có.
- Cùng input và configuration phải cho cùng output.
- Factory nhận dependency/configuration rõ ràng, không dùng global mutable state.
- Sequence/object trả ra phải immutable hoặc được copy defensively.
- Call log phải ghi đủ method, request ID và thứ tự gọi nhưng không chứa secret/raw provider payload.
- Có thể cấu hình riêng từng trạng thái: success, empty, timeout, unavailable và invalid reference.
- Không đọc DB, không gọi model/network và không resolve đường dẫn ảnh thật.
- Bundle không import `retrieval_api` và không tạo FastAPI app; composition thuộc C.

## A3-2. Tạo blocking/release fakes cho lifecycle test

A bổ sung fake hỗ trợ kiểm tra shutdown trong lúc request đang chạy.

Fake phải có khả năng:

1. báo hiệu request đã bắt đầu;
2. chặn execution bằng `asyncio.Event`;
3. cho test chủ động release request;
4. ghi nhận thời điểm `close()` được gọi;
5. phát hiện nếu resource bị close trước khi active request kết thúc;
6. hỗ trợ close idempotent.

Các fake này phải sử dụng được cho TRAKE và VQA composition tests mà không phụ thuộc trực tiếp vào code API của C.

## A3-3. Viết reusable conformance tests

File dự kiến:

- `tests/online/contract/test_advanced_runtime_conformance.py`
- `tests/online/contract/test_advanced_lifecycle_fakes.py`

Test bắt buộc:

- mỗi fake thỏa runtime-checkable protocol tương ứng;
- input ID, video ID, frame ID, evidence ID và provenance không bị đổi qua port boundary;
- success, empty, timeout và unavailable là các trạng thái phân biệt;
- invalid image/evidence reference fail-safe;
- fake không tùy tiện đọc filesystem;
- cùng input cho kết quả deterministic;
- concurrent calls không lẫn call log hoặc response;
- blocking fake không bị close trước release;
- gọi close nhiều lần không gây lỗi hoặc double-release;
- call log và lỗi không làm lộ dữ liệu nhạy cảm.

## A3-4. Contract A bàn giao cho C

A phải export rõ:

- tên advanced bundle;
- factory tạo happy-path bundle;
- cách cấu hình timeout/unavailable;
- cách lấy start/release event;
- cách kiểm tra call log;
- close/lifecycle behavior.

C chỉ được sử dụng public factory/export này, không truy cập private field của fake.

## A3-5. Điều kiện hoàn thành của A

- Advanced bundle dùng lại đúng ports và models của Wave 2.
- Không có network, DB, provider hoặc image-path access thật.
- Có deterministic behavior và safe logs.
- Có blocking/release support cho lifecycle tests.
- Conformance, concurrency và lifecycle tests pass.
- Regression tests của Wave 1/Wave 2 liên quan vẫn pass.
- Không sửa domain/ports, query rewrite, mode, API hoặc composition.

---

# 5. Người B — Query rewrite và VQA-to-KIS retrieval

## B3-1. Xây dựng query rewrite core

File dự kiến:

- `query_understanding/rewrite.py`
- `query_understanding/__init__.py`
- `tests/online/retrieval/test_query_rewrite.py`

B xây dựng rewrite contract nội bộ bằng frozen dataclass/protocol, tối thiểu biểu diễn được:

- rewrite purpose: KIS hoặc VQA evidence;
- original text;
- primary rewrite;
- optional q1/q2 variants;
- rewrite status: success hoặc degraded/no-op;
- bounded warnings/diagnostics;
- optional provider/model/prompt version dưới dạng định danh an toàn.

Không đưa rewrite models vào `online/domain`, vì đây chưa phải shared/public contract.

### Quy tắc KIS rewrite

- q0 luôn là query gốc.
- q1/q2 là structured paraphrases.
- Trim whitespace.
- Loại bỏ chuỗi rỗng.
- Deduplicate nhưng giữ nguyên thứ tự.
- Loại q1/q2 nếu trùng q0.
- Giới hạn số variants đúng policy, không để provider trả danh sách vô hạn.
- Timeout/provider unavailable/invalid output phải trả q0-only.
- Degraded fallback phải có bounded warning, không chứa raw exception/prompt/secret.

### Quy tắc VQA evidence rewrite

- Primary rewrite mô tả loại evidence hình ảnh cần tìm.
- Không sinh đáp án và không thêm fact không có trong question.
- q1/q2 chỉ là retrieval variants.
- Nếu rewriter không hoạt động, dùng question gốc làm retrieval text fallback.
- Fallback phải đánh dấu degraded nhưng vẫn cho phép KIS retrieval chạy.

### Implementations cần có trong Wave 3

- `NoOpQueryRewriter` cho baseline/fallback.
- Deterministic hoặc mapping rewriter để fake integration test.
- Không tích hợp LLM provider thật trong Wave 3.

## B3-2. Xây dựng VQA-to-KIS retrieval adapter

File dự kiến:

- `online/retrieval/vqa.py`
- `online/retrieval/__init__.py`
- `tests/online/retrieval/test_vqa_retrieval.py`

Luồng bắt buộc:

```text
VQAQuestion
  -> evidence-query rewrite
  -> KIS QueryBundle
  -> KISSearchOrchestrator.search(...)
  -> KISSearchResult.candidates
  -> tuple[FusedFrameCandidate, ...]
```

Adapter phải:

- dùng lại `KISQueryBuilder` và KIS seven-branch retrieval/ranking hiện có;
- tạo `QueryBundle` với mode KIS phù hợp;
- tạo query ID deterministic và truy vết được từ VQA question ID;
- gọi đúng một lần vào public KIS search orchestration;
- trả ranked `FusedFrameCandidate` theo đúng thứ tự KIS result;
- không tự gọi riêng OCR/ASR/summary/visual branches;
- không fuse/rank lần thứ hai;
- không hydrate evidence;
- không gọi VLM;
- không lưu kết quả request vào mutable `last_result`;
- an toàn khi có nhiều request đồng thời.

Public handoff cho C phải thỏa structural contract hiện tại:

```python
async def retrieve_candidates(
    self,
    question: VQAQuestion,
) -> Sequence[FusedFrameCandidate]:
    ...
```

B có thể trả tuple để bảo đảm bất biến.

## B3-3. Cung cấp execution diagnostics nội bộ

B có thể bổ sung `VQARetrievalExecution` để unit/integration test quan sát:

- rewrite status;
- bounded rewrite warnings;
- query variants sau normalization;
- KIS diagnostics;
- ranked candidates.

Nếu có `execute(...)`, `retrieve_candidates(...)` chỉ là public compatibility method lấy candidates từ execution result. C không được phụ thuộc vào private diagnostics structure này.

Diagnostics phải được trả theo từng request, không lưu trong shared mutable state.

## B3-4. Viết integration tests của phần B

File dự kiến:

- `tests/online/integration/test_vqa_retrieval_handoff.py`

Test bắt buộc:

- VQA question → deterministic rewrite → real KIS orchestration với fake indexes → fused candidates;
- KIS q0 luôn giữ query gốc;
- VQA primary rewrite chỉ mô tả evidence;
- q1/q2 được trim, loại rỗng và deduplicate;
- no-op/degraded rewrite vẫn chạy baseline retrieval;
- rewrite timeout không làm hỏng q0 retrieval;
- empty candidates là kết quả hợp lệ;
- KIS timeout/unavailable tuân theo typed error policy hiện có;
- concurrent requests không lẫn query ID, variants hoặc diagnostics;
- returned candidates đúng thứ tự KIS ranking;
- adapter thỏa `VQACandidateRetrievalPort`;
- adapter không gọi evidence hydrator hoặc VLM.

## B3-5. Điều kiện hoàn thành của B

- KIS q0 invariant được giữ nguyên.
- VQA rewrite chỉ phục vụ tìm evidence, không answer.
- Rewrite failure degraded an toàn về baseline.
- VQA adapter tái sử dụng đúng KIS retrieval/ranking.
- Adapter trả đúng candidate contract cho C.
- Unit, integration, concurrency và regression tests pass.
- Không sửa VQA orchestrator, A fakes, C mode/API/composition hoặc shared domain/ports.

---

# 6. Người C — TRAKE/VQA modes, internal API và composition

## C3-1. Xây dựng TRAKE mode adapter

File dự kiến:

- `online/modes/trake.py`
- `online/modes/__init__.py`
- `tests/online/modes/test_trake.py`

TRAKE mode adapter phải:

- validate request ở mode boundary;
- giữ nguyên thứ tự event;
- chuyển request thành `TRAKEQuery` hiện có;
- gọi đúng một lần vào `TRAKEService`;
- không truy cập trực tiếp DANTE optimizer/event encoder/visual corpus;
- không cài lại DANTE hoặc QUEST logic;
- giữ video ID, ordered sequence, score breakdown, provenance và diagnostics;
- không âm thầm đổi policy/lambda/top-k;
- map invalid input, timeout và unavailable thành typed/sanitized errors;
- không có mutable request-global state;
- hỗ trợ concurrent requests.

## C3-2. Xây dựng VQA mode adapter

File dự kiến:

- `online/modes/vqa.py`
- `online/modes/__init__.py`
- `tests/online/modes/test_vqa.py`

VQA mode adapter phải:

- validate question và answer type;
- gọi `VQAOrchestrator` hiện có;
- inject candidate retriever của B qua `VQACandidateRetrievalPort`;
- không tự rewrite query;
- không tự chạy KIS retrieval/ranking;
- không tự selection/hydration evidence;
- không gọi VLM ngoài orchestrator;
- giữ answer, evidence references, confidence và diagnostics;
- coi `INSUFFICIENT_EVIDENCE` là domain result hợp lệ, không biến thành internal error;
- map VLM timeout/unavailable theo error policy hiện có;
- hỗ trợ concurrent requests mà không lẫn ID/diagnostics.

Trong lúc B chưa bàn giao adapter thật, C dùng local test double thỏa đúng method `retrieve_candidates(...)`. C không tạo retrieval implementation thay thế.

## C3-3. Tạo internal unstable API models

File dự kiến:

- `retrieval_api/advanced_models.py`

Models chỉ dành cho API nội bộ và không được đặt vào `online/domain`.

TRAKE request model tối thiểu gồm:

- request/query ID;
- ordered event texts;
- optional event IDs nếu contract hiện có hỗ trợ;
- `top_k_videos`;
- DANTE configuration chỉ khi public service contract đã hỗ trợ.

VQA request model tối thiểu gồm:

- question ID;
- question text;
- answer type;
- optional evidence budget theo domain contract hiện có.

Response models phải:

- dùng domain results hiện có;
- giữ request/query ID;
- giữ provenance và diagnostics;
- thể hiện rõ schema là `unstable`/internal;
- không được gọi là competition-ready contract.

## C3-4. Thêm TRAKE và VQA internal routes

File dự kiến:

- `retrieval_api/search_engine.py`
- `tests/online/api/test_advanced_routes.py`

Routes bắt buộc:

```text
POST /internal/unstable/trake
POST /internal/unstable/vqa
```

Yêu cầu:

- Existing `/search` không đổi schema hoặc hành vi.
- Mode routing explicit; không dùng LLM để đoán mode.
- TRAKE route chỉ gọi TRAKE mode adapter.
- VQA route chỉ gọi VQA mode adapter.
- Invalid request trả 422.
- Disabled/unavailable dependency trả 503.
- Timeout trả 504.
- Unexpected internal failure trả sanitized 500.
- `INSUFFICIENT_EVIDENCE` vẫn là successful domain response.
- Không trả raw exception, stack trace, provider payload, secret hoặc filesystem path.
- Response giữ request ID/query ID để truy vết.
- OpenAPI description ghi rõ đây là unstable internal API.

## C3-5. Mở rộng composition root

File dự kiến:

- `retrieval_api/composition.py`
- `tests/online/api/test_advanced_composition.py`

Composition phải wire:

- advanced fake bundle của A;
- `TRAKEService` của Wave 2;
- VQA candidate retriever của B;
- `EvidenceSelector`/evidence hydration flow hiện có;
- VLM port;
- `VQAOrchestrator`;
- TRAKE mode adapter;
- VQA mode adapter;
- internal routes.

Quy tắc composition:

- dependency được inject rõ ràng;
- không âm thầm tạo DB/index/model/image resolver thật;
- advanced route chỉ enabled khi đủ dependency tương ứng;
- thiếu dependency phải thành readiness/503 rõ ràng;
- không duplicate service/orchestrator trong mỗi request;
- không import test module từ production code;
- không tạo circular dependency giữa modes, API và testing.

## C3-6. Readiness và lifecycle

Readiness phải phân biệt:

- KIS readiness;
- TRAKE readiness;
- VQA readiness;
- enabled/disabled/unavailable trạng thái từng mode.

Nếu VQA enabled, composition phải có VLM readiness probe phù hợp. Không kiểm tra VLM bằng cách gửi dummy image/evidence request đến provider.

Shutdown phải:

1. ngừng nhận request mới;
2. chờ active request kết thúc;
3. close VQA/mode resources;
4. close TRAKE/KIS services;
5. close shared executors/index resources theo dependency order;
6. hỗ trợ gọi close nhiều lần.

C dùng blocking/release fake của A để chứng minh resource không bị close khi request vẫn đang chạy.

## C3-7. Viết shared fake end-to-end tests

File dự kiến:

- `tests/online/integration/test_trake_mode_api_e2e.py`
- `tests/online/integration/test_vqa_rewrite_api_e2e.py`

TRAKE E2E phải kiểm tra:

```text
internal request
  -> TRAKE route
  -> TRAKE mode
  -> TRAKEService/DANTE
  -> ranked video results
  -> internal response
```

VQA E2E phải kiểm tra:

```text
internal question
  -> VQA route/mode
  -> VQAOrchestrator
  -> B evidence rewrite
  -> KIS seven-branch retrieval/ranking
  -> evidence selection/hydration
  -> fake VLM
  -> VQAResult
  -> internal response
```

Test cases bắt buộc:

- TRAKE happy path và ordered events;
- TRAKE empty results;
- TRAKE timeout/unavailable;
- VQA happy path;
- VQA degraded/no-op rewrite nhưng baseline retrieval vẫn chạy;
- VQA empty/insufficient evidence;
- VLM unavailable và timeout;
- invalid request → 422;
- disabled dependency → 503;
- timeout → 504;
- unexpected error được sanitize;
- concurrent requests không lẫn IDs/diagnostics;
- shutdown khi một request đang bị block;
- existing KIS `/search` vẫn hoạt động như trước.

## C3-8. Điều kiện hoàn thành của C

- TRAKE và VQA internal routes chạy fake E2E.
- VQA E2E sử dụng adapter thật của B, không dùng candidate list dựng sẵn ở final test.
- Composition sử dụng advanced fake bundle public của A.
- Không duplicate DANTE hoặc KIS retrieval.
- Error mapping, readiness, concurrency và graceful shutdown có test.
- Không có real provider, DB, model hoặc image access.
- `/search` regression pass.
- Không sửa shared domain/ports hoặc code thuộc ownership A/B.

---

# 7. Phần tích hợp chung của cả ba người

## 7.1 Handoff A → C

C phải dùng factory/bundle public của A cho composition tests. Nếu C phải đọc private field hoặc tự dựng lại fake port, handoff A chưa hoàn tất.

## 7.2 Handoff B → C

C phải inject object của B thỏa `VQACandidateRetrievalPort`. Final VQA E2E phải đi qua rewrite và real KIS orchestration với fake indexes, không được bắt đầu từ danh sách ranked candidates dựng sẵn.

## 7.3 Contract mismatch handling

- Sai hành vi trong file owner: `IMPLEMENTATION_BUG`.
- Tài liệu khác frozen code contract: `CONTRACT_MISMATCH`.
- Thiếu dependency test: `ENVIRONMENT_BLOCKER`.
- Thiếu data/model/provider thật: `NEED_RUNTIME_VERIFICATION`.

Không tự sửa domain/ports hoặc nới validation để né lỗi.

## 7.4 Full test gate

Sau khi ghép A, B và C, toàn bộ các lệnh sau phải pass trong môi trường có đủ test dependencies:

```powershell
python -m compileall online query_understanding retrieval_api tests/online
python -m pytest -q tests/online/contract
python -m pytest -q tests/online/retrieval
python -m pytest -q tests/online/modes
python -m pytest -q tests/online/trake tests/online/vqa
python -m pytest -q tests/online/integration
python -m pytest -q tests/online/api
python -m pytest -q tests/online
```

Điều kiện hoàn thành Wave 3:

- không collection error;
- không skipped test cho critical fake E2E;
- không coroutine/executor/resource leak warning;
- không network/model/database access;
- KIS regression pass;
- TRAKE và VQA internal route tests pass;
- concurrency/lifecycle/error-sanitization tests pass.

Máy chạy final gate phải cài FastAPI và toàn bộ test dependencies. Nếu API tests chưa collection được vì thiếu dependency thì phải ghi `ENVIRONMENT_BLOCKER`; chưa được công bố Wave 3 hoàn tất.

## 7.5 Những phần không được xem là đã hoàn thành sau Wave 3

Ngay cả khi fake E2E pass, các mục sau vẫn còn chờ runtime thật:

- competition/public API contract — OQ-002;
- Offline database/index mapping thật;
- encoder/checkpoint/index compatibility;
- image-path resolution — OQ-012;
- LLM rewrite provider thật;
- VLM/Gemini provider thật;
- dữ liệu, ảnh, video, metadata và evidence thật;
- latency/memory/concurrency benchmark trên máy thi;
- retrieval-quality evaluation và tuning;
- Stable Diffusion/QUEST enhancement nếu nhóm quyết định bật;
- submission/competition integration test.

Các mục này phải giữ nhãn `NEED_RUNTIME_VERIFICATION` hoặc open question tương ứng, không dùng fake tests làm bằng chứng production-ready.
