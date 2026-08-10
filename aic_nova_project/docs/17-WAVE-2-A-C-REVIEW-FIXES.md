# 17 — Các lỗi Người A và Người C phải sửa sau review Wave 2

## 1. Mục đích và snapshot đã review

Tài liệu này là yêu cầu sửa code chính thức sau khi review hai nhánh:

```text
Người A
Branch: feature/online-phase-Qluan
Commit đã review: 24bcf3c

Người C
Branch: feature/online-phase-Tngoc
Commit đã review: 0fb677f

Shared Wave 2 base: c1ef183
Người B Wave 2: 6b0948a
```

Review đã xác nhận:

- Hai nhánh đều bắt đầu từ đúng shared base `c1ef183`.
- Phạm vi file của A và C không lấn ownership của B.
- Merge thử theo thứ tự A → B → C không có conflict.
- `git diff --check` và `compileall` pass trên cả hai nhánh.
- A standalone: `12 passed`; A contract + adapter: `81 passed`.
- C VQA suite: `67 passed`; C broader unaffected suite: `233 passed`.
- A+B+C merge thử: `153` TRAKE/VQA tests và `331` unaffected Online tests pass.
- Full `tests/online` vẫn `NEED_RUNTIME_VERIFICATION` trong môi trường hiện tại vì
  thiếu `fastapi`, gây hai collection errors ở API tests.

Các test xanh trên chưa đủ để chốt Wave 2 vì còn contract mismatch và các failure
path chưa được kiểm tra. A và C phải sửa đúng nội dung bên dưới; không tự đổi public
domain/port contracts, không thêm database/model SDK và không sửa phần B.

---

# 2. Các quyết định sửa đã được chốt

Không cần A hoặc C lựa chọn lại cách sửa. Dùng đúng các quyết định sau:

1. Fixture A phải tách hai khái niệm:
   - toàn bộ evidence mà selector dự kiến chọn;
   - subset evidence mà fake VLM dùng để ground câu trả lời.
2. C dùng hai deadline khác nhau:
   - total orchestration deadline mặc định 30 giây;
   - từng VLM request có timeout tối đa 15 giây theo DD-031.
3. Một VLM attempt bao gồm cả `VLMPort.answer()` và response validation. Lỗi
   malformed ở bất kỳ bước nào đều đi qua cùng retry boundary.
4. Chỉ retry tối đa một lần và không bao giờ vượt total deadline.
5. OCR/ASR/summary chỉ được degrade đối với timeout hoặc resource unavailable.
   Invalid query, contract mismatch, dimension mismatch và sai kiểu dữ liệu phải
   fail rõ ràng; không được biến thành warning rồi trả kết quả có vẻ thành công.
6. Mọi output sai contract từ fake/internal port phải được chuyển thành shared
   domain error, không để rò `TypeError`, `AttributeError` hoặc `ValueError` thô.
7. Shared-fixture E2E được thêm sau khi A đã được merge trước. Không copy fixture
   của A thủ công sang nhánh C.

---

# 3. Người A — các việc phải sửa

## A-FIX-01 — Tách selected evidence và answer-grounding evidence

### Vấn đề

Trong `online/testing/advanced_fakes.py`, `AdvancedModesFixture` hiện có field:

```python
expected_vqa_evidence_ids
```

Tên field làm người dùng hiểu đây là toàn bộ expected evidence của VQA pipeline.
Tuy nhiên, giá trị hiện tại chỉ có bốn evidence của `V001`:

```text
image:V001_00004_010
ocr:V001_00004_010
asr:V001:interval-1
summary:V001
```

Khi chạy evidence selector của C bằng chính fixture A và default DD-030 budget,
output thật có chín evidence IDs:

```text
image:V001_00004_010
image:V002_00002_010
image:V001_00003_010
ocr:V001_00004_010
ocr:V002_00002_010
asr:V001:interval-1
asr:V002:interval-1
summary:V001
summary:V002
```

`CONTRACT_MISMATCH`: một field đang bị dùng cho hai ý nghĩa khác nhau. Nếu shared
E2E dùng field hiện tại làm golden output thì test sẽ sai; nếu chỉ xem nó là
grounding subset thì tên field không đúng.

### Cách sửa bắt buộc

Thay field mơ hồ bằng hai field rõ nghĩa:

```python
expected_vqa_selected_evidence_ids: tuple[str, ...]
expected_vqa_answer_evidence_ids: tuple[str, ...]
```

Giá trị bắt buộc:

```python
expected_vqa_selected_evidence_ids = (
    "image:V001_00004_010",
    "image:V002_00002_010",
    "image:V001_00003_010",
    "ocr:V001_00004_010",
    "ocr:V002_00002_010",
    "asr:V001:interval-1",
    "asr:V002:interval-1",
    "summary:V001",
    "summary:V002",
)

expected_vqa_answer_evidence_ids = (
    "image:V001_00004_010",
    "ocr:V001_00004_010",
    "asr:V001:interval-1",
    "summary:V001",
)
```

Các invariants bắt buộc:

- Hai tuple giữ deterministic order.
- ID trong mỗi tuple phải unique.
- `expected_vqa_answer_evidence_ids` phải là subset của
  `expected_vqa_selected_evidence_ids`.
- Mỗi ID phải trỏ tới evidence record thật có trong fixture.
- Không giữ alias `expected_vqa_evidence_ids`; xóa tên cũ để mọi chỗ dùng phải sửa
  rõ ý nghĩa thay vì âm thầm tiếp tục dùng contract mơ hồ.

### Sửa factory fake VLM

`AdvancedModesFixture.vlm()` phải dùng answer-grounding IDs, không dùng selected
IDs. Factory nên hỗ trợ override rõ ràng để unit test có thể tạo request nhỏ:

```python
def vlm(
    self,
    mode: FakeVLMMode | str = FakeVLMMode.ANSWERED,
    *,
    grounded_evidence_ids: Sequence[str] | None = None,
) -> FakeVLM:
    ...
```

Quy tắc:

- Không truyền override: dùng toàn bộ `expected_vqa_answer_evidence_ids`.
- Có override: dùng đúng tuple caller cung cấp sau validation.
- Không dùng `or` để fallback vì tuple rỗng phải bị xem là input không hợp lệ,
  không được âm thầm đổi sang evidence khác.
- ANSWERED mode phải reject grounding IDs rỗng hoặc không phải subset của request.

### Files A được phép sửa

```text
online/testing/advanced_fakes.py
online/testing/__init__.py                 # chỉ khi export symbol thay đổi
tests/online/contract/test_advanced_fakes.py
```

Không import `online.vqa.EvidenceSelector` vào contract test của A. A chỉ kiểm tra
fixture invariants; exact selector output sẽ được kiểm tra trong shared integration
test sau khi A merge.

### Tests A phải bổ sung/cập nhật

1. Hai expected-ID tuples deterministic qua hai lần build.
2. Mỗi tuple không có duplicate.
3. Answer IDs là subset selected IDs.
4. Mọi expected ID tồn tại trong image/OCR/ASR/summary fixture records.
5. Default `fixture.vlm()` ground bằng answer IDs khi request có đủ evidence.
6. Explicit override hoạt động với một image-only request.
7. Explicit empty override bị reject, không fallback.
8. Unknown override ID bị reject khi fake xử lý request.
9. Xóa/cập nhật tất cả test hoặc source reference đến tên field cũ.

### Definition of Done của A

- Không còn symbol `expected_vqa_evidence_ids`.
- Hai field mới có đúng ý nghĩa và đúng order nêu trên.
- Tất cả test A và regression contract/adapter pass.
- Không sửa TRAKE service, VQA selector/orchestrator, modes hoặc API.
- Commit/push lại cùng branch và gửi commit hash mới.

### Commands A phải chạy

```powershell
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/contract/test_advanced_fakes.py -q
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/contract tests/online/adapters -q
python -m compileall -q online tests/online
git diff --check c1ef183..HEAD
git diff --name-status c1ef183..HEAD
```

---

# 4. Người C — các việc phải sửa

## C-FIX-01 — Tách total deadline và VLM request timeout 15 giây

### Vấn đề

`online/vqa/orchestrator.py` hiện chỉ có một `timeout_sec=30.0` bao toàn bộ
orchestration. Không có timeout 15 giây riêng cho từng VLM request như DD-031.
Một VLM call có thể dùng gần hết 30 giây và retry không có per-attempt bound rõ ràng.

### Cách sửa bắt buộc

Constructor phải biểu diễn rõ hai giá trị:

```python
total_timeout_sec: float = 30.0
vlm_timeout_sec: float = 15.0
```

Không giữ tên `timeout_sec` mơ hồ. Wave 2 chưa có public API cho orchestrator nên
được phép rename nội bộ và cập nhật toàn bộ tests.

Tại đầu `execute()`:

```text
deadline = loop.time() + total_timeout_sec
```

Mỗi VLM attempt dùng:

```text
remaining = deadline - loop.time()
attempt_timeout = min(vlm_timeout_sec, remaining)
```

Rules:

- Cả hai timeout reject bool, non-number, NaN, infinity và giá trị <= 0.
- Total deadline bao gồm retrieval, evidence selection, VLM và retry.
- Nếu `remaining <= 0`, raise `BranchTimeoutError` trước khi submit thêm work.
- VLM attempt vượt `attempt_timeout` raise explicit `BranchTimeoutError`.
- Không retry VLM timeout.
- Không trả partial/false-success sau timeout.

## C-FIX-02 — Sửa retry boundary của VLM

### Vấn đề

Hiện `VLMPort.answer()` được gọi ngoài `try/except ContractMismatchError`. Nếu một
adapter conforming tự parse provider response và raise `ContractMismatchError` cho
malformed schema, orchestrator gọi đúng một lần rồi fail. Review đã tái hiện:

```text
Raised error: ContractMismatchError
VLM calls: 1
```

Ngoài ra lần retry hiện chưa kiểm tra remaining total deadline trước khi submit.

### Cách sửa bắt buộc

Một attempt phải bao trọn:

```text
call VLMPort.answer(request)
→ materialize raw response
→ validate_vlm_response(raw_response, request)
```

Catch và retry policy:

- `ContractMismatchError` từ VLM call hoặc validator: retry tối đa một lần.
- `ResourceUnavailableError` chỉ retry nếu error details có
  `retryable=True`; generic unavailable không retry.
- `BranchTimeoutError`: không retry.
- Mọi retry phải tính lại `remaining` và bị chặn bởi total deadline.
- Không đặt một guessed minimum retry window mới; nếu remaining dương thì second
  attempt dùng `min(15 giây, remaining)` và vẫn không thể vượt deadline.
- `vlm_retry_count` là số retry đã submit, không phải tổng số attempts.
- Malformed lần hai phải fail bằng `ContractMismatchError`, không trả answer giả.

Không retry mọi `ResourceUnavailableError` vô điều kiện vì missing credential/model
configuration không phải transient 429/5xx.

## C-FIX-03 — Không để invalid candidate-retrieval output làm rò exception thô

### Vấn đề

Code hiện chuyển output thành tuple trước khi kiểm tra:

```python
candidates = tuple(await retriever.retrieve_candidates(question))
```

Nếu retriever trả `None`, review tái hiện được:

```text
TypeError: 'NoneType' object is not iterable
```

### Cách sửa bắt buộc

- Materialize raw output trong một contract boundary riêng.
- Reject `None`, string, bytes và non-sequence bằng `ContractMismatchError`.
- Reject sequence chứa phần tử không phải `FusedFrameCandidate` bằng
  `ContractMismatchError`.
- `DataInfrastructureError` từ retriever phải được giữ nguyên.
- Unexpected ordinary exception từ retriever phải được wrap thành
  `ResourceUnavailableError` với safe details chỉ gồm stage và exception type.
- Không catch `CancelledError`/`BaseException`.

## C-FIX-04 — Siết contract validation cho evidence ports

### Vấn đề

Một số output sai kiểu từ metadata/evidence ports hiện có thể làm rò `TypeError`,
`AttributeError` hoặc `ValueError` thay vì shared `ContractMismatchError`. ASR path
đọc `item.video_id` trước khi xác nhận `item` là `ASREvidence`. Metadata neighbor
selection cũng có thể ném `ValueError` thô khi port trả sequence sai.

Image validation hiện chỉ đối chiếu mapping key với `frame_id`; chưa đảm bảo
`video_id`, `shot_id` và `timestamp_sec` của resolved image khớp frame đã yêu cầu.

### Cách sửa bắt buộc

1. Materialize từng port output trong helper nhỏ và map non-sequence/sai item type
   thành `ContractMismatchError`.
2. ASR phải xác nhận `ASREvidence` trước khi đọc field; sau đó mới kiểm tra đúng
   video và overlap requested time range.
3. OCR phải là `OCREvidence` và thuộc resolved image frame IDs.
4. Summary phải là `SummaryEvidence` và thuộc selected primary video IDs.
5. Metadata sequence phải chỉ chứa `FrameMetadata`, đúng video, không duplicate,
   deterministic ordered stream; lỗi của pure selector phải được map về
   `ContractMismatchError` tại adapter boundary.
6. Image resolver output phải là `Mapping[str, ImageEvidence]`; mỗi item phải khớp
   requested frame ở `frame_id`, `video_id`, `shot_id` và `timestamp_sec`.
7. Không rewrite ID, timestamp hoặc metadata để che contract lỗi.

## C-FIX-05 — Chỉ degrade đúng loại optional-backend failure

### Vấn đề

Evidence selector hiện catch gần như mọi `DataInfrastructureError` ngoài
`ContractMismatchError` rồi biến thành warning. Điều đó có thể che:

- `InvalidQueryError` do C gọi port sai;
- `DimensionMismatchError`;
- `MissingMetadataError` hoặc lỗi contract khác.

Các lỗi này không phải optional backend outage và không được trả như degraded
success.

### Cách sửa bắt buộc

Đối với OCR/ASR/summary:

```text
ResourceUnavailableError → degraded warning
BranchTimeoutError       → degraded warning
ContractMismatchError    → propagate
InvalidQueryError        → propagate
DimensionMismatchError   → propagate
MissingMetadataError     → propagate
unexpected exception     → wrap/surface; không degrade âm thầm
```

Warnings phải chỉ chứa bounded error code, không chứa backend message, prompt,
answer, local path hoặc secret.

## C-FIX-06 — Phản ánh text truncation trong diagnostics

### Vấn đề

`dropped_count` hiện chỉ tính record bị loại hoàn toàn hoặc evidence-ID collision.
Một OCR/ASR/summary record bị cắt text để vừa budget vẫn không được phản ánh trong
diagnostics, dù kế hoạch yêu cầu dropped/truncated evidence được báo rõ.

### Cách sửa bắt buộc

- Đếm mỗi evidence record bị rút ngắn text là một truncated/dropped diagnostic
  item.
- Không đếm số ký tự bị cắt vào field đếm evidence.
- Không thay public `VQADiagnostics` trong Wave 2; cộng số record bị truncate vào
  `dropped_evidence_count` hiện có.
- Thứ tự/caps/evidence output không được thay đổi chỉ để làm diagnostics đẹp hơn.

## C-FIX-07 — Bổ sung đầy đủ tests còn thiếu

### Tests trong `tests/online/vqa/test_evidence_selector.py`

1. ASR wrong item type → `ContractMismatchError`.
2. ASR trả sai video → `ContractMismatchError`.
3. ASR trả interval không overlap requested window → `ContractMismatchError`.
4. Summary trả unrequested video → `ContractMismatchError`.
5. Image trả đúng frame ID nhưng sai timestamp/video/shot → `ContractMismatchError`.
6. Metadata trả sai type/wrong video/duplicate → `ContractMismatchError`.
7. OCR/ASR/summary `ResourceUnavailableError` → degraded warning, không fabricate.
8. OCR/ASR/summary `BranchTimeoutError` → degraded warning, không fabricate.
9. `InvalidQueryError` và các contract-family errors không bị degrade.
10. Text bị truncate làm tăng `dropped_evidence_count` đúng một record.

### Tests trong `tests/online/vqa/test_orchestrator.py`

11. `total_timeout_sec` và `vlm_timeout_sec` reject mọi invalid value.
12. Một slow VLM call dừng ở per-call timeout 15 giây policy, không chờ total 30.
    Test dùng timeout nhỏ được inject, không sleep 15 giây thật.
13. VLM trả malformed object → retry một lần rồi success.
14. VLM adapter trực tiếp raise `ContractMismatchError` → retry một lần rồi success.
15. Hai malformed attempts → đúng hai calls rồi fail.
16. Total deadline hết trước retry → không submit call thứ hai.
17. Retryable unavailable (`details.retryable=True`) → tối đa một retry.
18. Generic unavailable → không retry.
19. VLM timeout → không retry.
20. Retriever trả `None`, string, non-sequence hoặc wrong item type →
    `ContractMismatchError`, không rò `TypeError`.
21. Retriever unexpected exception → safe `ResourceUnavailableError`.
22. Image resolver trả empty mapping → no-answer và VLM không được gọi.
23. Image resolver raise unavailable → explicit failure và VLM không được gọi.
24. Optional ASR/summary failure vẫn giữ image evidence và không fabricate text.
25. `close()`/drain/idempotence vẫn pass sau timeout và retry paths.

### Shared-fixture integration test sau khi A merge

Sau khi integration owner merge A trước, C cập nhật từ integration branch rồi thêm:

```text
tests/online/integration/test_vqa_fake_e2e.py
```

Test bắt buộc dùng trực tiếp `build_advanced_modes_fixture()` của A và xác nhận:

- ranked candidates → selector → VLM request → fake VLM → grounded `VQAResult`;
- selected evidence IDs bằng chính xác
  `fixture.expected_vqa_selected_evidence_ids` theo đúng order;
- response evidence IDs là subset của
  `fixture.expected_vqa_answer_evidence_ids` và request evidence IDs;
- missing images/degraded text/insufficient/malformed paths không fabricate;
- chạy hai lần cho cùng logical input cho cùng output ngoài latency fields;
- không network, database, filesystem image read hoặc provider SDK.

Không đặt shared integration test này trong `tests/online/vqa/` bằng fixture copy.

### Files C được phép sửa

```text
online/vqa/evidence_selector.py
online/vqa/vlm_request.py             # chỉ nếu helper validation cần cập nhật
online/vqa/orchestrator.py
online/vqa/__init__.py                # chỉ cập nhật exports cần thiết
tests/online/vqa/test_evidence_selector.py
tests/online/vqa/test_vlm_contract.py
tests/online/vqa/test_orchestrator.py
tests/online/integration/test_vqa_fake_e2e.py   # chỉ sau khi A merge
```

Không sửa:

```text
online/domain/
online/ports/
online/testing/advanced_fakes.py
query_understanding/
online/trake/
online/modes/
retrieval_api/
```

### Definition of Done của C

- Có per-VLM timeout 15 giây và total deadline riêng.
- Retry boundary bao gồm cả VLM call và response validation.
- Không quá một retry và không vượt total deadline.
- Invalid port output không làm rò exception thô.
- Chỉ timeout/unavailable optional text backend được degrade.
- Budget, order, stable IDs và grounding contract vẫn giữ nguyên.
- Tất cả C tests và unaffected regression pass.
- Shared A+C fake E2E pass sau khi A được merge.
- Không thêm actual Gemini/network/database/image resolver.
- Commit/push lại cùng branch và gửi commit hash mới.

### Commands C phải chạy trước khi A merge

```powershell
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/vqa -q
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/contract tests/online/adapters tests/online/retrieval tests/online/integration tests/online/ranking tests/online/modes tests/online/vqa -q
python -m compileall -q online tests/online
git diff --check c1ef183..HEAD
git diff --name-status c1ef183..HEAD
```

Sau khi A merge và C cập nhật branch:

```powershell
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/integration/test_vqa_fake_e2e.py -q
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/vqa tests/online/integration -q
```

---

# 5. Thứ tự sửa, review và merge bắt buộc

```text
1. A sửa A-FIX-01, test, commit và push.
2. Integration owner review lại A.
3. Nếu A pass, merge A vào branch tích hợp của B.
4. C có thể sửa C-FIX-01 đến C-FIX-07 song song bằng local fakes.
5. Sau khi A merge, C cập nhật branch từ integration branch.
6. C thêm shared `test_vqa_fake_e2e.py`, test, commit và push.
7. Integration owner review lại C.
8. Merge C sau A và B.
9. Chạy final Wave 2 gate.
```

Không merge C trước A. Không sửa fixture A từ branch C. Không chỉnh expected output
để che algorithm/contract failure.

---

# 6. Final Wave 2 gate sau khi merge A → B → C

Chạy từ `aic_nova_project/`:

```powershell
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/trake tests/online/vqa -q
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/contract tests/online/adapters tests/online/retrieval tests/online/integration tests/online/ranking tests/online/modes tests/online/trake tests/online/vqa -q
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online -q
python -m compileall -q online tests/online
git diff --check
git status --short
```

Nếu full suite vẫn bị chặn vì thiếu `fastapi`:

- Không cài dependency âm thầm.
- Ghi đúng hai collection blockers.
- Unaffected suite bắt buộc phải pass.
- Không gọi Wave 2 hoàn tất chỉ dựa trên targeted tests.

Wording được phép dùng sau khi mọi fake/integration gate pass:

```text
CONFIRMED_CODE: TRAKE và VQA chạy fake end-to-end, deterministic tests pass.
NEED_RUNTIME_VERIFICATION: database, image resolver, PE-Core checkpoint,
Gemini/VLM provider và dữ liệu thật chưa được kiểm tra.
```

