# 12 — CÁC VIỆC NGƯỜI A PHẢI HOÀN THÀNH ĐỂ BÀN GIAO CHO NGƯỜI B

## 0. Mục đích

Tài liệu này là checklist sửa lỗi và hoàn thiện lớp **Data & Infrastructure** do
Người A sở hữu trước khi Người B phụ thuộc ổn định vào lớp này.

Mục tiêu không phải làm thay phần Query & Retrieval của Người B. Mục tiêu là
đảm bảo:

```text
Người B nhận contract ổn định
→ dùng fake ports để code/test độc lập
→ thay fake bằng adapter thật mà không đổi business logic
→ lỗi database/contract được báo đúng, không bị giả thành empty result
→ có thể tích hợp database thật bằng một quy trình kiểm chứng lặp lại được
```

Tài liệu dựa trên audit source và test của commit phần A:

```text
ffb1747 — feat(online): add data infrastructure layer
```

Các kết luận runtime vẫn phải được kiểm tra lại trên branch/commit mới nhất sau
khi Người A sửa.

---

# 1. Phạm vi Người A

Người A sở hữu:

```text
online/domain/
online/ports/
online/adapters/
online/config.py
online/lifecycle.py
online/testing/
online/validate_contract.py
tests/online/contract/
tests/online/adapters/
tests/online/fixtures/
```

Người A có thể sửa test/package configuration cần thiết để test các phần trên.

Người A không làm trong PR này:

- `QueryBundle` và query parser của Người B.
- PE-Core/Vietnamese encoder implementation của Người B.
- Bảy retrieval branches.
- `RetrievalService` orchestration.
- ASR interval-to-frame mapping.
- Normalization, fusion, summary boost hoặc dedup.
- Public API cuối.
- Thay đổi Offline database schema.

Nếu một yêu cầu dưới đây cần đổi shared contract, Người A phải thông báo Người B
và C trước khi merge.

---

# 2. Trạng thái audit hiện tại

## 2.1 Phần đã có

`CONFIRMED_CODE`:

- Strict Pydantic domain models.
- Ba candidate levels: frame, ASR interval và video.
- SDK-neutral ports và records.
- Config cho Milvus, Elasticsearch và SQLite.
- Read-only SQLite adapter.
- Milvus adapter cho 4 collections.
- Elasticsearch adapter cho 3 indexes.
- Infrastructure lifecycle/health skeleton.
- Offline contract validator.
- Shared deterministic fakes.
- 29 unit tests chạy đạt bằng `unittest discovery`.

## 2.2 Phần chưa đạt

`CONFIRMED_CODE`:

- Pytest chuẩn không collect được test do package `tests/online` che package
  source `online`.
- Validator có thể `PASS` khi tất cả database cùng dùng một `frame_id` sai format.
- Validator có thể `PASS` khi không có encoder smoke check nào được chạy.
- Fakes chưa mô phỏng đầy đủ validation/error behavior của adapter thật.
- Fixture hiện chủ yếu là domain fake, chưa phải một integration fixture hoàn chỉnh
  cho database boundary.
- Một số nested dictionaries vẫn mutate được dù model khai báo frozen.
- Một số invalid database rows chưa được dịch nhất quán thành
  `ContractMismatchError`.

`NEED_RUNTIME_VERIFICATION`:

- Pymilvus adapter với SDK và Milvus thật.
- Elasticsearch adapter với SDK và Elasticsearch thật.
- Schema, vector dimension/norm và JOIN trên dữ liệu thật.
- Thread/concurrency behavior của long-lived adapters.

---

# 3. Mức ưu tiên

| ID | Mức | Nội dung | Chặn B |
|---|---|---|---|
| `A-FIX-001` | P0 | Sửa pytest package collision | Chặn mọi PR/CI test của B |
| `A-FIX-002` | P0 | Validator kiểm canonical `frame_id` | Chặn integration DB thật |
| `A-FIX-003` | P0 | Không `PASS` khi encoder smoke chưa chạy | Chặn xác nhận vector compatibility |
| `A-FIX-004` | P1 | Chốt và test shared enum/model contract | Chặn QueryBundle/parser ổn định |
| `A-FIX-005` | P1 | Làm fakes tương đương adapter behavior | Chặn unit test đáng tin cậy của B |
| `A-FIX-006` | P1 | Hoàn thiện integration fixture/handoff data | Chặn integration lặp lại được |
| `A-FIX-007` | P1 | Chuẩn hóa exception translation | Chặn graceful degradation chính xác |
| `A-FIX-008` | P1 | Chốt sync/concurrency contract | Chặn B7 chạy branches song song an toàn |
| `A-FIX-009` | P2 | Deep immutability và validation consistency | Không chặn B0, cần trước full integration |
| `A-FIX-010` | P2 | Dependency/dev setup tái lập được | Chặn máy mới/CI chạy adapter tests |
| `A-FIX-011` | P2 | Runtime validator/report completeness | Chặn tuyên bố production readiness |

Người A nên sửa theo đúng thứ tự trên. Không gom tất cả thành một commit khó review.

---

# 4. `A-FIX-001` — Sửa pytest package collision

## 4.1 Lỗi hiện tại

File:

```text
tests/online/__init__.py
```

Trong khi thư mục cha `tests/` không phải package, pytest có thể collect
`tests/online` thành top-level package tên `online`, che khuất source package:

```text
aic_nova_project/online/
```

Kết quả đã xác minh:

```text
ModuleNotFoundError: No module named 'online.domain'
ModuleNotFoundError: No module named 'online.adapters...'
7 collection errors
```

## 4.2 Yêu cầu sửa

Chọn một cấu trúc package test không thể che source package. Hướng nhỏ nhất nên
được kiểm tra trước:

```text
Thêm tests/__init__.py
→ test modules được import dưới tests.online...
→ source package online không bị thay thế trong sys.modules
```

Nếu cách đó không hoạt động ổn định với pytest/importlib mode, chọn một trong:

- Đổi tên test package để không trùng `online`.
- Bỏ các `__init__.py` không cần thiết và dùng layout pytest không-package.
- Chuẩn hóa package/project configuration bằng `pyproject.toml`.

Không sửa bằng cách chèn path thủ công trong từng test file.

Không thêm:

```python
sys.path.insert(...)
```

vào từng test module.

## 4.3 Acceptance tests

Lệnh chuẩn bắt buộc chạy được từ repository root:

```powershell
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online -q
```

Điều kiện đạt:

- Không có collection error.
- Không có package shadowing.
- Toàn bộ test A hiện tại chạy xanh.
- Test mới của B trong `tests/online/retrieval/` cũng import được `online.*`.
- Không yêu cầu người dùng tự đặt `PYTHONPATH` khác nhau trên từng máy.

## 4.4 Tests cần thêm

- Smoke test import `online.domain`.
- Smoke test import `online.adapters`.
- Smoke test import một module dự kiến của `online.retrieval`.

---

# 5. `A-FIX-002` — Validator phải kiểm canonical `frame_id`

## 5.1 Lỗi hiện tại

Validator hiện kiểm equality JOIN và `video_id/shot_id`, nhưng chưa kiểm format
canonical.

Một bộ database cùng dùng ID sai:

```text
shot_00000_pos_050
```

vẫn có thể nhận:

```text
ValidationStatus.PASS
```

## 5.2 Contract bắt buộc

```text
{video_id}_{shot_id:05d}_{position_3_digits}
```

Ví dụ:

```text
V001_00000_050
```

Không chỉ kiểm regex tổng quát. Validator phải xác minh semantic fields:

- ID bắt đầu đúng canonical `video_id`.
- Suffix shot có đúng 5 chữ số.
- Suffix position có đúng 3 chữ số.
- Shot suffix parse ra bằng field `shot_id`.
- Không chấp nhận local stem `shot_NNNNN_pos_PPP`.
- Không chấp nhận empty/whitespace ID.
- Không rewrite ID trong Online để làm test pass.

Nếu `video_id` có thể chứa underscore, parser phải tách từ suffix bên phải thay
vì giả định `video_id` chỉ có một token.

## 5.3 Nơi phải kiểm

Sample từ:

- Milvus `visual_features`.
- Milvus `ocr_features` khi có.
- Elasticsearch `ocr_texts` khi có.
- SQLite `metadata`.
- SQLite `objects` khi có.

Equality JOIN vẫn phải giữ; format validation không thay thế JOIN validation.

## 5.4 Error/report behavior

Core visual/metadata malformed ID:

```text
CheckStatus.FAIL
ValidationStatus.FAIL
ErrorCode.CONTRACT_MISMATCH
```

Malformed ID trong optional OCR/object resource:

```text
CheckStatus.WARNING hoặc FAIL theo required policy đã ghi rõ
ValidationStatus tối thiểu PARTIAL
```

Report phải nêu resource và số sample lỗi, nhưng không cần in dữ liệu nhạy cảm.

## 5.5 Acceptance tests

Phải có test cho:

1. Canonical ID hợp lệ.
2. Local filename stem bị reject.
3. Visual và SQLite cùng sai format vẫn không `PASS`.
4. ID đúng format nhưng `shot_id` field không khớp.
5. `video_id` chứa underscore.
6. OCR/objects malformed tạo PARTIAL/FAIL đúng policy.

---

# 6. `A-FIX-003` — Encoder smoke check không được bị bỏ qua

## 6.1 Lỗi hiện tại

`OfflineContractValidator` mặc định:

```python
encoder_smoke_vectors = None
```

CLI hiện không truyền encoder factory. Khi mapping rỗng, validator không tạo bất
kỳ `encoder.*` check nào nhưng report vẫn có thể `PASS`.

## 6.2 Yêu cầu sửa

Validator report phải phân biệt:

```text
PASS       = check đã chạy và đạt
WARNING    = optional check chưa chạy/không đạt
FAIL       = required check chưa chạy/không đạt
NOT_RUN    = có thể thêm status riêng nếu team chấp nhận
```

Ít nhất phải đảm bảo:

- Không có visual encoder smoke vector thì report không được `PASS` cho trạng
  thái sẵn sàng full Online integration.
- Nếu visual collection tồn tại, visual encoder dimension/norm mismatch là FAIL.
- Nếu OCR/ASR/summary collection tồn tại nhưng Vietnamese encoder chưa được cung
  cấp, report phải ghi rõ checks chưa chạy và status tối thiểu PARTIAL.
- Không load model GPU tự động trong validator nếu người chạy không yêu cầu.
- Cho phép inject callable/factory từ phần B sau khi encoder B2 hoàn thành.

## 6.3 Handoff interface cho Người B

Người A phải cung cấp một cách rõ ràng để B đăng ký:

```text
visual collection → PE-Core text smoke vector factory
ocr/asr/summary collections → Vietnamese text smoke vector factory
```

Không yêu cầu Người B import private symbol của adapter.

## 6.4 Acceptance tests

1. Không truyền encoder factories: report không PASS.
2. Visual vector sai dimension: FAIL + `DIMENSION_MISMATCH`.
3. Visual vector sai norm: FAIL.
4. Vietnamese vector đúng dimension/norm cho cả ba collections: PASS.
5. Factory ném exception: report rõ check failure, không crash toàn CLI.
6. Không có optional collection: không yêu cầu encoder tương ứng.

---

# 7. `A-FIX-004` — Chốt shared enum/model contract

## 7.1 Enum mode

Contract nội bộ hiện tại theo `AGENTS.md`, `docs/11` và source đã merge:

```python
KIS_TEXT = "kis_text"
KIS_VIDEO = "kis_video"
TRAKE = "trake"
VQA = "vqa"
```

`KIS_VISUAL` chỉ là wording cũ trong một số docs. Không thêm alias âm thầm nếu
chưa có quyết định migration/public API.

OQ-002 vẫn mở cho exact public request/response schema.

## 7.2 Candidate/BranchResult contract

Người A phải đảm bảo các model B cần ổn định:

- `CandidateProvenance`.
- `FrameCandidate`.
- `ASRIntervalCandidate`.
- `VideoCandidate`.
- `BranchResult[T]`.
- `QueryMode`, `RetrievalBranch`, `CandidateLevel`, `BranchStatus`.
- `FrameSearchHit`, `ASRSearchHit`, `VideoSearchHit`, `FrameMetadata`.
- Search/metadata/encoder ports.

## 7.3 Validation tests còn thiếu

Thêm tests cho:

- Tất cả `QueryMode` values.
- Tất cả `RetrievalBranch` values.
- Branch/candidate provenance branch mismatch.
- Branch/query variant mismatch.
- Wrong candidate type sau JSON round-trip.
- Whitespace-only warnings cho failed result.
- `DISABLED` và `DEGRADED` branch behavior đã chốt rõ.
- Empty `SUCCESS` khác `FAILED`.

## 7.4 Change policy

Sau khi merge task này:

- Không rename shared fields trong PR adapter nhỏ.
- Mọi shared-model change phải có review B và C.
- Model, JSON fixture và contract tests phải đổi trong cùng PR.

---

# 8. `A-FIX-005` — Fakes phải đủ giống adapter thật

## 8.1 Mục tiêu

Người B sẽ code phần lớn retrieval bằng fakes trước khi database thật sẵn sàng.
Nếu fake behavior khác adapter thật, test xanh không có ý nghĩa.

## 8.2 Fake Milvus requirements

`FakeMilvusSearchPort` phải:

- Validate `top_k` giống adapter thật, bao gồm reject bool/zero/negative.
- Có optional expected dimension để reject vector sai shape.
- Reject NaN/Inf.
- Có tùy chọn yêu cầu L2 norm như adapter thật.
- Cắt kết quả đúng `top_k`.
- Lưu call history gồm branch/vector/top_k để B assert.
- Có thể cấu hình empty result.
- Có thể cấu hình ném `ResourceUnavailableError`, `BranchTimeoutError` hoặc
  `ContractMismatchError` theo từng branch.
- Không expose SDK object hoặc `pk`.

## 8.3 Fake Elasticsearch requirements

- Reject empty/whitespace query bằng cùng domain error class như adapter thật.
- Validate `top_k` giống adapter thật.
- Lưu query, top-k và fuzzy option trong call history.
- Có thể cấu hình success, empty, timeout và unavailable cho từng branch.
- Trả đúng frame/interval/video boundary records.

## 8.4 Fake metadata/object requirements

- Batch lookup giữ behavior mapping giống adapter thật.
- Missing IDs không xuất hiện trong metadata result.
- Object result có empty tuple cho requested frame không có object.
- Validate label/min-confidence giống adapter thật.
- Ordered frames sort `(timestamp_sec, frame_id)`.
- Duplicate frame IDs trong fixture phải bị reject hoặc có policy rõ, không âm
  thầm last-write-wins.

## 8.5 Timeout/concurrency support

Để B7 test orchestration, fakes cần một cách deterministic để mô phỏng:

- Delay của từng branch.
- Một branch timeout.
- Một branch failure trong khi branch khác thành công.
- Call cancellation/late completion nếu implementation B hỗ trợ.

Không dùng sleep dài trong unit test. Dùng injectable behavior/event hoặc delay
rất nhỏ, deterministic.

## 8.6 Acceptance tests

Chạy cùng một conformance suite cho fake và adapter với mocked SDK đối với:

- Invalid top-k.
- Empty query.
- Empty result.
- Timeout.
- Resource unavailable.
- Candidate-level mapping.
- Call arguments.

`isinstance(fake, Protocol)` không đủ để coi là behavior-conformant.

---

# 9. `A-FIX-006` — Hoàn thiện fixtures cho B

## 9.1 Domain fixture tối thiểu

Fixture phải có ít nhất:

- 2 videos.
- Nhiều shots/frames trong mỗi video.
- Visual hits.
- OCR dense và OCR lexical hits riêng.
- ASR dense và ASR lexical hits riêng.
- Summary dense và summary lexical hits riêng.
- Một frame xuất hiện trong nhiều branches.
- Một frame xuất hiện trong nhiều query variants.
- Một visual hit thiếu SQLite metadata.
- Một hit có metadata mismatch để test contract behavior.
- ASR overlap, no-overlap và boundary intervals.
- Object zero/one/many, confidence thấp/cao.
- Empty branch result.
- Configurable branch timeout/failure.

## 9.2 Canonical IDs

Mọi fixture bình thường dùng canonical format:

```text
V001_00000_015
```

Invalid IDs chỉ xuất hiện trong fixture/test được đánh dấu explicit invalid.

## 9.3 Backend fixture

Tối thiểu phải có SQLite fixture thật tạo từ schema hiện hành để kiểm:

- Read-only connection.
- Batch hydration.
- Ordered frames.
- Object lookup.
- Missing metadata.

Milvus/ES integration fixture có thể được cung cấp bằng Docker/test profile hoặc
script riêng. Nếu chưa thể chạy trong CI, phải có:

- Dữ liệu seed rõ ràng.
- Command dựng/xóa môi trường test.
- Không dùng production database.
- Marker để test runtime không chạy mặc định.
- Report `NEED_RUNTIME_VERIFICATION`, không giả là đã test.

## 9.4 Fixture versioning

Thêm một version/schema marker hoặc fixture contract test để khi shared model đổi,
fixture cũ không bị deserialize âm thầm theo shape sai.

---

# 10. `A-FIX-007` — Chuẩn hóa exception translation

## 10.1 Error types canonical

Adapter boundary phải dùng:

```text
InvalidQueryError
ContractMismatchError
DimensionMismatchError
BranchTimeoutError
ResourceUnavailableError
MissingMetadataError khi policy sử dụng
```

## 10.2 Quy tắc

- Invalid caller input → `InvalidQueryError`.
- Backend response/row thiếu hoặc sai type → `ContractMismatchError`.
- Query vector dimension sai → `DimensionMismatchError`.
- Backend timeout → `BranchTimeoutError`.
- Connection/service failure → `ResourceUnavailableError`.
- Empty valid result → empty tuple/mapping, không phải exception.
- Không trả raw SDK exception ra phần B.

## 10.3 Các path cần bổ sung test

- SQLite object row có confidence ngoài `[0,1]`.
- SQLite bbox đảo chiều hoặc field null.
- SQLite metadata row invalid.
- Milvus score NaN/Inf.
- Milvus missing entity/output field.
- ES missing `_source`, `_score` hoặc required source field.
- ES malformed `hits` response.
- Invalid `find_documents` limit/filter.
- Timeout exception của SDK không kế thừa built-in `TimeoutError` nhưng có tên
  timeout theo mapping hiện tại.

## 10.4 Safe diagnostics

Error details không được chứa:

- Password/token.
- Full connection URI nếu URI chứa credentials.
- Full vector.
- Raw database response quá lớn.
- Stack trace trong public serialization.

---

# 11. `A-FIX-008` — Chốt sync/concurrency contract cho B7

## 11.1 Hiện trạng

Search ports hiện là synchronous:

```python
def search_visual(...)
def search_ocr(...)
def search_asr(...)
def search_summary(...)
```

Người B dự kiến viết async `RetrievalService` chạy branches song song.

## 11.2 Người A phải cung cấp

Không nhất thiết đổi ports sang async. Người A phải chốt và ghi rõ:

- Adapter nào thread-safe cho concurrent read calls.
- SQLite adapter serialize calls bằng lock như thế nào.
- Milvus alias/connection có dùng chung an toàn không.
- Elasticsearch client có dùng chung long-lived connection không.
- Connection được mở/đóng ở lifecycle nào.
- Ai sở hữu timeout: adapter, orchestrator hay cả hai với hai mức khác nhau.
- Có được gọi `close()` trong khi query đang chạy không.
- Một adapter instance dùng per process hay per request.

## 11.3 Acceptance tests

- Nhiều concurrent read calls không corrupt call/result state.
- SQLite batch calls không lỗi `check_same_thread`/closed connection.
- Lifecycle start idempotent.
- Lifecycle close idempotent theo policy hoặc lỗi rõ.
- Required/optional health status đúng.
- Connection failure không bị báo healthy.

Nếu chưa runtime-test được SDK thật, ghi rõ `NEED_RUNTIME_VERIFICATION` và cung
cấp strategy mà B có thể dùng tạm, ví dụ controlled thread executor.

Không thay toàn bộ ports sang async trong một PR mà không review với B/C.

---

# 12. `A-FIX-009` — Deep immutability và validation consistency

## 12.1 Lỗi hiện tại

Pydantic `frozen=True` ngăn gán lại field nhưng không tự freeze nested `dict`.

Các field như:

```text
branch_scores
stage_latencies_ms
branches
fusion_weights
report.dimensions
```

vẫn có thể bị mutate tại chỗ.

## 12.2 Yêu cầu

Chọn một contract nhất quán:

1. Dùng immutable mapping/tuple thực sự trong domain; hoặc
2. Không tuyên bố deep immutable và luôn copy tại boundary.

Hướng ưu tiên cho shared domain/result models là immutable snapshot. Serialization
vẫn phải tạo JSON object bình thường.

## 12.3 Validation consistency

- ID/string phải reject whitespace-only.
- Quyết định có trim hay preserve leading/trailing spaces và test rõ.
- Bbox pixel coordinates không được âm nếu Offline contract không cho phép.
- `x_max >= x_min`, `y_max >= y_min`.
- `warnings/errors` không chứa whitespace-only entries nếu status phụ thuộc chúng.
- Mutable default values phải dùng factory/copy an toàn.

---

# 13. `A-FIX-010` — Dependency và test setup tái lập được

## 13.1 Hiện trạng audit

Máy audit có:

```text
pydantic: installed
pymilvus: not installed
elasticsearch: not installed
```

Unit tests fake vẫn chạy, nhưng adapter runtime chưa được xác minh.

## 13.2 Yêu cầu

- Xác định Python version được nhóm hỗ trợ.
- Tách runtime và dev/test dependencies nếu cần.
- Có pytest trong dev/test dependency.
- Pin compatible ranges hoặc lockfile phù hợp, tránh chỉ dùng lower bound vô hạn
  cho SDK database.
- Có command setup cho Windows/PowerShell mà cả ba người dùng được.
- Không buộc cài GPU dependencies để chạy contract/adapter unit tests.
- Encoder dependencies của B nên tách khỏi adapter-only test environment nếu có
  thể.

## 13.3 Commands phải ghi trong README/handoff

```text
Create/activate environment
Install Online runtime dependencies
Install Online test dependencies
Run contract tests
Run adapter tests
Run all Online unit tests
Run optional runtime integration tests
```

Không commit secret hoặc machine-specific absolute path.

---

# 14. `A-FIX-011` — Hoàn thiện contract validator/report

Ngoài canonical ID và encoder smoke, validator cần bảo đảm report không tạo false
confidence.

## 14.1 Mọi required check phải xuất hiện

Report phải thể hiện rõ:

- Check đã PASS.
- Check đã FAIL.
- Optional check WARNING/PARTIAL.
- Check không chạy và lý do.

Không được bỏ qua một check rồi vẫn tổng hợp `PASS`.

## 14.2 Resources

Kiểm:

- 4 Milvus collections.
- 3 Elasticsearch indexes.
- 2 SQLite tables.
- Required fields/types.
- HNSW/IP.
- Dynamic dimensions.
- Sample records không rỗng theo required policy.
- Stored vector finite và norm gần 1.
- Vietnamese analyzer.
- `analysis-icu`.

## 14.3 JOINs

Kiểm:

- Visual → metadata.
- OCR dense → metadata.
- OCR lexical → metadata.
- OCR dense ↔ OCR lexical.
- Objects → metadata.
- ASR `(video_id, interval_id)` Milvus ↔ ES.
- Summary `video_id` Milvus ↔ ES.

Sample JOIN phải dùng cùng logical key, không lấy các record độc lập rồi kết luận.

## 14.4 Report summary

Report nên có:

```text
overall status
checks
dimensions
resources checked
checks skipped
sample counts
timestamp/version nếu cần
```

Không in full embeddings.

## 14.5 CLI exit codes

Chốt và test:

```text
PASS              → 0
PARTIAL            → 0 hoặc 1 tùy --fail-on-partial
FAIL               → non-zero cố định
startup/CLI error  → non-zero khác hoặc error rõ
```

Nếu lifecycle start cho required resource thất bại, CLI vẫn phải tạo report hợp
lý hoặc trả lỗi rõ; không crash khó hiểu trong finally/close.

---

# 15. Contract cụ thể Người B cần nhận

Sau khi Người A bàn giao, Người B phải có thể chỉ import public symbols, không
import private adapter implementation.

## 15.1 Search ports

```python
MilvusSearchPort.search_visual(vector, top_k) -> Sequence[FrameSearchHit]
MilvusSearchPort.search_ocr(vector, top_k) -> Sequence[FrameSearchHit]
MilvusSearchPort.search_asr(vector, top_k) -> Sequence[ASRSearchHit]
MilvusSearchPort.search_summary(vector, top_k) -> Sequence[VideoSearchHit]

ElasticsearchSearchPort.search_ocr(query, top_k, fuzzy=...) -> Sequence[FrameSearchHit]
ElasticsearchSearchPort.search_asr(query, top_k, fuzzy=...) -> Sequence[ASRSearchHit]
ElasticsearchSearchPort.search_summary(query, top_k, fuzzy=...) -> Sequence[VideoSearchHit]
```

## 15.2 Metadata/object ports

```python
MetadataReaderPort.get_frames_by_ids(frame_ids)
MetadataReaderPort.get_ordered_frames_by_video(video_id)
ObjectReaderPort.get_objects_by_frame_ids(...)
```

## 15.3 Encoder ports

Người A chỉ sở hữu port shape; Người B sở hữu implementation:

```python
TextEncoderPort.dimension
TextEncoderPort.encode_texts(texts)
ImageEncoderPort  # optional SD/QUEST, không phải baseline v-KIS
```

## 15.4 Candidate conversion responsibility

```text
Person A adapter
→ FrameSearchHit / ASRSearchHit / VideoSearchHit

Person B direct frame branch
→ hydrate through MetadataReaderPort
→ FrameCandidate

Person B ASR branch
→ ASRIntervalCandidate, không map frame

Person B summary branch
→ VideoCandidate, không tạo frame
```

---

# 16. Hai mức bàn giao

Không nên dùng một chữ “xong” cho cả fake development và real database integration.

## 16.1 `B-DEV-READY` — B có thể code mượt trên fakes

Tất cả điều kiện bắt buộc:

- [ ] `A-FIX-001` pytest collect và pass.
- [ ] Shared enums/models/ports đã chốt.
- [ ] Không còn pending rename field/method ảnh hưởng B.
- [ ] Fakes behavior-conformant cho success/empty/failure/timeout.
- [ ] Fixture có đủ bảy branch levels và missing metadata case.
- [ ] Contract/fixture tests xanh.
- [ ] Người A cung cấp public import paths.
- [ ] Người B review và chạy được một smoke test trên máy mình.

Khi đạt mức này, B0–B7 có thể phát triển độc lập với database thật.

## 16.2 `B-RUNTIME-READY` — B có thể thay fakes bằng adapters thật

Ngoài toàn bộ `B-DEV-READY`:

- [ ] `A-FIX-002` canonical ID validation đạt.
- [ ] `A-FIX-003` encoder smoke semantics đạt.
- [ ] Runtime dependencies cài được bằng documented command.
- [ ] Milvus/ES/SQLite reachable.
- [ ] Contract validator không false PASS.
- [ ] Core resources PASS.
- [ ] Optional missing resources xuất hiện PARTIAL đúng.
- [ ] Real cross-DB JOIN samples đạt.
- [ ] B2 encoders đúng dimension/norm.
- [ ] Concurrency/thread strategy đã chốt.
- [ ] Minimal vertical slice chạy:

```text
text
→ PE-Core text encoder
→ Milvus visual_features
→ SQLite hydration
→ FrameCandidate
```

Chỉ khi đạt mức này mới được nói “Online retrieval đã tích hợp database thật”.

---

# 17. Thứ tự PR/commit đề xuất

## PR A-1 — Test/package unblock

```text
A-FIX-001
enum/model contract tests cần thiết từ A-FIX-004
```

Mục tiêu: Người B có thể bắt đầu viết/running tests ngay.

## PR A-2 — Validator correctness

```text
A-FIX-002
A-FIX-003
A-FIX-011
```

Mục tiêu: loại false PASS.

## PR A-3 — Fake/fixture parity

```text
A-FIX-005
A-FIX-006
```

Mục tiêu: B code trên fakes mà không đổi logic khi dùng adapters thật.

## PR A-4 — Adapter hardening

```text
A-FIX-007
A-FIX-008
A-FIX-009
```

Mục tiêu: error/concurrency/model behavior rõ.

## PR A-5 — Reproducible runtime handoff

```text
A-FIX-010
runtime integration tests/report
```

Mục tiêu: máy B có thể dựng và chạy lại cùng kết quả.

Không bắt buộc đúng 5 PR nếu nhóm dùng commit nhỏ trong một branch, nhưng không
được tạo một PR khổng lồ trộn infrastructure fixes với retrieval code của B.

---

# 18. Checklist test cuối của Người A

## Static/contract

- [ ] Import public domain/port symbols.
- [ ] Enum values đúng.
- [ ] Candidate validation đúng.
- [ ] BranchResult homogeneous/provenance đúng.
- [ ] Canonical ID parser/validator đúng.
- [ ] Frozen/deep-copy policy đúng.

## SQLite

- [ ] Read-only.
- [ ] Batch hydration.
- [ ] Missing ID.
- [ ] Timestamp ordering.
- [ ] Object label/confidence.
- [ ] Invalid rows → contract error.
- [ ] Parameter binding.

## Milvus

- [ ] Four collection methods.
- [ ] Dynamic dimensions.
- [ ] Finite + L2 query vector.
- [ ] HNSW/IP search params.
- [ ] Empty vs timeout/unavailable.
- [ ] Missing fields/score.
- [ ] No `pk` leak.

## Elasticsearch

- [ ] Three index methods.
- [ ] Exact/fuzzy bodies.
- [ ] `shot_id` → int.
- [ ] `start_time/end_time` → `_sec`.
- [ ] Empty vs timeout/unavailable.
- [ ] Malformed response.

## Validator

- [ ] Resource fields/types.
- [ ] Dynamic dimensions.
- [ ] Index/metric/analyzer/plugin.
- [ ] Non-empty policy.
- [ ] Vector norm.
- [ ] Canonical IDs.
- [ ] Frame/ASR/summary/object JOINs.
- [ ] Encoder smoke or explicit not-run status.
- [ ] PASS/PARTIAL/FAIL and exit codes.

## Fakes/integration

- [ ] Protocol conformance.
- [ ] Behavior conformance.
- [ ] Call history.
- [ ] Success/empty/failure/timeout.
- [ ] Two videos and all candidate levels.
- [ ] Missing/mismatched metadata cases.
- [ ] Standard pytest command passes.

---

# 19. Nội dung handoff Người A phải gửi cho B

Người A không chỉ nhắn “đã sửa xong”. Handoff phải gồm:

```text
1. Branch/commit SHA
2. Files changed
3. Shared contract changes, nếu có
4. Public import paths B phải dùng
5. Test commands đã chạy
6. Exact test results
7. Validator command/report
8. Runtime services/dependencies đã hoặc chưa kiểm
9. Known limitations còn lại
10. B-DEV-READY hay B-RUNTIME-READY
```

Mẫu:

```text
Commit:
Readiness: B-DEV-READY / B-RUNTIME-READY

Contract changes:
- ...

Commands:
- python -m pytest ...

Results:
- N passed

Runtime verification:
- Milvus: PASS / NOT RUN
- Elasticsearch: PASS / NOT RUN
- SQLite: PASS / NOT RUN
- Encoder smoke: PASS / NOT RUN

Remaining limitations:
- ...
```

---

# 20. Definition of Done của phần sửa A

Phần sửa A chỉ hoàn tất khi:

- Không còn pytest collection error.
- Không còn validator false PASS đã biết.
- Shared contract ổn định và được B/C review.
- Fakes và adapters có behavior nhất quán ở các case B cần.
- Test fixtures đủ để B viết bảy branches và orchestration failure tests.
- Exceptions được dịch thành domain errors nhất quán.
- Sync/concurrency/lifecycle contract được ghi rõ.
- Test/dependency setup chạy được trên máy khác bằng command đã tài liệu hóa.
- Người B clone/fetch branch, chạy test và thực hiện được visual fake smoke flow.
- Mọi mục chưa runtime-test được ghi `NEED_RUNTIME_VERIFICATION`, không báo PASS.
- Không có workaround trong code B để che lỗi A.

Mục tiêu thực tế là một boundary ổn định, testable và có thể thay fake bằng real
adapter. Không thể bảo đảm “100%” chỉ bằng unit tests; mức tin cậy đầy đủ yêu cầu
contract tests, runtime database validation, encoder smoke test và một vertical
slice end-to-end trên dữ liệu thật.
