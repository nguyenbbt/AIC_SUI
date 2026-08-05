# Báo cáo trạng thái Online - Self-Indexed V2

Ngày cập nhật: 2026-08-05

Branch tích hợp: `feature/online-phase-Knguyen`

Contract hiện hành: `self-indexed-v2`

Nguồn thể lệ: `Thong tin vong So tuyen AIC2026 (1).pdf`

## 1. Kết luận hiện tại

`CONFIRMED_CODE`: Online đã hoàn thành toàn bộ phần có thể triển khai và kiểm
thử độc lập với dữ liệu/model production. Hệ thống đã có KIS, TRAKE/DANTE, VQA
evidence flow, adapter cho self-indexed data, manifest startup gate và logical
submission serializer theo thể lệ vòng sơ tuyển.

`NEED_RUNTIME_VERIFICATION`: chưa được gọi là hệ thống sẵn sàng thi cho đến khi
nhận dataset READY thật từ Offline, cài/chạy các SDK và model thật, chọn VLM/LLM
production, benchmark ranking/latency và chạy một vertical slice thật.

## 2. Định dạng câu trả lời BTC đã công bố

Tài liệu vòng sơ tuyển đã chốt các tuple logic sau:

| Loại truy vấn | Định dạng câu trả lời |
|---|---|
| Textual KIS | `<video_id>, <frame_id>` |
| Q&A / VQA | `<video_id>, <frame_id>, <answer>` |
| TRAKE | `<video_id>, <frame_id_1>, ..., <frame_id_n>` |

Quy tắc bắt buộc:

- `frame_id` trong định dạng BTC là chỉ số frame của video gốc.
- Online lấy giá trị này trực tiếp từ `source_frame_idx` nội bộ.
- Không được nộp internal JOIN key dạng `L21_V001_00003_050` thay cho frame số.
- TRAKE giữ các frame theo đúng thứ tự event trong truy vấn.
- Mỗi truy vấn được gửi tối đa 100 câu trả lời.
- Thứ tự xếp hạng ảnh hưởng các mốc `R@1`, `R@5`, `R@20`, `R@50`, `R@100`.

`OPEN_QUESTION`: PDF chưa quy định transport cụ thể như endpoint, HTTP body,
JSON wrapper, CSV header/delimiter, authentication hoặc quy trình upload. Vì
vậy code hiện cung cấp các logical submission rows; transport adapter cuối sẽ
được bổ sung khi BTC công bố giao thức gửi bài.

## 3. Những phần đã hoàn thành

### 3.1 Offline-to-Online contract

`CONFIRMED_CODE`:

- Chuyển Online về kiến trúc chỉ nhận raw video từ BTC và dùng dataset tự index.
- Canonical internal `frame_id`:
  `{video_id}_{shot_id:05d}_{position_code:03d}`.
- Giữ riêng `source_frame_idx` là zero-based decoded frame thật của raw video.
- KIS dedup theo `(video_id, source_frame_idx)`.
- Bổ sung `videos`, `metadata`, `objects` và các JOIN rule self-indexed-v2.
- Object bbox được đọc dưới dạng absolute pixel XYXY và normalize trong Online.
- ASR dùng `start_time_sec`/`end_time_sec` và identity
  `(video_id, interval_id)`.
- Keyframe và video path là POSIX relative path; không chấp nhận absolute path
  hoặc parent traversal.

Nguồn chuẩn: `docs/22-OFFLINE-TO-ONLINE-DATA-CONTRACT-SELF-INDEXED-V2.md`.

### 3.2 KIS retrieval và ranking

`CONFIRMED_CODE`:

- Textual KIS và Video KIS dùng chung text-to-keyframe pipeline.
- v-KIS nhận mô tả do thí sinh tự viết sau khi xem clip BTC trình chiếu; không
  nhận file video/image query trong baseline.
- Bảy retrieval branches:
  - `visual_dense`.
  - `ocr_dense`.
  - `ocr_bm25`.
  - `asr_dense`.
  - `asr_bm25`.
  - `summary_dense`.
  - `summary_bm25`.
- OpenCLIP `ViT-B-32::openai` text encoder cho visual retrieval.
- Vietnamese semantic encoder đúng identity của contract.
- Concurrent retrieval, per-branch timeout và diagnostics.
- Metadata hydration, ASR interval-to-frame mapping, branch normalization,
  query aggregation, fusion, summary propagation, object constraints và dedup.
- Final candidate giữ `video_id`, internal `frame_id`, `source_frame_idx`,
  timestamp, path, score, evidence và diagnostics.

### 3.3 TRAKE/DANTE

`CONFIRMED_CODE`:

- Parse chuỗi ordered events.
- Encode từng event trong cùng OpenCLIP space với visual corpus.
- Đọc toàn bộ ordered keyframes theo từng video.
- Tính similarity matrix cho từng video.
- DANTE dynamic programming chạy độc lập trong từng video.
- Backtracking deterministic và giữ một frame match cho mỗi event.
- Trả top-k video sequences cùng diagnostics.
- Production-side `MilvusSQLiteVisualCorpusAdapter` đọc vector Milvus, exact
  JOIN SQLite metadata, kiểm tra norm/dimension/identity và tạo local timeline.

### 3.4 VQA

`CONFIRMED_CODE`:

- Rewrite/reuse KIS retrieval để tìm candidate frames.
- Evidence budget, primary/neighbor selection và text caps.
- `FilesystemImageResolver` kiểm tra file nằm dưới configured data root, không
  lộ absolute local path.
- `ElasticsearchEvidenceHydrator` đọc:
  - OCR theo `frame_id`.
  - ASR theo closed-interval overlap.
  - Summary theo `video_id`.
- Evidence-only VLM request, response validation và evidence-ID grounding.
- Explicit `insufficient_evidence`, timeout, retry và degradation diagnostics.
- VQA chỉ được bật production khi host inject một `VLMPort` cụ thể.

### 3.5 Dataset manifest và startup safety

`CONFIRMED_CODE`:

- Strict `DatasetManifest` đúng `self-indexed-v2`.
- Chỉ chấp nhận `status=READY`.
- Kiểm tra contract/model identity, dimensions, normalization flags, bbox
  space, frame-index base và record-count relations.
- `DatasetManifestGate` pin dataset ID/fingerprint khi startup.
- Phát hiện manifest đổi sau startup và yêu cầu restart.
- Production bắt buộc bật manifest gate và cấu hình expected fingerprint.
- `online.validate_contract` kiểm tra manifest cùng SQLite, Milvus và
  Elasticsearch bằng read-only operations.

### 3.6 Submission logical serializers

`CONFIRMED_CODE`:

- KIS xuất `{video_id, frame_id}` với `frame_id = source_frame_idx`.
- VQA xuất `{video_id, frame_id, answer}` từ image evidence được chọn rõ ràng.
- TRAKE xuất `{video_id, frame_ids}` theo đúng event order.
- Enforce giới hạn tối đa 100 answer rows.
- Không tính ngược frame bằng `timestamp * fps`.

### 3.7 Runtime composition

`CONFIRMED_CODE`:

- KIS composition dùng adapter/model cấu hình qua environment.
- TRAKE data-backed có thể bật bằng `AIC_ONLINE_TRAKE_ENABLED=true`.
- VQA data-backed có thể bật bằng `AIC_ONLINE_VQA_ENABLED=true` khi đã inject
  production `VLMPort`.
- Advanced modes có readiness riêng; lifecycle drain executor trước khi đóng
  infrastructure.
- `.env.example`, `AGENTS.md` và `online/README.md` đã được cập nhật.

## 4. Bằng chứng kiểm thử

Lệnh chuẩn đã chạy:

```powershell
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online -q
```

Kết quả:

```text
492 passed, 2 warnings
```

Hai warning là Starlette deprecation cho tên constant HTTP 422; không phải test
failure và không ảnh hưởng logic hiện tại.

Các gate bổ sung:

```text
python -m compileall -q online query_understanding retrieval_api -> PASS
git diff --check                                           -> PASS
```

Vertical slice bằng fixture self-indexed đã đi qua adapter shape thật của:

```text
Milvus records + SQLite metadata
-> TRAKE similarity/DANTE

SQLite image path + Elasticsearch OCR/ASR/summary
-> VQA evidence selection
-> fake grounded VLM
```

Fixture chứng minh code path và contract; nó không chứng minh SDK/service/model
production hoạt động trên máy thi.

## 5. Những phần chưa hoàn thành

### 5.1 Chờ Offline/data thật

`NEED_RUNTIME_VERIFICATION`:

- Manifest READY thật và expected fingerprint.
- SQLite `videos/metadata/objects` thật.
- Milvus bốn collections thật và `query_iterator` của deployed pymilvus.
- Elasticsearch ba indexes thật cùng ICU analyzer.
- Keyframe files thật dưới configured data root.
- Exact cross-database JOIN trên toàn dataset.
- Record counts, vector dimensions và L2 norms trên dataset thật.
- Một KIS, một TRAKE và một VQA vertical slice thật.

### 5.2 Chờ quyết định model/runtime

`OPEN_QUESTION`:

- Chọn VLM production cho VQA.
- Chạy VLM local/GPU hay gọi external API.
- Cách VLM adapter upload/encode ảnh từ relative `image_rel_path`.
- Chọn LLM rewrite production; hiện có structured port, timeout và safe no-op
  fallback nhưng chưa tự chọn provider.
- Credential, rate limit, cost và retry policy của provider.

### 5.3 Chờ benchmark/tuning

`NEED_RUNTIME_VERIFICATION`:

- Per-branch top-k.
- Query rewrite quality.
- Ranking normalization/fusion weights.
- Summary/object boost.
- TRAKE latency khi scan full visual corpus.
- VQA evidence budget và VLM latency/cost.
- Throughput, memory, GPU/CPU và concurrent-user limits.

Ranking policy hiện vẫn mang trạng thái `experimental`; production mode không
cho phép đánh dấu sẵn sàng cho đến khi policy được benchmark và approve.

### 5.4 Chờ BTC công bố transport nộp bài

`OPEN_QUESTION`:

- HTTP endpoint hoặc file upload.
- JSON/CSV wrapper và header chính xác.
- Delimiter/encoding/filename.
- Authentication/session/query identifier.
- Submit/update/delete semantics và timeout.

Logical rows và external field meanings đã biết; chỉ transport wrapper còn mở.

### 5.5 UI và vận hành thi

`NEED_RUNTIME_VERIFICATION`:

- UI hiển thị keyframes/video từ artifact thật.
- Chọn/đổi thứ tự tối đa 100 đáp án trước khi nộp.
- UI cho TRAKE chọn đúng một frame mỗi event.
- UI cho Q&A chọn frame và answer.
- Logging/metrics, warm-up, backup/rollback và rehearsal trên máy thi.

### 5.6 Không thuộc baseline bắt buộc

`OPTIONAL`:

- Stable Diffusion branch.
- QUEST expansion/reranking.

Hai phần này không được phép chặn KIS/TRAKE/VQA baseline.

## 6. Gói bàn giao cần nhận từ Offline

Offline cần cung cấp tối thiểu:

1. Commit hash producer/indexer cuối.
2. Dataset ID và `sha256` fingerprint.
3. `dataset-manifest.json` trạng thái READY.
4. SQLite database read-only.
5. Milvus collection names/schema/index parameters.
6. Elasticsearch index names/mappings/analyzer.
7. Configured data root chứa keyframe và raw video relative paths.
8. Fixture nhỏ tối thiểu hai video.
9. Log Offline contract verifier PASS.
10. Known missing/optional artifacts.

Không được kết nối production nếu manifest, database và artifact thuộc các lần
Offline chạy khác nhau.

## 7. Thứ tự công việc tiếp theo

1. Nhận fixture và manifest từ Offline.
2. Cài runtime/encoder dependencies trong đúng environment triển khai.
3. Chạy `python -m online.validate_contract --fail-on-partial`.
4. Chạy real KIS vertical slice và kiểm tra `(video_id, frame_id)` BTC.
5. Bật TRAKE, chạy một ordered-event query thật và đo latency.
6. Chọn/integrate VLM và LLM rewrite provider.
7. Chạy VQA thật với image/OCR/ASR/summary evidence.
8. Benchmark và approve ranking policy.
9. Bổ sung transport adapter ngay khi BTC công bố cách gửi bài.
10. Hoàn thiện UI và rehearsal end-to-end.

## 8. Readiness declaration

```text
ONLINE_CODE_READY_FOR_OFFLINE_HANDOFF = YES
SDK_FREE_ONLINE_TESTS                 = PASS (492)
REAL_DATA_VERIFIED                    = NO
PRODUCTION_VLM_SELECTED               = NO
BTC_LOGICAL_ANSWER_FORMAT_KNOWN       = YES
BTC_SUBMISSION_TRANSPORT_KNOWN        = NO
COMPETITION_READY                     = NO
```
