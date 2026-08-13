# Báo cáo Phase Online gửi nhóm Offline kiểm tra

Ngày cập nhật: 2026-08-13

Online branch: `feature/online-phase-Knguyen`

Online revision: commit mới nhất trên `feature/online-phase-Knguyen`

Contract áp dụng: `self-indexed-v2`

Tài liệu contract đầy đủ:
`docs/22-OFFLINE-TO-ONLINE-DATA-CONTRACT-SELF-INDEXED-V2.md`

## 1. Mục đích của báo cáo

File này dùng để nhóm Offline:

1. Biết chính xác Phase Online đã code đến đâu.
2. Kiểm tra dữ liệu Offline có đúng với những gì Online đang đọc hay không.
3. Biết những phần nào chưa thể xác nhận nếu chưa có dataset thật.
4. Chuẩn bị một gói bàn giao đủ để chạy kiểm thử end-to-end.
5. Phản hồi các `CONTRACT_MISMATCH` trước khi hai phase tích hợp.

Đây là báo cáo tích hợp, không thay thế schema và quy tắc chi tiết trong
`docs/22-OFFLINE-TO-ONLINE-DATA-CONTRACT-SELF-INDEXED-V2.md`.

## 2. Kết luận nhanh

```text
ONLINE_CODE_READY_FOR_OFFLINE_HANDOFF = YES
ONLINE_SDK_FREE_TESTS                 = PASS (500)
OFFLINE_REAL_DATA_RECEIVED            = NO
REAL_DATABASES_VERIFIED               = NO
REAL_MODEL_COMPATIBILITY_VERIFIED     = NO
REAL_END_TO_END_VERIFIED              = NO
COMPETITION_READY                     = NO
```

Ý nghĩa:

- `CONFIRMED_CODE`: logic Online và adapter cần thiết đã được cài đặt, có unit
  test và fake-data integration test.
- `NEED_RUNTIME_VERIFICATION`: Online chưa được chạy với dataset READY thật do
  Offline sản xuất, vì vậy chưa thể xác nhận schema thực tế, JOIN thực tế, model
  identity, vector dimension/norm, file ảnh hoặc hiệu năng.
- Online không yêu cầu Offline chờ VLM, LLM rewrite, UI hoặc giao thức submit của
  BTC mới bàn giao dataset. Các phần này không thuộc producer contract của
  Offline.

## 3. Kiến trúc hai phase đã thống nhất

Đội chỉ lấy raw video của Ban tổ chức. Các keyframe, feature và index dùng bởi
hệ thống sẽ do đội tự tạo.

```text
Raw videos
→ Offline phát hiện shot và trích keyframe
→ Offline tạo visual/OCR/ASR/summary/object artifacts
→ Offline index SQLite + Milvus + Elasticsearch
→ Offline publish dataset-manifest.json trạng thái READY
→ Online kiểm tra manifest và contract
→ Online mở SQLite read-only, truy vấn Milvus/Elasticsearch
→ Online retrieval/ranking theo KIS, TRAKE hoặc VQA
→ Online đổi source_frame_idx thành trường frame_id bên ngoài để nộp BTC
```

Online business logic không đọc trực tiếp JSON/Parquet trung gian của Offline.
Các artifact trung gian chỉ dùng cho build, audit hoặc debug ở phía Offline.

## 4. Những phần Phase Online đã hoàn thành

### 4.1 Domain contract và định danh

`CONFIRMED_CODE`:

- Đã có model bất biến cho frame, ASR interval, video, object, branch result,
  diagnostics, TRAKE và VQA.
- Đã phân biệt rõ ba định danh:
  - `video_id`: ID video xuyên suốt toàn hệ thống.
  - `frame_id`: JOIN key nội bộ của keyframe.
  - `source_frame_idx`: chỉ số frame thật trong raw video, zero-based.
- Đã validate canonical internal `frame_id`:

```text
{video_id}_{shot_id:05d}_{position_code:03d}
```

Ví dụ:

```text
L21_V001_00003_050
```

- Đã giữ `source_frame_idx` xuyên suốt retrieval, ranking và output.
- Đã dedup kết quả KIS theo `(video_id, source_frame_idx)`.
- Online không tính lại frame bằng `timestamp * fps`, không suy ra từ filename
  và không dùng Milvus `pk` làm domain ID.

### 4.2 Cấu hình và lifecycle dữ liệu

`CONFIRMED_CODE`:

- URI, database path, table/index/collection name và timeout đều cấu hình được.
- SQLite được mở read-only và bật `PRAGMA query_only=ON`.
- Adapter không để SDK object thoát ra business logic.
- Runtime có lifecycle đóng connection và drain executor đúng thứ tự.
- Production composition yêu cầu manifest gate.

Các biến cấu hình chính Online đang chờ từ gói triển khai:

```text
AIC_ONLINE_DATASET_MANIFEST_PATH
AIC_ONLINE_DATA_ROOT
AIC_ONLINE_DATASET_CONTRACT_VERSION
AIC_ONLINE_DATASET_EXPECTED_FINGERPRINT
AIC_ONLINE_SQLITE_PATH
AIC_ONLINE_MILVUS_URI
AIC_ONLINE_MILVUS_VISUAL_COLLECTION
AIC_ONLINE_MILVUS_OCR_COLLECTION
AIC_ONLINE_MILVUS_ASR_COLLECTION
AIC_ONLINE_MILVUS_SUMMARY_COLLECTION
AIC_ONLINE_ES_URI
AIC_ONLINE_ES_OCR_INDEX
AIC_ONLINE_ES_ASR_INDEX
AIC_ONLINE_ES_SUMMARY_INDEX
```

Secrets không được ghi vào manifest, Markdown hoặc Git.

### 4.3 SQLite adapter

`CONFIRMED_CODE`:

- Đọc bảng `videos`, `metadata`, `objects`.
- Batch hydrate frame hit thành `FrameCandidate`.
- Giữ `frame_id`, `video_id`, `shot_id`, `source_frame_idx`, timestamp và
  `image_rel_path`.
- Đọc width/height để Online normalize object bbox khi cần.
- Phân biệt record thiếu với lỗi database.
- Không ghi hoặc thay đổi SQLite.

### 4.4 Milvus adapters

`CONFIRMED_CODE`:

- Hỗ trợ bốn collection:
  - `visual_features`.
  - `ocr_features`.
  - `asr_features`.
  - `summary_features`.
- Hỗ trợ dense search theo đúng candidate level:
  - visual/OCR trả frame hit.
  - ASR trả interval hit.
  - summary trả video hit.
- Đã kiểm tra dimension, finite vector, L2 norm và model compatibility tại các
  boundary liên quan.
- Có `MilvusSQLiteVisualCorpusAdapter` đọc toàn bộ ordered visual corpus theo
  từng video cho TRAKE/DANTE và exact-hydrate qua SQLite.
- Online không dùng Milvus internal `pk` để JOIN.

### 4.5 Elasticsearch adapters

`CONFIRMED_CODE`:

- Hỗ trợ lexical search trên:
  - `ocr_texts`.
  - `asr_transcripts`.
  - `video_summaries`.
- Có evidence hydrator cho VQA:
  - OCR theo exact `frame_id`.
  - ASR theo closed-interval overlap với timestamp frame.
  - summary theo exact `video_id`.
- Không parse Elasticsearch `_id` để dựng `(video_id, interval_id)`.

### 4.6 Manifest gate và contract validator

`CONFIRMED_CODE`:

- Chỉ chấp nhận `contract_version=self-indexed-v2`.
- Chỉ chấp nhận manifest `status=READY`.
- Validate fingerprint dạng `sha256:<64 lowercase hex>`.
- Validate frame-index base, bbox space, model IDs, revision, dimension,
  normalization và record-count relations.
- Pin `dataset_id` và fingerprint tại startup.
- Nếu active manifest đổi trong lúc chạy, Online báo lỗi và yêu cầu restart;
  không trộn hai dataset version.
- `python -m online.validate_contract --fail-on-partial` hỗ trợ kiểm tra read-only
  manifest, SQLite, Milvus và Elasticsearch.
- Validator full-scan theo batch, không tải toàn dataset vào RAM.
- Mọi vector/key/path được kiểm tra; duplicate domain key bị từ chối.
- Complete key sets được so bằng deterministic digest có exact duplicate gate.
- Count thật của 10 resource được đối chiếu với `manifest.record_counts`.
- Sample checks chỉ là diagnostics và không thể tạo `audit_scope=FULL`.

### 4.7 KIS retrieval và ranking

`CONFIRMED_CODE`:

- Textual KIS và Video KIS dùng chung text-to-keyframe pipeline.
- v-KIS: thí sinh xem clip trên màn hình rồi tự viết text query; Online không
  chờ file clip query từ BTC.
- Đã có bảy retrieval branches:

```text
visual_dense
ocr_dense
ocr_bm25
asr_dense
asr_bm25
summary_dense
summary_bm25
```

- Có OpenCLIP text encoder cho visual branch và Vietnamese text encoder cho ba
  semantic text branches.
- Có concurrent execution, per-branch timeout và diagnostics.
- Có ASR interval-to-frame mapping, query-variant aggregation, per-branch
  normalization, fusion, summary propagation, object constraints và dedup.
- Final candidate vẫn giữ provenance để truy lỗi hoặc tune ranking.

### 4.8 TRAKE/DANTE

`CONFIRMED_CODE`:

- Parse ordered events trong query.
- Encode từng event bằng cùng visual-semantic model space với visual corpus.
- Tính similarity matrix độc lập cho từng video.
- Chạy DANTE dynamic programming và deterministic backtracking.
- Không cho transition giữa hai video.
- Trả một `source_frame_idx` cho mỗi event theo đúng thứ tự event.
- Có data-backed visual corpus adapter nối Milvus với SQLite.

### 4.9 VQA

`CONFIRMED_CODE`:

- Reuse KIS để tìm frame bằng câu hỏi/rewrite.
- Chọn primary/neighbor evidence trong budget có giới hạn.
- Resolve `image_rel_path` an toàn dưới configured data root.
- Hydrate OCR, ASR và summary evidence từ Elasticsearch.
- Có VLM request/response contract, evidence-ID grounding, timeout, retry và
  `insufficient_evidence`.
- Không gửi toàn bộ dataset vào VLM.

`OPEN_QUESTION`: VLM production cụ thể chưa được chọn; đây không phải lỗi hay
thiếu dữ liệu của Offline.

### 4.10 Output theo thể lệ BTC

`CONFIRMED_CODE`:

- Đã có logical serializer:
  - KIS: `<video_id>, <frame_id>`.
  - Q&A/VQA: `<video_id>, <frame_id>, <answer>`.
  - TRAKE: `<video_id>, <frame_id_1>, ..., <frame_id_n>`.
- `frame_id` bên ngoài có giá trị bằng `source_frame_idx` nội bộ.
- Không xuất internal key dạng `L21_V001_00003_050` vào vị trí frame số.
- Enforce tối đa 100 answer rows cho mỗi query.
- TRAKE giữ frame theo đúng thứ tự event.

`OPEN_QUESTION`: BTC chưa mô tả endpoint, JSON/CSV wrapper, authentication và
submit/update semantics. Đây là transport layer, không ảnh hưởng Offline schema.

## 5. Bằng chứng kiểm thử Online

Tại revision full-audit mới nhất trên branch Online, các lệnh sau đã chạy thành công:

```powershell
python -m compileall -q online query_understanding retrieval_api
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online -q
git diff --check
```

Kết quả:

```text
compileall       PASS
Online tests     500 passed, 2 warnings
diff check       PASS
```

Hai warning là Starlette deprecation cho tên HTTP 422 constant, không phải test
failure và không liên quan Offline contract.

Fake-data vertical slice đã chứng minh các đường code:

```text
Milvus-shaped visual records + SQLite metadata
→ TRAKE similarity matrix
→ DANTE sequence

SQLite image path + Elasticsearch-shaped OCR/ASR/summary
→ VQA evidence selection
→ fake grounded VLM
```

Các test trên chỉ xác nhận code và contract shape. Chúng không thay thế kiểm
thử trên database, SDK, model và artifact thật do Offline tạo.

## 6. Contract dữ liệu Offline cần đối chiếu

### 6.1 Identity và path

| Thành phần | Online đang yêu cầu |
|---|---|
| `video_id` | Stem nguyên vẹn của raw video; không trim; không parse bằng `split("_")` |
| `frame_id` | `{video_id}_{shot_id:05d}_{position_code:03d}` |
| `shot_id` | Integer, bắt đầu từ 0 trong từng video |
| `source_frame_idx` | Integer zero-based, frame thật đã decode |
| `interval_id` | Chuỗi số không âm, unique trong từng video |
| `image_rel_path` | POSIX relative path, không absolute, không `..` |
| `source_video_rel_path` | Relative path dưới data root |

Nếu extractor fallback sang frame khác, `source_frame_idx` phải ghi frame thật
đã decode và lưu. Không được ghi frame mục tiêu ban đầu nếu ảnh thực tế thuộc
frame fallback.

### 6.2 SQLite

Online yêu cầu `data/metadata.db` hoặc đường dẫn cấu hình tương đương, gồm:

```text
videos(video_id, source_video_rel_path, fps, duration_sec,
       frame_count, width, height)

metadata(frame_id, video_id, shot_id, source_frame_idx,
         timestamp, image_rel_path)

objects(id, frame_id, label, confidence,
        x_min, y_min, x_max, y_max, model_source)
```

Quy tắc quan trọng:

- Một metadata row tương ứng một keyframe thật.
- `source_frame_idx < videos.frame_count`.
- `image_rel_path` phải tồn tại và decode được.
- `(video_id, source_frame_idx)` không bắt buộc unique; Online tự dedup.
- Object label phải lowercase/casefold-normalized.
- Object confidence thuộc `[0,1]`.
- Object bbox dùng absolute pixel XYXY:

```text
0 <= x_min < x_max <= video width
0 <= y_min < y_max <= video height
```

### 6.3 Milvus

Online yêu cầu:

| Collection | Domain key | Vector |
|---|---|---|
| `visual_features` | `frame_id` | OpenCLIP visual, 512 |
| `ocr_features` | `frame_id` | Vietnamese text vector |
| `asr_features` | `(video_id, interval_id)` | Vietnamese text vector |
| `summary_features` | `video_id` | Vietnamese text vector |

Tất cả vector phải:

```text
1-D
finite
L2-normalized
HNSW index
IP metric
```

Baseline index/search:

```text
M = 16
efConstruction = 256
search ef = 128
```

Model identity Online đang khóa:

```text
visual_model_id = ViT-B-32::openai
visual_dimension = 512
visual_normalized = true

text_model_name = dangvantuan/vietnamese-embedding
text_model_revision = 4ab46e46ba5902328ba0742e489e75f787932f2b
text_dimension = 768
text_max_length = 256
ASR/OCR pooling = direct_l2
summary pooling = chunk_mean_l2
text_normalized = true
```

Nếu Offline dùng khác revision, dimension, max length hoặc pooling, hai bên phải
đổi contract/dataset version trước khi kết nối. Online không được tự đoán hoặc
âm thầm dùng encoder khác.

### 6.4 Elasticsearch

Online yêu cầu Elasticsearch 8.x và ba index:

```text
ocr_texts
asr_transcripts
video_summaries
```

Analyzer baseline:

```text
vietnamese_analyzer
tokenizer: icu_tokenizer
filters: icu_folding, lowercase
```

Các field bắt buộc:

```text
ocr_texts:
  frame_id, video_id, shot_id, ocr_text_concat

asr_transcripts:
  video_id, interval_id, start_time_sec, end_time_sec, cleaned_text

video_summaries:
  video_id, summary
```

### 6.5 Exact JOIN matrix

| Nguồn A | Khóa | Nguồn B | Điều kiện |
|---|---|---|---|
| Milvus visual | `frame_id` | SQLite metadata | Tập ID phải bằng nhau |
| Milvus OCR | `frame_id` | Elasticsearch OCR | Tập ID phải bằng nhau |
| Milvus OCR | `frame_id` | SQLite metadata | Phải là tập con metadata |
| SQLite objects | `frame_id` | SQLite metadata | Phải là tập con metadata |
| Milvus ASR | `(video_id, interval_id)` | Elasticsearch ASR | Tập key phải bằng nhau |
| Milvus summary | `video_id` | Elasticsearch summary | Tập ID phải bằng nhau |
| SQLite metadata | `video_id` | SQLite videos | Mọi row phải JOIN |

Mọi JOIN dùng exact equality. Không dùng `LIKE`, full-text match, Milvus `pk`
hoặc Elasticsearch `_id` đã parse để thay domain key.

### 6.6 Dataset manifest

Offline phải xuất `dataset-manifest.json` chứa tối thiểu:

```json
{
  "contract_version": "self-indexed-v2",
  "dataset_id": "aic2026-team-run-001",
  "dataset_fingerprint": "sha256:<64-lowercase-hex>",
  "status": "READY",
  "frame_index_base": 0,
  "bbox_space": "absolute_pixel_xyxy",
  "visual_model_id": "ViT-B-32::openai",
  "visual_dimension": 512,
  "visual_normalized": true,
  "text_model_name": "dangvantuan/vietnamese-embedding",
  "text_model_revision": "4ab46e46ba5902328ba0742e489e75f787932f2b",
  "text_dimension": 768,
  "text_max_length": 256,
  "record_counts": {
    "videos": 0,
    "metadata": 0,
    "objects": 0,
    "visual_features": 0,
    "ocr_features": 0,
    "asr_features": 0,
    "summary_features": 0,
    "ocr_texts": 0,
    "asr_transcripts": 0,
    "video_summaries": 0
  },
  "created_at_utc": "2026-08-05T00:00:00Z"
}
```

Các số `0` trong ví dụ phải được thay bằng count thật. Fingerprint tối thiểu
phải phụ thuộc vào raw video IDs/content hashes, contract version, model
identity/revision và producer configuration ảnh hưởng artifact.

Offline không được ghi `READY` trước khi toàn bộ artifact và database cùng một
run đã qua validation.

## 7. Những phần Online chưa hoàn thành

### 7.1 Chưa thể hoàn thành nếu chưa có handoff thật từ Offline

`NEED_RUNTIME_VERIFICATION`:

- Chưa mở SQLite thật và kiểm tra đủ ba table/schema/index.
- Chưa kiểm tra keyframe file thật tồn tại và decode được.
- Chưa kết nối bốn Milvus collection thật.
- Chưa kết nối ba Elasticsearch index thật và kiểm tra ICU analyzer.
- Chưa đối chiếu record counts trong manifest với database.
- Chưa kiểm tra exact JOIN toàn dataset.
- Chưa kiểm tra vector dimension, norm và model identity trên record thật.
- Chưa xác nhận `source_frame_idx` đúng pixel/frame thực tế của ảnh.
- Chưa chạy KIS, TRAKE và VQA vertical slice với dataset thật.
- Chưa benchmark latency, RAM/VRAM và concurrency trên kích thước dataset thật.

Đây là các gate tích hợp, không có nghĩa Online cần viết lại thuật toán trước khi
nhận dữ liệu.

### 7.2 Phần Online tự chịu trách nhiệm sau handoff

`OPEN_QUESTION` hoặc `NEED_RUNTIME_VERIFICATION` nhưng không yêu cầu Offline sửa:

- Chọn và tích hợp VLM production cho VQA.
- Chọn LLM rewrite provider hoặc giữ safe no-op fallback.
- Benchmark/tune branch top-k, normalization/fusion weights, summary/object
  boosts và TRAKE latency.
- Chuyển ranking policy từ `experimental` sang approved sau benchmark.
- Hoàn thiện UI, observability, warm-up và rehearsal.
- Viết transport adapter khi BTC công bố endpoint/file wrapper chính xác.

### 7.3 Optional, không chặn baseline

```text
Stable Diffusion branch
QUEST expansion/reranking
```

Offline không cần tạo thêm artifact cho hai nhánh này ở gói handoff baseline.

## 8. Checklist Offline phải chạy trước khi bàn giao

Offline vui lòng đánh dấu từng dòng `PASS`, `FAIL` hoặc `NOT_AVAILABLE` và gửi
kèm bằng chứng/lệnh kiểm tra:

- [ ] 1. Mỗi raw video có đúng một row hợp lệ trong `videos`.
- [ ] 2. Mọi `metadata.video_id` exact-JOIN được `videos.video_id`.
- [ ] 3. Mọi `source_frame_idx` là zero-based, nằm trong frame count và đúng ảnh.
- [ ] 4. Mọi `image_rel_path` là relative path, tồn tại và decode được.
- [ ] 5. Không publish keyframe rỗng, đen giả hoặc synthetic placeholder.
- [ ] 6. Mọi `frame_id` unique, đúng regex và suffix khớp `shot_id`.
- [ ] 7. Tập Milvus visual `frame_id` bằng tập SQLite metadata `frame_id`.
- [ ] 8. OCR semantic IDs bằng OCR lexical IDs và là tập con metadata.
- [ ] 9. Object frame IDs là tập con metadata.
- [ ] 10. ASR semantic keys bằng ASR lexical keys theo `(video_id, interval_id)`.
- [ ] 11. Summary semantic IDs bằng summary lexical video IDs.
- [ ] 12. Mọi vector finite, đúng dimension và L2-normalized.
- [ ] 13. Mọi object bbox nằm đúng absolute pixel bounds của video.
- [ ] 14. Manifest counts bằng count thật trong databases.
- [ ] 15. Manifest fingerprint, model IDs/revision và producer config đúng.
- [ ] 16. Nếu một bước build/audit fail, dataset đó không được publish `READY`.

Offline verifier phải trả exit code khác 0 nếu bất kỳ gate bắt buộc nào fail.

## 9. Gói bàn giao Online cần nhận

Không chỉ báo “Offline đã chạy xong”. Vui lòng gửi đủ:

1. Commit hash cuối của Offline producer/indexer.
2. `contract_version`.
3. `dataset_id`.
4. `dataset_fingerprint`.
5. Path của `dataset-manifest.json`.
6. Path SQLite database.
7. Keyframe root.
8. Raw video root.
9. Milvus URI và bốn collection names.
10. Elasticsearch URI và ba index names.
11. Model identities/revisions/pooling/dimensions.
12. Record counts của mọi table/collection/index.
13. Full validation command và nguyên output PASS.
14. Danh sách artifact/branch optional chưa có.
15. Một fixture nhỏ tối thiểu hai video để Online kiểm tra nhanh.

Gói fixture cần đủ cả:

```text
raw video
keyframe files
SQLite rows
Milvus records/vectors hoặc script index fixture
Elasticsearch documents hoặc script index fixture
READY manifest
```

## 10. Quy trình tích hợp sau khi nhận handoff

Online sẽ thực hiện theo thứ tự:

1. Pin đúng Offline commit, dataset ID và fingerprint.
2. Cấu hình đường dẫn/URI bằng environment variables.
3. Cài đúng runtime SDK/model dependencies trong environment triển khai.
4. Chạy:

```powershell
python -m online.validate_contract --fail-on-partial
```

5. Nếu validator fail, phân loại rõ schema, JOIN, model, artifact hay service.
6. Chạy một KIS query thật và xác nhận output BTC dùng `source_frame_idx`.
7. Chạy một TRAKE ordered-event query thật.
8. Chạy VQA evidence hydration thật; VLM có thể dùng fake trước để tách lỗi data.
9. Chạy benchmark và tune ranking/latency.
10. Chỉ tuyên bố integration PASS khi cả hai bên cùng ký nhận contract version,
    commit hash và fingerprint.

Không kết nối production nếu manifest, SQLite, Milvus, Elasticsearch hoặc file
artifact thuộc các lần Offline build khác nhau.

## 11. Mẫu phản hồi đề nghị nhóm Offline điền

```text
OFFLINE PRODUCER COMMIT:
CONTRACT VERSION:
DATASET ID:
DATASET FINGERPRINT:
MANIFEST PATH:
SQLITE PATH:
KEYFRAME ROOT:
RAW VIDEO ROOT:
MILVUS URI/COLLECTIONS:
ELASTICSEARCH URI/INDEXES:

VISUAL MODEL ID / DIMENSION / NORMALIZED:
TEXT MODEL NAME / REVISION / DIMENSION / MAX LENGTH:
OCR-ASR POOLING:
SUMMARY POOLING:

VALIDATION COMMAND:
VALIDATION RESULT:
CHECKLIST 1-16 RESULT:
KNOWN MISSING OPTIONAL ARTIFACTS:

CONTRACT_MISMATCHES FOUND:
- ...

OFFLINE_READY_FOR_ONLINE_VERTICAL_SLICE = YES / NO
```

Nếu có mismatch, vui lòng ghi rõ field/table/collection/index, giá trị Offline
đang sinh, giá trị Online yêu cầu và lý do muốn giữ hoặc đổi. Không tự thêm
compatibility workaround ở một phía vì sẽ che lỗi dữ liệu.

## 12. Quyết định readiness hiện tại

Phase Online đã sẵn sàng **nhận handoff và bắt đầu kiểm tra dữ liệu thật**.
Phase Online chưa thể được gọi là **competition-ready** vì chưa nhận dataset READY
thật, chưa chạy real vertical slice và chưa benchmark production.

Điểm chặn tiếp theo không phải viết thêm retrieval baseline. Điểm chặn là nhóm
Offline xác nhận contract, gửi fixture/manifest/databases nhất quán, sau đó hai
bên cùng chạy validation và vertical slice.
