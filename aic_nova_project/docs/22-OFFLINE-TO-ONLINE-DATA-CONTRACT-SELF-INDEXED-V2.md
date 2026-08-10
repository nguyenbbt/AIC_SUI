# Offline → Online Data Contract — Self-Indexed V2

## 0. Yêu cầu gửi cho nhóm Offline

Tài liệu này là yêu cầu sửa chính thức gửi cho nhóm Offline sau khi review:

```text
Branch đã review: origin/main
Commit đã review: e11cdbf41ee79e62c77bfea6d398568337f7d8d4
Ngày review: 2026-08-05
Trạng thái hiện tại: CONTRACT_MISMATCH — chưa được dùng làm contract production
Contract đích: self-indexed-v2
```

Nhóm Offline cần thực hiện các mục P0 trong Mục 13, bổ sung test ở Mục 12 và
gửi lại đầy đủ gói handoff ở Mục 14. Không cần thay đổi kiến trúc retrieval hoặc
thay model theo dữ liệu keyframe/feature của Ban tổ chức. Quyết định kiến trúc đã
chốt là chỉ nhận raw video, sau đó đội tự sinh toàn bộ keyframe và feature.

Online có thể code song song dựa trên schema đích trong tài liệu này, nhưng chỉ
được nối end-to-end và chốt adapter sau khi Offline xác nhận các mục P0 đã hoàn
thành. Nếu Offline đề nghị thay đổi bất kỳ tên trường, kiểu dữ liệu hoặc semantics
nào trong tài liệu, hai bên phải thống nhất trước khi sửa code; không tự đổi contract.

### 0.1 Kết quả cần nhận lại từ Offline

Tối thiểu phải có:

1. Commit hash chứa toàn bộ sửa đổi.
2. SQLite schema mới và migration/reset instruction.
3. Một fixture nhỏ có ít nhất hai video để Online chạy integration test.
4. Manifest của fixture, gồm model identity và record counts.
5. Log test và log full contract verification đều pass.
6. Xác nhận chính xác quy tắc `source_frame_idx`, bbox và đường dẫn ảnh/video.

### 0.2 Bốn điều kiện chặn bàn giao hiện tại

1. SQLite đang bỏ `keyframes[].frame_index`, nên Online không có
   `source_frame_idx` chính xác để nộp bài.
2. SQLite đang bỏ `keyframes[].file_path`, nên VQA/UI không resolve được ảnh qua
   database contract.
3. Keyframe extractor có thể lưu frame fallback hoặc ảnh đen nhưng vẫn ghi metadata
   của frame đích.
4. Bbox đang là pixel tuyệt đối nhưng contract Online trước đó hiểu là normalized;
   cần chốt pixel XYXY cùng `width/height` như tài liệu này.

## 1. Trạng thái và quyết định kiến trúc

Tài liệu này là contract bàn giao mới giữa Phase Offline và Phase Online sau
quyết định của nhóm:

```text
Chỉ nhận raw video từ Ban tổ chức
→ đội tự chia shot và trích keyframe
→ đội tự chạy visual embedding, OCR, ASR, summary và object detection
→ đội tự index Milvus, Elasticsearch và SQLite
→ Online chỉ đọc các database/artifact đã publish READY
```

Contract này thay thế contract `organizer-v1` trong code Online. Nó kế thừa
schema tự index trước đây nhưng bổ sung những trường bắt buộc để:

- Nộp đúng frame gốc cho Ban tổ chức.
- Resolve ảnh keyframe thật cho VQA và UI.
- Stream visual corpus thật cho TRAKE/DANTE.
- Diễn giải đúng bounding box object.
- Ngăn Online đọc lẫn artifact từ nhiều lần chạy Offline.

Tên contract:

```text
self-indexed-v2
```

Mọi thay đổi phá vỡ schema hoặc semantics bên dưới phải tăng
`contract_version`; không được âm thầm thay đổi dữ liệu nhưng giữ nguyên version.

---

## 2. Nguyên tắc nguồn sự thật

Offline sở hữu việc sinh dữ liệu và publish dataset. Online không đọc raw JSON
hoặc Parquet trong business flow thông thường; Online đọc:

| Resource | Vai trò |
|---|---|
| SQLite | Video metadata, keyframe metadata và object detection |
| Milvus | Visual, OCR, ASR và summary vectors |
| Elasticsearch | OCR, ASR và summary lexical text |
| Dataset manifest | Version, model identity, record counts và trạng thái READY |
| Keyframe root | Ảnh WebP được resolve bằng relative path trong SQLite |

Artifact JSON/Parquet vẫn là nguồn audit, re-index và debugging. Chúng không
được dùng như một database Online thay thế.

Thứ tự nguồn sự thật tại runtime:

1. Code producer/indexer đang deploy.
2. Dataset manifest của đúng lần chạy.
3. Schema database đã audit.
4. Tài liệu này.

Nếu bốn nguồn trên không đồng ý, dataset không được đánh dấu `READY`.

---

## 3. Contract định danh

### 3.1 `video_id`

- Chuỗi Unicode không rỗng, không có whitespace đầu/cuối.
- Bằng stem của raw video do đội nhận từ Ban tổ chức.
- Không parse `video_id` bằng `split("_")`; bản thân ID có thể chứa `_`.
- Là khóa xuyên SQLite, Milvus, Elasticsearch và submission serializer.

Ví dụ:

```text
L21_V001
```

### 3.2 `frame_id`

Canonical internal frame ID:

```text
{video_id}_{shot_id:05d}_{position_code:03d}
```

Ví dụ:

```text
L21_V001_00003_050
```

Semantics:

- `shot_id` bắt đầu từ `0` trong từng video.
- `position_code = int(position * 100)` theo implementation hiện tại.
- Các position mặc định `0.15`, `0.50`, `0.85` tạo mã `015`, `050`, `085`.
- Regex validation: `^.+_[0-9]{5}_[0-9]{3}$`.
- Database chỉ lưu Global ID, không lưu local name
  `shot_00003_pos_050` làm domain ID.
- Không dựng `frame_id` từ timestamp.
- Không dùng Milvus `pk` hoặc SQLite `objects.id` làm frame ID.

`frame_id` là khóa JOIN nội bộ. Nó không thay thế frame index gốc khi nộp bài.

### 3.3 `source_frame_idx`

`source_frame_idx` là chỉ số frame thật trong raw video:

```text
source_frame_idx = keyframes[].frame_index
```

Quy tắc bắt buộc:

- Kiểu integer, `>= 0`.
- Chuẩn `0-based`.
- Là frame thực sự đã decode và lưu thành ảnh keyframe.
- Không được tính lại từ `timestamp_sec * fps` trong Online.
- Không được suy ra từ `frame_id`, `shot_id`, `position_code` hoặc filename.
- Nhiều `frame_id` được phép trỏ tới cùng `source_frame_idx` nếu shot quá ngắn.
- Cặp `(video_id, source_frame_idx)` là identity dùng cho KIS submission và
  competition-frame dedup.

Nếu extractor seek frame đích nhưng decode thất bại rồi fallback sang frame
khác, metadata phải ghi index của frame fallback thực sự. Không được lưu ảnh
frame A nhưng ghi `source_frame_idx` của frame B.

Không được sinh ảnh đen/synthetic placeholder rồi publish như keyframe hợp lệ.
Nếu không decode được frame thật, Offline phải fail hoặc loại keyframe/video có
diagnostics rõ ràng.

### 3.4 `interval_id`

- Chuỗi chữ số ASCII không âm, không có leading zero trừ giá trị `"0"`.
- Chỉ unique trong một video.
- Khóa ASR xuyên database là `(video_id, interval_id)`.
- Không parse Elasticsearch `_id` bằng dấu `_`.

---

## 4. Artifact Module 1 bắt buộc

`processed/metadata/{video_id}.json` phải giữ tối thiểu:

```json
{
  "contract_version": "self-indexed-v2",
  "video_id": "L21_V001",
  "source_path": "videos/L21_V001.mp4",
  "fps": 30.0,
  "duration_sec": 120.5,
  "frame_count": 3615,
  "width": 1920,
  "height": 1080,
  "num_shots": 42,
  "shots": [
    {
      "shot_id": 3,
      "start_frame": 80,
      "end_frame": 110,
      "start_time_sec": 2.667,
      "end_time_sec": 3.667,
      "keyframes": [
        {
          "position": 0.5,
          "position_code": 50,
          "frame_index": 95,
          "time_sec": 3.167,
          "file_path": "keyframes/L21_V001/shot_00003_pos_050.webp"
        }
      ]
    }
  ]
}
```

Yêu cầu:

- `source_path` và `file_path` phải là relative path dưới configured data root.
- Không publish absolute Windows/Linux path vào database.
- `frame_count`, `width`, `height` phải lấy từ video/decoded frame thật.
- `frame_index < frame_count`.
- `time_sec >= 0` và nằm trong duration với tolerance hợp lý.
- Mỗi `frame_id` phải unique.
- Mọi `file_path` phải tồn tại, đọc được và không phải ảnh placeholder.

---

## 5. SQLite contract

SQLite path mặc định:

```text
data/metadata.db
```

Online bắt buộc mở bằng read-only URI và bật `PRAGMA query_only=ON`.

### 5.1 Bảng `videos`

```sql
CREATE TABLE videos (
    video_id              TEXT PRIMARY KEY,
    source_video_rel_path TEXT NOT NULL,
    fps                   REAL NOT NULL,
    duration_sec          REAL NOT NULL,
    frame_count           INTEGER NOT NULL,
    width                 INTEGER NOT NULL,
    height                INTEGER NOT NULL
);
```

Invariants:

- `fps > 0` và finite.
- `duration_sec >= 0` và finite.
- `frame_count > 0`, `width > 0`, `height > 0`.
- `source_video_rel_path` là relative path.

### 5.2 Bảng `metadata`

```sql
CREATE TABLE metadata (
    frame_id         TEXT PRIMARY KEY,
    video_id         TEXT NOT NULL,
    shot_id          INTEGER NOT NULL,
    source_frame_idx INTEGER NOT NULL,
    timestamp        REAL NOT NULL,
    image_rel_path   TEXT NOT NULL,
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);

CREATE INDEX idx_metadata_video_id
    ON metadata(video_id);

CREATE INDEX idx_metadata_video_timeline
    ON metadata(video_id, timestamp, frame_id);

CREATE INDEX idx_metadata_video_source_frame
    ON metadata(video_id, source_frame_idx);
```

Một row tương ứng một keyframe do đội tự trích.

Invariants:

- `shot_id >= 0`.
- `source_frame_idx >= 0` và nhỏ hơn `videos.frame_count`.
- `timestamp >= 0` và finite.
- `image_rel_path` là relative path, tồn tại dưới keyframe root.
- `frame_id` semantic suffix phải khớp `shot_id`.
- Không đặt UNIQUE cho `(video_id, source_frame_idx)`; duplicate source frame
  giữa các keyframe IDs là hợp lệ và được Online dedup sau ranking.

### 5.3 Bảng `objects`

```sql
CREATE TABLE objects (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id     TEXT NOT NULL,
    label        TEXT NOT NULL,
    confidence   REAL NOT NULL,
    x_min        REAL NOT NULL,
    y_min        REAL NOT NULL,
    x_max        REAL NOT NULL,
    y_max        REAL NOT NULL,
    model_source TEXT,
    FOREIGN KEY (frame_id) REFERENCES metadata(frame_id) ON DELETE CASCADE
);

CREATE INDEX idx_objects_frame_id ON objects(frame_id);
CREATE INDEX idx_objects_label ON objects(label);
```

Object semantics:

- `label` phải lowercase/casefold-normalized, không rỗng.
- `confidence` nằm trong `[0,1]`.
- Bounding box là **absolute pixel XYXY**, không phải normalized `[0,1]`.
- `0 <= x_min < x_max <= videos.width`.
- `0 <= y_min < y_max <= videos.height`.
- `model_source` nullable nhưng nếu có phải là chuỗi không rỗng.
- `objects.id` chỉ là khóa nội bộ.

Nếu Online cần position constraint, Online normalize bằng `width/height` từ
`videos`; Offline không được để downstream đoán bbox coordinate space.

---

## 6. Milvus contract

Mọi vector:

- Một chiều, finite.
- L2-normalized.
- HNSW index.
- Inner Product metric.

Thông số baseline:

```text
M = 16
efConstruction = 256
search ef = 128
```

Dimension phải được audit từ schema và manifest; Online không chỉ tin constant.

### 6.1 `visual_features`

```text
pk          INT64 auto_id, internal only
frame_id    VARCHAR
video_id    VARCHAR
shot_id     INT64
embedding   FLOAT_VECTOR(512)
```

Model identity bắt buộc:

```text
model_id = ViT-B-32::openai
dimension = 512
normalized = true
```

Tập `visual_features.frame_id` phải bằng tập `SQLite metadata.frame_id`.

### 6.2 `ocr_features`

```text
pk          INT64 auto_id, internal only
frame_id    VARCHAR
video_id    VARCHAR
embedding   FLOAT_VECTOR(text_dimension)
```

OCR vector chỉ tồn tại cho frame có OCR text không rỗng.

### 6.3 `asr_features`

```text
pk              INT64 auto_id, internal only
video_id        VARCHAR
interval_id     VARCHAR
start_time_sec  FLOAT
end_time_sec    FLOAT
embedding       FLOAT_VECTOR(text_dimension)
```

### 6.4 `summary_features`

```text
pk          INT64 auto_id, internal only
video_id    VARCHAR
embedding   FLOAT_VECTOR(text_dimension)
```

Text model hiện tại:

```text
model_name = dangvantuan/vietnamese-embedding
model_revision = 4ab46e46ba5902328ba0742e489e75f787932f2b
max_length = 256
dimension = 768
ASR/OCR pooling = direct_l2
summary pooling = chunk_mean_l2
normalized = true
```

Nếu Offline thay revision, pooling, max length hoặc dimension thì phải tạo
dataset version mới và re-index toàn collection liên quan.

---

## 7. Elasticsearch contract

Runtime baseline:

```text
Elasticsearch 8.x
Python client >=8,<9
ICU plugin installed
```

Analyzer:

```text
vietnamese_analyzer:
  tokenizer: icu_tokenizer
  filters: icu_folding, lowercase
```

### 7.1 `ocr_texts`

```text
_id             = frame_id
frame_id         keyword
video_id         keyword
shot_id          keyword
ocr_text_concat  text using vietnamese_analyzer
```

### 7.2 `asr_transcripts`

```text
_id             = implementation detail; Online không parse
video_id        keyword
interval_id     keyword
start_time_sec  float
end_time_sec    float
cleaned_text    text using vietnamese_analyzer
```

Domain key là `(video_id, interval_id)`.

### 7.3 `video_summaries`

```text
_id       = video_id
video_id  keyword
summary   text using vietnamese_analyzer
```

---

## 8. Ma trận JOIN bắt buộc

| Nguồn | Khóa | Đích | Quan hệ |
|---|---|---|---|
| Milvus visual | `frame_id` | SQLite metadata | Bằng nhau |
| Milvus OCR | `frame_id` | Elasticsearch OCR | Bằng nhau |
| OCR | `frame_id` | SQLite metadata | Tập con |
| Objects | `frame_id` | SQLite metadata | Tập con |
| Milvus ASR | `(video_id, interval_id)` | Elasticsearch ASR | Bằng nhau |
| Milvus summary | `video_id` | Elasticsearch summary | Bằng nhau |
| SQLite metadata | `video_id` | SQLite videos | Mọi row phải JOIN |

JOIN ID phải dùng exact equality. Không dùng `LIKE`, full-text match hoặc
Milvus `pk` để JOIN.

---

## 9. Dataset manifest

Offline phải publish một manifest JSON cùng dataset:

```json
{
  "contract_version": "self-indexed-v2",
  "dataset_id": "aic2026-team-run-001",
  "dataset_fingerprint": "sha256:...",
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

Fingerprint phải phụ thuộc tối thiểu vào:

- Danh sách raw video IDs và content hashes.
- Contract version.
- Model identities/revisions.
- Producer configuration ảnh hưởng artifact.

Online so sánh manifest fingerprint với expected deployment fingerprint.
Online không tự hash toàn dataset khi startup.

---

## 10. Publish và tính nhất quán

Offline không được ghi trực tiếp đè lên dataset đang được Online phục vụ.

Luồng bắt buộc:

```text
BUILDING namespace/path
→ ghi tất cả artifacts và databases
→ audit schema
→ kiểm tra vector dimension/norm
→ kiểm tra record counts
→ kiểm tra toàn bộ JOIN invariants
→ kiểm tra keyframe files
→ ghi manifest status=READY
→ atomic switch active dataset pointer
```

Nếu bất kỳ bước nào fail:

- Không ghi `READY`.
- Không switch active dataset.
- Giữ dataset READY trước đó để rollback.

Online chỉ mở SQLite read-only và chỉ phục vụ dataset có manifest `READY` đúng
fingerprint.

---

## 11. Boundary output cho Online và submission

Mỗi final KIS candidate nội bộ phải giữ:

```text
frame_id
video_id
shot_id
source_frame_idx
timestamp_sec
image_rel_path
score
provenance
diagnostics
```

Competition serializer không parse hoặc tính lại identity. Nó lấy trực tiếp:

```json
{
  "video_id": "L21_V001",
  "frame_id": 95
}
```

Theo tài liệu vòng sơ tuyển AIC 2026, tên trường logic bên ngoài là
`frame_id`, nhưng giá trị của nó là chỉ số frame gốc mà Online đang giữ nội bộ
dưới tên `source_frame_idx`. Không được xuất chuỗi internal JOIN key
`L21_V001_00003_050` vào vị trí này.

Quy tắc:

- KIS dedup key: `(video_id, source_frame_idx)`.
- `frame_id` chỉ là internal JOIN key.
- `shot_id`, `position_code`, filename và timestamp không thay thế
  `source_frame_idx`.
- Các tuple logic đã được BTC chốt: KIS là `<video_id>, <frame_id>`, Q&A là
  `<video_id>, <frame_id>, <answer>`, TRAKE là
  `<video_id>, <frame_id_1>, ..., <frame_id_n>` và tối đa 100 câu trả lời cho
  mỗi truy vấn. Exact JSON/CSV wrapper, delimiter và endpoint truyền tải vẫn
  chưa được tài liệu này quy định.
- Việc wrapper bên ngoài chưa chốt không cho phép bỏ `source_frame_idx` khỏi
  Offline contract.

TRAKE giữ một source-frame match cho mỗi event theo đúng DANTE sequence;
không dùng KIS deduplicator để xóa event.

VQA phải giữ `video_id`, `source_frame_idx` và evidence IDs của ảnh dùng để trả
lời.

---

## 12. Validation gate bắt buộc trước handoff

Offline chỉ bàn giao khi tất cả gate sau pass:

1. Mọi raw video có một row `videos` hợp lệ.
2. Mọi metadata row JOIN được video.
3. `source_frame_idx` đúng ảnh keyframe thực tế và nằm trong frame count.
4. Mọi `image_rel_path` tồn tại và decode được.
5. Không có synthetic/empty placeholder keyframe.
6. Canonical `frame_id` unique và semantic suffix khớp `shot_id`.
7. Milvus visual frame IDs bằng SQLite metadata frame IDs.
8. OCR vector IDs bằng OCR lexical IDs và là tập con metadata.
9. Object frame IDs là tập con metadata.
10. ASR semantic/lexical keys bằng nhau.
11. Summary semantic/lexical video IDs bằng nhau.
12. Tất cả vectors finite, đúng dimension và L2-normalized.
13. Object bbox đúng absolute-pixel bounds.
14. Manifest record counts bằng database thực tế.
15. Dataset fingerprint/model identities đúng.
16. Indexing failure không publish dataset READY một phần.

Script cross-database verification phải trả non-zero exit code khi bất kỳ gate
nào fail.

---

## 13. Công việc Offline phải sửa so với code hiện tại

### P0 — Bắt buộc trước khi nối Online end-to-end

#### P0.1 Sửa Module 1 và metadata video/keyframe

Các file dự kiến bị ảnh hưởng:

```text
data_pipeline/shot_keyframe/metadata_schema.py
data_pipeline/shot_keyframe/keyframe_extractor.py
data_pipeline/shot_keyframe/pipeline.py
data_pipeline/shot_keyframe/cli.py
data_pipeline/shot_keyframe/resume_validation.py
data_pipeline/shot_keyframe/tests/*
tests/test_module1_resume_artifacts.py
```

Yêu cầu:

1. Mở rộng `VideoMetadata` với `contract_version`, `frame_count`, `width` và
   `height`; lấy từ raw video/decoder thật, không dùng giá trị giả.
2. Đổi `source_path` thành relative path ổn định dưới data root hoặc thêm riêng
   `source_video_rel_path`; không publish đường dẫn tuyệt đối của Modal/Windows.
3. Khi seek `target_idx` thành công, lưu đúng index frame thực sự đã decode.
4. Khi fallback sang `start_frame`, phải lưu `frame_index=start_frame` và timestamp
   tương ứng; không tiếp tục ghi metadata của `target_idx`.
5. Khi không decode được frame thật, phải fail keyframe/video với diagnostics;
   xóa hành vi tạo ảnh đen `np.zeros(...)` rồi coi là output hợp lệ.
6. Resume validation phải kiểm tra source video fingerprint/config, không chỉ kiểm
   tra file ảnh tồn tại và đọc được.
7. `-Force` của one-click runner phải thực sự truyền tới Module 1 hoặc Module 1
   phải có cơ chế invalidation tương đương.
8. Quy định rõ `frame_index` là zero-based absolute frame index trong raw video.

Test bắt buộc:

- Target seek thành công → image và `frame_index` cùng một frame.
- Target seek fail, fallback thành công → metadata ghi index của fallback.
- Target và fallback cùng fail → pipeline fail, không có ảnh placeholder/metadata READY.
- Video raw thay đổi nhưng giữ cùng `video_id` → resume không được dùng artifact cũ.
- `frame_index < frame_count`; `time_sec` hợp lệ; relative path tồn tại và decode được.

#### P0.2 Mở rộng SQLite thành nguồn hydration đầy đủ

Các file dự kiến bị ảnh hưởng:

```text
indexing/src/indexing/data_loader.py
indexing/src/indexing/clients/tabular_client.py
indexing/src/indexing/orchestrator.py
indexing/tests/test_data_loader.py
indexing/tests/test_schema_audit.py
indexing/tests/test_post_index_validation.py
indexing/tests/test_full_rollback.py
```

Yêu cầu:

1. Thêm bảng `videos` đúng schema ở Mục 5.1.
2. Loader phải copy trực tiếp:

   ```text
   keyframes[].frame_index → metadata.source_frame_idx
   keyframes[].file_path  → metadata.image_rel_path
   keyframes[].time_sec   → metadata.timestamp
   ```

3. Không tính `source_frame_idx` bằng `round(timestamp * fps)`.
4. `TabularClient` phải hỗ trợ create, audit, insert, snapshot, restore, delete và
   reset cho bảng/schema mới.
5. Snapshot/rollback phải giữ cả row `videos`, `source_frame_idx` và
   `image_rel_path`; không được rollback về state thiếu trường.
6. Có index timeline và source-frame như Mục 5.2.
7. Schema audit phải fail rõ ràng khi gặp database v1. Offline phải cung cấp lệnh
   reset/re-index hoặc migration chính thức; không tự âm thầm dùng database cũ.

Test bắt buộc:

- Module 1 fixture → SQLite giữ nguyên chính xác `frame_index` và `file_path`.
- Snapshot → replace thất bại → rollback khôi phục byte-for-byte các giá trị nghiệp vụ.
- Foreign key `metadata.video_id → videos.video_id` hoạt động.
- `image_rel_path` là relative path và không thoát khỏi configured data root.
- Mỗi Milvus visual `frame_id` hydrate được đúng một SQLite metadata row.

#### P0.3 Chốt object bbox contract

Các file dự kiến bị ảnh hưởng:

```text
feature_extraction/object_detection/src/object_detection/*
indexing/src/indexing/data_loader.py
indexing/tests/test_data_loader.py
verify_frame_id_consistency.py
```

Quyết định đã chọn:

```text
SQLite objects.x_min/y_min/x_max/y_max = absolute pixel XYXY
SQLite videos.width/height             = kích thước để Online normalize
Object JSON                            = giữ bbox pixel gốc
```

Offline phải validate `confidence`, finite coordinates, thứ tự XYXY và bounds theo
`videos.width/height`. Không nhận bbox thiếu phần tử, NaN/Inf, âm hoặc vượt kích
thước ảnh. Online sẽ normalize tại adapter boundary khi cần position constraint.

#### P0.4 Khóa model identity và embedding compatibility

Các file dự kiến bị ảnh hưởng:

```text
scripts/run_offline_pipeline.ps1
scripts/download_text_models.py
feature_extraction/visual_embedding/requirements.txt
feature_extraction/text_embedding/requirements.txt
feature_extraction/text_embedding/src/text_embedding/cli.py
feature_extraction/text_embedding/src/text_embedding/artifact_contract.py
```

Yêu cầu:

1. Giữ visual model `ViT-B-32::openai`, dimension 512, L2-normalized.
2. Pin immutable revision cho `dangvantuan/vietnamese-embedding`; baseline hiện
   quan sát được là `4ab46e46ba5902328ba0742e489e75f787932f2b`, Offline phải xác
   nhận revision này trước khi chốt.
3. One-click runner phải truyền revision đã pin, không để `None/default`.
4. Ghi model name/revision, dimension, max length, pooling và normalization vào
   manifest.
5. Khóa các phiên bản runtime ảnh hưởng preprocessing/tokenization bằng lock hoặc
   exact pins; tránh chỉ dùng `>=` cho deployment competition.
6. Khi đổi model/revision/preprocessing, phải tạo dataset fingerprint mới và
   re-index toàn bộ collection liên quan.

Test bắt buộc:

- Query smoke vector sinh bằng cấu hình Online có cùng dimension, finite và L2 norm.
- Artifact/model revision khác manifest → validation fail.
- Collection dimension/model contract cũ → không được publish READY.

#### P0.5 Publish manifest và mở rộng verifier

Các file dự kiến bị ảnh hưởng:

```text
verify_frame_id_consistency.py
indexing/src/indexing/orchestrator.py
scripts/run_offline_pipeline.ps1
tests/test_frame_id_verifier.py
tests/test_offline_one_click_script.py
```

Yêu cầu:

1. Sinh manifest theo Mục 9 sau khi mọi artifact/database validation đã pass.
2. `READY` phải là bước cuối; failure hoặc rollback không được để lại manifest READY
   của dataset mới.
3. Verifier hiện chủ yếu so ID sets; phải kiểm tra thêm schema mới, source frame,
   image path, bbox bounds, model identity, dimensions/norm và record counts.
4. Verification command phải trả exit code khác 0 khi fail để runner/CI dừng thật.
5. Handoff phải chứa output verifier, không chỉ log “indexing complete”.

### P1 — Bắt buộc trước real-data completion

1. Sửa resume/skip của indexer: hiện cùng record IDs có thể khiến indexer bỏ qua dù
   vector, text, timestamp hoặc model đã đổi. So sánh content fingerprint hoặc luôn
   force trên đường chạy production.
2. Thêm atomic publish/rollback hoặc active-dataset switch tương đương để Online
   không nhìn thấy trạng thái đang delete/insert giữa ba database.
3. Chốt cách xử lý shot cực ngắn khi `015`, `050`, `085` cùng trỏ một
   `source_frame_idx`. Cho phép nhiều internal `frame_id`, nhưng verifier và Online
   phải biết KIS dedup theo `(video_id, source_frame_idx)`.
4. Chạy full suite trong đúng Docker/runtime dependency thay vì chỉ môi trường dev.
5. Tạo fixture `self-indexed-v2` nhỏ nhưng đầy đủ visual/OCR/ASR/summary/object để
   Online kiểm tra read-only integration mà không cần toàn bộ dữ liệu thật.

### 13.1 Những phần Offline không cần đổi

- Không đổi lại sang keyframe/CLIP feature do Ban tổ chức cung cấp.
- Không đổi canonical internal ID khỏi
  `{video_id}_{shot_id:05d}_{position_code:03d}`.
- Không đổi tên Milvus collections hoặc Elasticsearch indices nếu không có lý do
  bắt buộc và chưa thống nhất với Online.
- Không đổi 7 retrieval data streams hiện có.
- Không quay lại PE-Core; Online sẽ đổi sang text tower tương ứng với
  `ViT-B-32::openai`.

---

## 14. Handoff Offline phải gửi cho Online

Không chỉ gửi tin nhắn “đã chạy xong”. Handoff phải gồm:

```text
1. contract_version
2. dataset_id
3. dataset_fingerprint
4. manifest path
5. SQLite path
6. keyframe root
7. raw video root
8. Milvus URI + collection names
9. Elasticsearch URI + index names
10. model identities/revisions
11. record counts
12. full validation command + output
13. known missing optional branches
14. commit hash của Offline producer/indexer
```

Secrets không được ghi vào Markdown, manifest hoặc Git. Chỉ truyền tên biến môi
trường/config key.

---

## 15. Definition of Done của contract

Contract chỉ được xem là chốt khi:

- Offline xác nhận có thể sinh đầy đủ schema trên.
- Online xác nhận đủ dữ liệu để hydrate KIS, TRAKE và VQA.
- Một fixture `self-indexed-v2` được commit cho tests.
- Offline index fixture thành công.
- Online đọc fixture read-only thành công.
- Candidate cuối giữ đúng `source_frame_idx`.
- Serializer lấy đúng `(video_id, source_frame_idx)` nội bộ rồi xuất tên trường
  BTC là `(video_id, frame_id)` mà không tính ngược.
- Hai bên ký nhận cùng `contract_version` và commit hash.

Cho đến lúc đó, code chỉ được ghi là `NEED_RUNTIME_VERIFICATION`, không được gọi
competition-ready.
