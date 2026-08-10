# 20 — KẾ HOẠCH SỬA OFFLINE VÀ ONLINE THEO DỮ LIỆU BTC AIC 2026

## 1. Mục đích tài liệu

Tài liệu này trả lời hai câu hỏi triển khai:

1. Code Offline hiện tại phải sửa, bỏ qua hoặc bổ sung những gì để ingest đúng
   dữ liệu Ban tổ chức (BTC) cung cấp.
2. Code Online hiện tại phải migration những gì để truy vấn, xếp hạng, TRAKE,
   VQA và xuất kết quả đúng trên dữ liệu đó.

Tài liệu này là kế hoạch sửa code, không phải tuyên bố rằng các hạng mục đã được
triển khai.

Nguồn sự thật cho data contract là:

- `docs/19-AIC2026-ORGANIZER-DATA-CONTRACT.md`.
- Các file mẫu và toàn bộ support package đã kiểm tra ngày 2026-08-04.
- Code Offline tại `origin/develop_mixi`.
- Code Online tại branch `feature/online-phase-Knguyen`.

Nhãn bằng chứng dùng trong tài liệu:

- `CONFIRMED_CODE`: đã đọc trực tiếp từ code hiện tại.
- `CONFIRMED_DATA`: đã kiểm tra trực tiếp trên dữ liệu BTC.
- `CONTRACT_MISMATCH`: code và dữ liệu/contract không khớp.
- `NEED_RUNTIME_VERIFICATION`: chỉ xác nhận được khi có Milvus,
  Elasticsearch, SQLite, model và dữ liệu thật đang chạy.
- `OPTIONAL`: không chặn baseline đem đi thi.

---

## 2. Kết luận ngắn gọn

### 2.1 Offline

Offline **chưa thể index trực tiếp dữ liệu BTC theo đúng contract**, dù branch
`develop_mixi` đã đổi visual model mặc định sang `ViT-B-32::openai`.

Các blocker chính:

1. Chưa có organizer-data adapter đọc trực tiếp `clip-features-32`,
   `map-keyframes`, `media-info`, `objects`, `keyframes` và `video`.
2. Indexer vẫn chỉ hiểu artifact tự sinh theo shot và Parquet.
3. `frame_id` vẫn theo `{video_id}_{shot_id:05d}_{position:03d}`.
4. SQLite vẫn lưu `shot_id` và `timestamp`, chưa có `keyframe_no`,
   `local_index`, `pts_time_sec`, `source_frame_idx`, `image_rel_path`.
5. Milvus `visual_features` vẫn lưu `shot_id`, chưa lưu `local_index`.
6. Object loader vẫn hiểu JSON tự sinh, không hiểu năm parallel arrays của BTC
   và chưa chuyển YXYX sang XYXY.

### 2.2 Online

Online hiện có baseline logic tốt và test fake/SDK-free đang xanh, nhưng **chưa
thể chạy đúng trên database mới từ dữ liệu BTC**.

Kết quả kiểm tra hiện tại:

```text
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online -q
466 passed, 3 warnings
```

Kết quả này chỉ chứng minh contract cũ chạy với fake fixtures. Nó không chứng
minh tương thích dữ liệu BTC.

Các blocker chính:

1. Online vẫn mặc định dùng PE-Core text encoder.
2. Domain, adapter, validator, ranking và API vẫn dựa vào `shot_id`/frame ID cũ.
3. Kết quả thi chưa lấy `source_frame_idx` nguyên văn từ CSV BTC.
4. TRAKE chưa có production `VisualCorpusPort`.
5. VQA chưa có production `ImageResolverPort` và VLM adapter thật.
6. Readiness validator chưa kiểm tra ingestion manifest/model identity.

### 2.3 Quyết định kiến trúc

Thực hiện **migration sạch cho competition dataset**, không tạo `shot_id=0` giả
và không parse/gán nghĩa mới cho ID cũ.

```text
Internal keyframe ID:
    frame_id = {video_id}_{keyframe_no:03d}

Competition frame:
    source_frame_idx = CSV.frame_idx

Ordering:
    local_index = keyframe_no - 1

Time:
    SQLite pts_time_sec = CSV.pts_time
    Online timestamp_sec = SQLite.pts_time_sec
```

`timestamp_sec` được giữ trong Online domain để giảm thay đổi không cần thiết,
nhưng từ sau migration nó có nghĩa chính xác là `pts_time` do BTC cung cấp.
Không được tính lại từ `source_frame_idx / fps`.

---

## 3. Contract không được vi phạm

### 3.1 JOIN key

```text
video_id + keyframe_no
```

Với một row CSV có `n = 1`:

```text
frame_id          = L21_V001_001
keyframe_no       = 1
local_index       = 0
clip_row_index    = 0
keyframe file     = keyframes/L21_V001/001.jpg
object file       = objects/L21_V001/001.json
pts_time_sec      = CSV.pts_time
source_frame_idx  = CSV.frame_idx
```

### 3.2 Những field không được dùng thay thế nhau

| Field | Ý nghĩa | Được dùng cho |
|---|---|---|
| `frame_id` | ID keyframe nội bộ | JOIN database/artifact |
| `keyframe_no` | Cột `n`, bắt đầu từ 1 | JOIN file JPG/JSON/NPY row |
| `local_index` | `keyframe_no - 1` | NPY row, TRAKE ordering |
| `pts_time_sec` | Cột `pts_time` | Temporal mapping/display |
| `source_frame_idx` | Cột `frame_idx` | Submission cho BTC |
| `fps` | Cột `fps` | Metadata/diagnostics |

### 3.3 Quy tắc đặc biệt

- `source_frame_idx` có thể trùng trong cùng video.
- Không đặt unique constraint cho `(video_id, source_frame_idx)`.
- Không order TRAKE bằng `source_frame_idx`.
- Không tính lại `source_frame_idx` từ timestamp/FPS.
- Không dùng OpenCV frame count để sửa giá trị CSV.
- Không chạy lại CLIP cho baseline khi đã có NPY từ BTC.
- Không chạy lại object detector cho baseline khi đã có object JSON từ BTC.
- Visual query encoder phải là text tower của đúng
  `ViT-B-32::openai`, dimension 512.

---

## 4. Trạng thái code Offline hiện tại

### 4.1 Những phần đã đúng hoặc có thể giữ

`CONFIRMED_CODE` trên `origin/develop_mixi`:

- `feature_extraction/visual_embedding/config.py` đã đặt:

  ```text
  DEFAULT_VISUAL_MODEL_ID = "ViT-B-32::openai"
  ```

- `OpenCLIPEncoder` đã L2-normalize và trả float32.
- Milvus index dùng HNSW + IP và dimension runtime.
- Text semantic model vẫn là
  `dangvantuan/vietnamese-embedding`, khớp giữa Offline và Online.
- ASR contract đã dùng `start_time_sec` và `end_time_sec`.
- Indexing đã có snapshot/rollback, schema audit và read-after-write validation.

Những phần trên nên được tái sử dụng, không viết lại tùy tiện.

### 4.2 Những phần đang sai contract BTC

| Vị trí | Code hiện tại | Vấn đề |
|---|---|---|
| `data_pipeline/shot_keyframe/` | Tự detect shot, xuất WebP | BTC đã có JPG keyframe/map |
| `visual_embedding/metadata_reader.py` | Sinh ID từ shot/position | Không đọc `n` CSV |
| `indexing/data_loader.py` | Chỉ đọc visual Parquet | Không đọc NPY BTC |
| `normalize_frame_id()` | Chỉ nhận ID shot cũ | Reject ID BTC mới |
| SQLite `metadata` | `frame_id, video_id, shot_id, timestamp` | Thiếu field competition |
| Milvus visual | Có `shot_id` | Thiếu `local_index` |
| Object loader | Đọc `frames[].objects[]` | Không hiểu parallel arrays BTC |
| OCR reader | Tìm `.webp` theo metadata shot | BTC là `{NNN}.jpg` |
| Artifact regex | Kiểm tra ID shot cũ | Reject `L21_V001_001` |

---

## 5. Offline — danh sách sửa chi tiết

## O0 — Cập nhật contract dùng chung

### Mục tiêu

Loại bỏ việc mỗi module tự tạo/parse frame ID theo cách riêng.

### File hiện tại bị ảnh hưởng

- `AGENTS.md`.
- `verify_frame_id_consistency.py`.
- Các tài liệu `docs/00` đến `docs/18` có mô tả PE-Core/shot ID cũ.
- Các regex/generator nằm rải rác trong:
  - `feature_extraction/visual_embedding/metadata_reader.py`.
  - `feature_extraction/ocr/src/ocr_module/metadata_reader.py`.
  - `feature_extraction/object_detection/src/object_detection/metadata_reader.py`.
  - `feature_extraction/text_embedding/src/text_embedding/artifact_contract.py`.
  - `indexing/src/indexing/data_loader.py`.

### Cần làm

- Tạo một module contract dependency-light dùng chung, đề xuất:

  ```text
  shared/organizer_contract/
      __init__.py
      identifiers.py
      records.py
      constants.py
  ```

- Định nghĩa duy nhất:

  ```python
  make_frame_id(video_id, keyframe_no)
  parse_frame_id(frame_id, expected_video_id=None)
  keyframe_filename(keyframe_no)
  object_filename(keyframe_no)
  local_index_from_keyframe_no(keyframe_no)
  ```

- Không cung cấp hàm “đoán và normalize” nhiều format trong production path.
- Nếu giữ legacy generated-data path, tách namespace rõ:

  ```text
  organizer-v1
  generated-shot-v1
  ```

  Hai contract không được trộn trong cùng một indexing run.

### Tiêu chí nghiệm thu

- Một test table-driven xác nhận ID `L21_V001_001`, `L26_V498_123`.
- Reject whitespace, `n=0`, negative, thiếu video ID và semantic mismatch.
- Không module Offline nào tự format organizer frame ID bằng string riêng.

---

## O1 — Tạo Organizer Dataset Adapter

### Mục tiêu

Đọc trực tiếp layout BTC và tạo record chuẩn cho các module sau.

### File mới đề xuất

```text
data_pipeline/organizer_data/__init__.py
data_pipeline/organizer_data/config.py
data_pipeline/organizer_data/discovery.py
data_pipeline/organizer_data/map_reader.py
data_pipeline/organizer_data/clip_reader.py
data_pipeline/organizer_data/media_reader.py
data_pipeline/organizer_data/object_reader.py
data_pipeline/organizer_data/validator.py
data_pipeline/organizer_data/manifest.py
data_pipeline/organizer_data/cli.py
```

### Input config

Không hardcode đường dẫn máy cá nhân. Cần một root cấu hình:

```text
dataset_root/
├── videos/
├── keyframes/
├── clip-features-32/
├── map-keyframes/
├── media-info/
└── objects/
```

Cho phép tên thư mục override qua CLI/env, nhưng default theo trên.

### Discovery

- Tạo set video ID từ từng family.
- Xuất báo cáo:
  - present in all required families;
  - missing video;
  - missing NPY;
  - missing CSV;
  - missing keyframe dir;
  - missing media-info;
  - missing object dir.
- Sort theo natural video ID, không dựa vào filesystem traversal order.

### Required/optional

Baseline required:

```text
clip-features-32
map-keyframes
keyframes
media-info
```

Required cho ASR/VQA đầy đủ:

```text
videos
```

Object là optional branch nhưng nếu folder tồn tại thì file/count sai phải báo
contract error, không im lặng bỏ qua.

### Tiêu chí nghiệm thu

- Support dataset hiện có: 873 video ID đồng bộ giữa NPY, CSV, media JSON,
  keyframe counts và object directories.
- Tổng 177.321 map rows/keyframes/feature rows.
- Báo rõ dataset local hiện có thể thiếu video/keyframe ngoài L21 thay vì coi là
  một dataset hoàn chỉnh.

---

## O2 — Validate map-keyframes

### Schema bắt buộc

```text
n,pts_time,fps,frame_idx
```

### Validation mỗi video

- Header đúng tên và đúng nghĩa.
- `n` integer, bắt đầu 1, tăng liên tục từng 1.
- `pts_time` finite, non-negative, không giảm.
- `fps` finite, positive.
- `frame_idx` integer, non-negative, không bắt buộc unique.
- Số row = số JPG = số NPY row.
- Với mỗi `n`, phải tồn tại `{n:03d}.jpg`.
- Nếu có object package, kiểm tra file `{n:03d}.json` theo policy được chọn.

### Không được validate sai

- Không yêu cầu `frame_idx` strictly increasing.
- Không yêu cầu `frame_idx` unique.
- Không reject FPS 26.44 hoặc 29.97.
- Không yêu cầu `frame_idx == round(pts_time * fps)`.

### Test bắt buộc

- Fixture có duplicate frame index vẫn pass.
- Fixture thiếu `n` giữa chuỗi fail.
- Fixture số NPY row lệch fail.
- Fixture `pts_time` giảm fail.
- Fixture Unicode dataset root trên Windows pass.

---

## O3 — Ingest visual NPY trực tiếp

### Code hiện tại

`indexing/src/indexing/data_loader.py::load_visual_embeddings()` chỉ đọc
Parquet do Module 2 tự sinh.

### Cần làm

- Thêm loader riêng, đề xuất:

  ```python
  load_organizer_visual_embeddings(dataset_root, video_id)
  ```

- Dùng `numpy.load(..., mmap_mode="r", allow_pickle=False)` để không nạp toàn bộ
  dataset vào RAM.
- Kiểm tra array:
  - rank 2;
  - shape `(N, 512)`;
  - source dtype `float16` theo package hiện tại;
  - finite;
  - row count khớp CSV;
  - norm gần 1 trong tolerance.
- Mỗi batch:
  - cast float16 sang float32;
  - re-normalize an toàn;
  - reject zero norm/non-finite;
  - tạo record bằng CSV row cùng `local_index`.

### Record Milvus

```text
frame_id
video_id
local_index
embedding
```

Không cần lưu `source_frame_idx` trong Milvus vì SQLite là nguồn metadata. Không
được lưu `shot_id` giả.

### Model provenance

Ghi chính xác:

```text
visual_model_id = ViT-B-32::openai
visual_dimension = 512
visual_source_dtype = float16
visual_stored_dtype = float32
visual_normalized = true
```

Không suy ra model identity chỉ từ dimension 512.

### Xử lý Module 2 hiện tại

- Giữ `feature_extraction/visual_embedding/` làm fallback/debug cho dataset không
  có feature BTC.
- Không gọi Module 2 trong organizer baseline.
- Đổi tên alias `PECoreEncoder` để tránh hiểu sai hoặc đánh dấu legacy rõ ràng.
- Không tạo lại NPY/Parquet chỉ để đi vòng qua loader cũ.

### Tiêu chí nghiệm thu

- `L21_V001.npy`: 307 record, dim 512, ID từ `001` đến `307`.
- Milvus sample vector có norm 1 trong tolerance.
- NPY row 0 JOIN `L21_V001_001`.
- Manifest và Online encoder báo cùng model ID.

---

## O4 — Migration SQLite schema

### File chính

- `indexing/src/indexing/clients/tabular_client.py`.
- `indexing/src/indexing/data_loader.py`.
- `indexing/src/indexing/orchestrator.py`.
- `indexing/tests/test_schema_audit.py`.
- `indexing/tests/test_data_loader.py`.

### Schema `videos`

```sql
CREATE TABLE videos (
    video_id                    TEXT PRIMARY KEY,
    media_title                TEXT,
    media_author               TEXT,
    media_description          TEXT,
    media_keywords_json        TEXT,
    media_length_sec           REAL,
    publish_date               TEXT,
    watch_url                  TEXT,
    video_rel_path             TEXT
);
```

### Schema `metadata`

```sql
CREATE TABLE metadata (
    frame_id          TEXT PRIMARY KEY,
    video_id          TEXT NOT NULL,
    keyframe_no       INTEGER NOT NULL CHECK (keyframe_no >= 1),
    local_index       INTEGER NOT NULL CHECK (local_index >= 0),
    pts_time_sec      REAL NOT NULL CHECK (pts_time_sec >= 0),
    fps               REAL NOT NULL CHECK (fps > 0),
    source_frame_idx  INTEGER NOT NULL CHECK (source_frame_idx >= 0),
    image_rel_path    TEXT NOT NULL,
    FOREIGN KEY (video_id) REFERENCES videos(video_id),
    UNIQUE (video_id, keyframe_no),
    UNIQUE (video_id, local_index)
);
```

Không thêm:

```sql
UNIQUE (video_id, source_frame_idx)
```

### Index cần có

```sql
CREATE INDEX idx_metadata_video_local
ON metadata(video_id, local_index);

CREATE INDEX idx_metadata_video_pts
ON metadata(video_id, pts_time_sec);

CREATE INDEX idx_metadata_video_source_frame
ON metadata(video_id, source_frame_idx);
```

### Migration policy

- Tăng schema/contract version.
- Không tự ALTER mù trên database cũ.
- Với competition dataset, tạo database mới từ source artifacts là an toàn nhất.
- CLI phải fail rõ nếu gặp schema cũ thay vì tiếp tục ghi record mới vào cột cũ.
- Snapshot/rollback hiện có phải mở rộng cho table `videos` và schema mới.

### Tiêu chí nghiệm thu

- Count metadata đúng tổng map rows.
- JOIN `metadata.frame_id` với Milvus visual đạt 100%.
- Duplicate `(video_id, source_frame_idx)` vẫn insert được.
- `ORDER BY local_index` trả chuỗi 0..N-1.
- `image_rel_path` resolve dưới configured dataset root và không thoát root.

---

## O5 — Ingest media-info

### Cần làm

- Parse UTF-8 JSON.
- Validate payload là object.
- Map các field có mặt sang table `videos`.
- Lưu keyword dưới JSON text chuẩn, không dùng Python repr.
- `video_rel_path` lấy theo cấu hình dataset, không lấy đường dẫn absolute máy
  ingestion.
- Không dùng media-info làm nguồn timestamp/keyframe mapping.

### Text enrichment

Media title/description/keywords có thể tạo lexical support về sau, nhưng cần:

- bỏ boilerplate URL/channel template;
- không đưa `watch_url` vào semantic text;
- áp trọng số thấp hơn OCR/ASR/visual;
- không hard-prefilter video chỉ vì title/keyword không match.

### Tiêu chí nghiệm thu

- Unicode tiếng Việt giữ nguyên.
- Missing optional field thành NULL, không thành chuỗi `"None"`.
- Video ID từ filename phải khớp record được ingest.

---

## O6 — Ingest object JSON BTC

### Schema nguồn

Mỗi file chứa năm array song song:

```text
detection_class_entities
detection_class_names
detection_class_labels
detection_scores
detection_boxes
```

### Validation

- Tất cả array phải tồn tại và có cùng chiều dài.
- Score numeric finite trong `[0, 1]`.
- Box có đúng bốn numeric finite values trong `[0, 1]`.
- Box nguồn là normalized YXYX:

  ```text
  [y_min, x_min, y_max, x_max]
  ```

- Sau conversion:

  ```text
  x_min = source[1]
  y_min = source[0]
  x_max = source[3]
  y_max = source[2]
  ```

### Schema `objects`

```sql
CREATE TABLE objects (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id           TEXT NOT NULL,
    label_display      TEXT NOT NULL,
    label_normalized   TEXT NOT NULL,
    class_mid          TEXT,
    class_label_id     TEXT,
    confidence         REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    x_min              REAL NOT NULL,
    y_min              REAL NOT NULL,
    x_max              REAL NOT NULL,
    y_max              REAL NOT NULL,
    model_source       TEXT NOT NULL,
    FOREIGN KEY (frame_id) REFERENCES metadata(frame_id) ON DELETE CASCADE
);
```

Index:

```sql
CREATE INDEX idx_objects_frame_confidence
ON objects(frame_id, confidence DESC);

CREATE INDEX idx_objects_label_confidence
ON objects(label_normalized, confidence DESC);

CREATE INDEX idx_objects_mid
ON objects(class_mid);
```

### Filter/NMS quyết định

Baseline được chốt:

```text
ingestion threshold = 0.10
per-class NMS IoU   = 0.50
query default       = 0.50
```

- Threshold/NMS phải configurable và ghi vào manifest.
- NMS chạy theo class identity, không gộp hai class khác nhau.
- Không đếm raw duplicate boxes cho count constraint.
- Giữ cả label hiển thị, label normalized, MID và label ID.

### Module 5 hiện tại

- Giữ detector tự chạy làm fallback cho dataset không có organizer objects.
- Không chạy Module 5 trong organizer baseline.
- Không merge organizer boxes với YOLO/Co-DETR mặc định; chỉ làm ensemble khi có
  benchmark chứng minh tốt hơn.

### Tiêu chí nghiệm thu

- 307 file object của `L21_V001` đọc hợp lệ.
- Visual spot-check cho bbox xác nhận conversion YXYX→XYXY.
- Frame không còn box sau threshold vẫn có metadata và trả object tuple rỗng.

---

## O7 — Sửa OCR để đọc keyframe JPG BTC

### File chính

- `feature_extraction/ocr/src/ocr_module/metadata_reader.py`.
- `feature_extraction/ocr/src/ocr_module/pipeline.py`.
- `feature_extraction/ocr/src/ocr_module/resume_validation.py`.
- `feature_extraction/ocr/tests/`.

### Code hiện tại

- Đọc metadata shot.
- Tạo frame ID từ shot/position.
- Tìm ảnh `.webp`.

### Cần làm

- Reader nhận canonical organizer records hoặc đọc SQLite/manifests sau O1.
- Dùng `image_rel_path`, không đoán filename từ frame ID.
- Hỗ trợ JPG RGB của BTC.
- Output OCR giữ:

  ```text
  frame_id
  video_id
  keyframe_no
  ocr_text_concat
  regions/raw OCR nếu cần debug
  OCR model/prompt/version provenance
  ```

- Resume validation so sánh ordered frame ID và source fingerprint.
- Không coi OCR rỗng là pipeline failure; phải phân biệt empty-success với file
  thiếu/hỏng.

### Tiêu chí nghiệm thu

- OCR mở đúng `keyframes/L21_V001/001.jpg` qua `image_rel_path`.
- Output ID `L21_V001_001` JOIN SQLite.
- Text embedding OCR và Elasticsearch OCR dùng cùng ID.

---

## O8 — ASR và Summary trên MP4 BTC

### File chính

- `feature_extraction/asr_transcript/`.
- `feature_extraction/text_embedding/src/text_embedding/data_readers.py`.
- `indexing/src/indexing/data_loader.py`.

### Cần làm

- Video discovery lấy từ organizer adapter/table `videos`.
- Resolve MP4 qua `video_rel_path` dưới configured dataset root.
- Không dựa vào việc video ID liên tục; dataset có ID bị khuyết.
- Giữ ASR interval key:

  ```text
  video_id + interval_id
  ```

- Giữ field chuẩn:

  ```text
  start_time_sec
  end_time_sec
  cleaned_text
  ```

- Summary vẫn video-level, không tạo frame giả.
- Ghi provenance cho ASR model, language, cleaning model/prompt và summary model.

### Kiểm tra đặc biệt

- Duration ASR không được vượt media duration quá tolerance hợp lý.
- Interval order không giảm và end >= start.
- Video không có speech phải tạo empty-success artifact hợp lệ.
- Summary thiếu vì ASR rỗng không được làm fail visual baseline.

### Tiêu chí nghiệm thu

- ASR dense và lexical JOIN bằng `(video_id, interval_id)`.
- ASR-to-frame Online map theo `pts_time_sec`.
- Summary dense và lexical JOIN bằng `video_id`.

---

## O9 — Sửa Vietnamese text embedding artifact contract

### File chính

- `feature_extraction/text_embedding/src/text_embedding/artifact_contract.py`.
- `feature_extraction/text_embedding/src/text_embedding/data_readers.py`.
- `feature_extraction/text_embedding/src/text_embedding/pipeline.py`.
- Tests tương ứng.

### Cần làm

- OCR artifact chấp nhận organizer frame ID mới.
- Loại bỏ yêu cầu `shot_id` cho OCR record.
- ASR và summary contracts giữ nguyên candidate level.
- Ghi và kiểm tra:

  ```text
  model_name = dangvantuan/vietnamese-embedding
  model_revision
  embedding_dimension
  normalized = true
  source artifact fingerprint
  ```

- Indexer phải reject việc trộn nhiều text model/revision trong cùng collection.

### Không cần đổi

Không đổi Vietnamese semantic model chỉ vì visual model đổi. Visual CLIP space và
Vietnamese OCR/ASR/summary space là hai vector spaces độc lập.

---

## O10 — Sửa Milvus schema và indexer orchestration

### File chính

- `indexing/src/indexing/clients/milvus_client.py`.
- `indexing/src/indexing/data_loader.py`.
- `indexing/src/indexing/orchestrator.py`.
- `indexing/src/indexing/cli.py`.

### Visual schema mới

```text
pk          INT64 auto ID
frame_id    VARCHAR
video_id    VARCHAR
local_index INT64
embedding   FLOAT_VECTOR dim=512
```

### Các collection text giữ candidate level

```text
ocr_features:     frame_id + video_id + embedding
asr_features:     video_id + interval_id + start/end + embedding
summary_features: video_id + embedding
```

### Orchestrator input mode

Thêm explicit mode:

```text
--input-layout organizer
--input-layout generated
```

Không dùng `auto` trong production vì có thể chọn nhầm artifact family khi cả
hai cùng tồn tại. `auto` chỉ nên dùng cho local exploration và phải log mode.

### Transaction order đề xuất

```text
validate all source families for video
→ build immutable batch records
→ snapshot old backend records
→ delete old video records
→ insert Milvus
→ insert Elasticsearch
→ insert SQLite videos/metadata/objects
→ read-after-write validation
→ commit success manifest
→ rollback all backends nếu bất kỳ gate nào fail
```

### Validation sau ghi

- Visual count = metadata count.
- Visual frame ID set = metadata frame ID set.
- Visual dimension = 512.
- Visual norm pass.
- `local_index` contiguous 0..N-1.
- OCR/object frame IDs là subset metadata.
- ASR vector IDs = ASR lexical IDs.
- Summary vector video IDs = summary lexical video IDs.
- Manifest model ID đúng.

### Schema migration

Collection visual cũ có `shot_id` không được reuse. Tạo collection mới hoặc reset
có chủ đích sau khi backup; schema audit phải fail trước destructive operation.

---

## O11 — Ingestion manifest và dataset fingerprint

### File/table đề xuất

```text
data/index-manifest.json
SQLite table dataset_manifest
```

### Field bắt buộc

```text
dataset_name
dataset_batches
contract_version = organizer-v1
ingestion_version
visual_model_id = ViT-B-32::openai
visual_dimension = 512
visual_source_dtype = float16
visual_stored_dtype = float32
visual_normalized = true
object_source
object_threshold
object_nms_iou
frame_id_contract_version
record counts per artifact family
created_at UTC
dataset fingerprint/source hashes
```

### Quy tắc

- Manifest chỉ publish sau read-after-write gate.
- Một database không được tự nhận READY nếu manifest thiếu hoặc khác collection.
- Online đọc manifest read-only trong startup validation.
- Không ghi absolute local source path vào public diagnostics.

---

## O12 — CLI, Docker và documentation

### Cần cập nhật

- `.env.example` cho organizer dataset root và layout.
- `docker-compose.yml` volume mounts read-only cho source dataset.
- `indexing/Dockerfile`/requirements có NumPy/Pandas cần thiết.
- `scripts/offline_modal_runner.py` để có organizer ingestion stage.
- `scripts/run_all_tests.py` thêm organizer-contract tests.
- README Offline ghi rõ module nào bypass khi dùng organizer data.

### Không được làm

- Không mount đường dẫn OneDrive cá nhân vào config commit.
- Không download lại dữ liệu BTC trong unit test.
- Không log token/URL nhạy cảm.
- Không xóa source archive/dataset khi `--force`.

---

## O13 — Bộ test Offline bắt buộc

### Unit tests mới

```text
tests/test_organizer_identifiers.py
tests/test_organizer_discovery.py
tests/test_organizer_map_reader.py
tests/test_organizer_clip_reader.py
tests/test_organizer_media_reader.py
tests/test_organizer_object_reader.py
tests/test_organizer_manifest.py
tests/test_organizer_index_contract.py
```

### Sửa tests hiện có

- `tests/test_frame_id_verifier.py`.
- `tests/test_visual_model_contract.py`.
- `tests/test_offline_producer_consumer_integration.py`.
- `indexing/tests/test_data_loader.py`.
- `indexing/tests/test_schema_audit.py`.
- `indexing/tests/test_post_index_validation.py`.
- OCR/text embedding tests chứa ID `V001_00000_015`.

### Integration fixtures cần có

Một mini dataset 2 video, trong đó:

- video A có 3 keyframe;
- video B có 2 keyframe;
- có duplicate `source_frame_idx`;
- có FPS decimal;
- có object file với YXYX bbox;
- có một frame object rỗng sau threshold;
- NPY float16 `(N, 512)` normalized;
- media-info có Unicode tiếng Việt.

### Gate với dữ liệu thật

Chạy ít nhất:

1. `L21_V001` vertical slice.
2. Một video có FPS 29.97.
3. `L28_V006` hoặc fixture tương đương có nhiều duplicate frame index.
4. Full support consistency scan 873 video.

---

## 6. Offline — phần nào được bypass, phần nào vẫn phải chạy

| Module | Organizer baseline | Lý do |
|---|---|---|
| Shot/keyframe extraction | Bypass | BTC đã cung cấp keyframe/map |
| Visual image embedding | Bypass | BTC đã cung cấp CLIP NPY |
| Object detection | Bypass | BTC đã cung cấp object JSON |
| OCR | Vẫn chạy | BTC không cung cấp OCR text |
| ASR | Vẫn chạy | BTC không cung cấp transcript |
| Summary | Vẫn chạy | BTC không cung cấp summary semantic |
| Vietnamese embeddings | Vẫn chạy | Cần OCR/ASR/summary dense retrieval |
| Multi-database indexing | Phải sửa và chạy | Online chỉ đọc database |

Fallback modules không bị xóa; chúng chỉ không nằm trên competition organizer
baseline path.

---

## 7. Trạng thái code Online hiện tại

### 7.1 Những phần có thể giữ

- QueryBundle và t-KIS/v-KIS dùng chung text retrieval pipeline.
- Bảy retrieval branch phân biệt đúng frame/ASR interval/video candidate level.
- Concurrency, timeout và branch diagnostics đã có.
- Ranking có aggregation, normalization, fusion, ASR mapping, summary boost,
  object processing và dedup.
- TRAKE/DANTE algorithm và VQA evidence orchestration có contracts/fakes/tests.
- Database adapters read-only và lỗi được map qua domain error boundary.

### 7.2 Những mismatch đã xác nhận

| Vị trí | Hiện tại | Cần đổi |
|---|---|---|
| `online/retrieval/encoders.py` | PE-Core default | `ViT-B-32::openai` |
| `online/domain/identifiers.py` | ID shot/position | ID video/keyframe_no |
| `FrameMetadata` | `shot_id, timestamp_sec` | organizer fields |
| `FrameCandidate` | `shot_id` | keyframe/source fields |
| SQLite adapter | SELECT `shot_id,timestamp` | SELECT schema mới |
| Milvus adapter | output `shot_id` | hydrate metadata; visual local index |
| Contract validator | audit schema cũ | schema + manifest mới |
| Deduplicator | group `(video_id, shot_id)` | group competition frame |
| Competition serializer | frame ID/time/score | exact `source_frame_idx` |
| TRAKE runtime | fake corpus | production corpus adapter |
| VQA runtime | fake image resolver | safe JPG resolver |
| Rewrite runtime | NoOp mặc định | production provider nếu bật |
| VLM runtime | protocol/fake | production VLM adapter |

---

## 8. Online — danh sách sửa chi tiết

## N0 — Cập nhật tài liệu và source precedence

### File

- `AGENTS.md`.
- `docs/00`–`docs/11` ở các đoạn visual model, identifier, SQLite/Milvus schema.
- README Online và `.env.example`.

### Cần làm

- `docs/19` và tài liệu này phải đứng trên tài liệu legacy trong precedence.
- Thay PE-Core bằng `ViT-B-32::openai` cho organizer visual space.
- Thay canonical shot ID bằng organizer keyframe ID.
- Ghi rõ `source_frame_idx` là submission field.
- Ghi rõ generated-shot path là fallback khác contract version.

### Tiêu chí nghiệm thu

Không còn instruction hiện hành nào yêu cầu Online organizer path dùng PE-Core,
`shot_id` hoặc `{video_id}_{shot_id}_{position}`.

---

## N1 — Migration identifier domain

### File

- `online/domain/identifiers.py`.
- `online/domain/__init__.py`.
- Toàn bộ tests đang dùng `V001_00000_015`.

### Contract mới

```python
CanonicalFrameId(
    video_id="L21_V001",
    keyframe_no=1,
)
```

Parser nên parse suffix ba chữ số từ bên phải và luôn semantic-check với
`video_id` khi backend đã trả video ID.

### Cần làm

- Đổi `shot_id` thành `keyframe_no`.
- Require `keyframe_no >= 1`.
- Không parse `source_frame_idx` từ `frame_id`.
- Reject frame ID legacy trong organizer-v1 runtime.
- Nếu cần hỗ trợ hai dataset contract, chọn validator theo manifest version;
  không tự đoán format.

### Test

- Video ID có underscore.
- Semantic video mismatch.
- Leading/trailing whitespace.
- `n=000`.
- Legacy shot ID.
- Organizer ID round-trip.

---

## N2 — Migration domain models và ports

### File chính

- `online/ports/records.py`.
- `online/domain/candidates.py`.
- `online/domain/trake.py`.
- `online/domain/vqa.py`.
- `online/ports/metadata.py`.
- `online/ports/visual_corpus.py`.

### `FrameMetadata` mới

```text
frame_id
video_id
keyframe_no
local_index
timestamp_sec       # lấy từ SQLite pts_time_sec
fps
source_frame_idx
image_rel_path
```

### `FrameSearchHit`

Đề xuất giữ hit tối thiểu:

```text
frame_id
video_id
raw_score
```

Không tin/copy duplicate metadata field từ Milvus hit nếu SQLite sẽ hydrate.

### `FrameCandidate` và `FusedFrameCandidate`

Phải mang ít nhất:

```text
frame_id
video_id
keyframe_no
local_index
timestamp_sec
source_frame_idx
scores/provenance
```

Không đưa `image_rel_path` vào public candidate. Image resolver đọc nó qua
metadata port.

### `OrderedVisualFrame`

Thay `shot_id` bằng:

```text
keyframe_no
local_index
timestamp_sec
source_frame_idx
```

Tiếp tục require L2-normalized vector và local indices contiguous từ 0.

### `TRAKEFrameMatch`

Thêm `source_frame_idx`, giữ `local_index`; bỏ `shot_id`.

### `ImageEvidence`

Thêm `keyframe_no` và `source_frame_idx`; bỏ `shot_id`. `image_reference` vẫn là
opaque safe reference, không trả absolute local path.

### Metadata port mở rộng

```python
get_frames_by_ids(frame_ids)
get_ordered_frames_by_video(video_id)
list_video_ids()
```

`get_ordered_frames_by_video` bắt buộc order theo `local_index`, sau đó mới dùng
frame ID làm deterministic tie-breaker nếu cần.

---

## N3 — Sửa SQLite read adapter

### File

- `online/adapters/sqlite.py`.
- `online/config.py`.
- `tests/online/adapters/test_sqlite_adapter.py`.
- `online/testing/sqlite_fixture.py`.

### Query mới

Hydration:

```sql
SELECT
    frame_id,
    video_id,
    keyframe_no,
    local_index,
    pts_time_sec,
    fps,
    source_frame_idx,
    image_rel_path
FROM metadata
WHERE frame_id IN (...)
```

Ordered frames:

```sql
SELECT ...
FROM metadata
WHERE video_id = ?
ORDER BY local_index ASC
```

Video IDs:

```sql
SELECT video_id FROM videos ORDER BY video_id
```

### Object query mới

- Select label display/normalized/MID/label ID.
- Filter bằng `label_normalized` hoặc `class_mid`.
- `min_confidence` vẫn configurable.
- Không leak absolute path qua diagnostics.

### Safety giữ nguyên

- `mode=ro`.
- `PRAGMA query_only=ON`.
- Chunk size dưới SQLite parameter limit.
- Validate finite numeric values.

### Test

- Duplicate `source_frame_idx` hydration.
- Ordered local indices.
- Unicode path.
- Missing row.
- Malformed bbox.
- Attempted write vẫn fail.

---

## N4 — Sửa Milvus search adapter

### File

- `online/adapters/milvus.py`.
- `online/ports/search.py` nếu signature cần đổi.
- `tests/online/adapters/test_milvus_adapter.py`.

### KIS search

- Visual search output fields chỉ cần:

  ```text
  frame_id, video_id
  ```

- `shot_id` không còn tồn tại.
- Visual branch hydrate organizer metadata qua SQLite như hiện tại hydrate
  timestamp.

### Production corpus cho TRAKE

Không nhét logic full-corpus vào KIS search method. Thêm adapter chuyên biệt,
đề xuất:

```text
online/adapters/visual_corpus.py
```

Adapter kết hợp:

```text
Milvus visual vectors
+ SQLite metadata
→ OrderedVisualFrame batches
```

Milvus backend cần read-only pagination/query theo:

```text
video_id
local_index ascending
```

Không dùng `pk` làm ordering/domain ID.

### Tiêu chí nghiệm thu

- Search vector dim 512.
- Search/query metric IP.
- Visual hits hydrate 100% metadata.
- Corpus stream local indices contiguous.
- Không load toàn bộ 177.321 × 512 vectors cùng lúc.
- Batch size và timeout configurable.

---

## N5 — Thay Online visual text encoder

### File

- `online/retrieval/encoders.py`.
- `online/retrieval/factory.py`.
- `retrieval_api/composition.py`.
- `online/requirements-encoders.txt`.
- `tests/online/retrieval/test_encoders.py`.

### Code hiện tại

```text
PE_CORE_MODEL_ID = hf-hub:timm/PE-Core-bigG-14-448
PECoreTextEncoder
```

### Code mục tiêu

```text
OPEN_CLIP_MODEL_ID = ViT-B-32::openai
OpenCLIPTextEncoder
```

Backend OpenCLIP hiện tại đã gần dùng lại được vì đã hiểu syntax `model::tag`.
Cần đổi tên/class message/default và test, không cần viết lại encode logic.

### Bắt buộc

- `open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")`.
- Tokenizer của `ViT-B-32`.
- `encode_text`.
- Output float32, finite, L2-normalized.
- Dimension 512.
- Startup check model ID từ ingestion manifest, không chỉ dimension.

### TRAKE

TRAKE event encoder phải dùng cùng instance/config visual encoder với KIS hoặc
một instance khác nhưng cùng exact model identity.

### Không đổi

`VietnameseTextEncoder` tiếp tục phục vụ OCR/ASR/summary semantic collections.

---

## N6 — Sửa retrieval hydration

### File

- `online/retrieval/branches.py`.
- `online/retrieval/service.py`.
- Tests visual/OCR branches.

### Cần làm

- `_FrameBranchBase` tạo `FrameCandidate` bằng organizer metadata.
- Không so semantic `shot_id` giữa Milvus và SQLite.
- Missing metadata vẫn là contract mismatch/core failure cho visual branch.
- OCR frame hits dùng cùng hydration.
- ASR và summary candidate levels không đổi.
- Provenance giữ raw score, backend, resource, query variant.

### Test

- Visual hit → candidate giữ đúng `source_frame_idx`.
- OCR lexical/dense cùng ID hydrate cùng record.
- Duplicate source frame từ hai keyframe IDs vẫn là hai candidate ở retrieval
  stage; dedup xảy ra sau fusion.

---

## N7 — Sửa ranking, ASR mapping và dedup

### File chính

- `online/ranking/asr_mapper.py`.
- `online/ranking/fusion.py`.
- `online/ranking/dedup.py`.
- `online/ranking/object_filter.py`.
- `online/ranking/summary.py`.
- `online/modes/kis.py`.

### ASR mapping

- `get_ordered_frames_by_video()` trả local order.
- Match interval bằng `timestamp_sec = pts_time_sec`.
- Candidate sinh ra giữ organizer metadata/source frame.
- Không dùng `source_frame_idx/fps` để tạo timestamp.

### Fusion

- Khi gộp evidence theo `frame_id`, các metadata field của cùng frame phải giống
  nhau; mismatch phải fail.
- Tie-breaker deterministic:

  ```text
  final_score desc
  video_id asc
  local_index asc
  frame_id asc
  ```

### Dedup

Xóa `ShotDeduplicator` semantics cũ. Đổi thành competition deduplicator:

```text
primary dedup key = (video_id, source_frame_idx)
```

Vì CSV có duplicate frame index, nhiều internal keyframe có thể trỏ cùng
competition frame. Giữ candidate score cao nhất và đưa candidate còn lại vào
near/evidence diagnostics.

Near-frame grouping theo time/local index là bước riêng, không gọi là shot.

### Objects

- Match label qua `label_normalized`/MID.
- Thêm synonym map có version, ví dụ Vietnamese query → Open Images English
  canonical label.
- Count sau ingestion NMS, không đếm raw boxes.
- Position constraint dùng normalized XYXY sau conversion đúng.
- Confidence default 0.5, configurable/tunable.

### Tiêu chí nghiệm thu

- Không output hai row cùng `(video_id, source_frame_idx)` cho KIS.
- ASR mapping không mất metadata competition.
- Object count test không bị duplicate box làm tăng số lượng.

---

## N8 — Production TRAKE corpus và output

### File

- `online/ports/visual_corpus.py`.
- File adapter mới `online/adapters/visual_corpus.py`.
- `online/trake/similarity.py`.
- `online/trake/service.py`.
- `online/domain/trake.py`.
- `online/modes/trake.py`.
- `retrieval_api/advanced_models.py`.
- `retrieval_api/composition.py`.

### Cần làm

- Wire production corpus adapter vào lifecycle, thay fake runtime.
- Encode mọi event bằng `ViT-B-32::openai` text tower.
- Stream per video, batch, không giữ toàn corpus trong RAM.
- DANTE transition chỉ trong cùng video.
- Sequence order theo strictly increasing `local_index`.
- `source_frame_idx` chỉ dùng khi serialize kết quả.
- Kết quả mỗi event giữ:

  ```text
  event_id
  video_id
  frame_id
  keyframe_no
  local_index
  timestamp_sec
  source_frame_idx
  similarity_score
  ```

### Competition serialization

Khi thể lệ yêu cầu frame list:

```text
video_id, source_frame_idx_1, ..., source_frame_idx_N
```

Không output internal `frame_id` thay cho frame number BTC.

### Performance gate

- Benchmark toàn corpus 873 video.
- Có cancellation/timeout.
- Không materialize toàn bộ vectors toàn dataset trong một tuple.
- Cache text event embeddings trong request.
- Có diagnostics số video/frame/batch và latency, không log vector.

---

## N9 — Production VQA image resolver

### File

- `online/ports/images.py`.
- File mới đề xuất `online/adapters/images.py`.
- `online/vqa/evidence_selector.py`.
- `online/domain/vqa.py`.
- `retrieval_api/composition.py`.

### Resolver flow

```text
frame_ids
→ SQLite metadata hydration
→ image_rel_path
→ resolve dưới configured keyframe/dataset root
→ verify path containment
→ return opaque image reference/bytes handle
```

### Safety

- Không chấp nhận path từ request.
- Không cho `..` thoát dataset root.
- Không trả absolute path trong API.
- Validate file tồn tại, regular file, extension allowed.
- Bound kích thước ảnh/bytes và số ảnh theo VQA budget.
- Missing image là diagnostics/evidence loss, không tự thay ảnh sai.

### Tiêu chí nghiệm thu

- Resolve `L21_V001_001` đúng `001.jpg`.
- Symlink/path traversal bị reject.
- VQA evidence giữ đúng source frame metadata.

---

## N10 — Production VLM adapter

### Trạng thái

VQA orchestration, evidence budget và VLM port đã có, nhưng adapter model thật
chưa phải production baseline.

### File mới đề xuất

```text
online/adapters/vlm.py
online/vqa/prompts.py
```

### Cần làm

- Chọn provider/model qua config.
- Gửi chỉ evidence đã retrieve, không gửi toàn dataset.
- Structured response theo `VLMResponse`.
- Grounding evidence IDs phải là subset evidence request.
- Timeout, retry bounded, token/image budget.
- Prompt version/model ID trong diagnostics.
- Không log ảnh, API key hoặc full sensitive response.

### Output Q&A

Competition adapter phải có khả năng xuất:

```text
video_id, source_frame_idx, answer
```

Exact format cuối cùng vẫn phải đối chiếu endpoint/rule BTC khi công bố.

---

## N11 — Production LLM query rewrite

### Trạng thái

`QueryRewriteService` đã có timeout, validation và safe degradation, nhưng
composition mặc định dùng `NoOpQueryRewriter`.

### Cần làm

- Thêm concrete provider adapter nếu đội quyết định bật LLM rewrite.
- KIS luôn giữ q0 là query gốc; q1/q2 chỉ là paraphrase.
- VQA rewrite tạo visual evidence description, không trả lời câu hỏi.
- Structured output, dedup variant, timeout và fallback q0.
- Config feature flag; provider failure không làm visual baseline chết.
- Benchmark có/không rewrite trước khi bật mặc định.

### Mức ưu tiên

`P1`: quan trọng cho chất lượng nhưng không chặn organizer data integration.

---

## N12 — Sửa contract validator/readiness

### File

- `online/adapters/contract_validator.py`.
- `online/validate_contract.py`.
- `retrieval_api/composition.py`.
- `tests/online/adapters/test_contract_validator.py`.

### Schema audit mới

Milvus visual:

```text
frame_id VARCHAR
video_id VARCHAR
local_index INT64
embedding FLOAT_VECTOR 512
HNSW/IP
```

SQLite metadata/objects/videos đúng schema O4/O6.

### Manifest audit

Required startup gates:

```text
contract_version == organizer-v1
visual_model_id == ViT-B-32::openai
visual_dimension == 512
visual_normalized == true
frame_id_contract_version đúng
object threshold/NMS có giá trị hợp lệ
```

### JOIN audit

- Visual sample JOIN metadata.
- OCR dense/lexical JOIN metadata.
- Object JOIN metadata.
- ASR dense/lexical JOIN composite ID.
- Summary dense/lexical JOIN video ID.
- Local indices sampled đúng order/range.
- Competition source frame field tồn tại và non-negative.

### Readiness rule

Core required:

```text
Milvus visual
SQLite metadata
visual encoder/model manifest match
```

Optional/degradable:

```text
OCR
ASR
summary
objects
LLM rewrite
VLM, tùy mode đang bật
```

Nếu TRAKE endpoint enabled thì production visual corpus adapter phải required.
Nếu VQA endpoint enabled thì image resolver và VLM phải required.

---

## N13 — Sửa API và competition serializers

### File

- `retrieval_api/search_engine.py`.
- `retrieval_api/advanced_models.py`.
- `retrieval_api/main.py`.
- API tests.

### Internal/UI response

Có thể trả cả:

```text
frame_id
video_id
keyframe_no
timestamp_sec
source_frame_idx
score
```

### Submission response/file

Không dùng current `competition_candidates()` đang trả frame ID/timestamp/score
làm format nộp bài.

Tạo serializer theo mode:

```text
KIS:   video_id + source_frame_idx
TRAKE: video_id + ordered source_frame_idx list
VQA:   video_id + source_frame_idx + answer
```

### Quy tắc

- Serializer chỉ nhận domain result đã hydrate.
- Không parse source frame từ filename/frame ID.
- Dedup trước serialize.
- Validate row count/top-k theo rule BTC.
- Có golden tests exact delimiter/column/order khi submission format chốt.

---

## N14 — Config, lifecycle và production wiring

### File

- `.env.example`.
- `online/config.py`.
- `retrieval_api/composition.py`.
- `online/lifecycle.py`.
- `online/requirements-runtime.txt`.

### Config mới cần có

```text
AIC_ONLINE_DATASET_ROOT
AIC_ONLINE_KEYFRAME_ROOT
AIC_ONLINE_DATASET_MANIFEST_PATH hoặc manifest table config
AIC_ONLINE_VISUAL_MODEL_ID=ViT-B-32::openai
AIC_ONLINE_VISUAL_ENCODER_DIMENSION=512
AIC_ONLINE_VISUAL_CORPUS_BATCH_SIZE
AIC_ONLINE_OBJECT_MIN_CONFIDENCE
AIC_ONLINE_IMAGE_MAX_BYTES
AIC_ONLINE_ENABLE_TRAKE
AIC_ONLINE_ENABLE_VQA
AIC_ONLINE_ENABLE_QUERY_REWRITE
```

### Wiring

- KIS: SQLite + Milvus + ES + encoders.
- TRAKE enabled: thêm production visual corpus.
- VQA enabled: thêm image resolver + VLM.
- Optional component disabled phải có explicit status, không giả healthy.
- Shutdown phải drain executor/request trước đóng adapters.

---

## N15 — Sửa fakes, fixtures và toàn bộ Online tests

### Phạm vi

Mọi fixture chứa dạng ID `V001_00000_015` hoặc `shot_id` phải migration.

Thư mục chịu ảnh hưởng:

```text
online/testing/
tests/online/contract/
tests/online/adapters/
tests/online/retrieval/
tests/online/ranking/
tests/online/trake/
tests/online/vqa/
tests/online/modes/
tests/online/api/
tests/online/integration/
tests/online/fixtures/
```

### Fixture chuẩn đề xuất

```text
video_id          = L21_V001
frame_id          = L21_V001_001
keyframe_no       = 1
local_index       = 0
timestamp_sec     = 0.0
fps               = 30.0
source_frame_idx  = 0
image_rel_path    = keyframes/L21_V001/001.jpg
```

### Các case mới bắt buộc

- Duplicate source frame dedup.
- Variable FPS.
- Model ID mismatch cùng dim 512.
- Missing manifest.
- Visual corpus real adapter pagination.
- Image resolver containment.
- Competition serializer exact output.
- Object YXYX conversion/NMS/count.
- TRAKE local order khác source-frame order.

### Gate

```text
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online -q
```

Sau migration, 466 test cũ phải được update đúng semantics và toàn suite phải
xanh; không được chỉ xóa assertions cũ để đạt pass.

---

## 9. Những hạng mục Online không bị data migration thay đổi thuật toán

Các phần sau chủ yếu chỉ cần đổi record fields/fixtures, không cần thiết kế lại:

- t-KIS/v-KIS shared pipeline.
- Query variants q0/q1/q2 retrieval độc lập.
- OCR/ASR/summary branch candidate levels.
- Per-branch normalization.
- Fusion/provenance framework.
- DANTE dynamic programming policy.
- VQA evidence budget/orchestrator contract.
- Timeout/concurrency/error taxonomy.

Không tranh thủ data migration để thay ranking weights hoặc DANTE formula nếu
không có benchmark riêng.

---

## 10. Hạng mục optional, không chặn baseline

### Stable Diffusion

`OPTIONAL`. Không cần để ingest/query CLIP NPY BTC. Chỉ nghiên cứu như một query
expansion/generation branch khi baseline thật đã đo xong.

### QUEST

`OPTIONAL`. Không nằm trong baseline DANTE matrix hiện đã chốt. Chỉ bật sau khi
reproduce paper và benchmark trên validation queries.

### Ensemble object detector

`OPTIONAL`. BTC đã cung cấp object detections. Không chạy YOLO/Co-DETR lại trước
khi chứng minh organizer objects thiếu nghiêm trọng.

### Temporal local refinement trên video

`OPTIONAL/P1`. Có thể đọc vài frame quanh `source_frame_idx` sau retrieval để
tinh chỉnh, nhưng submission vẫn dùng frame index đúng rule và không được làm
baseline phụ thuộc vào video decode latency.

---

## 11. Ma trận handoff Offline → Online

| Artifact/resource | Offline phải bảo đảm | Online dùng như thế nào |
|---|---|---|
| Manifest | Model/contract/version/count/fingerprint | Startup readiness |
| Milvus visual | ID, video, local index, vector 512 | KIS + TRAKE corpus |
| Milvus OCR | frame ID, video ID, Vietnamese vector | OCR semantic |
| Milvus ASR | video+interval+time+vector | ASR semantic |
| Milvus summary | video+vector | Summary semantic |
| ES OCR | frame ID + OCR text | OCR lexical/VQA |
| ES ASR | video+interval+time+text | ASR lexical/VQA |
| ES summary | video+summary | Summary lexical/VQA |
| SQLite videos | media fields + relative video path | metadata/VQA/debug |
| SQLite metadata | exact map row + image path | hydration/output/order |
| SQLite objects | normalized objects/NMS/provenance | object constraints |

Một handoff được xem là đạt chỉ khi Online validator đọc trực tiếp database thật
và xác nhận JOIN, không chỉ so JSON/Parquet trung gian.

---

## 12. Thứ tự triển khai bắt buộc

## Phase M0 — Freeze contract

- [ ] Merge `docs/19` và tài liệu này vào shared branch.
- [ ] Chốt `organizer-v1` identifier/schema.
- [ ] Chốt object threshold/NMS đã nêu.
- [ ] Chốt manifest location.

Không code ba hướng schema khác nhau trước khi M0 xong.

## Phase M1 — Offline organizer adapter

- [ ] O0 shared contract.
- [ ] O1 discovery.
- [ ] O2 CSV validation.
- [ ] O3 NPY loader.
- [ ] O5 media loader.
- [ ] O6 object loader.

Output M1: canonical in-memory records + validation report, chưa cần database.

## Phase M2 — Offline database migration

- [ ] O4 SQLite schema.
- [ ] O10 Milvus schema/indexer.
- [ ] O11 manifest.
- [ ] Elasticsearch existing indexes revalidated.

Output M2: `L21_V001` index được vào ba database.

## Phase M3 — Online shared contract migration

- [ ] N0 docs.
- [ ] N1 identifiers.
- [ ] N2 models/ports.
- [ ] N3 SQLite adapter.
- [ ] N4 Milvus KIS adapter.
- [ ] N12 validator.
- [ ] N15 core fixtures.

Output M3: Person A infrastructure đọc đúng organizer DB.

## Phase M4 — Online KIS migration

- [ ] N5 OpenCLIP text encoder.
- [ ] N6 retrieval hydration.
- [ ] N7 ranking/dedup.
- [ ] N13 KIS serializer.

Output M4: one real text query → correct source frame result.

## Phase M5 — OCR/ASR/summary enrichment

- [ ] O7 OCR.
- [ ] O8 ASR/summary.
- [ ] O9 Vietnamese embeddings.
- [ ] Online semantic/lexical vertical slice.

Output M5: bảy KIS branches chạy trên dữ liệu thật.

## Phase M6 — TRAKE production

- [ ] N4 production visual corpus.
- [ ] N8 production wiring/output/performance.

Output M6: ordered event query → video + source frame sequence.

## Phase M7 — VQA production

- [ ] N9 image resolver.
- [ ] N10 VLM adapter.
- [ ] N13 VQA serializer.

Output M7: question → grounded answer + exact source frame.

## Phase M8 — Quality features và tuning

- [ ] N11 real LLM rewrite nếu benchmark tốt.
- [ ] Object synonyms/position tuning.
- [ ] Ranking weights/top-k/latency tuning.
- [ ] Optional QUEST/Stable Diffusion experiments.

---

## 13. Việc có thể làm song song và dependency

### Sau M0, có thể song song

```text
Lane Offline A:
organizer discovery + CSV + NPY + media

Lane Offline B:
object loader + OCR/ASR path adaptation

Lane Online A:
domain/ports/adapters/validator migration

Lane Online B:
OpenCLIP encoder + retrieval hydration

Lane Online C:
ranking/API fixtures migration
```

### Không được làm song song trước upstream contract

- Online adapter không thể hoàn tất trước schema SQLite/Milvus được freeze.
- Ranking/API không thể hoàn tất trước `FrameCandidate` fields được freeze.
- TRAKE real adapter không thể hoàn tất trước visual `local_index` schema.
- VQA real resolver không thể hoàn tất trước `image_rel_path` contract.

### Interface freeze bắt buộc trước merge

Freeze cùng lúc:

```text
FrameMetadata
FrameCandidate
FusedFrameCandidate
OrderedVisualFrame
ImageEvidence
TRAKEFrameMatch
Milvus visual schema
SQLite schema
competition serializer inputs
```

---

## 14. P0/P1/P2 backlog

## P0 — Blocker không thể chạy đúng dữ liệu BTC

- [ ] Organizer adapter và validation.
- [ ] Internal frame ID mới.
- [ ] SQLite schema mới.
- [ ] Direct NPY visual ingestion.
- [ ] Milvus visual `local_index` schema.
- [ ] Object YXYX loader.
- [ ] Online OpenCLIP text encoder.
- [ ] Online domain/adapter/ranking migration bỏ shot semantics.
- [ ] `source_frame_idx` propagation và output.
- [ ] Manifest/model identity validation.
- [ ] One real vertical slice.

## P1 — Cần để hệ thống thi mạnh/đầy đủ

- [ ] OCR trên JPG organizer keyframes.
- [ ] ASR/summary trên organizer MP4.
- [ ] Vietnamese semantic artifacts/indexing.
- [ ] Production TRAKE corpus.
- [ ] Production VQA resolver/VLM.
- [ ] Object synonym normalization.
- [ ] Production LLM rewrite nếu benchmark tốt.
- [ ] Latency/memory benchmark và tuning.
- [ ] UI thumbnail/video seek sử dụng metadata mới.

## P2 — Optional experiment

- [ ] Stable Diffusion.
- [ ] QUEST.
- [ ] Additional object detector ensemble.
- [ ] Video local temporal refinement.

---

## 15. Validation gates cuối cùng

## G1 — Raw data gate

- [ ] Required folders readable.
- [ ] Video-family coverage report.
- [ ] CSV schema/count/order pass.
- [ ] NPY shape/dtype/norm pass.
- [ ] JPG existence/count pass.
- [ ] Media JSON pass.
- [ ] Object arrays/bboxes pass.

## G2 — Offline artifact/database gate

- [ ] Manifest published only after success.
- [ ] SQLite schema exact.
- [ ] Milvus schema/index exact.
- [ ] ES mapping exact.
- [ ] Record counts correct.
- [ ] Cross-database joins correct.
- [ ] Rollback tested.

## G3 — Online startup gate

- [ ] Visual model ID exact match.
- [ ] Visual dimension/norm match.
- [ ] Required core services healthy.
- [ ] Optional failures accurately degraded.
- [ ] Enabled advanced modes have production adapters.

## G4 — KIS gate

- [ ] t-KIS and v-KIS same pipeline.
- [ ] Query text → CLIP query vector → visual candidates.
- [ ] Metadata hydration includes source frame.
- [ ] Seven branches operate at correct candidate levels.
- [ ] Final dedup by video/source frame.
- [ ] Submission serializer golden test.

## G5 — TRAKE/VQA gate

- [ ] TRAKE corpus streams by local index.
- [ ] DANTE never crosses video.
- [ ] TRAKE source frame list exact.
- [ ] VQA image resolver safe.
- [ ] VLM answer grounded to evidence IDs.
- [ ] VQA output exact source frame.

## G6 — Full-system gate

- [ ] Full Offline tests pass.
- [ ] Full Online tests pass.
- [ ] `L21_V001` real vertical slice pass.
- [ ] Variable-FPS sample pass.
- [ ] Duplicate-frame sample pass.
- [ ] Full support consistency pass.
- [ ] Performance test under contest hardware constraints.
- [ ] Restart/recovery/read-only behavior pass.

---

## 16. Definition of Done

Offline được xem là hoàn thành khi:

1. Có thể ingest nguyên layout BTC mà không đổi tên/re-encode source files.
2. Tạo đúng Milvus, Elasticsearch, SQLite và manifest.
3. Không sinh shot ID giả.
4. Cross-database JOIN và count pass trên dữ liệu thật.
5. OCR/ASR/summary có empty-success/failure semantics rõ.

Online được xem là hoàn thành khi:

1. Startup xác nhận đúng organizer manifest/model/schema.
2. KIS dùng exact OpenAI CLIP ViT-B/32 text tower.
3. Mọi frame result giữ `source_frame_idx` từ CSV.
4. Ranking/dedup/serializer không còn phụ thuộc `shot_id`.
5. TRAKE và VQA dùng production adapters khi endpoint enabled.
6. Full test suite và ít nhất một real-data vertical slice pass.

Toàn hệ thống chỉ được gọi là competition-ready sau khi cả hai Definition of
Done cùng đạt và output được đối chiếu với exact submission rule/endpoint BTC.

---

## 17. Các quyết định đã chốt trong tài liệu này

1. Organizer data là primary competition source.
2. Shot/keyframe extraction, visual re-embedding và object re-detection bị
   bypass trong organizer baseline.
3. Internal ID là `{video_id}_{keyframe_no:03d}`.
4. `source_frame_idx` giữ nguyên từ CSV.
5. TRAKE order dùng `local_index`.
6. Offline visual source là BTC NPY; Online text tower là
   `ViT-B-32::openai`.
7. SQLite giữ exact mapping metadata và relative image path.
8. Object bbox chuyển YXYX→XYXY; threshold 0.10; per-class NMS IoU 0.50;
   query default 0.50.
9. Online domain giữ tên `timestamp_sec`, nhưng nguồn duy nhất là CSV
   `pts_time` qua SQLite `pts_time_sec`.
10. Competition dedup dùng `(video_id, source_frame_idx)`.
11. Generated-shot pipeline chỉ là fallback với contract version riêng.
12. Stable Diffusion và QUEST không chặn baseline.

