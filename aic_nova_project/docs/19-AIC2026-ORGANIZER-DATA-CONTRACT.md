# 19 — AIC2026 ORGANIZER DATA CONTRACT AND PROCESSING PLAN

## 1. Mục đích và trạng thái tài liệu

Tài liệu này là nguồn tham chiếu tập trung cho dữ liệu hỗ trợ do Ban tổ chức
(BTC) cung cấp cho vòng Sơ tuyển AIC 2026, bao gồm:

- Cấu trúc archive và thư mục sau khi giải nén.
- Schema thực tế của video, keyframe, CLIP feature, map-keyframes, media-info và
  object detection.
- Quy tắc JOIN chính xác giữa các nguồn.
- Các ngoại lệ đã phát hiện trên toàn bộ dữ liệu hỗ trợ hiện có.
- Data contract mục tiêu giữa Phase Offline và Phase Online.
- Hướng xử lý được đề xuất cho ingestion, retrieval, TRAKE, Q&A và output nộp
  bài.



- Đọc trực tiếp các file mẫu của `L21_V001`.
- Đối chiếu trực tiếp video và 307 keyframe của `L21_V001`.
- Kiểm tra schema/count của toàn bộ 873 bộ support record từ L21 đến L30.
- Kiểm tra tất cả 873 file CLIP feature và map-keyframes.
- Kiểm tra toàn bộ 307 object JSON của `L21_V001`.
- Đối chiếu với code Offline trên `origin/develop_mixi` và Online trên
  `feature/online-phase-Knguyen` tại thời điểm audit.

Tài liệu mô tả dữ liệu đã kiểm chứng và hướng triển khai mục tiêu. Nó không có
nghĩa là source code hiện tại đã thực hiện đầy đủ contract này.

Các nhãn bằng chứng dùng trong tài liệu:

- `CONFIRMED_DESIGN`: quy tắc đã được BTC công bố hoặc đã được nhóm chốt làm
  contract mục tiêu.
- `CONFIRMED_CODE`: hành vi đã được kiểm tra trực tiếp trong source code hiện
  tại.
- `CONTRACT_MISMATCH`: source/schema hiện tại không khớp dữ liệu BTC hoặc
  contract mục tiêu.
- `NEED_RUNTIME_VERIFICATION`: cần model, database hoặc vertical slice thật để
  xác nhận.

---

## 2. Các gói dữ liệu BTC cung cấp

BTC cung cấp hai nhóm archive.

### 2.1 Video và keyframe chia theo batch

```text
Videos_L21_a.zip
Videos_L22_a.zip
...
Videos_L26_a.zip ... Videos_L26_e.zip
...
Videos_L30_a.zip

Keyframes_L21.zip
Keyframes_L22.zip
...
Keyframes_L26_a.zip ... Keyframes_L26_e.zip
...
Keyframes_L30.zip
```

Video và keyframe được chia nhỏ để tải/giải nén. Tên video là domain ID chính,
ví dụ:

```text
L21_V001.mp4
L21_V002.mp4
L22_V001.mp4
```

Không được giả định các số video liên tục. Ví dụ L21 không có `L21_V004` và
`L21_V020`; sự thiếu này nhất quán ở mọi nguồn hỗ trợ đã kiểm tra.

### 2.2 Dữ liệu hỗ trợ dùng chung

```text
clip-features-32-aic25-b1.zip
map-keyframes-aic25-b1.zip
media-info-aic25-b1.zip
objects-aic25-b1.zip
```

Tên archive có nhãn `aic25-b1` vì dữ liệu Batch 1 kế thừa dữ liệu AIC 2025.
Khi lưu cục bộ, tên folder thực tế đang dùng là:

```text
clip-features-32/
keyframes/
map-keyframes/
media-info/
objects/
video/
```

Tên folder phải cấu hình được qua `AIC_DATA_ROOT`; source code không được
hardcode đường dẫn máy cá nhân hoặc OneDrive.

---

## 3. Cấu trúc thư mục chuẩn sau khi giải nén

```text
<AIC_DATA_ROOT>/
├── video/
│   ├── L21_V001.mp4
│   ├── L21_V002.mp4
│   └── ...
├── keyframes/
│   ├── L21_V001/
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   └── ...
│   └── ...
├── clip-features-32/
│   ├── L21_V001.npy
│   ├── L21_V002.npy
│   └── ...
├── map-keyframes/
│   ├── L21_V001.csv
│   ├── L21_V002.csv
│   └── ...
├── media-info/
│   ├── L21_V001.json
│   ├── L21_V002.json
│   └── ...
└── objects/
    ├── L21_V001/
    │   ├── 001.json
    │   ├── 002.json
    │   └── ...
    └── ...
```

Một `video_id` hợp lệ trong dữ liệu mẫu có dạng:

```text
L<batch>_V<video-number>
```

Ví dụ:

```text
L21_V001
L26_V499
L30_V096
```

Không dùng tên archive, tên YouTube hoặc Milvus internal primary key làm
`video_id`.

---

## 4. Phạm vi dữ liệu đã kiểm chứng

Các support package hiện có chứa:

```text
873 video IDs
177,321 keyframe records
873 CLIP .npy files
873 map-keyframes CSV files
873 media-info JSON files
873 object directories
```

Số video theo batch:

| Batch | Video IDs | Keyframe records |
|---|---:|---:|
| L21 | 29 | 7,800 |
| L22 | 31 | 9,096 |
| L23 | 25 | 2,326 |
| L24 | 43 | 6,781 |
| L25 | 88 | 37,445 |
| L26 | 498 | 79,590 |
| L27 | 16 | 4,914 |
| L28 | 24 | 10,683 |
| L29 | 23 | 10,771 |
| L30 | 96 | 7,915 |
| **Tổng** | **873** | **177,321** |

Kiểm tra toàn bộ 873 video ID cho kết quả:

```text
NPY row count = CSV data-row count = object JSON file count
media-info JSON tương ứng tồn tại
NPY dtype = float16
NPY vector dimension = 512
```

Ở máy audit, video và keyframe mới được giải nén đầy đủ cho L21. Toàn bộ 29
video L21 đều thỏa:

```text
CSV rows = NPY rows = JPG count = object JSON count
```

Điều này đủ để chốt schema và viết adapter. Muốn chạy full-corpus vertical
slice cho L22–L30 vẫn cần giải nén các archive video/keyframe còn lại.

---

## 5. Quy tắc JOIN trung tâm

### 5.1 Hai thành phần khóa nguồn

Mọi artifact keyframe-level của BTC JOIN bằng:

```text
video_id + keyframe_no
```

Trong đó:

- `video_id` lấy từ tên video/folder/file, ví dụ `L21_V001`.
- `keyframe_no` là cột `n` trong CSV, bắt đầu từ 1.

### 5.2 Ví dụ `L21_V001`, keyframe đầu tiên

```text
video_id:          L21_V001
keyframe_no:       1
local_index:       0
video file:        video/L21_V001.mp4
keyframe image:    keyframes/L21_V001/001.jpg
CLIP vector:       np.load("L21_V001.npy")[0]
map CSV row:       n = 1
object detections: objects/L21_V001/001.json
media metadata:    media-info/L21_V001.json
```

### 5.3 Công thức tổng quát

```text
keyframe_filename = f"{keyframe_no:03d}.jpg"
object_filename   = f"{keyframe_no:03d}.json"
local_index       = keyframe_no - 1
clip_row_index    = keyframe_no - 1
timestamp_sec     = CSV.pts_time
source_frame_idx  = CSV.frame_idx
```

`CONFIRMED_DESIGN`: thứ tự row của CLIP feature tương ứng thứ tự keyframe.

`CONFIRMED_DESIGN`: `source_frame_idx` lấy nguyên giá trị `frame_idx` BTC cung
cấp và là giá trị dùng cho competition output.

### 5.4 Không được JOIN bằng các giá trị sau

- Không JOIN bằng `frame_idx`: nhiều keyframe có thể map về cùng frame index.
- Không JOIN bằng `pts_time`: đây là số thực và có sai số làm tròn.
- Không JOIN bằng `shot_id`: dữ liệu BTC không cung cấp shot.
- Không JOIN bằng Milvus `pk`: đây chỉ là khóa nội bộ database.
- Không JOIN bằng thứ tự filesystem chưa sort.

---

## 6. Schema `map-keyframes/*.csv`

Tất cả 873 CSV đã kiểm tra có header giống nhau:

```text
n,pts_time,fps,frame_idx
```

### 6.1 Ý nghĩa field

| Field | Type sau parse | Ý nghĩa |
|---|---|---|
| `n` | integer, >= 1 | Số thứ tự keyframe trong video, 1-based |
| `pts_time` | finite float, >= 0 | Presentation timestamp, đơn vị giây |
| `fps` | finite float, > 0 | FPS dùng trong map của video |
| `frame_idx` | integer, >= 0 | Frame index do BTC cung cấp, dùng khi nộp |

### 6.2 Invariant bắt buộc

Trong một video:

```text
n == 1, 2, ..., N
pts_time tăng nghiêm ngặt
fps hữu hạn, dương và nhất quán trong video
frame_idx không âm
```

Không được yêu cầu `frame_idx` tăng nghiêm ngặt.

### 6.3 FPS thực tế

Trên 873 video:

| FPS | Số video |
|---:|---:|
| 25.0 | 781 |
| 26.44 | 1 |
| 29.97 | 30 |
| 30.0 | 61 |

Không hardcode 25 hoặc 30 FPS.

### 6.4 `frame_idx` trùng là dữ liệu hợp lệ cần xử lý

Có 192/873 video có ít nhất một cặp keyframe liên tiếp cùng `frame_idx`.
Nguyên nhân quan sát được là phép làm tròn/lấy phần nguyên khi chuyển timestamp
sang frame index. Ví dụ:

```text
n=1, pts_time=0.0000000, fps=30, frame_idx=0
n=2, pts_time=0.0333333, fps=30, frame_idx=0
```

Video có nhiều duplicate nhất trong dữ liệu đã kiểm tra là `L28_V006`, với 15
row frame index trùng.

Hệ quả bắt buộc:

- Primary key nội bộ phải dùng `video_id + keyframe_no`.
- TRAKE order phải dùng `local_index` hoặc `pts_time`.
- Competition output phải dedup theo `video_id + source_frame_idx`.
- Validator không được fail chỉ vì hai keyframe map cùng frame index.

### 6.5 Không tự tính lại frame index

Sai số lớn nhất đã quan sát giữa:

```text
pts_time - frame_idx / fps
```

là khoảng `0.04` giây, tương đương một frame ở 25 FPS.

So pixel mẫu cho thấy một số JPEG khớp nhất với frame video `frame_idx + 1`
dù CSV ghi `frame_idx`. Đây có thể là khác biệt giữa cách trích ảnh theo PTS và
cách map frame bằng phép làm tròn.

Quy tắc cuối cùng:

```text
Luôn nộp CSV.frame_idx.
Không sửa CSV theo OpenCV.
Không suy ra frame_idx từ pts_time × fps.
```

---

## 7. Schema `clip-features-32/*.npy`

### 7.1 Dữ liệu thực tế

Tất cả 873 file đã kiểm tra:

```text
ndim:      2
shape:     (N, 512)
dtype:     float16
row count: N = số keyframe của video
```

Với `L21_V001`:

```text
shape: (307, 512)
dtype: float16
finite values: true
```

Norm vector của `L21_V001`:

```text
min:  0.999553
mean: 1.000019
max:  1.000489
```

Các vector đã L2-normalized trong sai số lượng tử float16.

### 7.2 Model space

BTC công bố feature được tạo bằng OpenAI CLIP ViT-B/32. Vector dimension 512
phù hợp với contract này.

Online visual text encoder bắt buộc dùng text tower cùng model space:

```text
OpenCLIP model name: ViT-B-32
pretrained weights: openai
internal model id:  ViT-B-32::openai
```

Không dùng PE-Core, BEiT-3 hoặc checkpoint CLIP khác để query collection này.
Cùng dimension không có nghĩa là cùng embedding space.

### 7.3 Ingestion rule

```text
raw = np.load(path, allow_pickle=False)
assert raw.ndim == 2
assert raw.shape == (map_row_count, 512)
assert raw.dtype == np.float16
assert finite(raw)

vector = raw[local_index].astype(np.float32)
vector = vector / ||vector||_2
```

Không chạy lại image encoder cho baseline khi BTC đã cung cấp feature tương
ứng chính xác với keyframe/map/object.

`NEED_RUNTIME_VERIFICATION`: chạy text-encoder smoke test thật và xác nhận
dimension 512, norm 1, sau đó thực hiện một real Milvus search.

---

## 8. Schema keyframe image

Keyframe được lưu theo video:

```text
keyframes/<video_id>/<keyframe_no:03d>.jpg
```

Với `L21_V001`:

```text
307 files
001.jpg ... 307.jpg
resolution: 1280 × 720
```

Trong toàn bộ support data, số keyframe/video nằm trong khoảng 24 đến 632, nên
ba chữ số đủ cho Batch 1. Adapter vẫn phải parse số từ file stem thay vì giả
định vĩnh viễn rằng các batch tương lai không vượt 999 keyframe.

Image resolver phải trả reference an toàn từ data root; không đưa absolute path
máy cá nhân ra public API.

---

## 9. Schema `media-info/*.json`

Các field đã quan sát:

```text
author
channel_id
channel_url
description
keywords
length
publish_date
thumbnail_url
title
watch_url
```

Đây là metadata nguồn/YouTube, không phải technical media probe.

### 9.1 Quy tắc sử dụng

- Đọc bằng UTF-8.
- Dùng `video_id` từ filename làm khóa JOIN.
- Có thể lưu title, author, description, keywords, publish date và source URL.
- `length` là số giây làm tròn; chỉ dùng cho display hoặc validation mềm.
- Không lấy FPS, exact frame count hoặc exact duration từ media-info.
- Không dùng raw YouTube description như một video summary đã làm sạch.

### 9.2 Hướng retrieval

Title/keywords có thể là video-level lexical evidence trọng số thấp. Description
thường chứa boilerplate của kênh và link quảng bá, vì vậy phải được làm sạch
trước khi dùng cho semantic retrieval hoặc summarization.

Khuyến nghị lưu media-info trong SQLite table `videos`; không trộn thẳng vào
`video_summaries` mà không ghi provenance.

---

## 10. Schema `objects/<video_id>/<NNN>.json`

### 10.1 Các array song song

Mỗi file object mẫu có:

```text
detection_scores
detection_class_names
detection_class_entities
detection_boxes
detection_class_labels
```

Với toàn bộ 307 file của `L21_V001`:

- Năm array luôn cùng length.
- Mỗi file chứa đúng top 100 detection.
- Tổng cộng 30,700 raw detection.
- Score và bbox được serialize dưới dạng string.
- Không có NaN/Infinity hoặc bbox ngoài `[0, 1]`.
- Detection được sắp theo score giảm dần.

### 10.2 Ý nghĩa field

| Field | Ý nghĩa mục tiêu |
|---|---|
| `detection_scores[i]` | Confidence float trong `[0,1]` |
| `detection_class_names[i]` | Open Images MID, ví dụ `/m/079cl` |
| `detection_class_entities[i]` | Nhãn đọc được, ví dụ `Person` |
| `detection_class_labels[i]` | Numeric/string class ID |
| `detection_boxes[i]` | Bbox normalized theo thứ tự YXYX |

### 10.3 Bbox order bắt buộc

Raw box có dạng:

```text
[y_min, x_min, y_max, x_max]
```

Domain Online đang dùng:

```text
x_min, y_min, x_max, y_max
```

Adapter phải chuyển:

```python
y_min, x_min, y_max, x_max = raw_box
```

Không copy raw index 0→`x_min`, 1→`y_min`; làm vậy sẽ đảo trục.

### 10.4 Confidence và nhiễu

Trên `L21_V001`:

| Threshold | Detection còn lại | Frame không có detection |
|---:|---:|---:|
| >= 0.10 | 5,662 | 1 |
| >= 0.25 | 2,690 | 6 |
| >= 0.30 | 2,262 | 6 |
| >= 0.50 | 1,186 | 22 |

Raw top 100 chứa nhiều detection score rất thấp và nhiều box trùng/gần trùng.
Không được dùng raw array length để trả lời object count.

### 10.5 Object processing đề xuất

```text
parse numeric strings
→ validate parallel arrays
→ validate normalized YXYX bbox
→ confidence prefilter
→ per-class NMS
→ normalize label/MID
→ insert SQLite
```

Baseline đề xuất ban đầu:

```text
ingestion threshold: configurable, start at 0.10
per-class NMS IoU:   configurable, start at 0.50
query threshold:     do UI/query quyết định, default hiện có thể là 0.50
```

Các giá trị trên phải ghi vào ingestion manifest. Cần tuning trên validation
query thật trước khi chốt production.

Nên giữ cả:

```text
label_display
label_normalized
class_mid
class_label_id
confidence
x_min, y_min, x_max, y_max
model_source
```

Open Images có nhãn phân cấp/gần nghĩa như `Person`, `Man`, `Human body`,
`Vehicle`, `Land vehicle`, `Car`. Query layer cần synonym/ontology mapping;
không so chuỗi người dùng với một label duy nhất một cách tuyệt đối.

---

## 11. Data contract mục tiêu

### 11.1 Hai loại ID phải tách riêng

```text
Internal frame_id/keyframe_id:
    khóa JOIN ổn định giữa database và artifact

Competition source_frame_idx:
    frame số từ CSV dùng để nộp bài
```

Không gọi cả hai cùng một nghĩa `frame_id` trong code mà không có qualifier.

### 11.2 Internal keyframe ID đề xuất

```text
frame_id = f"{video_id}_{keyframe_no:03d}"
```

Ví dụ:

```text
L21_V001_001
L21_V001_307
```

Tên field `frame_id` có thể được giữ để giảm thay đổi ở retrieval/ranking, nhưng
semantics phải là **keyframe identity nội bộ**, không phải competition frame
index.

Không tạo `shot_id=0` giả để ép dữ liệu BTC vào contract cũ. Nếu pipeline tự
sinh shot cho thử nghiệm khác, `shot_id` phải là field nullable/provenance-specific
và không tham gia canonical key BTC.

### 11.3 Record keyframe mục tiêu

```text
frame_id:          L21_V001_001
video_id:          L21_V001
keyframe_no:       1
local_index:       0
pts_time_sec:      0.0
fps:               30.0
source_frame_idx:  0
clip_row_index:    0
image_rel_path:    keyframes/L21_V001/001.jpg
```

### 11.4 SQLite schema mục tiêu

Khuyến nghị thêm/chuyển thành ba table:

```text
videos
metadata
objects
```

`videos`:

```text
video_id TEXT PRIMARY KEY
media_title TEXT
media_author TEXT
media_description TEXT
media_keywords_json TEXT
media_length_sec REAL
publish_date TEXT
watch_url TEXT
video_rel_path TEXT
```

`metadata`:

```text
frame_id TEXT PRIMARY KEY
video_id TEXT NOT NULL
keyframe_no INTEGER NOT NULL
local_index INTEGER NOT NULL
pts_time_sec REAL NOT NULL
fps REAL NOT NULL
source_frame_idx INTEGER NOT NULL
image_rel_path TEXT NOT NULL
UNIQUE(video_id, keyframe_no)
UNIQUE(video_id, local_index)
```

Không đặt `UNIQUE(video_id, source_frame_idx)` vì duplicate frame index là dữ
liệu thực tế hợp lệ.

`objects`:

```text
id INTEGER PRIMARY KEY
frame_id TEXT NOT NULL
label_display TEXT NOT NULL
label_normalized TEXT NOT NULL
class_mid TEXT
class_label_id TEXT
confidence REAL NOT NULL
x_min REAL NOT NULL
y_min REAL NOT NULL
x_max REAL NOT NULL
y_max REAL NOT NULL
model_source TEXT NOT NULL
FOREIGN KEY(frame_id) REFERENCES metadata(frame_id)
```

### 11.5 Milvus `visual_features` mục tiêu

```text
pk                 auto internal primary key
frame_id           internal keyframe ID
video_id
local_index
embedding          FLOAT_VECTOR dim=512
```

Metric/index:

```text
metric: IP
index: HNSW
stored vectors: L2-normalized float32
query vectors: L2-normalized float32
```

`local_index` cần cho production `VisualCorpusPort` và TRAKE ordering.

### 11.6 Ingestion manifest bắt buộc

Database cần một manifest/version record chứa tối thiểu:

```text
dataset_name
dataset_batch
ingestion_version
visual_model_id = ViT-B-32::openai
visual_dimension = 512
visual_source_dtype = float16
visual_stored_dtype = float32
visual_normalized = true
object_source = organizer Open Images detections
object_threshold
object_nms_iou
frame_id_contract_version
created_at
source file hashes or dataset fingerprint
```

Dimension và norm không đủ để phát hiện hai model space khác nhau. Online
readiness phải kiểm tra manifest/model ID, không chỉ collection dimension.

---

## 12. Hướng xử lý Phase Offline

### 12.1 Nguồn chính và fallback

Baseline AIC 2026 nên dùng dữ liệu BTC theo thứ tự ưu tiên:

```text
BTC keyframes + map + CLIP feature + object data
→ direct organizer ingestion
→ databases
```

Không chạy lại Module 1/2/5 cho baseline nếu support artifact tương ứng đã tồn
tại và qua validation.

Fallback chỉ dùng khi support file thiếu hoặc cần thí nghiệm:

```text
official video only
→ self-generated keyframes/features/objects
→ ghi provenance khác rõ ràng
→ không trộn âm thầm với organizer artifact
```

### 12.2 Organizer dataset adapter

Cần một adapter duy nhất chịu trách nhiệm:

1. Discover `video_id` từ intersection/expected dataset manifest.
2. Đọc CSV bằng UTF-8/BOM-safe parser.
3. Validate `n`, PTS, FPS và non-negative frame index.
4. Load NPY với `allow_pickle=False`.
5. Validate `(N,512)`, float16, finite, norm.
6. Resolve `NNN.jpg` và `NNN.json` bằng `n`.
7. Parse/reorder/NMS object detections.
8. Sinh canonical internal `frame_id`.
9. Ghi visual vector vào Milvus.
10. Ghi metadata/video/object vào SQLite.
11. Ghi manifest và validation report.

Adapter không được phụ thuộc vào schema shot/keyframe tự sinh cũ.

### 12.3 OCR, ASR và summary

BTC không cung cấp OCR/ASR/summary trong các package đã kiểm tra. Các module này
vẫn phải chạy Offline trên organizer video/keyframe:

```text
organizer keyframe JPG → OCR → OCR text + Vietnamese embedding
organizer video MP4    → ASR → cleaned transcript + Vietnamese embedding
media/ASR              → summary → Vietnamese embedding
```

OCR và object phải dùng cùng internal `frame_id` tạo từ `video_id + n`.

ASR interval giữ:

```text
video_id
interval_id
start_time_sec
end_time_sec
```

Summary giữ video-level `video_id`; không giả thành frame.

### 12.4 Validation trước khi index

Mỗi video phải qua:

```text
map row count == NPY rows
map row count == keyframe JPG count (khi archive đã giải nén)
map row count == object JSON count
media-info exists
n contiguous from 1
PTS strictly increasing
FPS valid and constant within video
frame_idx non-negative (duplicate allowed)
NPY shape (N,512), finite, unit-norm
object arrays parallel and bbox valid
```

Không được insert một video partially rồi coi là thành công. Cần per-video
transaction/rollback hoặc staged replace giống Module 7 contract hiện tại.

---

## 13. Hướng xử lý Phase Online

### 13.1 Visual KIS

```text
query text
→ OpenAI CLIP ViT-B/32 text encoder
→ normalized 512-d vector
→ Milvus visual_features
→ hydrate SQLite metadata
→ fusion/dedup
→ competition output adapter
```

`CONTRACT_MISMATCH`: Online hiện tại vẫn mặc định dùng PE-Core. Phải đổi cả KIS
visual encoder và TRAKE event encoder sang `ViT-B-32::openai` trong một contract
change thống nhất.

### 13.2 Textual KIS và v-KIS

Textual KIS và v-KIS vẫn dùng chung text-to-keyframe retrieval pipeline:

```text
t-KIS: mô tả text BTC cung cấp
v-KIS: thí sinh xem clip và tự viết mô tả text
→ cùng query pipeline
```

Thay đổi dataset/model visual không tạo pipeline riêng cho v-KIS.

### 13.3 TRAKE

```text
ordered event texts
→ CLIP ViT-B/32 text vectors
→ full ordered keyframe vectors per video
→ similarity matrix
→ DANTE
→ event-to-keyframe sequence
→ source_frame_idx mapping
```

Ordering bắt buộc:

```text
local_index = n - 1
```

Không order theo `source_frame_idx` vì duplicate frame index tồn tại.

Do khoảng cách keyframe có thể lớn hơn cửa sổ chấm, cần một stage sau DANTE:

```text
winning semantic keyframe
→ optional local temporal refinement quanh source_frame_idx
→ final source frame
```

`NEED_RUNTIME_VERIFICATION`: chỉ bật refinement sau khi đo recall trên ground
truth/validation query. DANTE baseline vẫn dùng toàn bộ ordered keyframes.

### 13.4 Q&A/VQA

```text
question
→ reuse KIS retrieval
→ ranked internal keyframes
→ resolve organizer JPG
→ hydrate OCR/ASR/summary evidence
→ VLM
→ choose grounded frame evidence
→ <video_id, source_frame_idx, answer>
```

ImageResolver phải map:

```text
internal frame_id → image_rel_path
```

Competition adapter phải lấy `source_frame_idx` từ SQLite metadata; không parse
nó từ internal ID.

### 13.5 Object constraint

```text
user label/constraint
→ normalize + synonym/MID mapping
→ SQLite object query
→ per-frame post-NMS detections
→ hard filter hoặc soft boost
```

Position constraint có thể dùng normalized bbox sau khi YXYX→XYXY đúng. Không
cần image dimensions cho normalized-region comparison, nhưng phải ghi rõ tất cả
tọa độ đang ở `[0,1]`.

### 13.6 Final dedup và competition output

Internal ranking có thể chứa hai keyframe khác nhau map cùng competition frame.
Trước output:

```text
group by (video_id, source_frame_idx)
keep highest-ranked representative
preserve internal evidence/provenance in diagnostics
truncate to competition limit
```

KIS output:

```text
<video_id>, <source_frame_idx>
```

Q&A output:

```text
<video_id>, <source_frame_idx>, <answer>
```

TRAKE output:

```text
<video_id>, <source_frame_idx_1>, ..., <source_frame_idx_N>
```

Public/competition output không được trả internal keyframe ID thay cho source
frame index.

---

## 14. Mismatch với code hiện tại

### 14.1 Offline organizer ingestion chưa tồn tại

`CONFIRMED_CODE`, `CONTRACT_MISMATCH`:

- Visual loader hiện đọc Parquet nội bộ, không đọc organizer `.npy`.
- Metadata loader đòi `metadata/<video_id>.json` có `shots[].keyframes[]`.
- Canonical frame ID cũ đòi `shot_id` và relative position.
- Object loader đòi một JSON envelope/video, không đọc một JSON/keyframe với năm
  array song song.
- `map-keyframes` và `media-info` chưa được ingest.
- `source_frame_idx` chưa được giữ trong SQLite.

### 14.2 Online visual model mismatch

`CONFIRMED_CODE`, `CONTRACT_MISMATCH`:

```text
Offline/organizer visual space: ViT-B-32::openai, dim 512
Online current visual encoder:  PE-Core-bigG-14-448
```

KIS visual và TRAKE đều bị ảnh hưởng.

### 14.3 Internal ID và competition ID đang bị trộn

`CONFIRMED_CODE`, `CONTRACT_MISMATCH`:

- Online domain chỉ mang canonical string cũ.
- SQLite không có `source_frame_idx`.
- Hàm `competition_candidates()` hiện vẫn trả internal string frame ID.
- TRAKE/VQA internal results chưa có competition serializer hoàn chỉnh.

### 14.4 Production advanced adapters còn thiếu

`CONFIRMED_CODE`:

- `VisualCorpusPort` mới có fake/testing implementation.
- `ImageResolverPort` mới có fake/testing implementation.
- `EvidenceHydrationPort` mới có fake/testing implementation.
- TRAKE/VQA production composition chưa được attach từ environment mặc định.

### 14.5 Elasticsearch ASR field mismatch độc lập

Offline mới dùng:

```text
start_time_sec
end_time_sec
```

Online Elasticsearch adapter/validator còn đọc:

```text
start_time
end_time
```

Mục này phải sửa cùng đợt contract integration nhưng không phụ thuộc organizer
CLIP feature.

---

## 15. Thứ tự triển khai được đề xuất

### Milestone D1 — Organizer contract models

- Tạo model/path config cho dataset root.
- Tạo `OrganizerMapRecord`, `OrganizerKeyframeRecord`, object raw model và
  manifest model.
- Chốt internal frame ID mới và `source_frame_idx` riêng.
- Unit tests cho duplicate frame index, variable FPS và Unicode path.

### Milestone D2 — Organizer reader và validator

- Discover 873 video IDs.
- Parse CSV/JSON/NPY read-only.
- Cross-count validation.
- NPY shape/dtype/norm validation.
- Object YXYX validation.
- Dataset report trước ingestion.

### Milestone D3 — Offline database migration

- Migrate SQLite metadata/videos/objects.
- Migrate Milvus visual schema thêm `local_index`, dimension 512.
- Ingest organizer vectors trực tiếp.
- Ghi manifest.
- Giữ rollback per video.

### Milestone D4 — OCR/ASR/summary trên organizer IDs

- OCR đọc organizer JPG.
- ASR đọc organizer MP4.
- Summary dùng cleaned ASR và media metadata có provenance.
- Text embeddings/indexes giữ Vietnamese model contract.

### Milestone D5 — Online contract migration

- Đổi identifier/domain/SQLite adapter sang keyframe contract mới.
- Thêm `source_frame_idx`, `keyframe_no`, `local_index`, `image_rel_path`.
- Sửa Elasticsearch ASR `_sec` fields.
- Update fakes/tests/docs atomically.

### Milestone D6 — CLIP Online encoder

- Thay PE-Core bằng OpenAI CLIP ViT-B/32 text tower.
- Readiness kiểm tra model ID, dimension 512 và norm.
- Real query smoke test với organizer Milvus collection.

### Milestone D7 — Production TRAKE/VQA adapters

- Real `VisualCorpusPort` từ ordered organizer vectors.
- Real `ImageResolverPort` từ SQLite/data root.
- Real evidence hydrator.
- Production composition/readiness.

### Milestone D8 — Competition output

- KIS top-100 serializer.
- TRAKE sequence serializer.
- Q&A answer serializer.
- Dedup theo competition frame.
- Submission format tests.

### Milestone D9 — Real vertical slices

- Một KIS query thật.
- Một TRAKE query thật.
- Một Q&A query thật.
- Đo latency, memory, recall và error diagnostics.
- Chỉ sau đó tuning fusion weights, object threshold/NMS và temporal refinement.

---

## 16. Validation gates bắt buộc

### Gate G1 — Dataset completeness

- [ ] Expected video IDs được xác định.
- [ ] Map/NPY/media/object cùng video ID set.
- [ ] Keyframe/video archive cần dùng đã giải nén.
- [ ] Count invariant pass mỗi video.

### Gate G2 — Identifier correctness

- [ ] `n` 1-based liên tục.
- [ ] `local_index = n - 1`.
- [ ] Internal ID unique.
- [ ] Duplicate `source_frame_idx` được chấp nhận.
- [ ] Competition dedup test pass.

### Gate G3 — Visual compatibility

- [ ] Stored vectors dim 512.
- [ ] Stored/query vectors unit norm.
- [ ] Manifest model `ViT-B-32::openai`.
- [ ] Online encoder output dim 512.
- [ ] Real Milvus search trả canonical internal ID hydrate được.

### Gate G4 — Object correctness

- [ ] Numeric strings parse an toàn.
- [ ] Parallel arrays cùng length.
- [ ] YXYX→XYXY đúng.
- [ ] Bbox trong `[0,1]`.
- [ ] Confidence threshold được ghi manifest.
- [ ] Per-class NMS test pass.
- [ ] Count constraint không đếm raw duplicate boxes.

### Gate G5 — Cross-database joins

- [ ] Visual Milvus hit JOIN SQLite metadata.
- [ ] OCR Milvus/ES hit JOIN cùng frame ID.
- [ ] Object row JOIN cùng frame ID.
- [ ] ASR interval map bằng timestamp.
- [ ] Summary giữ video-level.

### Gate G6 — Competition output

- [ ] Không output internal ID.
- [ ] KIS dùng `source_frame_idx`.
- [ ] TRAKE giữ event order và cùng video.
- [ ] Q&A answer có grounded video/frame.
- [ ] Không duplicate `(video_id, source_frame_idx)` trong ranked KIS output.
- [ ] Không vượt giới hạn số đáp án/query.

---

## 17. Những điều tuyệt đối không được làm

1. Không tiếp tục query organizer CLIP vectors bằng PE-Core.
2. Không chạy lại keyframe extraction rồi ghép tùy tiện với organizer NPY/object.
3. Không dùng `frame_idx` làm primary key nội bộ.
4. Không giả định `frame_idx` luôn tăng nghiêm ngặt.
5. Không tính lại `frame_idx` từ `pts_time × fps`.
6. Không hardcode FPS.
7. Không tạo `shot_id=0` giả như domain truth của dữ liệu BTC.
8. Không copy bbox YXYX thành XYXY mà không reorder.
9. Không đếm raw top-100 object boxes trước NMS.
10. Không đưa absolute path máy cá nhân ra API.
11. Không dùng `media-info.length` như exact technical duration.
12. Không coi dimension/norm pass là bằng chứng hai model space tương thích.
13. Không output internal keyframe ID thay cho competition frame index.
14. Không claim full-data readiness khi mới giải nén video/keyframe của L21.

---

## 18. Contract tóm tắt để triển khai

```text
SOURCE IDENTITY
video_id + n

INTERNAL KEYFRAME ID
frame_id = {video_id}_{n:03d}

ORDERING
local_index = n - 1

VISUAL VECTOR
clip_features[n - 1]
OpenAI CLIP ViT-B/32
512 dimensions
L2-normalized

TIMESTAMP
pts_time from CSV

COMPETITION FRAME
source_frame_idx = frame_idx from CSV

IMAGE
keyframes/{video_id}/{n:03d}.jpg

OBJECT
objects/{video_id}/{n:03d}.json
parallel arrays
normalized YXYX → normalized XYXY
confidence filter + per-class NMS

TRAKE ORDER
local_index/PTS, never source_frame_idx

FINAL DEDUP
video_id + source_frame_idx
```

Đây là contract nền phải được merge trước khi sửa đồng bộ Offline ingestion và
Online retrieval/output. Mọi thay đổi model, identifier, database schema hoặc
competition serializer sau này phải được đối chiếu lại với tài liệu này.
