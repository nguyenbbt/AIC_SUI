# 03 — DATABASE SCHEMA CURRENT

## 1. Tổng quan

Hệ thống dùng Polyglot Persistence:

```text
Milvus       → dense vector retrieval
Elasticsearch→ Vietnamese full-text retrieval
SQLite       → relational metadata and object constraints
```

---

# 2. Khóa liên kết

| Đơn vị | Canonical key | Nơi sử dụng |
|---|---|---|
| Video | `video_id` | Tất cả DB |
| Keyframe | `frame_id` | Milvus visual/OCR, ES OCR, SQLite metadata/objects |
| ASR interval | `video_id + interval_id` | Milvus ASR + ES ASR |
| Summary | `video_id` | Milvus summary + ES summary |

Không dùng Milvus `pk` để JOIN.

---

# 3. Canonical `frame_id`

Target format:

```text
{video_id}_{shot_id:05d}_{position_3_digits}
```

Ví dụ:

```text
V001_00000_015
```

Local filename stem:

```text
shot_00000_pos_015
```

phải được chuyển thành global ID trước khi index.

Yêu cầu runtime validation:

- `visual_features.frame_id`
- `ocr_features.frame_id`
- `ocr_texts.frame_id`
- `metadata.frame_id`
- `objects.frame_id`

phải bằng nhau cho cùng một keyframe.

---

# 4. Milvus

## 4.1 Common settings

```text
Index type: HNSW
Metric: IP
Index params:
  M = 16
  efConstruction = 256
Search params:
  ef = 128
```

Stored vectors phải L2-normalized.

Online query vectors cũng phải L2-normalized.

Với vector norm 1:

```text
Inner Product = Cosine Similarity
```

---

## 4.2 Collection `visual_features`

| Field | Type | Required | Mô tả |
|---|---|---:|---|
| `pk` | `INT64`, auto ID, primary | Yes | Internal Milvus ID |
| `frame_id` | `VARCHAR(256)` | Yes | Cross-DB keyframe key |
| `video_id` | `VARCHAR(512)` | Yes | Video ID |
| `shot_id` | `INT64` | Yes | Zero-based shot index |
| `embedding` | `FLOAT_VECTOR(dim)` | Yes | L2-normalized visual vector |

`dim` được phát hiện động từ visual Parquet/model output.

Search output fields:

```text
frame_id
video_id
shot_id
```

---

## 4.3 Collection `ocr_features`

| Field | Type | Required | Mô tả |
|---|---|---:|---|
| `pk` | `INT64`, auto ID, primary | Yes | Internal Milvus ID |
| `frame_id` | `VARCHAR(256)` | Yes | Keyframe JOIN key |
| `video_id` | `VARCHAR(512)` | Yes | Video ID |
| `embedding` | `FLOAT_VECTOR(dim)` | Yes | OCR semantic vector |

`dim` được phát hiện từ `text_ocr` Parquet.

Logical result level:

```text
frame-level
```

Search output fields:

```text
frame_id
video_id
```

---

## 4.4 Collection `asr_features`

| Field | Type | Required | Mô tả |
|---|---|---:|---|
| `pk` | `INT64`, auto ID, primary | Yes | Internal Milvus ID |
| `video_id` | `VARCHAR(512)` | Yes | Video ID |
| `interval_id` | `VARCHAR(256)` | Yes | ASR interval key |
| `start_time_sec` | `FLOAT` | Yes | Interval start |
| `end_time_sec` | `FLOAT` | Yes | Interval end |
| `embedding` | `FLOAT_VECTOR(dim)` | Yes | ASR semantic vector |

Expected current text model dimension is commonly 768, nhưng Online phải đọc schema thật và validate.

Logical result level:

```text
ASR interval-level
```

Search output fields:

```text
video_id
interval_id
start_time_sec
end_time_sec
```

---

## 4.5 Collection `summary_features`

| Field | Type | Required | Mô tả |
|---|---|---:|---|
| `pk` | `INT64`, auto ID, primary | Yes | Internal Milvus ID |
| `video_id` | `VARCHAR(512)` | Yes | Video key |
| `embedding` | `FLOAT_VECTOR(dim)` | Yes | Video summary vector |

Logical result level:

```text
video-level
```

Search output:

```text
video_id
```

---

# 5. Elasticsearch

## 5.1 Connection

Default:

```text
http://localhost:9200
```

## 5.2 Required plugin

```text
analysis-icu
```

## 5.3 Common analyzer

```text
vietnamese_analyzer
```

Components:

```text
icu_tokenizer
icu_folding
lowercase
```

---

## 5.4 Index `ocr_texts`

Document `_id`:

```text
frame_id
```

| Field | ES type | Analyzer | Mô tả |
|---|---|---|---|
| `frame_id` | `keyword` | — | Keyframe JOIN key |
| `video_id` | `keyword` | — | Video ID |
| `shot_id` | `keyword` | — | Shot ID serialized by indexer |
| `ocr_text_concat` | `text` | `vietnamese_analyzer` | OCR text |

Frame không có OCR text có thể không tồn tại trong index.

Logical result level:

```text
frame-level
```

---

## 5.5 Index `asr_transcripts`

Document `_id`:

```text
{video_id}_{interval_id}
```

| Field | ES type | Analyzer | Mô tả |
|---|---|---|---|
| `interval_id` | `keyword` | — | Interval ID |
| `video_id` | `keyword` | — | Video ID |
| `start_time` | `float` | — | Start seconds |
| `end_time` | `float` | — | End seconds |
| `cleaned_text` | `text` | `vietnamese_analyzer` | Cleaned transcript |

Logical result level:

```text
ASR interval-level
```

---

## 5.6 Index `video_summaries`

Document `_id`:

```text
video_id
```

| Field | ES type | Analyzer | Mô tả |
|---|---|---|---|
| `video_id` | `keyword` | — | Video key |
| `summary` | `text` | `vietnamese_analyzer` | Video summary |

Logical result level:

```text
video-level
```

---

# 6. SQLite

## 6.1 File

```text
data/metadata.db
```

Docker mount có thể là:

```text
/data/metadata.db
```

Recommended settings:

```text
journal_mode=WAL
foreign_keys=ON
```

Online dùng read-only connection.

---

## 6.2 Table `metadata`

```sql
CREATE TABLE metadata (
    frame_id  TEXT PRIMARY KEY,
    video_id  TEXT NOT NULL,
    shot_id   INTEGER NOT NULL,
    timestamp REAL NOT NULL
);

CREATE INDEX idx_metadata_video_id
ON metadata(video_id);
```

| Column | Type | Mô tả |
|---|---|---|
| `frame_id` | `TEXT` | Keyframe key |
| `video_id` | `TEXT` | Video ID |
| `shot_id` | `INTEGER` | Shot index |
| `timestamp` | `REAL` | Seconds |

---

## 6.3 Table `objects`

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
    FOREIGN KEY (frame_id)
        REFERENCES metadata(frame_id)
        ON DELETE CASCADE
);

CREATE INDEX idx_objects_frame_id
ON objects(frame_id);

CREATE INDEX idx_objects_label
ON objects(label);
```

Một detection box là một row.

Object count của một label trong frame:

```sql
SELECT COUNT(*)
FROM objects
WHERE frame_id = ?
  AND label = ?
  AND confidence >= ?;
```

Bounding boxes hiện là pixel coordinates.

---

# 7. JOIN reference

| Từ | Key | Sang |
|---|---|---|
| Milvus `visual_features` | `frame_id` | SQLite `metadata` |
| Milvus `visual_features` | `frame_id` | SQLite `objects` |
| Milvus `visual_features` | `frame_id` | ES `ocr_texts` |
| Milvus `ocr_features` | `frame_id` | SQLite `metadata` |
| ES `ocr_texts` | `frame_id` | SQLite `metadata` |
| Milvus `asr_features` | `video_id + interval_id` | ES `asr_transcripts` |
| Milvus `summary_features` | `video_id` | ES `video_summaries` |
| SQLite `objects` | `frame_id` | SQLite `metadata` |

---

# 8. Online result levels

## Frame candidate

```json
{
  "candidate_type": "frame",
  "frame_id": "V001_00000_015",
  "video_id": "V001",
  "shot_id": 0,
  "timestamp_sec": null,
  "branch": "visual_semantic",
  "raw_score": 0.82,
  "normalized_score": null
}
```

## ASR interval candidate

```json
{
  "candidate_type": "asr_interval",
  "video_id": "V001",
  "interval_id": "interval_0001",
  "start_time_sec": 10.2,
  "end_time_sec": 17.8,
  "branch": "asr_semantic",
  "raw_score": 0.79,
  "normalized_score": null
}
```

## Video candidate

```json
{
  "candidate_type": "video",
  "video_id": "V001",
  "branch": "summary_semantic",
  "raw_score": 0.74,
  "normalized_score": null
}
```

---

# 9. Required runtime checks

Trước khi Online integration được xem là hoàn thành:

1. Đọc schema thật của cả 4 Milvus collections.
2. Đọc mapping thật của cả 3 ES indexes.
3. Đọc SQLite schema.
4. Kiểm tra one-record JOIN cho `frame_id`.
5. Kiểm tra one-record JOIN cho ASR interval.
6. Kiểm tra one-record JOIN cho summary.
7. Kiểm tra vector dimensions.
8. Kiểm tra vector norm.
9. Kiểm tra `analysis-icu`.
10. Kiểm tra collection/index/table không rỗng.
