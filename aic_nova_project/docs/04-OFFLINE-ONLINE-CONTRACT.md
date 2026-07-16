# 04 — OFFLINE–ONLINE CONTRACT

## 1. Mục đích

File này định nghĩa giao diện ổn định giữa Phase Offline và Phase Online.

Online chỉ được phụ thuộc vào:

- Database names.
- Field names và types.
- Identifier semantics.
- Vector compatibility.
- Search result levels.
- Required JOIN behavior.

Online không được phụ thuộc vào implementation detail bên trong từng Offline module.

---

# 2. Required resources

## Milvus

Default URI:

```text
http://localhost:19530
```

Collections:

```text
visual_features
ocr_features
asr_features
summary_features
```

## Elasticsearch

Default URI:

```text
http://localhost:9200
```

Indexes:

```text
ocr_texts
asr_transcripts
video_summaries
```

## SQLite

Default path:

```text
data/metadata.db
```

Tables:

```text
metadata
objects
```

Mọi endpoint/path phải configurable.

---

# 3. Identifier contract

## 3.1 `video_id`

- Type: string.
- Stable trong toàn pipeline.
- Không được tạo lại khác nhau ở Online.

## 3.2 `frame_id`

- Type: string.
- Canonical logical key của keyframe.
- Dùng equality JOIN.
- Target format:

```text
{video_id}_{shot_id:05d}_{position_3_digits}
```

Online không được tự đoán hoặc rewrite `frame_id`.

Nếu runtime records không khớp, dừng integration và báo `CONTRACT_MISMATCH`.

## 3.3 `interval_id`

- Type: string.
- Chỉ unique trong context của `video_id`.
- Canonical ASR key:

```text
video_id + interval_id
```

## 3.4 `shot_id`

- Logical type: integer.
- ES OCR có thể serialize dưới dạng keyword/string.
- Adapter phải normalize về integer trong domain model.

---

# 4. Vector contract

## 4.1 Common

- Milvus metric: `IP`.
- Stored vectors: L2-normalized.
- Query vectors: phải L2-normalized.
- Query dimension phải bằng collection dimension.
- Adapter phải validate dimension trước search.

## 4.2 Visual space

Offline encoder:

```text
PE-Core-bigG-14-448 compatible image encoder
```

Online phải dùng text/image encoder tương thích chính xác với visual embeddings đã lưu.

Không được dùng một CLIP checkpoint khác chỉ vì có cùng dimension.

## 4.3 Vietnamese text space

Dùng cho:

```text
ocr_features
asr_features
summary_features
```

Online phải dùng cùng model/pooling/preprocessing contract với Module 6.

---

# 5. Retrieval output contract

## 5.1 Milvus visual result

Required fields:

```text
frame_id
video_id
shot_id
distance/score
```

Converted to:

```text
FrameCandidate
```

## 5.2 Milvus OCR result

Required fields:

```text
frame_id
video_id
distance/score
```

Converted to:

```text
FrameCandidate
```

`shot_id` và `timestamp` được hydrate từ SQLite.

## 5.3 Elasticsearch OCR result

Required fields:

```text
frame_id
video_id
shot_id
_score
```

Converted to:

```text
FrameCandidate
```

## 5.4 Milvus/ES ASR result

Required fields:

```text
video_id
interval_id
start_time
end_time
score
```

Converted first to:

```text
ASRIntervalCandidate
```

Sau đó mới map sang frame.

## 5.5 Milvus/ES summary result

Required fields:

```text
video_id
score
```

Converted to:

```text
VideoCandidate
```

Summary result không phải frame evidence trực tiếp.

---

# 6. ASR interval-to-frame contract

Input:

```text
video_id
start_time_sec
end_time_sec
ASR score
```

Required SQLite query:

```sql
SELECT frame_id, video_id, shot_id, timestamp
FROM metadata
WHERE video_id = ?
ORDER BY timestamp;
```

Mapping strategy phải:

- Deterministic.
- Configurable.
- Không nhảy video.
- Giữ provenance của ASR branch.
- Không nhân bản score không kiểm soát.
- Có test cho overlap, no-overlap và boundary.

Exact baseline strategy nằm trong `08-OPEN-QUESTIONS.md` cho đến khi được chốt.

---

# 7. Summary propagation contract

Summary search trả score ở cấp video.

Quyết định bắt buộc:

```text
Summary không được prefilter cứng.
```

Summary score chỉ được dùng sau khi frame candidates đã được tạo.

Logical propagation:

```text
summary_score(video)
→ controlled boost/prior
→ frames belonging to video
```

Không được tạo một frame giả từ video summary.

---

# 8. Object contract

Object constraints đến từ UI.

Online nhận structured constraints, không tự trích bắt buộc từ text query trong baseline.

Supported logical fields:

```text
label
count operator
count value
position
minimum confidence
filter mode
```

SQLite source:

```text
objects
```

Object filter phải chạy trên candidate set, không full-scan toàn bộ database nếu không cần.

Position logic phải xử lý việc bbox trong SQLite đang là pixel coordinates.

---

# 9. Hydration contract

Frame result phải được hydrate bằng SQLite `metadata`.

Required hydrated fields:

```text
frame_id
video_id
shot_id
timestamp_sec
```

Optional hydration:

```text
objects
image_path
near frames
OCR text
ASR text
summary
```

`image_path` chưa thuộc SQLite schema hiện tại; Online không được giả định path có trong table.

Path resolution phải được cấu hình hoặc định nghĩa riêng.

---

# 10. Error contract

## Core failures

Phải fail query hoặc trả lỗi rõ:

- Milvus visual unavailable cho visual retrieval.
- Vector dimension mismatch.
- SQLite metadata unavailable.
- Invalid `frame_id`.
- Không thể map candidate bắt buộc.
- DANTE không có metadata thời gian.

## Optional branch failures

Có thể degrade:

- OCR branch rỗng.
- ASR branch rỗng.
- Summary branch rỗng.
- Stable Diffusion lỗi.
- QUEST lỗi.
- Object constraints không được chọn.

Mọi degradation phải được ghi vào diagnostics.

---

# 11. Contract validation checklist

Trước khi code orchestration:

- [ ] Milvus reachable.
- [ ] ES reachable.
- [ ] SQLite readable.
- [ ] `visual_features` exists.
- [ ] `ocr_features` exists.
- [ ] `asr_features` exists.
- [ ] `summary_features` exists.
- [ ] `ocr_texts` exists.
- [ ] `asr_transcripts` exists.
- [ ] `video_summaries` exists.
- [ ] `metadata` exists.
- [ ] `objects` exists.
- [ ] Visual dimension read successfully.
- [ ] OCR text dimension read successfully.
- [ ] ASR text dimension read successfully.
- [ ] Summary dimension read successfully.
- [ ] Sample `frame_id` joins across DB.
- [ ] Sample ASR interval joins across DB.
- [ ] Sample summary joins across DB.
- [ ] Stored vector norm approximately 1.
- [ ] Query encoder smoke test produces correct dimension and norm.

---

# 12. Contract violations

- Missing canonical ID.
- `frame_id` differs across DB.
- Wrong vector dimension.
- Non-normalized query vector.
- ASR interval treated as frame.
- Summary treated as frame.
- Summary used to remove videos before frame retrieval.
- Milvus `pk` exposed as domain ID.
- Hardcoded local absolute paths.
- Online writes to Offline SQLite without approval.
- Collection/index/table renamed without migration.
