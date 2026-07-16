# 02 — OFFLINE PIPELINE ACTUAL

## 1. Mục đích

File này mô tả luồng Offline hiện hành mà Phase Online phụ thuộc.

Codex phải dùng file này làm bản đồ, sau đó kiểm chứng từng chi tiết bằng source code và symbol cụ thể.

Các module logic:

1. Shot Detection & Keyframe Extraction.
2. Visual Embedding.
3. ASR, Transcript Cleaning & Video Summary.
4. OCR.
5. Object Detection.
6. Vietnamese Text Embedding.
7. Multi-DB Indexing.

---

# 2. Module 1 — Shot Detection & Keyframe Extraction

## 2.1 Input

```text
data/raw_videos/<video_id>.<video_extension>
```

Dữ liệu nguồn là video thô.

## 2.2 Xử lý logic

```text
Video
→ đọc metadata video
→ shot detection
→ keyframe selection trong từng shot
→ lưu ảnh WebP
→ lưu metadata JSON
```

Thiết kế hiện hành sử dụng shot-level structure để hỗ trợ:

- `shot_id`
- temporal ordering
- group near frames
- DANTE per video

Codex phải kiểm chứng từ code:

- Model shot detection thực tế.
- Công thức chọn keyframe.
- Số keyframe/shot.
- Quy tắc đặt tên file.
- Resume/force behavior.

## 2.3 Output ảnh

```text
data/processed/keyframes/<video_id>/shot_NNNNN_pos_PPP.webp
```

Ví dụ:

```text
data/processed/keyframes/V001/shot_00000_pos_015.webp
```

## 2.4 Output metadata JSON

```text
data/processed/metadata/<video_id>.json
```

Logical schema:

```json
{
  "video_id": "V001",
  "source_path": "...",
  "fps": 25.0,
  "duration_sec": 120.0,
  "num_shots": 2,
  "shots": [
    {
      "shot_id": 0,
      "start_frame": 0,
      "end_frame": 249,
      "start_time_sec": 0.0,
      "end_time_sec": 9.96,
      "keyframes": [
        {
          "position": 0.15,
          "frame_index": 37,
          "time_sec": 1.48,
          "file_path": "keyframes/V001/shot_00000_pos_015.webp"
        }
      ]
    }
  ]
}
```

## 2.5 Consumer

- Module 2: visual embedding.
- Module 4: OCR.
- Module 5: object detection.
- Module 7: SQLite metadata.
- Phase Online: thông qua SQLite sau khi indexing.

## 2.6 Database destination

Module 7 chuyển keyframe metadata thành SQLite:

```text
metadata(frame_id, video_id, shot_id, timestamp)
```

---

# 3. Module 2 — Visual Embedding

## 3.1 Input

- Keyframe WebP.
- Metadata JSON từ Module 1.

## 3.2 Model

```text
PE-Core-bigG-14-448
```

Được gọi qua `open_clip`.

## 3.3 Xử lý

```text
Image
→ model-specific preprocessing
→ PE-Core image encoder
→ float embedding
→ L2 normalization
```

## 3.4 Output

Một Parquet cho mỗi video:

```text
data/processed/embeddings/visual/<video_id>.parquet
```

Schema:

| Field | Type | Mô tả |
|---|---|---|
| `frame_id` | `str` | Global keyframe ID |
| `video_id` | `str` | Video ID |
| `shot_id` | `int` | Shot index |
| `position` | `float` | Vị trí trong shot |
| `file_path` | `str` | Đường dẫn ảnh |
| `model_name` | `str` | Checkpoint/model |
| `embedding_dim` | `int` | Số chiều |
| `embedding` | `list[float32]` | L2-normalized vector |

Ví dụ `frame_id`:

```text
V001_00000_015
```

## 3.5 Database destination

Milvus:

```text
visual_features
```

## 3.6 Điều kiện contract

- Dimension không được hardcode.
- Query encoder Online phải tạo đúng dimension.
- Query vector Online phải L2-normalize.
- `frame_id` phải JOIN được với SQLite và Elasticsearch.

---

# 4. Module 3 — ASR, Cleaning & Summary

## 4.1 Input

- Video gốc.
- Caption `.srt`/`.vtt` nếu có.
- Metadata liên quan.

## 4.2 Xử lý

```text
Video
→ extract 16 kHz mono WAV
→ ưu tiên caption có sẵn
→ nếu không có: PhoWhisper
→ timestamped raw segments
→ group segments thành interval
→ LLM cleaning
→ video-level summarization
```

Provider cleaning/summarization có thể là:

- Gemini.
- Azure OpenAI.
- Local LLM.

Codex phải đọc config/CLI để xác định default hiện tại.

## 4.3 Output audio

```text
data/processed/audio/<video_id>.wav
```

## 4.4 Output raw transcript

```text
data/processed/transcripts/<video_id>_raw.json
```

## 4.5 Output cleaned transcript

```text
data/processed/transcripts/<video_id>_cleaned.json
```

Logical item:

```json
{
  "interval_id": "interval_0001",
  "start_time": 10.2,
  "end_time": 17.8,
  "cleaned_text": "..."
}
```

## 4.6 Output summary

```text
data/processed/summaries/<video_id>.json
```

Logical schema:

```json
{
  "summary": "..."
}
```

## 4.7 Consumers

- Module 6 tạo ASR embeddings.
- Module 6 tạo summary embeddings.
- Module 7 index transcript vào Elasticsearch.
- Module 7 index summary vào Elasticsearch.

## 4.8 Database destinations

Elasticsearch:

```text
asr_transcripts
video_summaries
```

Milvus thông qua Module 6:

```text
asr_features
summary_features
```

---

# 5. Module 4 — OCR

## 5.1 Input

- Keyframe WebP.
- Metadata JSON.

## 5.2 Pipeline

```text
Keyframe
→ EasyOCR / CRAFT text detection
→ perspective correction
→ VietOCR recognition
→ spatial reading order
→ concatenate text
```

Default recognizer:

```text
VietOCR vgg_transformer
```

## 5.3 Output

```text
data/processed/ocr/<video_id>.json
```

Schema:

```json
{
  "video_id": "V001",
  "frames": [
    {
      "frame_id": "V001_00000_015",
      "shot_id": 0,
      "position": 0.15,
      "ocr_regions": [
        {
          "bbox": [[100, 50], [300, 50], [300, 100], [100, 100]],
          "text": "Detected text",
          "confidence": 0.93
        }
      ],
      "ocr_text_concat": "Detected text"
    }
  ]
}
```

## 5.4 Consumers

- Module 6 tạo OCR semantic embeddings.
- Module 7 index OCR lexical text.
- Online dùng OCR evidence cho VQA/debug.

## 5.5 Database destinations

Elasticsearch:

```text
ocr_texts
```

Milvus thông qua Module 6:

```text
ocr_features
```

Frame không có OCR text có thể bị bỏ khỏi OCR indexes mà không làm mất visual keyframe.

---

# 6. Module 5 — Object Detection

## 6.1 Input

- Keyframe WebP.
- Metadata JSON.

## 6.2 Models

- YOLO-World.
- Co-DETR.

Có thể bật một hoặc cả hai model.

## 6.3 Xử lý

```text
Keyframe
→ detector(s)
→ confidence filtering
→ NMS / box fusion
→ object list
```

Default documented parameters:

```text
confidence threshold = 0.25
NMS IoU threshold = 0.5
```

## 6.4 Output

```text
data/processed/object_detection/<video_id>.json
```

Logical schema:

```json
{
  "video_id": "V001",
  "frames": [
    {
      "frame_id": "V001_00000_015",
      "shot_id": 0,
      "position": 0.15,
      "objects": [
        {
          "label": "person",
          "score": 0.95,
          "box": [100, 50, 300, 450],
          "area": 80000.0
        }
      ]
    }
  ]
}
```

Module 7 chấp nhận alias:

```text
score / confidence
box / bbox
```

## 6.5 Database destination

SQLite:

```text
objects
```

Mỗi bounding box trở thành một row.

## 6.6 Online use

- Object presence.
- Object count.
- Co-occurrence.
- Confidence threshold.
- Position constraint.
- Hard filter hoặc soft boost.

---

# 7. Module 6 — Vietnamese Text Embedding

## 7.1 Input

Ba nguồn text:

1. Cleaned ASR intervals.
2. Video summary.
3. OCR text per keyframe.

## 7.2 Model

Default:

```text
dangvantuan/vietnamese-embedding
```

## 7.3 Xử lý

```text
Text
→ tokenizer/model
→ pooling
→ embedding
→ L2 normalization
```

Summary dài dùng:

```text
chunking
→ per-chunk embedding
→ mean-pooling
→ L2 normalization
```

## 7.4 ASR embedding output

```text
data/processed/embeddings/text_asr/<video_id>.parquet
```

Fields:

```text
video_id
interval_id
start_time_sec
end_time_sec
text
embedding
```

## 7.5 OCR embedding output

```text
data/processed/embeddings/text_ocr/<video_id>.parquet
```

Fields:

```text
video_id
frame_id
shot_id
text
embedding
```

## 7.6 Summary embedding output

```text
data/processed/embeddings/text_summary/<video_id>.parquet
```

Fields:

```text
video_id
text
embedding
```

## 7.7 Database destinations

Milvus:

```text
asr_features
ocr_features
summary_features
```

Dimension được Module 7 phát hiện từ artifact thực tế; không được giả định chỉ từ README.

---

# 8. Module 7 — Multi-DB Indexing

## 8.1 Input root

```text
data/processed/
```

Module 7 đọc output của Module 1–6.

## 8.2 Nhiệm vụ

- Discover `video_id`.
- Load JSON/Parquet.
- Detect embedding dimensions.
- Chuẩn hóa `frame_id`.
- Tạo schema nếu chưa có.
- Delete-then-insert theo video.
- Batch insertion.
- Rollback khi DB downstream lỗi.
- Graceful degradation khi một optional artifact rỗng.
- Reset toàn bộ chỉ khi có cờ explicit.

## 8.3 Milvus outputs

```text
visual_features
ocr_features
asr_features
summary_features
```

## 8.4 Elasticsearch outputs

```text
ocr_texts
asr_transcripts
video_summaries
```

## 8.5 SQLite outputs

```text
metadata
objects
```

## 8.6 Canonical frame ID normalization

Target:

```text
{video_id}_{shot_id}_{position}
```

Ví dụ:

```text
shot_00000_pos_015
→ V001_00000_015
```

Module 7 hiện có `normalize_frame_id()` cho nhiều loaders.

Codex phải audit riêng `load_visual_embeddings()` để xác nhận visual Parquet luôn mang global ID hoặc cũng được normalize.

## 8.7 OCR semantic fix

Module 7 hiện phải có đầy đủ:

```text
load_text_ocr_embeddings()
→ ocr_features schema
→ collection creation
→ insertion
→ delete/rollback/reset
→ tests
```

Nếu bất kỳ khâu nào thiếu, đánh dấu `CONTRACT_MISMATCH`.

---

# 9. Offline output cuối cùng

Sau khi indexing thành công, Online chỉ cần đọc:

```text
Milvus:
- visual_features
- ocr_features
- asr_features
- summary_features

Elasticsearch:
- ocr_texts
- asr_transcripts
- video_summaries

SQLite:
- metadata
- objects
```

Online không được phụ thuộc trực tiếp vào JSON/Parquet trung gian, trừ tool kiểm thử hoặc migration được duyệt.
