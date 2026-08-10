# 01 — SYSTEM OVERVIEW

## 1. Mục tiêu

Dự án xây dựng hệ thống truy xuất video đa phương thức cho HCM AI Challenge.

Hệ thống phải hỗ trợ bốn dạng bài toán chính:

1. **Textual KIS**: tìm đúng khoảnh khắc từ mô tả bằng văn bản.
2. **Video KIS (`v-KIS`)**: BTC trình chiếu một clip trên màn hình chung; thí sinh
   xem clip, tự viết mô tả bằng text, sau đó dùng cùng pipeline
   text-to-keyframe như Textual KIS.
3. **TRAKE**: tìm chuỗi keyframe tương ứng với nhiều sự kiện có thứ tự.
4. **VQA**: tìm bằng chứng trong video rồi trả lời bằng text.

Kiến trúc được chia thành hai phase:

```text
PHASE OFFLINE
Video thô
→ preprocessing
→ feature extraction
→ indexing

PHASE ONLINE
Query
→ query construction
→ retrieval
→ hydration
→ normalization
→ fusion/rerank
→ mode-specific output
```

---

## 2. Trách nhiệm của Phase Offline

Phase Offline chạy trước thời điểm truy vấn và không phụ thuộc query của người dùng.

Phase Offline chịu trách nhiệm:

- Phát hiện shot.
- Trích keyframe.
- Gán `video_id`, `shot_id`, `frame_id`.
- Tính visual embedding.
- Chuyển lời nói thành transcript.
- Làm sạch transcript.
- Sinh video summary.
- Trích OCR text.
- Phát hiện object và bounding box.
- Tính semantic embedding cho ASR, OCR và summary.
- Index dữ liệu vào Milvus, Elasticsearch và SQLite.
- Đảm bảo khóa JOIN nhất quán.

---

## 3. Trách nhiệm của Phase Online

Phase Online chạy khi người dùng nhập query.

Phase Online chịu trách nhiệm:

- Nhận mode và query.
- Sinh query bundle.
- Encode text query; image encoding chỉ phục vụ các nhánh ảnh optional đã được
  bật rõ ràng.
- Chạy các retrieval branches.
- Chuyển kết quả về các candidate type thống nhất.
- Ánh xạ ASR interval về keyframe.
- Hydrate metadata từ SQLite.
- Áp object constraints từ UI.
- Normalize score theo branch.
- Late fusion và rerank.
- Dedup/group near frames.
- Chạy DANTE cho TRAKE.
- Thu thập evidence và gọi VLM cho VQA.
- Đóng gói output theo yêu cầu cuộc thi.

Phase Online không được tính lại embedding cho toàn bộ dataset.

---

## 4. Kiến trúc dữ liệu tổng quát

```text
┌──────────────────────────────────────────────────────────────┐
│                       PHASE OFFLINE                          │
│                                                              │
│ Video ──► Shot Detection ──► Keyframes ──► Metadata          │
│                        │            │                         │
│                        │            ├──► Visual Embedding     │
│                        │            ├──► OCR                  │
│                        │            └──► Object Detection     │
│                        │                                      │
│                        └──► Audio ──► ASR ──► Summary         │
│                                             │                │
│ OCR / ASR / Summary ──► Vietnamese Text Embedding            │
│                                                              │
│ Artifacts ──► Module 7 Indexing                              │
└───────────────────────┬──────────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ Milvus                                                      │
│ - visual_features                                           │
│ - ocr_features                                              │
│ - asr_features                                              │
│ - summary_features                                          │
├──────────────────────────────────────────────────────────────┤
│ Elasticsearch                                               │
│ - ocr_texts                                                 │
│ - asr_transcripts                                           │
│ - video_summaries                                           │
├──────────────────────────────────────────────────────────────┤
│ SQLite                                                      │
│ - metadata                                                  │
│ - objects                                                   │
└───────────────────────┬──────────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────┐
│                        PHASE ONLINE                          │
│ Query Builder                                               │
│ ├── Visual semantic                                         │
│ ├── OCR lexical / semantic                                  │
│ ├── ASR lexical / semantic                                  │
│ ├── Summary lexical / semantic                              │
│ ├── Stable Diffusion image [optional]                        │
│ └── QUEST rewrite/image [optional]                           │
│                                                              │
│ Candidate conversion → Hydration → Object filter             │
│ → Normalize → Fusion → Dedup → Mode output                  │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. Ba cấp retrieval result

### 5.1 Frame-level

Có trực tiếp `frame_id`.

Nguồn:

- `visual_features`
- `ocr_features`
- `ocr_texts`
- Video KIS visual-semantic text search
- Stable Diffusion/QUEST image search

### 5.2 ASR interval-level

Có:

```text
video_id
interval_id
start_time_sec
end_time_sec
```

Nguồn:

- `asr_features`
- `asr_transcripts`

Phải ánh xạ sang keyframe qua `metadata.timestamp`.

### 5.3 Video-level

Có `video_id`, không có `frame_id`.

Nguồn:

- `summary_features`
- `video_summaries`

Video-level score chỉ là tín hiệu bổ trợ cho frame thuộc video đó, không được tự dùng để loại video.

---

## 6. Canonical identifiers

### Video

```text
video_id: string
```

### Shot

```text
video_id + shot_id
```

### Keyframe

```text
frame_id: string
```

Canonical target format:

```text
{video_id}_{shot_id:05d}_{position_3_digits}
```

Ví dụ:

```text
V001_00000_015
```

### ASR interval

```text
video_id + interval_id
```

Milvus internal `pk` không phải khóa JOIN.

---

## 7. Tổng quan từng mode

### Textual KIS

```text
Original query
→ 2 LLM paraphrases
→ parallel retrieval branches
→ interval/video evidence propagation
→ object filter
→ normalize + fusion
→ dedup/group
→ ranked keyframes
```

### Video KIS (`v-KIS`)

```text
BTC trình chiếu clip trên màn hình
→ thí sinh xem clip
→ thí sinh tự viết mô tả text
→ dùng chung text-to-keyframe retrieval với Textual KIS
→ ranked keyframes
```

Baseline không nhận file video, frame hoặc query image từ BTC. Khác biệt giữa
Textual KIS và Video KIS chỉ là nguồn của câu query:

- Textual KIS dùng mô tả text do đề cung cấp.
- Video KIS dùng mô tả text do thí sinh tự viết sau khi xem clip.

### TRAKE

```text
E1...EN
→ PE-Core text encoder
→ event-keyframe similarity
→ DANTE per video
→ backtracking
→ ordered sequence
```

### VQA

```text
Question
→ retrieval rewrite
→ multimodal retrieval
→ evidence collection
→ VLM
→ text answer
```

---

## 8. Hiện trạng project

Các module Offline và Module 7 đã có code để Codex phân tích.

Các thư mục `query_understanding`, `retrieval_api` và `ui` phải được xem xét bằng source code hiện tại để xác định phần nào đã implement, phần nào mới scaffold.

Không được suy luận trạng thái chỉ từ README cấp root.

---

## 9. Nguyên tắc kiến trúc

- Dùng đúng database cho đúng workload.
- Tích hợp giữa database bằng khóa logic và late fusion.
- Không dùng summary để prefilter cứng.
- Không dùng LLM tự trích object constraint trong baseline.
- Query vector phải tương thích với embedding đã lưu.
- Không cộng raw BM25 và raw vector score trực tiếp.
- DANTE không được nhảy giữa video.
- Optional branch thất bại không nhất thiết làm hỏng toàn query.
- Core branch thất bại phải được báo rõ.
