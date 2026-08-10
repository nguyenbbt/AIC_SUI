# 05 — ONLINE PIPELINE TARGET

> **Migration notice (2026-08-05):** Read this flow together with
> `docs/22-OFFLINE-TO-ONLINE-DATA-CONTRACT-SELF-INDEXED-V2.md`. Document 22 wins
> for identifiers, fields, model identity, image paths and submission identity.

## 1. Mục tiêu

Phase Online nhận query và trả kết quả với độ trễ thấp mà không chạy lại Offline preprocessing.

Modes:

```text
KIS_TEXT
KIS_VISUAL
TRAKE
VQA
```

`KIS_VISUAL` là working/legacy enum cho Video KIS (`v-KIS`) cho đến khi OQ-002
chốt public API schema. Tên enum này không có nghĩa baseline nhận image query.

Retrieval core vẫn phải nhận explicit mode.

Một Agent router có thể được bổ sung sau, nhưng không được làm retrieval core phụ thuộc hoàn toàn vào suy luận mode của LLM.

---

# 2. Luồng chung

```text
Input validation
→ Query Bundle
→ Parallel Retrieval
→ Candidate Conversion
→ ASR Mapping / Summary Propagation
→ Metadata Hydration
→ Object Constraints
→ Branch Normalization
→ Late Fusion
→ Dedup / Near-frame Grouping
→ Mode-specific Output
```

---

# 3. Domain candidate types

## 3.1 FrameCandidate

```text
frame_id
video_id
shot_id
timestamp_sec
branch
raw_score
normalized_score
provenance
metadata
```

## 3.2 ASRIntervalCandidate

```text
video_id
interval_id
start_time_sec
end_time_sec
branch
raw_score
normalized_score
text optional
```

## 3.3 VideoCandidate

```text
video_id
branch
raw_score
normalized_score
summary optional
```

## 3.4 FusedFrameCandidate

```text
frame_id
video_id
shot_id
timestamp_sec
final_score
branch_scores
near_frames
objects optional
diagnostics
```

---

# 4. Textual KIS

## 4.1 Input

```text
Original Vietnamese textual description
Optional structured object constraints
Options for optional branches
```

## 4.2 Query Builder

Baseline:

```text
q0 = original query
q1 = LLM paraphrase 1
q2 = LLM paraphrase 2
```

Output logical bundle:

```json
{
  "mode": "KIS_TEXT",
  "original_query": "...",
  "text_queries": ["q0", "q1", "q2"],
  "image_queries": [],
  "object_constraints": [],
  "options": {}
}
```

API field names chỉ trở thành canonical sau khi được chốt trong Design Decisions.

---

## 4.3 Branch A — Visual semantic

Mỗi text query chạy riêng:

```text
q0/q1/q2
→ PE-Core text encoder
→ L2 normalization
→ Milvus visual_features
→ FrameCandidate list
```

Không average embeddings trước retrieval trong baseline.

Có thể aggregate nhiều query variants sau retrieval.

---

## 4.4 Branch B — OCR lexical

```text
q0
→ Elasticsearch ocr_texts
→ FrameCandidate list
```

Search field:

```text
ocr_text_concat
```

Fuzzy matching là configurable.

---

## 4.5 Branch C — OCR semantic

```text
q0 và/hoặc paraphrases
→ Vietnamese text embedding encoder
→ L2 normalization
→ Milvus ocr_features
→ FrameCandidate list
```

OCR semantic là branch hiện hành vì Offline đã index `ocr_features`.

---

## 4.6 Branch D — ASR lexical

```text
q0
→ Elasticsearch asr_transcripts
→ ASRIntervalCandidate list
```

---

## 4.7 Branch E — ASR semantic

```text
q0 và/hoặc paraphrases
→ Vietnamese text embedding encoder
→ L2 normalization
→ Milvus asr_features
→ ASRIntervalCandidate list
```

---

## 4.8 Branch F — Summary lexical

```text
q0
→ Elasticsearch video_summaries
→ VideoCandidate list
```

---

## 4.9 Branch G — Summary semantic

```text
q0 và/hoặc paraphrases
→ Vietnamese text embedding encoder
→ L2 normalization
→ Milvus summary_features
→ VideoCandidate list
```

---

## 4.10 Branch H — Stable Diffusion [OPTIONAL]

```text
q0
→ Stable Diffusion
→ generated image
→ PE-Core image encoder
→ Milvus visual_features
→ FrameCandidate list
```

Dùng để trực quan hóa mô tả cảnh phổ biến.

Không bắt buộc cho baseline đầu tiên.

---

## 4.11 Branch I — QUEST [OPTIONAL]

### Rewrite branch

```text
query
→ LLM visual grounding rewrite
→ PE-Core text encoder
→ visual search
```

### External image branch

```text
query/OOK entity
→ external exemplar image
→ PE-Core image encoder
→ visual search
```

Không bắt buộc cho baseline đầu tiên.

---

# 5. ASR mapping

ASR branches trả interval candidates.

Pipeline:

```text
ASR interval
→ query SQLite metadata by video_id
→ select in-interval or nearest keyframes
→ copy ASR provenance and score
→ FrameCandidate(s)
```

Exact policy phải chốt trước milestone ASR mapper.

---

# 6. Summary propagation

Summary branches trả video candidates.

Bắt buộc:

```text
Không prefilter.
Không loại video.
Không thay frame-level evidence.
```

Sau normalization:

```text
video summary score
→ low/controlled weighted boost
→ frame candidates thuộc video đó
```

Nếu frame không xuất hiện ở bất kỳ frame/interval branch nào, summary không tự tạo frame candidate.

---

# 7. Object constraints từ UI

## 7.1 Source

Người dùng chọn trực tiếp trong UI.

Baseline không dùng LLM tự động suy luận object constraints.

## 7.2 Logical schema

```json
{
  "label": "person",
  "count_operator": "eq",
  "count": 3,
  "min_confidence": 0.4,
  "position": {
    "type": "region",
    "x_min": 0.0,
    "y_min": 0.0,
    "x_max": 0.5,
    "y_max": 1.0
  },
  "filter_mode": "hard"
}
```

## 7.3 Count operators

```text
eq
gte
lte
```

## 7.4 Position

UI nên dùng normalized coordinates `[0,1]`.

SQLite hiện lưu pixel coordinates; backend cần width/height hoặc một quy tắc resolution.

Điểm này đang là open question.

## 7.5 Hard filter

Candidate không thỏa bị loại.

## 7.6 Soft boost

Candidate thỏa được cộng điểm; candidate không thỏa vẫn giữ.

---

# 8. Metadata hydration

Mọi frame candidate phải được hydrate từ SQLite:

```text
frame_id
→ metadata
→ video_id, shot_id, timestamp
```

Candidate không có metadata được xem là contract error hoặc bị loại với diagnostics, tùy policy được chốt.

---

# 9. Branch normalization

Mỗi branch normalize độc lập.

Không cộng trực tiếp:

```text
Milvus IP
ES BM25
object count
summary score
```

Normalization interface phải hỗ trợ ít nhất:

- Min-max.
- Rank-based.
- Reciprocal-rank conversion.

Baseline method nằm trong `08-OPEN-QUESTIONS.md`.

---

# 10. Fusion

Frame-level fusion key:

```text
frame_id
```

Logical weighted fusion:

```text
final_score(frame)
=
Σ branch_weight × normalized_branch_score
+ summary_video_boost
+ optional_object_boost
```

Hoặc RRF nếu được chốt.

Mỗi final candidate phải giữ `branch_scores` để debug.

---

# 11. Dedup và near-frame grouping

Baseline:

1. Group theo `video_id + shot_id`.
2. Giữ frame có `final_score` lớn nhất.
3. Các frame còn lại trở thành `near_frames`.
4. Nếu `shot_id` thiếu, dùng temporal suppression window.

Output:

```json
{
  "frame_id": "V001_00012_050",
  "video_id": "V001",
  "shot_id": 12,
  "timestamp_sec": 44.2,
  "final_score": 0.91,
  "branch_scores": {},
  "near_frames": []
}
```

---

# 12. Video KIS (`v-KIS`)

## Input

```text
Textual description manually authored by the contestant
after watching the clip displayed by the organizer
Optional object constraints
```

BTC không cung cấp file video, frame hoặc query image cho hệ thống ở baseline.
Thí sinh quan sát clip trực tiếp và chuyển thông tin nhìn thấy thành câu
query text.

## Pipeline

```text
Organizer-displayed clip
→ contestant observation
→ manually authored textual description
→ reuse the Textual KIS text-to-keyframe pipeline
→ ranked keyframes
```

Textual KIS và Video KIS dùng chung retrieval mechanism. Khác biệt duy nhất
ở baseline là query text đến từ đề bài hay do thí sinh tự viết. Nếu cần
giữ mode riêng cho routing, UI hoặc diagnostics thì mode đó không được làm
thay đổi retrieval semantics này.

Baseline tái sử dụng các retrieval branches và quy tắc fusion của Textual KIS;
mode riêng chỉ ghi nhận nguồn query và phục vụ routing/UI/diagnostics.

---

# 13. TRAKE

## 13.1 Input

```text
E1, E2, ..., EN
```

Các event đã có narrative order.

## 13.2 Baseline decision

TRAKE baseline chỉ dùng visual-semantic DANTE.

Không fuse OCR, ASR, summary, SD hoặc QUEST vào DANTE matrix ở baseline.

## 13.3 Event encoding

```text
event text
→ PE-Core text encoder
→ L2-normalized vector
```

## 13.4 Similarity matrix

```text
S[i,t] = IP(event_i, visual_keyframe_t)
```

## 13.5 DANTE

Chạy riêng từng video.

Keyframes phải sort theo `metadata.timestamp`.

DP conceptual form:

```text
DP[i,t]
=
S[i,t]
+
max over τ<t:
  DP[i-1,τ] - λ × temporal_distance(τ,t)
```

Phải lưu backpointer.

## 13.6 Output

```json
{
  "video_id": "V001",
  "sequence_score": 3.42,
  "sequence": [
    {
      "event_id": "E1",
      "frame_id": "V001_00010_050",
      "timestamp_sec": 31.2,
      "event_score": 0.88
    }
  ]
}
```

---

# 14. VQA

## 14.1 Input

```text
question
answer_type
optional object constraints
```

## 14.2 Query Builder

Sinh:

```text
vqa_retrieval_rewrite
paraphrase_1
paraphrase_2
```

Rewrite mô tả bằng chứng thị giác cần tìm, không cố trả lời ngay.

## 14.3 Retrieval

Có thể dùng các branch của Textual KIS:

- Visual semantic.
- OCR lexical/semantic.
- ASR lexical/semantic.
- Summary lexical/semantic.
- Optional SD/QUEST.
- Object constraints từ UI.

## 14.4 Evidence

Thu thập:

```text
top keyframe images
near frames
OCR text
ASR transcript
video summary
timestamps
original question
```

## 14.5 VLM answer

```text
question + evidence
→ VLM
→ short answer
```

VLM phải được yêu cầu chỉ dựa trên evidence đã truy xuất.

## 14.6 Output

```json
{
  "answer": "7",
  "answer_type": "number",
  "evidence": [
    {
      "video_id": "V001",
      "frame_id": "V001_00012_050",
      "timestamp_sec": 44.2
    }
  ]
}
```

Competition adapter có thể chỉ gửi answer text.

---

# 15. Diagnostics

Mỗi query nên giữ:

- Branch enabled/disabled.
- Branch latency.
- Raw result count.
- Mapping losses.
- Missing metadata.
- Normalization config.
- Fusion weights.
- Object filter removals.
- Dedup removals.
- Optional branch failures.
