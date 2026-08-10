# 21 — PHÂN CÔNG CODE LẠI ONLINE CHO NGƯỜI A VÀ C

## 1. Mục tiêu

Code lại toàn bộ Phase Online để tương thích contract dữ liệu BTC AIC 2026 trong:

- `docs/19-AIC2026-ORGANIZER-DATA-CONTRACT.md`.
- `docs/20-AIC2026-OFFLINE-ONLINE-MIGRATION-PLAN.md`.

Nhân sự:

- **Người A**: Data contract, infrastructure, encoders và retrieval.
- **Người C**: Ranking, mode orchestration, TRAKE, VQA và API/output.
- **Người B**: Chỉ review contract, review code, kiểm tra integration và quyết
  định merge; không nhận coding milestone.

Có thể bắt đầu ngay dù Offline chưa hoàn thành. Hai người dùng synthetic
`organizer-v1` fixtures để code và test. Các test với database/dữ liệu thật được
đánh dấu `NEED_RUNTIME_VERIFICATION` cho đến khi Offline bàn giao.

---

## 2. Ranh giới hệ thống

```text
                  NGƯỜI A

organizer-v1 contract
→ domain models/ports
→ SQLite/Milvus/Elasticsearch adapters
→ OpenCLIP + Vietnamese encoders
→ query building/retrieval branches
→ BranchResult

                         │ frozen handoff
                         ▼

                  NGƯỜI C

BranchResult
→ ASR mapping
→ normalization/aggregation/fusion
→ summary/object processing
→ dedup
→ KIS/TRAKE/VQA orchestration
→ API/competition output
```

Người C không truy vấn database SDK trực tiếp. Người A không đặt ranking policy
trong adapter/retrieval.

---

## 3. Contract chung đã chốt

### 3.1 Frame identity

```text
frame_id = f"{video_id}_{keyframe_no:03d}"
```

Ví dụ:

```text
video_id          = L21_V001
frame_id          = L21_V001_001
keyframe_no       = 1
local_index       = 0
timestamp_sec     = CSV.pts_time
fps               = CSV.fps
source_frame_idx  = CSV.frame_idx
image_rel_path    = keyframes/L21_V001/001.jpg
```

### 3.2 Quy tắc bắt buộc

- Không còn `shot_id` trong organizer production contract.
- `source_frame_idx` được phép trùng.
- TRAKE order bằng `local_index`, không bằng `source_frame_idx`.
- Online không tính lại frame index từ timestamp/FPS.
- Visual model là `ViT-B-32::openai`, dimension 512.
- Vietnamese text model vẫn là `dangvantuan/vietnamese-embedding`.
- KIS và v-KIS dùng chung text-to-keyframe pipeline.
- Internal `frame_id` không phải competition frame number.

### 3.3 Shared handoff models

Người A là primary owner và phải freeze các model sau trước khi hai bên làm sâu:

```text
FrameSearchHit
FrameMetadata
FrameCandidate
ASRIntervalCandidate
VideoCandidate
BranchResult
FusedFrameCandidate
OrderedVisualFrame
ImageEvidence
TRAKEFrameMatch
```

Người C không tự sửa các model này trên branch của mình. Nếu thiếu field, gửi
change request cho A và B review.

---

## 4. Quyền sở hữu file

## 4.1 Người A sở hữu

```text
online/domain/
online/ports/
online/adapters/
online/retrieval/
online/config.py
online/lifecycle.py
online/validate_contract.py
online/testing/
online/requirements*.txt
query_understanding/parser.py
query_understanding/trake_parser.py
tests/online/contract/
tests/online/adapters/
tests/online/retrieval/
```

Người A cũng là primary owner của:

```text
.env.example
```

## 4.2 Người C sở hữu

```text
online/ranking/
online/modes/
online/trake/
online/vqa/
retrieval_api/
query_understanding/rewrite.py
tests/online/ranking/
tests/online/modes/
tests/online/trake/
tests/online/vqa/
tests/online/api/
tests/online/integration/
```

Provider code mới do C sở hữu:

```text
query_understanding/providers/
online/vqa/providers/
```

## 4.3 File không được hai người cùng sửa

- A không sửa `online/ranking/`, `online/modes/`, `online/trake/`,
  `online/vqa/`, `retrieval_api/`.
- C không sửa `online/domain/`, `online/ports/`, `online/adapters/`,
  `online/retrieval/`, `online/config.py`, `online/testing/`.
- C cần thay đổi shared contract phải gửi yêu cầu cho A.
- A cần mode/API behavior phải gửi yêu cầu cho C.

## 4.4 Fixtures tránh conflict

- A tạo canonical base fixtures trong `online/testing/`.
- C chỉ consume base fixtures.
- Nếu C cần mode-specific builder, đặt trong:

  ```text
  tests/online/fixtures/mode_fixtures.py
  ```

- C không copy lại một phiên bản khác của `FrameMetadata` hoặc
  `FrameCandidate`.

---

# PHẦN I — NGƯỜI A

## A0 — Freeze organizer-v1 domain contract

### Mục tiêu

Thay toàn bộ shared Online contract từ shot-based sang organizer keyframe-based.

### File chính

```text
online/domain/identifiers.py
online/domain/candidates.py
online/domain/trake.py
online/domain/vqa.py
online/ports/records.py
online/ports/metadata.py
online/ports/visual_corpus.py
online/ports/images.py
online/domain/__init__.py
online/ports/__init__.py
```

### Công việc

1. Đổi canonical frame parser sang `{video_id}_{keyframe_no:03d}`.
2. `video_id` có thể chứa underscore; parse suffix từ bên phải.
3. Semantic-check parsed video ID với field `video_id` backend trả về.
4. Reject:
   - whitespace;
   - keyframe number `000`;
   - ID shot cũ;
   - video mismatch;
   - malformed suffix.
5. Xóa `shot_id` khỏi organizer models.
6. `FrameMetadata` gồm:

   ```text
   frame_id
   video_id
   keyframe_no
   local_index
   timestamp_sec
   fps
   source_frame_idx
   image_rel_path
   ```

7. `FrameCandidate` gồm:

   ```text
   frame_id
   video_id
   keyframe_no
   local_index
   timestamp_sec
   source_frame_idx
   rank/raw_score/normalized_score/provenance
   ```

8. `FusedFrameCandidate` giữ cùng identity/mapping fields.
9. `OrderedVisualFrame` gồm vector và organizer ordering metadata.
10. `TRAKEFrameMatch` thêm `source_frame_idx`, bỏ `shot_id`.
11. `ImageEvidence` thêm `keyframe_no`, `source_frame_idx`, bỏ `shot_id`.
12. Giữ public evidence không chứa absolute local path.

### Canonical fixture

Tạo fixture tối thiểu:

```text
L21_V001_001
L21_V001_002
L21_V001_003
L21_V002_001
```

Trong đó hai keyframe của `L21_V001` có thể cùng `source_frame_idx` để test
duplicate competition frame.

### Test owner

```text
tests/online/contract/test_models.py
tests/online/contract/test_advanced_models.py
tests/online/contract/test_advanced_ports.py
tests/online/contract/test_package_imports.py
```

### Definition of Done A0

- Shared models không còn `shot_id`.
- Organizer ID validation pass.
- Fixtures serialize/deserialize đúng.
- C có thể code hoàn toàn bằng shared types mà không cần định nghĩa type riêng.
- B review và xác nhận contract trước khi A/C tiếp tục merge các milestone sâu.

---

## A1 — SQLite organizer adapter

### File

```text
online/adapters/sqlite.py
online/config.py
online/testing/sqlite_fixture.py
tests/online/adapters/test_sqlite_adapter.py
```

### Công việc

1. Đổi metadata SELECT sang:

   ```text
   frame_id
   video_id
   keyframe_no
   local_index
   pts_time_sec
   fps
   source_frame_idx
   image_rel_path
   ```

2. Map `pts_time_sec` sang Online domain `timestamp_sec`.
3. `get_ordered_frames_by_video()` order bằng `local_index ASC`.
4. Mở rộng `MetadataReaderPort`:

   ```python
   list_video_ids()
   get_frames_by_ids(...)
   get_ordered_frames_by_video(...)
   ```

5. Object query đọc:

   ```text
   label_display
   label_normalized
   class_mid
   class_label_id
   confidence
   x_min/y_min/x_max/y_max
   model_source
   ```

6. Cho phép filter bằng normalized label/MID theo port contract được chốt.
7. Giữ SQLite tuyệt đối read-only:

   ```text
   URI mode=ro
   PRAGMA query_only=ON
   ```

8. Không resolve `image_rel_path` trong SQLite adapter; việc này thuộc image
   resolver.

### Test

- Batch hydration.
- Missing metadata.
- Duplicate source frame.
- Ordered local indices.
- Decimal FPS.
- Unicode relative path.
- Malformed row.
- Read-only enforcement.
- Object empty-success.

### Definition of Done A1

- SQLite adapter conform `MetadataReaderPort` và `ObjectReaderPort` mới.
- Không query column cũ `shot_id`/`timestamp`.
- Không dùng source frame để order.

---

## A2 — Milvus organizer adapter

### File

```text
online/adapters/milvus.py
online/ports/search.py
tests/online/adapters/test_milvus_adapter.py
```

### Công việc KIS

1. Visual hit chỉ require:

   ```text
   frame_id
   video_id
   raw_score
   ```

2. Bỏ `shot_id` khỏi output fields và `FrameSearchHit`.
3. Giữ dimension discovery/runtime validation.
4. Giữ query-vector finite/norm validation.
5. Giữ HNSW/IP search parameters configurable.

### Công việc text collections

- OCR dense: frame-level.
- ASR dense: interval-level với start/end.
- Summary dense: video-level.
- Không thay đổi candidate level.

### Test

- Visual output schema mới.
- Dimension 512 pass, dimension mismatch fail.
- Same dimension/wrong manifest model sẽ được validator reject.
- Backend timeout/unavailable/invalid hit.
- OCR/ASR/summary records vẫn đúng level.

### Definition of Done A2

- Milvus KIS adapter không còn biết `shot_id`.
- Hit được hydrate ở retrieval layer, không chứa guessed metadata.

---

## A3 — Manifest adapter và contract validator

### File

```text
online/adapters/contract_validator.py
online/validate_contract.py
online/config.py
online/adapters/manifest.py        # file mới đề xuất
online/ports/manifest.py           # file mới đề xuất
tests/online/adapters/test_contract_validator.py
```

### Manifest fields bắt buộc

```text
contract_version = organizer-v1
visual_model_id = ViT-B-32::openai
visual_dimension = 512
visual_normalized = true
frame_id_contract_version
object_threshold
object_nms_iou
record counts
dataset fingerprint
```

### Công việc

1. Đọc manifest read-only từ SQLite table hoặc configured JSON path theo
   contract M0.
2. Audit Milvus visual schema:

   ```text
   frame_id VARCHAR
   video_id VARCHAR
   local_index INT64
   embedding FLOAT_VECTOR 512
   HNSW/IP
   ```

3. Audit SQLite tables `videos`, `metadata`, `objects`.
4. Audit ES mappings hiện hữu.
5. Audit canonical organizer IDs.
6. Audit sample JOIN:
   - visual → metadata;
   - OCR dense/lexical → metadata;
   - objects → metadata;
   - ASR dense ↔ lexical;
   - summary dense ↔ lexical.
7. Audit sampled local indices và source frame field.
8. Reject same dimension nhưng wrong visual model ID.
9. Diagnostics không leak URI/path/credentials.

### Definition of Done A3

- Production không READY nếu manifest thiếu/sai.
- Visual dimension và model identity đều được kiểm tra.
- Optional branch failure được phân loại riêng, không giả core READY.

---

## A4 — OpenCLIP visual text encoder

### File

```text
online/retrieval/encoders.py
online/retrieval/factory.py
retrieval_api/composition.py        # C sở hữu; A chỉ cung cấp requested wiring change
online/requirements-encoders.txt
tests/online/retrieval/test_encoders.py
```

`retrieval_api/composition.py` thuộc C. A không tự sửa file này; A bàn giao cách
khởi tạo encoder để C wire.

### Công việc

1. Thay:

   ```text
   PE_CORE_MODEL_ID
   PECoreTextEncoder
   ```

   bằng:

   ```text
   OPEN_CLIP_MODEL_ID = ViT-B-32::openai
   OpenCLIPTextEncoder
   ```

2. Dùng:

   ```python
   open_clip.create_model_and_transforms(
       "ViT-B-32",
       pretrained="openai",
   )
   ```

3. Tokenize bằng tokenizer của `ViT-B-32`.
4. `encode_text`, cast float32, finite, L2-normalize.
5. Dimension bắt buộc 512 trong organizer production composition.
6. CPU tự dùng fp32; CUDA cho phép fp16/bf16/fp32.
7. Lazy load và thread-safe inference giữ nguyên.
8. Error message không còn nói PE-Core.
9. `VietnameseTextEncoder` giữ nguyên model contract.

### Test

- Backend factory unit tests không cần tải model.
- Exact model/pretrained/tokenizer call.
- Dimension/norm/finite validation.
- Concurrent calls.
- Wrong dimension.
- Load failure.
- Empty/invalid text.

### Definition of Done A4

- Không còn production reference đến PE-Core visual encoder.
- KIS và TRAKE có thể dùng cùng exact visual model identity.

---

## A5 — Query builder và seven-branch retrieval

### File

```text
online/retrieval/query_builder.py
online/retrieval/branches.py
online/retrieval/service.py
online/retrieval/factory.py
query_understanding/parser.py
query_understanding/trake_parser.py
tests/online/retrieval/
```

### Query contract

```text
q0 = original text
q1 = optional paraphrase
q2 = optional paraphrase
```

Mỗi variant retrieval độc lập.

### Seven branches

```text
visual_dense
ocr_dense
ocr_bm25
asr_dense
asr_bm25
summary_dense
summary_bm25
```

### Công việc

1. Visual branch dùng `OpenCLIPTextEncoder`.
2. OCR/ASR/summary semantic branches dùng `VietnameseTextEncoder`.
3. Visual/OCR hits batch-hydrate `FrameMetadata` mới.
4. Tạo `FrameCandidate` có source-frame metadata.
5. ASR branches vẫn trả `ASRIntervalCandidate`.
6. Summary branches vẫn trả `VideoCandidate`.
7. Không normalize/fuse raw scores.
8. Giữ branch/query/backend/resource provenance.
9. Giữ controlled executor cho sync adapters.
10. Giữ deterministic output ordering.
11. Phân biệt empty-success, timeout, unavailable và contract mismatch.
12. t-KIS/v-KIS parity test tiếp tục bắt buộc.

### Test đặc biệt

- Visual hit → correct source frame metadata.
- OCR dense và lexical cùng hydrate một canonical frame.
- ASR không chứa frame ID trước mapper.
- Summary không chứa frame ID.
- Duplicate source frame chưa dedup ở retrieval layer.
- Query variants không average embeddings.
- Timeout không để background result lọt vào response sau deadline.

### Definition of Done A5

Người A bàn giao cho C đúng:

```text
QueryBundle → tuple[BranchResult]
```

Không có ranking hoặc serializer logic trong retrieval.

---

## A6 — Production VisualCorpus adapter cho TRAKE

### File

```text
online/adapters/visual_corpus.py    # mới
online/ports/visual_corpus.py
online/adapters/milvus.py
online/adapters/sqlite.py
tests/online/adapters/test_visual_corpus_adapter.py
```

### Flow

```text
list video IDs từ SQLite
→ query/paginate Milvus visual theo video_id
→ order bằng local_index
→ batch hydrate SQLite metadata
→ OrderedVisualFrame batches
```

### Công việc

1. Không dùng Milvus auto `pk` làm order/domain ID.
2. Không load toàn bộ corpus vào RAM.
3. Batch size configurable.
4. Verify mỗi video:
   - local index từ 0;
   - contiguous;
   - unique frame ID;
   - vector dimension nhất quán;
   - norm đúng;
   - metadata JOIN đầy đủ.
5. Hỗ trợ timeout/cancellation ở runtime wrapper.
6. Port chỉ trả SDK-neutral `OrderedVisualFrame`.

### Test

- Multi-batch video.
- Missing local index.
- Duplicate local index.
- Wrong-video record.
- Metadata missing.
- Dimension change giữa batch.
- Empty corpus.
- Large synthetic corpus không materialize toàn bộ trong adapter.

### Definition of Done A6

C có thể inject adapter vào `TRAKEExecutionService` mà không biết Milvus/SQLite
SDK.

---

## A7 — Production ImageResolver adapter cho VQA

### File

```text
online/adapters/images.py           # mới
online/ports/images.py
online/config.py
tests/online/adapters/test_image_resolver.py
```

### Flow

```text
frame IDs
→ SQLite metadata
→ image_rel_path
→ configured dataset/keyframe root
→ containment validation
→ ImageEvidence
```

### Công việc

1. Không nhận path từ public request.
2. Reject absolute path, `..`, symlink escape và NUL.
3. Allowed extension ít nhất `.jpg`/`.jpeg`; chỉ mở regular files.
4. Có max byte/file count config.
5. Không trả absolute path trong `ImageEvidence`/API.
6. Missing image trả missing evidence map/diagnostics theo port policy.
7. Resolver không gọi VLM.

### Test

- Resolve organizer JPG thành công.
- Unicode root.
- Missing file.
- Traversal/symlink escape.
- Oversized file.
- Duplicate requested IDs.
- Metadata mismatch.

### Definition of Done A7

C nhận được safe `ImageEvidence` từ frame IDs và dùng trực tiếp trong VQA
evidence selection.

---

## A8 — Config, lifecycle, health và infrastructure fakes

### File

```text
online/config.py
online/lifecycle.py
online/testing/
online/validate_contract.py
online/requirements*.txt
.env.example
tests/online/contract/
```

### Config cần thêm

```text
dataset/keyframe root
manifest location
visual model ID/dimension
visual corpus batch size
object default threshold
image max bytes
adapter timeouts
```

### Công việc

1. Tạo protocol-conformant organizer fakes.
2. Fakes phải hỗ trợ duplicate source frame và variable FPS.
3. Lifecycle mở core adapters trước, đóng theo reverse order.
4. Health/readiness phân biệt core và optional.
5. Không đưa mode policy vào lifecycle.
6. Bàn giao C factory inputs rõ ràng:

   ```text
   retrieval service
   metadata reader
   object reader
   visual corpus
   image resolver
   visual encoder/event encoder
   manifest/readiness checks
   ```

### Definition of Done A8

- A-side unit/contract/adapter/retrieval tests xanh.
- Có handoff note liệt kê constructor/signature để C wire.
- Các phần chưa có database thật được ghi `NEED_RUNTIME_VERIFICATION`.

---

# PHẦN II — NGƯỜI C

## C0 — Migration ranking models consumption

### Điều kiện đầu vào

C code theo A0 frozen contract. Không tự tạo compatibility dataclass.

### File

```text
online/ranking/asr_mapper.py
online/ranking/aggregation.py
online/ranking/normalizers.py
online/ranking/fusion.py
online/ranking/summary.py
online/ranking/object_filter.py
online/ranking/dedup.py
online/ranking/policy.py
tests/online/ranking/
```

### Công việc

1. Xóa mọi access đến `candidate.shot_id`.
2. Propagate:

   ```text
   keyframe_no
   local_index
   timestamp_sec
   source_frame_idx
   ```

3. Fusion reject metadata inconsistency của cùng `frame_id`.
4. Tie-breaker:

   ```text
   score desc
   video_id asc
   local_index asc
   frame_id asc
   ```

5. Không đổi normalization/fusion weights nếu migration không yêu cầu.

### Definition of Done C0

Ranking không còn shot semantics và giữ source-frame metadata tới final
candidate.

---

## C1 — ASR interval-to-frame mapping

### File

```text
online/ranking/asr_mapper.py
tests/online/ranking/test_asr_mapper.py
```

### Công việc

1. Lấy ordered frames từ `MetadataReaderPort`.
2. Match interval bằng:

   ```text
   start <= frame.timestamp_sec <= end
   ```

3. Không tính timestamp từ source frame/FPS.
4. Preserve organizer frame fields trong mapped `FrameCandidate`.
5. Giữ bounded frames per interval và diagnostics truncation.
6. Deterministic ordering bằng timestamp/local index/frame ID.

### Test

- Inclusive boundaries.
- Interval không có frame.
- Long interval truncation.
- Variable FPS không ảnh hưởng mapping.
- Duplicate source frame vẫn map thành internal candidates riêng trước dedup.

---

## C2 — Competition-frame dedup và near-frame grouping

### File

```text
online/ranking/dedup.py
tests/online/ranking/test_fusion_dedup_objects.py
```

### Công việc

1. Xóa/đổi tên `ShotDeduplicator`.
2. KIS primary dedup key:

   ```text
   (video_id, source_frame_idx)
   ```

3. Giữ candidate score cao nhất.
4. Tie bằng deterministic order.
5. Candidate bị gộp phải được lưu vào near/evidence diagnostics nếu hợp lệ.
6. Near-frame grouping theo timestamp/local index là bước khác, không gọi shot.
7. Không tái sử dụng KIS deduplicator cho TRAKE event sequence.

### Test

- Hai frame IDs cùng source frame.
- Hai video cùng frame number không bị gộp.
- Same frame repeated across branches đã fuse trước dedup.
- Stable result không phụ thuộc input order.

---

## C3 — Object constraint migration

### File

```text
online/ranking/object_filter.py
online/domain/query.py             # A sở hữu; gửi change request nếu cần field
query_understanding/providers/objects.py
tests/online/ranking/
```

C không tự sửa `online/domain/query.py`; nếu cần schema change, yêu cầu A.

### Công việc

1. Match `label_normalized` hoặc MID, không chỉ raw display label.
2. Tạo versioned synonym normalizer cho Vietnamese/English query labels.
3. Count detections sau Offline threshold/NMS.
4. Confidence query default 0.50.
5. Soft/hard behavior giữ policy rõ ràng.
6. Position constraint dùng normalized XYXY.
7. Bbox center/intersection policy phải có test và version nếu được bật.

### Test

- Vietnamese synonym → canonical English label.
- MID exact match.
- Confidence boundary.
- Count operators.
- Hard vs soft.
- Position region.
- Empty objects.

---

## C4 — KIS mode orchestration

### File

```text
online/modes/kis.py
online/ranking/policy.py
tests/online/modes/test_kis.py
tests/online/integration/test_retrieval_adapter_handoff.py
```

### Pipeline bắt buộc

```text
BranchResult
→ ASR map
→ aggregate query variants per branch
→ normalize each branch
→ frame fusion
→ controlled summary support
→ object processing
→ KIS competition-frame dedup
→ final sort/top-k
```

### Công việc

1. Consume A-side `BranchResult` only.
2. Không gọi Milvus/ES trực tiếp.
3. Preserve full provenance.
4. Optional branch failure xuất hiện trong diagnostics.
5. Visual core failure không bị che bằng empty result.
6. t-KIS/v-KIS vẫn chạy cùng orchestrator.
7. `source_frame_idx` có mặt trong mọi final KIS candidate.

### Definition of Done C4

Synthetic organizer query chạy end-to-end từ fake BranchResults tới ranked
competition frame candidates.

---

## C5 — KIS API và competition serializer

### File

```text
retrieval_api/search_engine.py
retrieval_api/main.py
retrieval_api/composition.py
tests/online/api/
```

### Internal/UI response

Cho phép trả:

```text
frame_id
video_id
keyframe_no
local_index
timestamp_sec
source_frame_idx
score
provenance/diagnostics nếu request
```

### Competition serializer

KIS serializer lấy trực tiếp:

```text
video_id
source_frame_idx
```

Không parse source frame từ `frame_id`, filename hoặc timestamp.

### Công việc

1. Xóa public validation phụ thuộc frame ID shot cũ.
2. Cập nhật safe error detail validation theo organizer ID.
3. Serializer mode-specific, không dùng một generic payload sai cho mọi mode.
4. Golden tests cho field names/order/type.
5. Exact external endpoint format vẫn cấu hình/adapt sau khi BTC chốt.

---

## C6 — TRAKE/DANTE migration và production orchestration

### File

```text
online/trake/similarity.py
online/trake/dante.py
online/trake/service.py
online/trake/config.py
online/modes/trake.py
retrieval_api/advanced_models.py
tests/online/trake/
tests/online/modes/test_trake.py
tests/online/integration/test_trake_*.py
```

### Input từ A

```text
VisualCorpusPort
OpenCLIP event encoder
OrderedVisualFrame organizer records
```

### Công việc

1. Xóa `shot_id` khỏi event/frame match logic.
2. Giữ DANTE transition trong cùng video.
3. Sequence strictly increasing theo `local_index`.
4. Similarity dùng event vectors và visual corpus cùng exact model space.
5. Output match giữ `source_frame_idx`.
6. Stream/batch corpus; service không materialize toàn dataset nếu không cần.
7. Timeout/cancellation và bounded diagnostics.
8. Competition TRAKE serializer trả ordered source frame list.

### Duplicate source frame rule

- Không gọi KIS dedup.
- DANTE phải giữ một match cho mỗi event.
- Nếu hai matches có cùng `source_frame_idx`, ghi diagnostics.
- Nếu external rule cấm duplicate, implement sequence-level alternative/tie-break
  sau khi rule BTC được xác nhận; không xóa event.

### Không thêm vào baseline

- OCR.
- ASR.
- Summary.
- Stable Diffusion.
- QUEST.

### Performance tests

- Multi-video isolation.
- Multi-batch corpus.
- Many events/frames synthetic load.
- Determinism.
- Timeout/cancel.

---

## C7 — VQA evidence selection migration

### File

```text
online/vqa/evidence_selector.py
online/vqa/selection.py
online/vqa/budget.py
online/vqa/orchestrator.py
online/retrieval/vqa.py          # A sở hữu; gửi requirement, không tự sửa
tests/online/vqa/
```

### Input từ A

- Ranked organizer frame candidates.
- `ImageResolverPort` implementation.
- Evidence/search ports.

### Công việc

1. Primary evidence giữ keyframe/source-frame metadata.
2. Resolve image theo frame ID qua A-side resolver.
3. OCR evidence JOIN frame ID.
4. ASR evidence chọn quanh `timestamp_sec`.
5. Summary evidence JOIN video ID.
6. Giữ max video/frame/image/text budgets.
7. Missing evidence có diagnostics, không thay evidence sai.
8. Không gửi toàn dataset vào VLM.

### Test

- Evidence budget.
- Duplicate frame/image elimination.
- Missing image/OCR/ASR/summary.
- Correct source frame grounding.
- Deterministic evidence ordering.

---

## C8 — Production VLM provider

### File mới/hiện tại

```text
online/vqa/providers/__init__.py
online/vqa/providers/<provider>.py
online/vqa/prompts.py
online/vqa/vlm_request.py
online/vqa/orchestrator.py
tests/online/vqa/test_vlm_contract.py
```

### Công việc

1. Chọn provider/model qua config/wiring do C sở hữu ở composition.
2. Structured request chỉ gồm selected evidence.
3. Structured response theo `VLMResponse`.
4. Answer evidence IDs phải là subset request evidence IDs.
5. Timeout, bounded retry, token/image limits.
6. Prompt version/model ID trong safe diagnostics.
7. Không log image bytes, API keys hoặc raw sensitive response.
8. Provider unavailable → explicit mode failure/insufficient evidence theo policy.

### Test

- Success.
- Insufficient evidence.
- Invalid structured output.
- Unknown evidence ID.
- Timeout/retry.
- Budget exceeded.
- Secret-safe diagnostics.

---

## C9 — Production LLM query rewrite provider

### File

```text
query_understanding/rewrite.py
query_understanding/providers/__init__.py
query_understanding/providers/<provider>.py
tests/online/retrieval/test_query_rewrite.py   # A owns folder; coordinate test ownership
```

Để tránh conflict, C viết provider tests trong:

```text
tests/online/integration/test_query_rewrite_provider.py
```

A chỉ giữ core rewrite/retrieval contract tests hiện hữu nếu cần sửa imports.

### Công việc

1. KIS q0 luôn là original query.
2. Provider chỉ đề xuất q1/q2.
3. VQA rewrite tạo evidence-search description, không trả lời.
4. Structured parsing/normalization/dedup.
5. Timeout và fallback q0.
6. Feature flag default off cho đến khi benchmark.
7. Provider/model/prompt version diagnostics.
8. Không để rewrite failure làm KIS visual chết.

### Definition of Done C9

Code provider production-ready nhưng activation mặc định phụ thuộc benchmark và
B review.

---

## C10 — VQA API và competition serializer

### File

```text
retrieval_api/advanced_models.py
retrieval_api/search_engine.py
retrieval_api/composition.py
tests/online/api/test_advanced_routes.py
tests/online/integration/test_vqa_mode_api_e2e.py
```

### Công việc

1. Wire A-side image resolver.
2. Wire C-side VLM provider/orchestrator.
3. Response giữ grounded evidence IDs.
4. Competition output có:

   ```text
   video_id
   source_frame_idx
   answer
   ```

5. Exact payload wrapper theo rule BTC cuối cùng.
6. VQA disabled/unconfigured phải trả safe explicit status.
7. Health requiredness phụ thuộc feature flag.

---

## C11 — Composition root và shutdown

### File

```text
retrieval_api/composition.py
retrieval_api/main.py
tests/online/api/test_composition.py
tests/online/api/test_advanced_composition.py
```

### Wiring A bàn giao

```text
retrieval service
metadata reader
object reader
visual corpus
image resolver
visual/event encoder
Vietnamese encoder
manifest/readiness validator
```

### Cần làm

1. KIS composition dùng OpenCLIP encoder mới.
2. TRAKE dùng same exact model identity.
3. VQA dùng production image resolver/VLM khi enabled.
4. Query rewrite provider theo feature flag.
5. Startup probe:
   - core KIS required;
   - TRAKE dependencies required khi TRAKE enabled;
   - VQA dependencies required khi VQA enabled.
6. Shutdown order:
   - ngừng nhận request;
   - drain retrieval/ranking/VLM tasks;
   - đóng mode services;
   - đóng adapters.
7. Không khởi tạo fake trong production mode.

---

## C12 — C-side integration tests

### Test flow bắt buộc

#### KIS synthetic vertical slice

```text
SearchRequest
→ QueryBundle
→ A-side fake BranchResults
→ C ranking
→ organizer FusedFrameCandidate
→ KIS competition serializer
```

#### TRAKE synthetic vertical slice

```text
ordered events
→ fake A visual corpus
→ CLIP-compatible event fake vectors
→ DANTE
→ source-frame sequence
```

#### VQA synthetic vertical slice

```text
question
→ retrieval/ranking fixture
→ A image resolver fake
→ evidence selection
→ VLM provider fake
→ grounded answer/source frame
```

### Failure cases

- Core visual unavailable.
- Optional OCR/ASR/summary unavailable.
- Missing metadata.
- Manifest mismatch surfaced by runtime.
- Duplicate source frame.
- TRAKE duplicate source frame across different events.
- Missing image.
- VLM timeout.
- Rewrite timeout.
- Shutdown with in-flight request.

### Definition of Done C12

- C-owned tests xanh.
- Full Online suite xanh sau merge A+C.
- Không có direct import/use của database SDK trong C-owned source.

---

# PHẦN III — NGƯỜI B REVIEW

## 5. Vai trò duy nhất của Người B

Người B không code implementation. Người B thực hiện các review gate sau.

## B-R0 — Review contract trước khi code sâu

Kiểm tra A0:

- [ ] `frame_id` đúng organizer contract.
- [ ] Không còn shot semantics.
- [ ] `source_frame_idx` tách khỏi internal ID.
- [ ] Candidate levels không bị trộn.
- [ ] Models đủ field cho C.
- [ ] Không có compatibility workaround tự đoán ID.

Kết quả: approve/reject shared contract.

## B-R1 — Review A-side

- [ ] SQLite read-only và đúng schema.
- [ ] Milvus output/schema đúng.
- [ ] Manifest/model identity validation.
- [ ] OpenCLIP exact checkpoint.
- [ ] Retrieval không normalize/fuse.
- [ ] Visual corpus streaming.
- [ ] Image resolver path-safe.
- [ ] A-side tests xanh.

## B-R2 — Review C-side

- [ ] Ranking không còn `shot_id`.
- [ ] KIS dedup đúng source frame.
- [ ] TRAKE không dùng KIS dedup.
- [ ] VQA grounded evidence.
- [ ] Serializer không parse frame number.
- [ ] LLM/VLM failures safe.
- [ ] C không bypass A ports.
- [ ] C-side tests xanh.

## B-R3 — Review integration

- [ ] Merge A contract/infrastructure trước.
- [ ] C cập nhật/reconcile trên exact A contract.
- [ ] Không còn duplicate model/port implementation.
- [ ] Full suite pass.
- [ ] Git diff không chứa unrelated changes.
- [ ] Docs/config/tests cùng contract.

## B-R4 — Review dữ liệu thật sau khi Offline bàn giao

- [ ] Startup validator pass database thật.
- [ ] One real `L21_V001` KIS query.
- [ ] Variable-FPS sample.
- [ ] Duplicate-frame sample.
- [ ] TRAKE corpus real adapter.
- [ ] VQA real image resolver.
- [ ] Competition serializer đối chiếu rule BTC.

---

## 6. Thứ tự thực hiện

## Wave 0 — Contract freeze

### Người A

- Hoàn thành A0.
- Tạo organizer base fixtures.

### Người C

- Đọc exact models A đề xuất.
- Chuẩn bị C-side test cases và expected outputs trong C-owned files.
- Không sửa shared domain/ports.

### Người B

- Review B-R0.

Chỉ sau B-R0 mới coi shared contract ổn định.

---

## Wave 1 — Core KIS migration, làm song song

### Người A

```text
A1 SQLite
A2 Milvus
A3 manifest/validator
A4 OpenCLIP encoder
A5 retrieval
```

### Người C

```text
C0 ranking migration
C1 ASR mapping
C2 dedup
C3 objects
C4 KIS orchestration
C5 KIS API/serializer
```

### Output Wave 1

```text
synthetic organizer KIS request
→ seven typed branches
→ ranked source-frame candidates
→ KIS competition output
```

---

## Wave 2 — Advanced production modes, làm song song

### Người A

```text
A6 VisualCorpus adapter
A7 ImageResolver adapter
A8 config/lifecycle/fakes
```

### Người C

```text
C6 TRAKE
C7 VQA evidence
C8 VLM provider
C9 LLM rewrite provider
C10 VQA API
C11 composition
```

### Output Wave 2

- TRAKE chạy qua production-shaped visual corpus port.
- VQA chạy qua production-shaped image/VLM ports.
- Feature flags và readiness đúng.

---

## Wave 3 — Integration và review

### Người A

- Chạy A-side suite.
- Bàn giao constructor/config/schema expectations.
- Sửa đúng A-owned defects do review phát hiện.

### Người C

- Hoàn thành C12.
- Wire toàn bộ A dependencies.
- Sửa đúng C-owned defects do review phát hiện.

### Người B

- B-R1, B-R2, B-R3.
- Chạy full Online suite.
- Kiểm tra architecture boundaries.
- Quyết định merge/chưa merge.

---

## Wave 4 — Real-data validation, chờ Offline

Không chặn code Wave 0–3.

Khi Offline có database thật:

1. A chạy validator/adapters/encoder smoke tests.
2. C chạy real KIS/TRAKE/VQA vertical slices.
3. B thực hiện B-R4 và chốt integration.

Mọi kết luận trước Wave 4 phải ghi:

```text
CODE_READY_AGAINST_SYNTHETIC_ORGANIZER_CONTRACT
NEED_RUNTIME_VERIFICATION_WITH_OFFLINE_DATABASES
```

---

## 7. Acceptance gates cho từng người

## Người A hoàn thành khi

- [ ] Shared organizer models/ports frozen.
- [ ] Không còn PE-Core/shot semantics trong A-owned production source.
- [ ] SQLite/Milvus/ES adapters conform contract.
- [ ] Manifest validator reject wrong model/schema.
- [ ] OpenCLIP text encoder đúng exact model.
- [ ] Seven branches trả đúng candidate levels.
- [ ] Production visual corpus adapter có streaming.
- [ ] Production image resolver path-safe.
- [ ] A-owned tests pass.
- [ ] Handoff interfaces được ghi rõ.

## Người C hoàn thành khi

- [ ] Ranking giữ organizer metadata.
- [ ] ASR mapping dùng CSV-derived timestamp.
- [ ] KIS dedup đúng source frame.
- [ ] Objects dùng normalized labels/XYXY.
- [ ] KIS output dùng source frame.
- [ ] TRAKE order local index và output source frames.
- [ ] VQA evidence/image/VLM flow grounded.
- [ ] Production rewrite/VLM providers có safe fallback/failure behavior.
- [ ] Composition/readiness theo mode.
- [ ] C-owned tests và integration tests pass.

## Người B approve khi

- [ ] Không có duplicate contract/adapter/ranking implementation.
- [ ] Full Online suite pass.
- [ ] No direct DB SDK access trong C layer.
- [ ] No ranking policy trong A layer.
- [ ] No guessed/recomputed competition frame.
- [ ] KIS/TRAKE/VQA serializers đúng semantics.
- [ ] Runtime limitations được báo trung thực.

---

## 8. Test commands bắt buộc

Từ application root:

```text
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/contract -q
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/adapters -q
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/retrieval -q
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/ranking -q
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/trake -q
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/vqa -q
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/api -q
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online/integration -q
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online -q
```

Không được xóa test cũ chỉ vì fixture contract thay đổi. Phải migration assertion
sang organizer semantics.

---

## 9. Những điều cấm để tránh lệch pha

1. C không tự thêm field vào bản copy riêng của candidate model.
2. A không thêm score normalization vào retrieval.
3. C không query Milvus/ES/SQLite SDK trực tiếp.
4. A không quyết định dedup/fusion/summary boost.
5. Không tạo `shot_id=0` compatibility field.
6. Không parse `source_frame_idx` từ `frame_id`.
7. Không hardcode local dataset path.
8. Không claim database/model thật đã chạy khi chỉ test fake.
9. Không bật Stable Diffusion/QUEST trong baseline migration.
10. Không áp KIS dedup lên TRAKE event sequence.
11. Không đổi shared model trong một unrelated PR.
12. Không merge một nhánh khi full Online suite chưa chạy sau integration.

---

## 10. Deliverables cuối cùng

### Người A bàn giao

```text
organizer-v1 domain/ports
read-only production adapters
manifest/readiness validator
OpenCLIP + Vietnamese encoders
seven-branch retrieval service
production VisualCorpusPort adapter
production ImageResolverPort adapter
organizer fakes/fixtures
A-side tests
```

### Người C bàn giao

```text
organizer-aware ranking pipeline
ASR mapper
source-frame dedup/object processing
KIS mode/API/serializer
TRAKE/DANTE mode/API/serializer
VQA evidence/VLM mode/API/serializer
production LLM rewrite/VLM providers
composition/feature flags
C-side integration tests
```

### Người B bàn giao

```text
contract review decision
A-side review report
C-side review report
integration test report
merge readiness decision
real-data verification report khi Offline sẵn sàng
```

