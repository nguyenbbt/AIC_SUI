# 10 — OFFLINE CODE ISSUES

## 1. Mục đích

Tài liệu này là issue register cho toàn bộ Phase Offline hiện tại, bao gồm:

- Module 1 — Shot Detection & Keyframe Extraction.
- Module 2 — Visual Embedding.
- Module 3 — ASR, Transcript Cleaning & Summary.
- Module 4 — OCR.
- Module 5 — Object Detection.
- Module 6 — Vietnamese Text Embedding.
- Module 7 — Multi-DB Indexing.
- Docker, dependency, model-download, validation và test tooling liên quan.

Mục tiêu là sửa và kiểm chứng các lỗi Offline có thể ảnh hưởng trực tiếp tới
Online trước khi bắt đầu implementation Online.

Tài liệu này chỉ ghi nhận lỗi và tiêu chí cần đạt. Nó không sửa source code,
không thay schema và không tự quyết định contract còn mở.

---

## 2. Phạm vi và giới hạn của kết quả audit

- Các lỗi `CONFIRMED_CODE` và `CONTRACT_MISMATCH` được phát hiện bằng static
  source inspection.
- Các mục `NEED_RUNTIME_VERIFICATION` có rủi ro rõ từ source nhưng cần chạy
  dependency, model, artifact hoặc database thật để xác nhận biểu hiện cuối.
- Chưa chạy GPU model, database, Docker build hoặc full test suite trong lần
  audit tạo tài liệu này.
- Danh sách bao phủ toàn bộ lỗi tĩnh đã phát hiện trong snapshot hiện tại; lỗi
  phụ thuộc phiên bản thư viện, driver, model checkpoint và dữ liệu thật vẫn có
  thể xuất hiện khi runtime verification bắt đầu.

---

## 3. Mức ưu tiên

| Mức | Ý nghĩa |
|---|---|
| `P0` | Blocker hoặc có nguy cơ mất dữ liệu/sai toàn bộ branch; phải xử lý trước Online. |
| `P1` | Lỗi nghiêm trọng làm module không chạy, index thiếu hoặc contract không đáng tin cậy. |
| `P2` | Lỗi gây output stale/partial, che giấu failure hoặc làm cấu hình không có tác dụng. |
| `P3` | Tooling, test coverage, resource handling hoặc maintainability có thể gây lỗi gián tiếp. |

---

## 4. Tóm tắt issue

| ID | Ưu tiên | Khu vực | Tóm tắt |
|---|---:|---|---|
| `OFF-001` | P0 | M3 → M6 | Cleaned ASR JSON envelope không khớp semantic embedding reader. |
| `OFF-002` | P0 | M3 → M7 | Cleaned ASR JSON không khớp Elasticsearch ASR loader. |
| `OFF-003` | P0 | M1 → M5 | Module 5 không đọc được metadata `shots[].keyframes[]` của Module 1. |
| `OFF-004` | P0 | M1 → M5 | Global `frame_id` và local keyframe filename chưa được tách rõ. |
| `OFF-005` | P1 | M4 → M6/M7 | OCR artifact dùng local frame stem trong khi docs mô tả global ID. |
| `OFF-006` | P0 | M2 → M7 | Visual loader không normalize `frame_id`. |
| `OFF-007` | P1 | M1 Docker | Build context, requirements và package copy path không hợp lệ. |
| `OFF-008` | P1 | M2 Docker | Entrypoint chạy top-level `cli` nhưng CLI dùng relative import. |
| `OFF-009` | P1 | M3 Docker | Entrypoint chạy top-level `cli` nhưng CLI dùng relative import. |
| `OFF-010` | P1 | M4 Docker | Dockerfile giả định module-local context, trái hướng dẫn root context. |
| `OFF-011` | P1 | M3 dependencies | Azure provider thiếu package `openai`; local provider có dependency gap. |
| `OFF-012` | P2 | Download tooling | Script tải model nuốt lỗi và có thể exit thành công khi weights bị thiếu. |
| `OFF-013` | P1 | M1 resume | Metadata hợp lệ đủ để skip dù keyframe file bị mất/corrupt. |
| `OFF-014` | P2 | M1 config | `--threshold` thường không được truyền cho TransNetV2. |
| `OFF-015` | P2 | M1 fallback | Fallback frame rỗng dùng API `cv2.Mat.zeros` cần runtime validation. |
| `OFF-016` | P2 | M1 parallelism | Nhiều worker có thể load nhiều GPU model độc lập và gây OOM. |
| `OFF-017` | P2 | M1 output | Metadata Parquet có thể partial nhưng CLI vẫn kết thúc thành công. |
| `OFF-018` | P1 | M2 resume | Resume chỉ so row count, không validate model/dim/norm/ID. |
| `OFF-019` | P1 | M2 model contract | CLI cho đổi model nhưng artifact luôn ghi tên PE-Core cố định. |
| `OFF-020` | P1 | M2 completeness | Missing image hoặc OOM có thể tạo visual Parquet thiếu frame. |
| `OFF-021` | P1 | M3 failure signaling | ASR/summary failure có thể bị biến thành output rỗng và exit thành công. |
| `OFF-022` | P2 | M3 resume | Empty/failed summary có thể được cache và skip ở lần chạy sau. |
| `OFF-023` | P2 | M3 async | Cleaning không hoạt động an toàn bên trong event loop đang chạy. |
| `OFF-024` | P2 | M3 summary | Full transcript được gửi cho LLM không có chunk/token budget. |
| `OFF-025` | P1 | M4 resume | OCR skip chỉ vì output file tồn tại, không validate completeness. |
| `OFF-026` | P1 | M4 completeness | Missing images có thể tạo OCR JSON partial rồi bị cache. |
| `OFF-027` | P2 | M4 CLI | `--workers` và `--batch-size` không thực sự hoạt động. |
| `OFF-028` | P1 | M5 resume | Object output partial/corrupt vẫn có thể làm Module 5 skip. |
| `OFF-029` | P3 | M5 resources | PIL images không được đóng rõ ràng trong batch loop. |
| `OFF-030` | P0 | M6 output | Lỗi ghi Parquet bị log rồi nuốt, pipeline vẫn tiếp tục. |
| `OFF-031` | P1 | M6 resume | Existing Parquet được skip mà không validate model/schema/input. |
| `OFF-032` | P1 | M6 provenance | Text Parquet không lưu model/pooling/version fingerprint. |
| `OFF-033` | P2 | M6 discovery/ID | ASR glob quá rộng và `video_id` phụ thuộc filename thay vì payload. |
| `OFF-034` | P0 | M7 transaction | Xóa dữ liệu cũ trước insert làm mất bản tốt khi reindex lỗi. |
| `OFF-035` | P0 | M7 Elasticsearch | Bulk lỗi một phần vẫn có thể được coi là thành công. |
| `OFF-036` | P0 | M7 rollback | Rollback không xóa đầy đủ partial ES/SQLite writes và không restore dữ liệu cũ. |
| `OFF-037` | P0 | M7 success criteria | Video rỗng hoặc thiếu core artifacts vẫn được báo success. |
| `OFF-038` | P2 | M7 CLI | `--force` được nhận nhưng không ảnh hưởng hành vi. |
| `OFF-039` | P1 | M7 identifiers | Invalid/empty frame IDs được prepend thay vì reject. |
| `OFF-040` | P1 | M7 vectors | Dimension chỉ lấy từ file đầu tiên; không validate norm/toàn bộ artifact. |
| `OFF-041` | P0 | M7 DB schema | Existing Milvus/ES/SQLite resources không được audit trước reuse. |
| `OFF-042` | P1 | M7 discovery | Video discovery bỏ qua embedding/transcript/summary-only artifacts. |
| `OFF-043` | P0 | M7 post-condition | Không có post-index cross-DB validation trước khi báo success. |
| `OFF-044` | P3 | M7 lifecycle | Database clients không được đóng trong `finally`. |
| `OFF-045` | P1 | Validation tool | Frame-ID verifier không JOIN cùng một record giữa các DB. |
| `OFF-046` | P0 | Tests | Không có producer-consumer integration fixture cho Module 1–7. |
| `OFF-047` | P2 | Test command | Root test command bỏ sót Module 6, Module 7 và Online contract tests. |
| `OFF-048` | P3 | Modal tooling | `modal_runner.py` chứa absolute path và service entrypoint không tồn tại. |

---

# 5. Cross-module contract issues

## OFF-001 — Module 3 output không khớp Module 6 ASR semantic reader

**Priority:** `P0`  
**Evidence:** `CONTRACT_MISMATCH`

### Producer hiện tại

`feature_extraction/asr_transcript/pipeline.py` —
`ASRTranscriptPipeline.process_video()` ghi:

```json
{
  "video_id": "V001",
  "source": "asr",
  "llm_provider": "...",
  "intervals": [
    {
      "interval_id": 0,
      "start_time_sec": 10.2,
      "end_time_sec": 17.8,
      "raw_text": "...",
      "cleaned_text": "..."
    }
  ]
}
```

`feature_extraction/asr_transcript/segment_grouper.py` —
`SegmentGrouper.group_segments()` tạo `start_time_sec` và `end_time_sec`.

### Consumer hiện tại

`feature_extraction/text_embedding/src/text_embedding/data_readers.py` —
`parse_asr_file()` yêu cầu top-level là list và đọc `start_time`/`end_time`:

```json
[
  {
    "interval_id": "0",
    "start_time": 10.2,
    "end_time": 17.8,
    "cleaned_text": "..."
  }
]
```

### Tác động

- `parse_asr_file()` thấy dict và trả `[]`.
- Module 6 không sinh `embeddings/text_asr/<video_id>.parquet`.
- Module 7 không có dữ liệu để insert Milvus `asr_features`.
- ASR semantic retrieval Online sẽ rỗng.

### Điều kiện cần đạt khi sửa

- Chốt một canonical envelope duy nhất.
- Chốt một cặp tên timestamp duy nhất.
- Chốt type/format của `interval_id`.
- Test phải truyền output thật của Module 3 trực tiếp vào Module 6.

---

## OFF-002 — Module 3 output không khớp Module 7 ASR lexical loader

**Priority:** `P0`  
**Evidence:** `CONTRACT_MISMATCH`

`indexing/src/indexing/data_loader.py` — `load_asr_transcripts()` cũng yêu cầu
top-level list và `start_time`/`end_time`.

### Tác động

- Cleaned transcript thật của Module 3 bị trả về `[]`.
- Elasticsearch `asr_transcripts` không được index.
- ASR lexical retrieval Online sẽ rỗng.

### Ghi chú

Output Parquet dự kiến của Module 6 và `load_text_asr_embeddings()` của Module 7
tương đối tương thích. Blocker chính nằm tại Module 3 → Module 6 và Module 3 →
ASR lexical loader của Module 7.

### Điều kiện cần đạt khi sửa

- Module 6 và Module 7 phải dùng cùng ASR artifact contract.
- Cần integration test xác nhận cùng `video_id + interval_id` xuất hiện trong
  Milvus `asr_features` và ES `asr_transcripts`.

---

## OFF-003 — Module 5 không đọc được metadata của Module 1

**Priority:** `P0`  
**Evidence:** `CONTRACT_MISMATCH`

### Module 1 sinh

`data_pipeline/shot_keyframe/metadata_schema.py` — `VideoMetadata`:

```text
shots[].keyframes[]
```

### Module 5 đọc

`feature_extraction/object_detection/src/object_detection/metadata_reader.py` —
`read_metadata()`:

```python
frames = data.get("frames", [])
if not frames:
    raise ValueError("No frames found in metadata")
```

### Tác động

- Module 5 dừng trước khi đọc ảnh.
- Object JSON không được sinh.
- SQLite `objects` không có dữ liệu.
- Object filtering/count/position của Online không hoạt động.

### Điều kiện cần đạt khi sửa

- Chốt Module 5 đọc trực tiếp `shots[].keyframes[]` hay Module 1 sinh thêm một
  artifact frame-level phẳng.
- Integration test phải dùng metadata thật do Module 1 tạo.

---

## OFF-004 — Global frame ID và local image filename bị trộn

**Priority:** `P0`  
**Evidence:** `CONTRACT_MISMATCH`

Module 1 lưu ảnh tên:

```text
shot_00000_pos_015.webp
```

Module 5 tìm ảnh bằng:

```python
image_path = keyframe_dir / f"{frame_id}.webp"
```

Nếu `frame_id` là canonical global ID `V001_00000_015`, Module 5 sẽ tìm file
`V001_00000_015.webp`, không tồn tại trong output Module 1.

### Tác động

- Sau khi sửa OFF-003, Module 5 vẫn có thể skip toàn bộ image nếu dùng global
  ID làm filename.
- Object output có thể rỗng hoặc partial.

### Điều kiện cần đạt khi sửa

- Domain record phải tách `frame_id` khỏi `file_path`/local filename.
- Global `frame_id` dùng cho JOIN; `file_path` dùng để mở ảnh.
- Test phải dùng đúng tên file thật của Module 1.

---

## OFF-005 — OCR artifact dùng local frame stem

**Priority:** `P1`  
**Evidence:** `CONTRACT_MISMATCH`

`feature_extraction/ocr/src/ocr_module/metadata_reader.py` —
`get_keyframes_from_metadata()` tạo:

```python
frame_id = Path(file_path).stem
```

Kết quả là `shot_00000_pos_015`, không phải `V001_00000_015`.

Module 7 hiện normalize OCR lexical, OCR semantic và SQLite-related IDs, nên
happy path có cơ chế bù. Tuy nhiên artifact schema thực tế khác docs và mọi
consumer ngoài Module 7 có thể hiểu sai.

### Tác động

- Debug/migration tool đọc trực tiếp OCR JSON có thể JOIN sai.
- Nếu một loader quên normalize, cùng keyframe có nhiều ID.

### Điều kiện cần đạt khi sửa

- Chốt artifact OCR dùng canonical global ID hay contract bắt buộc consumer
  normalize.
- Không được có hai cách hiểu ngầm cùng tồn tại.

---

## OFF-006 — Visual embedding loader không normalize frame ID

**Priority:** `P0`  
**Evidence:** `CONTRACT_MISMATCH`

`indexing/src/indexing/data_loader.py` — `load_visual_embeddings()` copy thẳng
`frame_id` từ Parquet, trong khi các loader OCR/metadata/object gọi
`normalize_frame_id()`.

### Tác động

Nếu visual Parquet cũ hoặc artifact ngoài Module 2 chứa local ID:

```text
Milvus visual_features: shot_00000_pos_015
SQLite/ES:              V001_00000_015
```

Online không hydrate được visual result.

### Test gap

Fixture `indexing/tests/test_data_loader.py` dùng local visual ID nhưng test
không assert canonical visual `frame_id`.

### Điều kiện cần đạt khi sửa

- Visual loader phải reject hoặc normalize theo contract đã chốt.
- Test consistency phải bao gồm `visual_features` cùng bốn frame-level stores.

---

# 6. Docker, dependency và model-download issues

## OFF-007 — Module 1 Dockerfile không build đúng từ project root

**Priority:** `P1`  
**Evidence:** `CONTRACT_MISMATCH`

`data_pipeline/shot_keyframe/Dockerfile`:

- `COPY requirements.txt .` nhưng project root không có file này.
- Module 1 cũng không có `requirements.txt` riêng trong snapshot.
- `COPY . /app/data_pipeline/shot_keyframe/` copy toàn repository vào bên trong
  package path, làm cấu trúc bị lồng.
- `COPY weights/ /app/weights/` fail nếu thư mục `weights/` chưa tồn tại; thư
  mục này đang bị Git ignore.

### Tác động

- Docker build có thể fail trước khi chạy CLI.
- Nếu build qua được bằng context khác, package import path vẫn có thể sai.

### Điều kiện cần đạt khi sửa

- Có build context duy nhất được tài liệu hóa và test.
- Dependency manifest và package path phải tồn tại đúng theo context đó.
- Build không được phụ thuộc thư mục ignored chưa được chuẩn bị mà không có
  kiểm tra rõ.

---

## OFF-008 — Module 2 Docker entrypoint sai package context

**Priority:** `P1`  
**Evidence:** `CONFIRMED_CODE`

Docker chạy:

```text
python -m cli
```

nhưng `feature_extraction/visual_embedding/cli.py` dùng:

```python
from .pipeline import run_pipeline
```

Top-level module `cli` không có parent package để resolve relative import.

### Tác động

- Container có thể fail ngay khi start với relative-import error.

---

## OFF-009 — Module 3 Docker entrypoint sai package context

**Priority:** `P1`  
**Evidence:** `CONFIRMED_CODE`

Module 3 có cùng pattern:

```text
ENTRYPOINT ["python", "-m", "cli"]
```

trong khi `cli.py` import `.pipeline`.

### Tác động

- ASR container có thể fail trước argument parsing.

---

## OFF-010 — Module 4 Docker context không thống nhất

**Priority:** `P1`  
**Evidence:** `CONTRACT_MISMATCH`

OCR Dockerfile dùng:

```text
COPY requirements.txt .
COPY src/ /app/src/
```

Các path này chỉ đúng với module-local build context. Root README lại yêu cầu
build module từ project root.

### Tác động

- Theo lệnh build root, cả `requirements.txt` và `src/` đều không nằm ở path
  Dockerfile chờ đợi.

---

## OFF-011 — ASR provider dependencies/config chưa đầy đủ

**Priority:** `P1`  
**Evidence:** `CONTRACT_MISMATCH` / `NEED_RUNTIME_VERIFICATION`

- `llm/azure_llm.py` import `openai.AzureOpenAI` nhưng Module 3
  `requirements.txt` không có package `openai`.
- `LocalTranscriptLLM` dùng `device_map`; một số Transformers versions cần
  package `accelerate`, nhưng requirements không khai báo.
- `.env.example` rỗng dù Gemini/Azure yêu cầu API keys và endpoint variables.

### Tác động

- `--llm-provider azure` có thể fail import ngay.
- Local provider có thể fail lúc load model.
- Người vận hành không có config template chính thức.

---

## OFF-012 — Download scripts che giấu failure

**Priority:** `P2`  
**Evidence:** `CONFIRMED_CODE`

`scripts/download_object_detectors.py` và
`scripts/download_transnet_weights.py` catch exception, chỉ `print()` lỗi và
không trả non-zero exit code.

Object Detection Dockerfile chạy downloader trong build. Nếu downloader bắt
lỗi rồi exit 0, image build có thể thành công nhưng weights/config bị thiếu.

### Tác động

- Failure chuyển từ build time sang runtime.
- CI không nhận biết model cache chưa hoàn chỉnh.

### Điều kiện cần đạt khi sửa

- Downloader phải xác nhận required files và trả failure status cho automation.

---

# 7. Module 1 issues

## OFF-013 — Resume không kiểm tra keyframe artifacts

**Priority:** `P1`  
**Evidence:** `CONFIRMED_CODE`

`VideoProcessor.process_video()` skip nếu metadata JSON parse được và qua
Pydantic validation. Nó không kiểm tra mọi `file_path` có tồn tại/đọc được.

### Tác động

```text
metadata hợp lệ + WebP bị mất
→ Module 1 báo already processed successfully
→ Module 2/4/5 nhận input thiếu
```

---

## OFF-014 — CLI threshold có thể bị bỏ qua

**Priority:** `P2`  
**Evidence:** `CONFIRMED_CODE`

`TransNetPredictor.predict_shots()` gọi `detect_scenes(video_path)` trước. Chỉ
khi lời gọi không-threshold ném `TypeError`, code mới thử
`detect_scenes(..., threshold=threshold)`.

### Tác động

- Trong version chấp nhận default threshold, `--threshold` không có tác dụng.
- Hai lần chạy với threshold khác nhau có thể cho kết quả giống nhau ngoài ý
  muốn.

---

## OFF-015 — Fallback empty frame cần runtime verification

**Priority:** `P2`  
**Evidence:** `NEED_RUNTIME_VERIFICATION`

Khi cả target frame và start frame đều không đọc được,
`KeyframeExtractor.extract_keyframes()` gọi:

```python
cv2.Mat.zeros((224, 224, 3), dtype="uint8")
```

API này không được kiểm chứng trong Python OpenCV version của project và code
không import NumPy để có fallback chuẩn khác.

### Tác động

- Nhánh recovery hiếm có thể tự phát sinh exception và fail toàn video.

---

## OFF-016 — Multi-worker có nguy cơ nhân bản GPU model

**Priority:** `P2`  
**Evidence:** `NEED_RUNTIME_VERIFICATION`

Mỗi `process_single_video()` tạo một `VideoProcessor`, từ đó tạo một
`TransNetPredictor` và load model. Với nhiều process workers trên CUDA, mỗi
process có thể giữ một model copy trong VRAM.

### Tác động

- CUDA OOM hoặc model initialization contention.
- `--workers` lớn không chắc tăng throughput.

---

## OFF-017 — Metadata Parquet có thể partial nhưng không fail job

**Priority:** `P2`  
**Evidence:** `CONFIRMED_CODE`

`build_parquet_index()` catch lỗi từng JSON rồi tiếp tục. CLI ghi số video
thành công nhưng không trả non-zero nếu một số video hoặc JSON parse thất bại.

### Tác động

- Automation có thể coi run thành công trong khi aggregate Parquet thiếu video.
- Không có manifest phân biệt complete/partial.

---

# 8. Module 2 issues

## OFF-018 — Resume chỉ kiểm tra row count

**Priority:** `P1`  
**Evidence:** `CONFIRMED_CODE`

Visual pipeline skip existing Parquet nếu `meta.num_rows == len(v_records)`.

Không kiểm tra:

- `model_id` hoặc checkpoint revision.
- Embedding dimension.
- L2 norm.
- Canonical `frame_id`.
- Input image checksum/mtime.
- Precision/preprocessing config.

### Tác động

- Artifact stale hoặc sai embedding space được xem là hợp lệ.

---

## OFF-019 — Model ID configurable nhưng artifact ghi tên cố định

**Priority:** `P1`  
**Evidence:** `CONFIRMED_CODE`

CLI cho truyền model bất kỳ qua `--model-id`, nhưng
`process_video_batch()` luôn ghi:

```python
record_out["model_name"] = "PE-Core-bigG-14-448"
```

### Tác động

- Artifact có thể khai sai model thực tế.
- Online chọn encoder sai nhưng không có metadata đáng tin để phát hiện.
- Cho phép đổi embedding space mà không có migration gate.

---

## OFF-020 — Visual output partial vẫn được ghi và coi là hoàn thành

**Priority:** `P1`  
**Evidence:** `CONFIRMED_CODE`

- Image load lỗi được filter khỏi batch.
- Sau ba lần OOM retry, batch có thể bị bỏ.
- Các record encode thành công còn lại vẫn được ghi ra Parquet.
- Pipeline cuối cùng log completed successfully.

### Tác động

- Milvus `visual_features` thiếu frame nhưng metadata SQLite vẫn có frame.
- Cross-DB cardinality không khớp.

### Điều kiện cần đạt khi sửa

- Có completeness threshold/manifest và failure status.
- Không publish artifact final khi thiếu core visual records ngoài policy.

---

# 9. Module 3 issues

## OFF-021 — ASR/LLM failure signaling không đủ

**Priority:** `P1`  
**Evidence:** `CONFIRMED_CODE`

- `ASREngine.transcribe()` catch exception và trả `[]`.
- `ASRTranscriptPipeline.run()` catch lỗi từng video rồi tiếp tục.
- Không có aggregate failure exit code/report bắt buộc.
- Summary failure được chuyển thành chuỗi rỗng.

### Tác động

- Job có thể exit thành công dù nhiều video không có ASR/summary.
- Module 7 xem artifact thiếu như graceful empty data.

---

## OFF-022 — Failed/empty summary có thể được cache

**Priority:** `P2`  
**Evidence:** `CONFIRMED_CODE`

`VideoSummarizer.summarize_video()` trả `""` khi LLM lỗi. Pipeline vẫn ghi JSON
summary. Lần sau, chỉ cần file tồn tại là skip nếu không `--force`.

### Tác động

- Lỗi tạm thời trở thành output rỗng lâu dài.
- Summary lexical và semantic branches biến mất cho video đó.

---

## OFF-023 — Async cleaning không an toàn khi đã có event loop

**Priority:** `P2`  
**Evidence:** `CONFIRMED_CODE`

Nếu có running loop, code gọi `loop.run_until_complete()`, vốn không được phép
trên loop đang chạy. Exception sau đó dẫn tới `asyncio.run()`, cũng không được
phép gọi bên trong running loop.

### Tác động

- CLI đồng bộ có thể chạy, nhưng khi nhúng pipeline vào async service/notebook,
  cleaning có thể fail.

---

## OFF-024 — Summary không có chunk/token budget

**Priority:** `P2`  
**Evidence:** `NEED_RUNTIME_VERIFICATION`

`VideoSummarizer` nối toàn bộ cleaned intervals thành một chuỗi rồi gửi một lần
cho LLM.

### Tác động

- Video dài có thể vượt context limit hoặc request-size limit.
- Retry cùng payload quá dài không giải quyết nguyên nhân.
- Chi phí/latency khó kiểm soát.

---

# 10. Module 4 issues

## OFF-025 — OCR resume dựa duy nhất vào file existence

**Priority:** `P1`  
**Evidence:** `CONFIRMED_CODE`

`OCRPipeline.process_video()` skip nếu output JSON tồn tại.

Không validate:

- JSON parse được.
- Số frame bằng metadata.
- Model/backbone/config.
- Output có bị partial không.

---

## OFF-026 — Missing images tạo artifact partial rồi bị cache

**Priority:** `P1`  
**Evidence:** `CONFIRMED_CODE`

Image không tồn tại hoặc `cv2.imread()` fail chỉ được log và skip. Pipeline vẫn
ghi JSON từ các frame còn lại; kể cả `frames` rỗng, file output vẫn có thể tồn
tại và làm lần chạy sau skip.

### Tác động

- `ocr_texts` và `ocr_features` thiếu frame mà không có hard failure.

---

## OFF-027 — OCR concurrency flags không hoạt động

**Priority:** `P2`  
**Evidence:** `CONFIRMED_CODE`

- `--batch-size` được CLI ghi rõ là chưa sử dụng.
- Khi `workers > 1`, pipeline chỉ log rồi vẫn chạy tuần tự.

### Tác động

- Cấu hình performance gây hiểu lầm.
- Benchmark/kế hoạch thời gian Offline có thể sai.

---

# 11. Module 5 issues

## OFF-028 — Object detection resume/completeness không được validate

**Priority:** `P1`  
**Evidence:** `CONFIRMED_CODE`

Module 5 skip nếu output tồn tại. Missing/corrupt images được warning và bỏ qua;
phần còn lại vẫn được ghi.

### Tác động

- SQLite `objects` thiếu detection cho frame nhưng không có diagnostics chuẩn.
- Online hard filter có thể loại nhầm frame do detector artifact bị thiếu, không
  chỉ vì detector false negative.

---

## OFF-029 — PIL image handles không được đóng rõ ràng

**Priority:** `P3`  
**Evidence:** `NEED_RUNTIME_VERIFICATION`

Batch loop dùng `Image.open(...).convert("RGB")` nhưng không dùng context manager
hoặc `close()` rõ ràng.

### Tác động

- Dataset lớn có nguy cơ tăng file descriptors/memory pressure tùy Pillow/GC.

---

# 12. Module 6 issues

## OFF-030 — Parquet write failure bị nuốt

**Priority:** `P0`  
**Evidence:** `CONFIRMED_CODE`

`embedding_writer.write_embeddings_to_parquet()` catch mọi exception, chỉ log
và không raise hoặc trả failure status.

### Tác động

```text
to_parquet fails
→ logger.error
→ pipeline tiếp tục
→ CLI có thể in Text embedding complete
```

- ASR/OCR/summary semantic artifact có thể thiếu mà orchestration vẫn coi run
  thành công.

---

## OFF-031 — Text embedding resume không validate artifact

**Priority:** `P1`  
**Evidence:** `CONFIRMED_CODE`

Mỗi pipeline ASR/OCR/summary skip nếu output Parquet tồn tại.

Không kiểm tra:

- Model/checkpoint revision.
- Pooling/chunking/max length.
- Dimension/norm.
- Input JSON thay đổi.
- Row completeness và canonical IDs.

---

## OFF-032 — Text embedding artifact thiếu provenance

**Priority:** `P1`  
**Evidence:** `CONFIRMED_CODE`

Parquet records không lưu model name/revision, pooling strategy, max length hoặc
embedding dimension metadata tương đương Module 2.

### Tác động

- Chỉ nhìn dimension không thể biết OCR/ASR/summary có cùng embedding space.
- Online khó xác định chính xác encoder contract.

---

## OFF-033 — Input discovery và video ID derivation dễ tạo sai artifact

**Priority:** `P2`  
**Evidence:** `CONFIRMED_CODE`

- `process_asr()` glob toàn bộ `*.json`, gồm cả raw và cleaned transcript.
- `video_id` của ASR/summary/OCR được suy từ filename thay vì validate với
  `video_id` trong payload.

### Tác động

- File rename hoặc file ngoài convention có thể tạo sai `video_id`.
- Raw transcript có thể bị đọc nhầm nếu schema thay đổi trong tương lai.

---

# 13. Module 7 issues

## OFF-034 — Delete-before-insert làm mất dữ liệu tốt

**Priority:** `P0`  
**Evidence:** `CONFIRMED_CODE`

`IndexingOrchestrator.process_video()` load artifacts rồi gọi
`_delete_video_from_all(video_id)` trước mọi insertion.

Nếu reindex thất bại, dữ liệu cũ đã bị xóa và không có snapshot/restore.

### Tác động

- Một retry lỗi có thể làm video biến mất khỏi một hoặc cả ba database.
- “Rollback” hiện tại chỉ xóa partial new data, không phục hồi last-known-good.

---

## OFF-035 — Elasticsearch partial bulk errors không fail video

**Priority:** `P0`  
**Evidence:** `CONFIRMED_CODE`

`ESClient.bulk_index()` dùng `raise_on_error=False`. Khi `errors` không rỗng,
hàm chỉ warning rồi trả `success_count`. Orchestrator bỏ qua return value.

### Tác động

- 800/1000 documents thành công vẫn có thể dẫn tới `process_video() == True`.
- ES lexical data không khớp Milvus/SQLite.

---

## OFF-036 — Rollback không đầy đủ

**Priority:** `P0`  
**Evidence:** `CONTRACT_MISMATCH`

### ES failure

- Orchestrator chỉ rollback Milvus.
- ES documents đã insert ở batch trước có thể còn lại.

### SQLite failure

- Orchestrator rollback Milvus và ES.
- SQLite metadata/object batches đã commit không được rollback.

### Mọi failure

- Dữ liệu cũ bị xóa trước đó không được restore.

### Điều kiện cần đạt khi sửa

- Phải định nghĩa rõ atomicity boundary và last-known-good behavior.
- Tests cần cover failure ở batch giữa, không chỉ exception ngay lần gọi đầu.

---

## OFF-037 — Success criteria cho phép thiếu core artifacts

**Priority:** `P0`  
**Evidence:** `CONFIRMED_CODE`

Nếu mọi loader trả `[]`, `process_video()` vẫn return `True`. Test hiện tại còn
xác nhận zero-data success là graceful degradation.

Ngoài ra, nếu có visual records nhưng `visual_dim is None`, insertion bị skip
im lặng vì condition `if visual_records and visual_dim`.

### Tác động

- Required `visual_features` và `metadata` có thể thiếu nhưng run báo success.
- Required Milvus collection có thể không được tạo nếu không có record.

### Điều kiện cần đạt khi sửa

- Phân loại core/optional artifacts rõ ràng.
- Missing core data hoặc dimension phải fail video/job.

---

## OFF-038 — `--force` không có tác dụng

**Priority:** `P2`  
**Evidence:** `CONFIRMED_CODE`

`IndexingOrchestrator.run()` nhận `force` nhưng không dùng khi chọn video hoặc
gọi `process_video()`. Mọi video luôn delete-then-insert.

---

## OFF-039 — Frame-ID normalizer quá dễ dãi

**Priority:** `P1`  
**Evidence:** `CONFIRMED_CODE`

`normalize_frame_id()`:

- Trả thẳng mọi chuỗi bắt đầu bằng `<video_id>_` mà không validate phần còn lại.
- Regex local ID không anchor cuối chuỗi.
- Format không nhận diện được sẽ bị prepend `video_id` thay vì reject.
- Empty string trở thành `<video_id>_`.

### Tác động

- Invalid IDs được insert và chỉ lộ ra khi Online hydration fail.

---

## OFF-040 — Vector validation chưa đủ

**Priority:** `P1`  
**Evidence:** `CONFIRMED_CODE`

`detect_embedding_dim()` chỉ đọc embedding đầu tiên của Parquet đầu tiên.

Không kiểm tra:

- Mọi video có cùng dimension.
- Mọi row có cùng vector length.
- Vector hữu hạn, không NaN/Inf.
- L2 norm xấp xỉ 1.
- ASR/OCR/summary cùng model space, không chỉ cùng dimension.

---

## OFF-041 — Existing DB resources không được validate

**Priority:** `P0`  
**Evidence:** `CONTRACT_MISMATCH`

### Milvus

Nếu collection tồn tại, `create_collection_if_not_exists()` trả collection ngay,
không kiểm tra fields, dimension, metric hoặc HNSW params.

### Elasticsearch

Nếu index tồn tại, code không audit mapping/analyzer/plugin compatibility.

### SQLite

`CREATE TABLE IF NOT EXISTS` không xác nhận existing columns, types, indexes hoặc
foreign-key behavior đúng contract.

### Tác động

- Module 7 có thể insert/query vào schema cũ hoặc sai mà không báo contract
  mismatch sớm.

---

## OFF-042 — Video discovery không bao phủ mọi artifact

**Priority:** `P1`  
**Evidence:** `CONFIRMED_CODE`

`discover_video_ids()` chỉ scan metadata và OCR directories. Nó không discover
từ visual/text embeddings, transcripts, summaries hoặc object detection.

### Tác động

- Artifact orphan ở các nguồn khác không được xử lý hoặc báo lỗi.
- Missing metadata có thể làm một video biến mất hoàn toàn khỏi indexing report
  thay vì được báo core-contract failure.

---

## OFF-043 — Không có post-index validation

**Priority:** `P0`  
**Evidence:** `CONTRACT_MISMATCH`

Sau insertion, Module 7 không kiểm tra:

- Expected record counts.
- Cùng `frame_id` giữa visual/OCR/metadata/objects.
- Cùng `video_id + interval_id` giữa Milvus và ES.
- Summary video IDs giữa Milvus và ES.
- Vector dimensions/norms trong DB.
- Collection/index/table existence và non-empty conditions.

### Tác động

- `Successfully processed video` không bảo đảm Online có thể JOIN/search.

---

## OFF-044 — Connection lifecycle không được đóng an toàn

**Priority:** `P3`  
**Evidence:** `CONFIRMED_CODE`

`IndexingOrchestrator.run()` connect ba clients nhưng không disconnect/close
trong `finally`.

### Tác động

- Connection/file handles có thể tồn tại nếu run được gọi nhiều lần trong cùng
  process hoặc exception xảy ra.

---

# 14. Validation và test issues

## OFF-045 — Frame-ID verifier không thực sự kiểm tra equality JOIN

**Priority:** `P1`  
**Evidence:** `CONFIRMED_CODE`

`verify_frame_id_consistency.py` lấy độc lập `LIMIT 1` từ mỗi collection,
index và table. Các record đó có thể thuộc video/frame khác nhau.

Script chỉ in kết quả; không:

- Chọn một canonical `frame_id` làm seed.
- Query chính ID đó trong năm frame-level resources.
- Assert equality.
- Trả PASS/PARTIAL/FAIL.

### Tác động

- Tool có thể tạo cảm giác contract đã được kiểm chứng dù chưa có JOIN thật.

---

## OFF-046 — Không có end-to-end producer-consumer fixture

**Priority:** `P0`  
**Evidence:** `CONTRACT_MISMATCH`

Tests hiện tại tự tạo fixture phù hợp với consumer thay vì dùng output producer:

- Module 5 test tạo top-level `frames`, khác Module 1.
- Module 6 ASR test tạo top-level list, khác Module 3.
- Module 7 ASR test cũng tạo top-level list, khác Module 3.
- Visual loader test dùng local ID nhưng không assert normalization.
- `tests/run_demo_e2e.py` rỗng.

### Tác động

- Unit tests riêng lẻ có thể pass trong khi pipeline thật bị đứt giữa module.

### Điều kiện cần đạt khi sửa

- Có một fixture video nhỏ đi xuyên Module 1–7.
- Hoặc tối thiểu có contract fixtures được sinh từ producer serializers thật.
- Test phải kiểm tra artifact schemas và cross-DB identifiers.

---

## OFF-047 — Root test command không chạy toàn bộ Offline tests

**Priority:** `P2`  
**Evidence:** `CONTRACT_MISMATCH`

Lệnh test trong root README không gồm:

- `feature_extraction/text_embedding/tests`.
- `indexing/tests`.
- Cross-module/integration tests.

### Tác động

- “Run all tests” thực tế bỏ sót Module 6 và Module 7.

---

## OFF-048 — Modal runner không khớp repository hiện tại

**Priority:** `P3`  
**Evidence:** `CONTRACT_MISMATCH`

`modal_runner.py` chứa:

- Absolute path `D:/Project/AI Challenge 2026/aic_nova_project/...`.
- Service command `uvicorn app.main:app` trong khi repository không có
  `app/main.py`.
- Chỉ đăng ký `ocr_app`, không đại diện toàn Offline pipeline.

### Tác động

- Tool không portable và có thể fail ngay trên workspace hiện tại.

---

# 15. Fix gates trước khi bắt đầu Online

## Gate 1 — Artifact contracts

- [ ] OFF-001 đến OFF-006 đã được quyết định và sửa.
- [ ] Một video fixture đi qua M1 → M5 và M3 → M6 → M7.
- [ ] Canonical ID và artifact schemas được assert tự động.

## Gate 2 — Reproducible module execution

- [ ] Docker M1–M4 build từ một documented context.
- [ ] Tất cả entrypoints start và parse `--help` thành công.
- [ ] Dependency/model download failures trả non-zero status.

## Gate 3 — Artifact completeness

- [ ] Resume validation không chỉ dựa vào file existence/row count.
- [ ] Missing core frame/vector records làm job fail rõ ràng.
- [ ] Artifact ghi model/config/schema provenance đủ để Online chọn encoder.

## Gate 4 — Safe indexing

- [ ] Reindex failure không làm mất last-known-good data.
- [ ] Partial ES/SQLite/Milvus writes được phát hiện và xử lý theo policy rõ.
- [ ] Existing DB schemas được audit trước reuse.
- [ ] Module 7 có required-resource success criteria.

## Gate 5 — Database contract audit

- [ ] Bốn Milvus collections đúng fields/dim/metric/index/norm.
- [ ] Ba ES indexes đúng mappings/analyzer và non-empty theo artifact.
- [ ] Hai SQLite tables đúng schema/FK/indexes.
- [ ] Cross-DB JOIN được kiểm tra bằng cùng một logical record.
- [ ] Report cuối trả PASS/PARTIAL/FAIL, không chỉ in sample độc lập.

## Gate 6 — Test coverage

- [ ] Root test command chạy Module 1–7.
- [ ] Có producer-consumer contract tests.
- [ ] Có failure-in-middle rollback tests theo batch.
- [ ] Có fixture kiểm tra empty/partial/corrupt/stale artifacts.

---

## 16. Kết luận

Phase Offline hiện có source tương đối đầy đủ cho Module 1–7 nhưng chưa thể xem
là Online-ready. Các blocker lớn nhất là:

1. ASR contract bị đứt giữa Module 3, 6 và 7.
2. Object detection contract bị đứt giữa Module 1 và 5.
3. Visual `frame_id` chưa được bảo vệ nhất quán tại Module 7.
4. Docker execution của nhiều module chưa reproducible.
5. Module 7 delete-before-insert và rollback hiện tại có nguy cơ mất dữ liệu.
6. Partial/empty artifacts thường bị xem là success.
7. Chưa có integration test hoặc post-index audit chứng minh Online có thể JOIN.

Chỉ nên bắt đầu implementation Online sau khi các mục `P0` được sửa, các mục
`P1` ảnh hưởng contract được đóng, và sáu fix gates ở trên có bằng chứng PASS.
