# 11 — PHÂN CÔNG PHÁT TRIỂN ONLINE CHO NHÓM 3 NGƯỜI

> **Contract migration notice (2026-08-05):** Ownership boundaries remain
> useful, but technical payload/model statements are governed by
> `docs/22-OFFLINE-TO-ONLINE-DATA-CONTRACT-SELF-INDEXED-V2.md`.

## 0. Mục đích và cách dùng tài liệu

Tài liệu này là kế hoạch làm việc chung cho ba người phụ trách Phase Online.
Nó chốt:

- Ranh giới trách nhiệm của từng người.
- Contract dữ liệu mà ba phần code phải tuân theo.
- Cách làm song song khi Offline chưa hoàn tất.
- Thứ tự merge và các mốc tích hợp.
- Test bắt buộc và Definition of Done.
- Các quyết định nào đã chốt, quyết định nào cần benchmark hoặc họp nhóm.

Tài liệu này **không sửa hoặc thay đổi code Offline**. Online chỉ đọc dữ liệu mà
Offline đã index vào Milvus, Elasticsearch và SQLite.

Tên tạm dùng trong tài liệu:

- **Người A — Data & Infrastructure**.
- **Người B — Query & Retrieval**.
- **Người C — Ranking, Orchestration & API**.

Nhóm thay A/B/C bằng tên thật trong buổi họp. Không đổi ranh giới module nếu
chưa có lý do kỹ thuật rõ ràng và chưa cập nhật tài liệu này.

---

# 1. Kết luận phân chia

Không chia ba người theo t-KIS, v-KIS và TRAKE/VQA. Cách chia đó làm mỗi người
tự xây lại adapter, encoder, retrieval và ranking, dẫn đến ba pipeline khác nhau.

Chia theo ba lớp ổn định:

```text
                         OFFLINE DATABASES
             Milvus + Elasticsearch + SQLite
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│ Người A — Data & Infrastructure                         │
│ Config, domain contract, ports, adapters, validation    │
└──────────────────────────────────────────────────────────┘
                              │
                  canonical hits / metadata
                              ▼
┌──────────────────────────────────────────────────────────┐
│ Người B — Query & Retrieval                             │
│ Query bundle, encoders, seven retrieval branches        │
└──────────────────────────────────────────────────────────┘
                              │
                       BranchResult[]
                              ▼
┌──────────────────────────────────────────────────────────┐
│ Người C — Ranking, Orchestration & API                  │
│ Mapping, normalization, fusion, objects, dedup, API     │
└──────────────────────────────────────────────────────────┘
                              │
                              ▼
                    FusedFrameCandidate[]
```

Nguyên tắc phụ thuộc:

```text
Người B chỉ phụ thuộc contract/ports của Người A.
Người C chỉ phụ thuộc BranchResult và read ports đã chốt.
Không phần nào đọc dictionary thô của SDK do phần khác truyền sang.
```

---

# 2. Những gì lấy từ Offline và những gì Online tự sở hữu

## 2.1 Bắt buộc bám Offline

| Dữ liệu | Nguồn canonical |
|---|---|
| `frame_id` | Milvus/ES/SQLite; equality JOIN key |
| `video_id` | Tất cả database |
| `shot_id` | Milvus visual, ES OCR, SQLite metadata |
| `timestamp_sec` | SQLite `metadata.timestamp` |
| `interval_id` | Milvus/ES ASR; chỉ unique trong một video |
| `start_time_sec`, `end_time_sec` | Milvus/ES ASR |
| OCR text | ES `ocr_texts` |
| ASR text | ES `asr_transcripts` |
| Summary | ES `video_summaries` |
| Object detection | SQLite `objects` |
| Vector dimension/checkpoint | Schema và encoder Offline thực tế |

Online không được:

- Tự tạo lại hoặc rewrite `frame_id`.
- Dùng Milvus `pk` làm domain ID.
- Xem ASR interval như frame.
- Xem summary result như frame.
- Giả định `image_path` có trong SQLite.
- Cộng trực tiếp điểm Milvus IP và Elasticsearch BM25.

## 2.2 Do Online sở hữu

- Query modes và query bundle.
- `FrameCandidate`, `ASRIntervalCandidate`, `VideoCandidate`.
- `BranchResult`, `FusedFrameCandidate`.
- Provenance, rank, raw score và normalized score.
- ASR interval-to-frame mapping.
- Multi-query aggregation.
- Normalization và fusion.
- Summary boost.
- Object constraint hard/soft behavior.
- Dedup/near-frame grouping.
- Diagnostics, error mapping và API orchestration.

---

# 3. Quyết định định dạng chung

Các quyết định dưới đây là contract nội bộ mà cả ba người phải dùng.

## 3.1 Công nghệ model

- Dùng **Pydantic v2** cho domain models, request/response và config.
- Model cấu hình `extra="forbid"` để phát hiện field viết sai.
- Model domain cấu hình `frozen=True` để tránh một người sửa object tại chỗ.
- JSON dùng `snake_case`.
- Enum kế thừa `str, Enum` để serialize ổn định.
- Không import SDK Milvus, Elasticsearch hoặc SQLite trong `domain/`.

Mẫu base model:

```python
from pydantic import BaseModel, ConfigDict

class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
```

## 3.2 Quy ước type

| Giá trị | Type/quy tắc |
|---|---|
| ID | `str`, không rỗng, giữ nguyên giá trị canonical |
| `shot_id` | `int >= 0` |
| Thời gian | `float >= 0`, đơn vị giây |
| Rank | `int >= 1` |
| Confidence | `float` trong `[0, 1]` |
| Normalized score | `float` trong `[0, 1]` |
| Latency | `float >= 0`, đơn vị millisecond |
| Bbox | `(x_min, y_min, x_max, y_max)` |

Mọi score trong domain đều theo chiều **lớn hơn là tốt hơn**. Adapter phải chuyển
chiều nếu backend nào trả distance theo chiều ngược lại, đồng thời giữ thông tin
chuyển đổi trong provenance/diagnostics.

## 3.3 Enum canonical

```python
class QueryMode(str, Enum):
    KIS_TEXT = "kis_text"
    KIS_VIDEO = "kis_video"
    TRAKE = "trake"
    VQA = "vqa"

class RetrievalBranch(str, Enum):
    VISUAL_DENSE = "visual_dense"
    OCR_DENSE = "ocr_dense"
    OCR_BM25 = "ocr_bm25"
    ASR_DENSE = "asr_dense"
    ASR_BM25 = "asr_bm25"
    SUMMARY_DENSE = "summary_dense"
    SUMMARY_BM25 = "summary_bm25"

class CandidateLevel(str, Enum):
    FRAME = "frame"
    ASR_INTERVAL = "asr_interval"
    VIDEO = "video"

class BranchStatus(str, Enum):
    SUCCESS = "success"
    DEGRADED = "degraded"
    FAILED = "failed"
    DISABLED = "disabled"

class CountOperator(str, Enum):
    EQ = "eq"
    GTE = "gte"
    LTE = "lte"

class FilterMode(str, Enum):
    HARD = "hard"
    SOFT = "soft"
```

Không dùng các alias tự phát như `clip`, `image_search`, `semantic_visual` hoặc
`visual_search` thay cho `visual_dense`.

---

# 4. Domain contract chính thức

## 4.1 `CandidateProvenance`

Mọi candidate phải giải thích được nó đến từ đâu.

```python
class CandidateProvenance(StrictFrozenModel):
    branch: RetrievalBranch
    backend: Literal["milvus", "elasticsearch", "derived"]
    source_resource: str
    query_variant_id: str
    query_text: str
```

`source_resource` là tên collection/index, ví dụ `visual_features` hoặc
`ocr_texts`, không phải URI chứa password.

## 4.2 `FrameCandidate`

Đại diện cho một keyframe từ một branch, trước fusion.

```python
class FrameCandidate(StrictFrozenModel):
    frame_id: str
    video_id: str
    shot_id: int
    timestamp_sec: float

    rank: int
    raw_score: float
    normalized_score: float | None = None
    provenance: CandidateProvenance
```

Invariants:

- Đã hydrate đầy đủ metadata trước khi đưa vào fusion.
- `frame_id` phải JOIN đúng record SQLite.
- `video_id` lấy từ hit phải bằng `video_id` trong SQLite.
- `shot_id` và `timestamp_sec` canonical lấy từ SQLite khi có khác biệt.
- `normalized_score=None` trước bước normalization.
- Không nhét dictionary SDK vào model.

## 4.3 `ASRIntervalCandidate`

```python
class ASRIntervalCandidate(StrictFrozenModel):
    video_id: str
    interval_id: str
    start_time_sec: float
    end_time_sec: float

    rank: int
    raw_score: float
    normalized_score: float | None = None
    text: str | None = None
    provenance: CandidateProvenance
```

Invariants:

- Canonical key là `(video_id, interval_id)`.
- `end_time_sec >= start_time_sec`.
- Đây chưa phải frame; chỉ ASR mapper được chuyển nó thành frame evidence.
- Adapter đổi ES `start_time/end_time` sang domain
  `start_time_sec/end_time_sec`.

## 4.4 `VideoCandidate`

```python
class VideoCandidate(StrictFrozenModel):
    video_id: str
    rank: int
    raw_score: float
    normalized_score: float | None = None
    summary: str | None = None
    provenance: CandidateProvenance
```

Invariants:

- Summary chỉ tạo `VideoCandidate`.
- Không tự tạo frame từ một `VideoCandidate`.
- Summary score chỉ được propagate vào những frame đã có frame/interval evidence.

## 4.5 `BranchResult[T]`

Mỗi result chỉ chứa một candidate level và một branch/query variant.

```python
CandidateT = TypeVar(
    "CandidateT",
    FrameCandidate,
    ASRIntervalCandidate,
    VideoCandidate,
)

class BranchResult(StrictFrozenModel, Generic[CandidateT]):
    branch: RetrievalBranch
    candidate_level: CandidateLevel
    query_variant_id: str
    candidates: tuple[CandidateT, ...]

    requested_top_k: int
    latency_ms: float
    status: BranchStatus
    warnings: tuple[str, ...] = ()
```

Derived properties, không cần lưu trùng trong JSON:

```text
returned_count = len(candidates)
```

Validation bắt buộc:

- `FRAME` chỉ chứa `FrameCandidate`.
- `ASR_INTERVAL` chỉ chứa `ASRIntervalCandidate`.
- `VIDEO` chỉ chứa `VideoCandidate`.
- `FAILED` phải có warning/error được ghi ở diagnostics.
- Empty list có thể là `SUCCESS`; nó khác với backend failure.

## 4.6 `ObjectConstraint` và `ObjectDetection`

`ObjectConstraint` là điều người dùng muốn tìm. `ObjectDetection` là dữ liệu
Offline đã phát hiện. Không dùng chung một model cho hai khái niệm.

```python
class NormalizedRegion(StrictFrozenModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

class ObjectConstraint(StrictFrozenModel):
    label: str
    count_operator: CountOperator
    count: int
    min_confidence: float = 0.5
    position: NormalizedRegion | None = None
    filter_mode: FilterMode = FilterMode.SOFT

class ObjectDetection(StrictFrozenModel):
    label: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    model_source: str | None = None
```

Baseline đầu tiên hỗ trợ label, count, confidence, hard/soft. `position` được giữ
trong schema nhưng backend phải trả lỗi validation rõ nếu image width/height
contract chưa được giải quyết; không được giả định một resolution cố định.

## 4.7 `FusedFrameCandidate`

```python
class NearFrameRef(StrictFrozenModel):
    frame_id: str
    timestamp_sec: float
    final_score: float

class CandidateEvidence(StrictFrozenModel):
    branch: RetrievalBranch
    query_variant_id: str
    raw_score: float
    normalized_score: float

class CandidateDiagnostics(StrictFrozenModel):
    summary_boost: float = 0.0
    object_boost: float = 0.0
    object_constraints_satisfied: int = 0

class FusedFrameCandidate(StrictFrozenModel):
    frame_id: str
    video_id: str
    shot_id: int
    timestamp_sec: float

    final_score: float
    branch_scores: dict[RetrievalBranch, float]
    evidence: tuple[CandidateEvidence, ...]
    near_frames: tuple[NearFrameRef, ...] = ()
    objects: tuple[ObjectDetection, ...] = ()
    diagnostics: CandidateDiagnostics
```

Invariants:

- Một `frame_id` chỉ xuất hiện một lần trong final list.
- `branch_scores` chứa normalized/aggregated score, không chứa raw BM25/IP lẫn lộn.
- `final_score` có deterministic tie-break.
- `near_frames` không chứa frame đại diện.
- Final sort dùng `final_score DESC`, sau đó tie-break bằng `frame_id ASC`.

## 4.8 Diagnostics cấp query

```python
class BranchDiagnostics(StrictFrozenModel):
    status: BranchStatus
    latency_ms: float
    requested_top_k: int
    raw_result_count: int
    output_candidate_count: int
    mapping_loss_count: int = 0
    warnings: tuple[str, ...] = ()

class QueryDiagnostics(StrictFrozenModel):
    query_id: str
    total_latency_ms: float
    stage_latencies_ms: dict[str, float]
    branches: dict[RetrievalBranch, BranchDiagnostics]

    missing_metadata_count: int = 0
    object_filter_removals: int = 0
    dedup_removals: int = 0
    normalization_method: str
    fusion_method: str
    fusion_weights: dict[RetrievalBranch, float]
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
```

Diagnostics không được chứa password, full vector, exception stack trace hoặc
toàn bộ query response của backend trong public API.

---

# 5. Contract giữa ba người

## 5.1 Ports do Người A định nghĩa và cung cấp

```python
class MilvusSearchPort(Protocol):
    def search_visual(...) -> Sequence[FrameSearchHit]: ...
    def search_ocr(...) -> Sequence[FrameSearchHit]: ...
    def search_asr(...) -> Sequence[ASRSearchHit]: ...
    def search_summary(...) -> Sequence[VideoSearchHit]: ...

class ElasticsearchSearchPort(Protocol):
    def search_ocr(...) -> Sequence[FrameSearchHit]: ...
    def search_asr(...) -> Sequence[ASRSearchHit]: ...
    def search_summary(...) -> Sequence[VideoSearchHit]: ...

class MetadataReaderPort(Protocol):
    def get_frames_by_ids(...) -> Mapping[str, FrameMetadata]: ...
    def get_ordered_frames_by_video(...) -> Sequence[FrameMetadata]: ...

class ObjectReaderPort(Protocol):
    def get_objects_by_frame_ids(...) -> Mapping[str, Sequence[ObjectDetection]]: ...
```

`FrameSearchHit`, `ASRSearchHit`, `VideoSearchHit` là boundary records trung lập,
không chứa SDK object. Người B hydrate direct frame hits thành `FrameCandidate`.
Người C dùng `MetadataReaderPort` để map ASR interval sang frame.

## 5.2 Interface Người B cung cấp cho Người C

```python
class RetrievalService(Protocol):
    async def retrieve(self, bundle: QueryBundle) -> tuple[BranchResult, ...]: ...
```

Người C không gọi trực tiếp Milvus/ES để chạy retrieval branch.

## 5.3 Interface Người C cung cấp cho API

```python
class RankingService(Protocol):
    def rank(
        self,
        bundle: QueryBundle,
        branch_results: Sequence[BranchResult],
    ) -> tuple[FusedFrameCandidate, ...]: ...

class SearchOrchestrator(Protocol):
    async def search(self, request: SearchRequest) -> SearchResponse: ...
```

---

# 6. Cấu trúc source đề xuất và quyền sở hữu

```text
online/
├── domain/                         # Contract chung; A primary, B/C review
│   ├── enums.py
│   ├── candidates.py
│   ├── query.py
│   ├── diagnostics.py
│   └── errors.py
├── ports/                          # A primary
│   ├── search.py
│   ├── metadata.py
│   ├── objects.py
│   └── encoders.py
├── adapters/                       # A owner
│   ├── milvus.py
│   ├── elasticsearch.py
│   ├── sqlite.py
│   └── contract_validator.py
├── retrieval/                      # B owner
│   ├── query_builder.py
│   ├── encoders.py
│   ├── branches.py
│   └── service.py
├── ranking/                        # C owner
│   ├── asr_mapper.py
│   ├── normalizers.py
│   ├── fusion.py
│   ├── summary.py
│   ├── object_filter.py
│   └── dedup.py
├── modes/
│   ├── kis.py                      # C owner, B reviewer
│   ├── trake.py                    # B owner, C reviewer
│   └── vqa.py                      # C owner, B reviewer
└── testing/
    └── fakes.py                    # Shared contract fakes

retrieval_api/
├── main.py                         # C owner
└── search_engine.py                # C owner; composition root/facade

query_understanding/
└── parser.py                       # B owner

tests/online/
├── contract/                       # A primary; all review
├── adapters/                       # A owner
├── retrieval/                      # B owner
├── ranking/                        # C owner
└── integration/                    # Shared; one owner/test case
```

Đây là target structure. Khi bắt đầu code phải kiểm tra lại package/import layout
thực tế và tạo thay đổi nhỏ; không di chuyển code Offline.

---

# 7. Người A — Data & Infrastructure

## 7.1 Mục tiêu

Biến Milvus/Elasticsearch/SQLite thành các read-only ports ổn định, để Người B
và C không cần biết chi tiết SDK hoặc schema khác nhau.

## 7.2 Danh sách task

### A-00 — Domain contract PR

Thực hiện:

- Tạo enums và Pydantic models ở mục 3–4.
- Tạo error codes: `CONTRACT_MISMATCH`, `RESOURCE_UNAVAILABLE`,
  `DIMENSION_MISMATCH`, `INVALID_QUERY`, `MISSING_METADATA`, `BRANCH_TIMEOUT`.
- Tạo serialization fixtures cho từng candidate type.
- Tạo Protocols ở mục 5.

Tests:

- Reject ID rỗng, negative time/rank/count.
- Reject unknown field.
- Reject `end_time_sec < start_time_sec`.
- Round-trip JSON không mất enum/provenance.
- Model frozen, không mutate tại chỗ.
- `BranchResult` không chấp nhận sai candidate level.

Definition of Done:

- B và C review, approve.
- Contract tests xanh.
- Không import database SDK trong domain/ports.
- Merge trước mọi implementation PR khác.

### A-01 — Config và lifecycle

Thực hiện:

- Configurable Milvus URI, ES URI, SQLite path.
- Configurable collection/index/table names.
- Timeout/search params đọc từ config, không hardcode trong adapter.
- Read-only SQLite connection.
- Startup/shutdown hooks; không tạo connection mới cho từng candidate.
- Health status phân biệt `healthy`, `degraded`, `unhealthy`.

Không đưa secret vào diagnostics hoặc log.

### A-02 — SQLite adapter

Methods tối thiểu:

- Batch hydrate `frame_id`.
- Lấy ordered frames theo `video_id`.
- Batch lấy objects theo candidate frame IDs.
- Trả mapping theo `frame_id`, không dựa vào thứ tự query result.

Tests:

- Batch hydration nhiều frame.
- Missing frame.
- Video có zero/one/many frames.
- Sort timestamp deterministic.
- Object count theo label/confidence.
- Empty input không query full table.
- Query sử dụng parameter binding.

### A-03 — Milvus adapter

Collections:

- `visual_features`.
- `ocr_features`.
- `asr_features`.
- `summary_features`.

Thực hiện:

- Đọc schema thật và dimension khi startup/validation.
- Validate query vector shape, finite values và norm.
- Search với metric/config tương thích Offline.
- Chuyển SDK hit thành boundary record.
- Không expose `pk`.
- Giữ raw similarity score.

Tests:

- Output field mapping từng collection.
- Empty result.
- Wrong/missing field.
- Dimension mismatch.
- NaN/Inf vector.
- Timeout/connection error.
- Score direction invariant.

### A-04 — Elasticsearch adapter

Indexes:

- `ocr_texts`.
- `asr_transcripts`.
- `video_summaries`.

Thực hiện:

- Query body chỉ request source fields cần thiết.
- Empty query bị reject trước backend.
- Convert `_score` và source fields sang boundary records.
- Normalize `shot_id` về int tại boundary.
- Convert `start_time/end_time` sang `_sec` domain naming.
- Timeout/error không bị giả thành empty successful result.

Tests:

- Exact generated query body.
- Fuzzy toggle/config.
- Mapping từng index.
- Missing `_score`/source field.
- Empty hits khác backend failure.

### A-05 — Offline contract validator

Read-only command/report kiểm tra:

- 4 Milvus collections, 3 ES indexes, 2 SQLite tables tồn tại.
- Required fields và types.
- Vector dimensions.
- Sample vector norms gần 1.
- `frame_id` JOIN cùng logical record giữa databases.
- `(video_id, interval_id)` JOIN ASR.
- `video_id` JOIN summary.
- Encoder smoke vector đúng dimension/norm khi encoder sẵn sàng.

Output status:

```text
PASS     = đủ điều kiện integration.
PARTIAL  = core chạy được, optional resource thiếu.
FAIL     = contract cốt lõi sai; không chạy production query.
```

Validator tuyệt đối không reset, delete hay rewrite database.

### A-06 — Shared fakes và integration fixture

Tạo fixture nhỏ, ID khớp:

- Ít nhất 2 video.
- Mỗi video có nhiều shot/frame.
- Visual, OCR, ASR, summary example.
- ASR interval có overlap, no-overlap và boundary case.
- Object cases: zero/one/many, confidence thấp/cao.
- Một missing-metadata record có chủ đích để test diagnostics.

Fake adapters phải implement đúng Protocol, không chỉ là `MagicMock` tùy ý.

### A-07 — Việc sau khi adapter ổn định

- Benchmark batch sizes và query latency.
- Health/readiness endpoints support cho Người C.
- Connection retry/circuit behavior sau khi OQ-021 được duyệt.
- Hỗ trợ image-path resolver sau khi OQ-012 được duyệt.
- Chủ trì integration với dữ liệu Offline thật.

## 7.3 Người A không làm

- Không quyết định fusion weight.
- Không viết t-KIS/v-KIS orchestration.
- Không tự map summary thành frame.
- Không sửa artifacts Offline trong runtime Online.

---

# 8. Người B — Query & Retrieval

## 8.1 Mục tiêu

Nhận query đã validate, chạy đúng encoder và bảy retrieval branches, trả về
`BranchResult[]` độc lập với ranking/fusion.

## 8.2 Danh sách task

### B-00 — Query models và parser

Tạo `QueryBundle` nội bộ gồm:

```text
query_id
mode
original_query
text_variants: [{variant_id, text, weight_hint}]
object_constraints
enabled_branches
options
```

Quy tắc:

- `q0` luôn là query gốc.
- Paraphrases có ID ổn định `q1`, `q2`.
- Không average embeddings của q0/q1/q2 trước retrieval baseline.
- t-KIS và v-KIS tạo cùng loại text bundle.
- v-KIS không nhận file video/frame/query image từ BTC.

Tests:

- Empty/whitespace query.
- Invalid mode.
- Duplicate variant IDs.
- t-KIS/v-KIS tạo retrieval contract tương đương.
- Structured object constraints được giữ nguyên.

### B-01 — Encoder interfaces và implementations

Encoders:

- PE-Core compatible text encoder cho visual space.
- Vietnamese text encoder cho OCR/ASR/summary semantic space.
- PE-Core image encoder chỉ là optional extension, không thuộc baseline v-KIS.

Thực hiện:

- Load đúng checkpoint/config tương thích Offline.
- Batch encode.
- L2 normalize.
- Validate dimension với metadata từ Người A.
- Cache model, không reload mỗi query.
- Mock/fake encoder cho unit tests.

Tests:

- Output shape/dtype.
- Norm gần 1.
- Empty input.
- Wrong dimension.
- Loading failure.
- Deterministic fake encoder.

### B-02 — Visual semantic branch

Pipeline:

```text
q0/q1/q2
→ PE-Core text encoder
→ Milvus visual_features
→ batch SQLite hydration
→ BranchResult[FrameCandidate]
```

Đây là branch cốt lõi và là vertical slice đầu tiên.

### B-03 — OCR lexical và semantic

```text
q0 → ES ocr_texts → hydrate → FrameCandidate
q variants → Vietnamese encoder → Milvus ocr_features → hydrate → FrameCandidate
```

Mỗi lexical/semantic branch trả `BranchResult` riêng.

### B-04 — ASR lexical và semantic

```text
q0 → ES asr_transcripts → ASRIntervalCandidate
q variants → Vietnamese encoder → Milvus asr_features → ASRIntervalCandidate
```

Người B không chọn frame cho interval. Người C chịu trách nhiệm mapper.

### B-05 — Summary lexical và semantic

```text
q0 → ES video_summaries → VideoCandidate
q variants → Vietnamese encoder → Milvus summary_features → VideoCandidate
```

Không prefilter video và không tạo frame giả.

### B-06 — Retrieval service

Thực hiện:

- Chạy các branch độc lập, có timeout riêng.
- Có thể chạy song song nhưng output order phải deterministic.
- Core visual failure được phân biệt với optional branch degradation.
- Một branch empty không làm mất kết quả branch khác.
- Thu branch latency, result count và warning.
- Không normalize/fuse score tại đây.

Tests:

- All branches success.
- One optional branch failure.
- Visual core failure.
- Timeout.
- Empty source.
- Deterministic BranchResult order.
- Provenance qua q0/q1/q2 không bị mất.

### B-07 — t-KIS và v-KIS query behavior

Chốt hành vi:

```text
t-KIS: mô tả chữ do đề cung cấp → text retrieval pipeline.
v-KIS: thí sinh xem clip BTC chiếu, tự viết mô tả → cùng text retrieval pipeline.
```

Hai mode có thể khác UI label/diagnostics, nhưng không fork retrieval algorithm.

### B-08 — TRAKE sau baseline KIS

Người B primary cho:

- Ordered event query models.
- Event PE-Core encoding.
- Candidate scope và similarity matrix theo từng video.
- DANTE DP/backpointer implementation sau khi OQ-013–OQ-016 được duyệt.

Người C review output/ranking contract và API serialization.

### B-09 — Hỗ trợ VQA retrieval

- Retrieval-oriented rewrite, không trả lời câu hỏi ngay.
- Reuse KIS branches.
- Trả evidence candidates cho VQA orchestrator của Người C.

## 8.3 Người B không làm

- Không truy cập SDK database ngoài ports của Người A.
- Không cộng điểm branches.
- Không quyết định summary boost.
- Không map ASR interval sang frame.
- Không tạo API response cuối.

---

# 9. Người C — Ranking, Orchestration & API

## 9.1 Mục tiêu

Biến `BranchResult[]` thành kết quả cuối có thể giải thích: mapping, normalize,
fusion, object constraints, summary boost, dedup, diagnostics và API routing.

## 9.2 Danh sách task

### C-00 — ASR interval-to-frame mapper

Input:

- `ASRIntervalCandidate`.
- Ordered frame metadata của đúng `video_id`.

Output:

- Zero/one/many `FrameCandidate` có provenance ASR.

Yêu cầu:

- Deterministic.
- Không map sang video khác.
- Không nhân score vô hạn khi một interval có nhiều frames.
- Ghi mapping losses.
- Policy configurable và có version/name trong diagnostics.

Test cases:

- Frame nằm trong interval.
- Nhiều frames trong interval.
- Không có frame trong interval.
- Interval ở đầu/cuối video.
- Overlapping intervals.
- Video không có metadata.

Implementation strategy chỉ merge sau khi OQ-005 được nhóm duyệt.

### C-01 — Query-variant aggregation

Gộp q0/q1/q2 trong cùng logical branch nhưng vẫn giữ evidence từng variant.

Phải test:

- Một frame xuất hiện ở nhiều variants.
- Equal scores.
- Missing variant result.
- Variant timeout.

Method production chỉ chốt sau OQ-004. Default để benchmark nên là RRF-based vì
ít phụ thuộc scale, nhưng không coi nó là quyết định cuối nếu chưa đo trên fixture
và validation queries.

### C-02 — Branch normalization

Tạo interface thay được implementation:

```python
class ScoreNormalizer(Protocol):
    def normalize(self, candidates: Sequence[...]) -> Sequence[...]: ...
```

Implement/test ít nhất:

- Rank/RRF conversion.
- Min-max có xử lý equal-score list.

Không dùng global min-max giữa Milvus và BM25. Production selection qua OQ-006.

### C-03 — Frame fusion

Thực hiện:

- Merge key bằng `frame_id`.
- Giữ branch scores và evidence.
- Missing branch không mặc định bằng một điểm giả không được ghi lại.
- Config validation cho weights.
- Deterministic sort/tie-break.

Weighted RRF là baseline an toàn để benchmark; method/weights production phải
được duyệt ở OQ-007 dựa trên validation set.

### C-04 — Summary propagation

Thực hiện:

- Aggregate lexical/semantic summary ở video level.
- Controlled boost vào frames cùng `video_id`.
- Không prefilter.
- Không tạo frame nếu video chỉ có summary evidence.
- Có cap/config và diagnostics.

Exact method/weight merge sau OQ-008.

### C-05 — Object constraints

Baseline:

- Label presence.
- `eq/gte/lte` count.
- Minimum detector confidence.
- Multiple labels/co-occurrence.
- Hard filter và soft boost.

Position chỉ bật sau OQ-010/OQ-011. Nếu request có position khi feature chưa
sẵn sàng, trả validation error cụ thể; không bỏ qua âm thầm.

### C-06 — Dedup và near-frame grouping

Baseline:

1. Group theo `(video_id, shot_id)`.
2. Giữ frame điểm cao nhất.
3. Các frame khác thành `near_frames`.
4. Tie-break bằng `frame_id ASC`.
5. Temporal fallback chỉ dùng khi shot ID thực sự unavailable theo policy.

Tests:

- Same shot/different shot.
- Same shot ID/different video.
- Equal score.
- Representative không xuất hiện trong `near_frames`.
- Output deterministic.

### C-07 — Search orchestrator

Pipeline:

```text
Validate request
→ build QueryBundle
→ RetrievalService
→ ASR mapping/direct frame hydration checks
→ query-variant aggregation
→ branch normalization
→ summary propagation
→ object hard/soft processing
→ fusion
→ dedup
→ diagnostics
→ response
```

Thứ tự object soft boost/fusion phải được version/config rõ. Không để hai code
path t-KIS/v-KIS chạy thứ tự khác nhau.

### C-08 — API layer

Thực hiện sau khi internal contract ổn định:

- FastAPI request validation.
- Mode routing.
- Error code → HTTP mapping.
- `/health/live` và `/health/ready`.
- Optional diagnostics flag.
- Competition response adapter.
- Không expose raw exception/secret/internal vector.

Exact public request/response, pagination và competition adapter cần chốt tại
OQ-002 trước khi tuyên bố API stable.

### C-09 — VQA orchestration sau baseline KIS

- Nhận question và answer type.
- Gọi retrieval rewrite/branches của Người B.
- Chọn evidence budget.
- Resolve image paths qua port được duyệt.
- Hydrate OCR/ASR/summary evidence.
- Gọi VLM qua interface mockable.
- Yêu cầu evidence-only answer.
- Trả answer và evidence references.

Không merge final VQA trước OQ-012, OQ-017 và OQ-018.

### C-10 — UI/backend object contract

- UI gửi structured constraints, không gửi câu text tự do để backend đoán.
- Serialize đúng enum/count/confidence.
- Disable position controls nếu backend chưa support.
- Hiển thị degraded/warning ở chế độ debug.

## 9.3 Người C không làm

- Không viết query SDK Milvus/ES trực tiếp.
- Không load encoder checkpoint trong ranking.
- Không sửa Offline schema để tiện API.
- Không hardcode fusion weight trong nhiều file.

---

# 10. Ma trận ownership và review

| Thành phần | Primary | Reviewer bắt buộc |
|---|---|---|
| Domain models/enums | A | B và C |
| Ports/config | A | B |
| SQLite/Milvus/ES adapters | A | B |
| Contract validator/fixtures | A | C |
| Query bundle/parser | B | C |
| Encoders | B | A |
| Seven branches/RetrievalService | B | A và C |
| ASR mapper | C | A và B |
| Normalization/fusion/summary | C | B |
| Objects/dedup | C | A |
| KIS orchestrator/API | C | A và B |
| TRAKE/DANTE | B | C |
| VQA orchestration | C | B |
| End-to-end fixture test | A | B và C |

Primary là người chịu trách nhiệm hoàn thành và sửa review. Reviewer không viết
lại toàn bộ phần của primary; reviewer kiểm contract, edge cases và integration.

---

# 11. Kế hoạch làm song song

## Wave 0 — Buổi họp khởi động

Cả nhóm thực hiện:

1. Gán tên thật cho A/B/C.
2. Xác nhận contract mục 3–5.
3. Chọn repository branch/PR convention.
4. Gán owner cho từng open question.
5. Không thảo luận weight/top-k quá lâu nếu chưa có validation queries.

Output buổi họp:

- Một contract PR owner A.
- Một fixture specification owner A.
- Một fake adapter plan cho B.
- Một fake BranchResult plan cho C.

## Wave 1 — Contract và minimal vertical slice

Làm song song sau khi contract PR mở:

| A | B | C |
|---|---|---|
| Domain/ports/config | Fake ports + query bundle | Fake BranchResults + ranking skeleton |
| SQLite adapter | PE-Core text encoder | Diagnostics models usage |
| Milvus visual adapter | Visual branch | Deterministic pass-through ranking |

Vertical slice phải chạy được:

```text
text query
→ PE-Core text encoder
→ Milvus visual_features
→ SQLite hydration
→ FrameCandidate
→ trivial single-branch rank
→ response
```

Mốc này chưa cần OCR/ASR/summary/fusion phức tạp.

## Wave 2 — Bảy retrieval branches

| A | B | C |
|---|---|---|
| ES + remaining Milvus adapters | OCR/ASR/summary branches | ASR mapper experiments |
| Contract validator | Async RetrievalService | Normalizer/fusion experiments |
| Integration fixture | Branch failure behavior | Object/dedup implementation |

Mỗi người dùng fakes của dependency để không chờ người khác.

## Wave 3 — Integration KIS baseline

- A chạy contract validator trên dữ liệu Offline đã fix.
- B chứng minh bảy branch trả đúng `BranchResult`.
- C chạy full KIS orchestration và diagnostics.
- Cả nhóm review một end-to-end trace cho một query.
- t-KIS và v-KIS phải dùng cùng retrieval/ranking service.

## Wave 4 — Tuning và advanced modes

Thứ tự:

1. Đo/tune KIS baseline.
2. TRAKE/DANTE.
3. VQA evidence pipeline.
4. UI hoàn chỉnh.
5. Stable Diffusion/QUEST chỉ khi core ổn định và còn thời gian.

---

# 12. Mock strategy để không chờ nhau

## 12.1 Người B mock Người A

Fake ports trả boundary hits cố định:

```text
FakeMilvusSearchPort
FakeElasticsearchSearchPort
FakeMetadataReaderPort
```

Người B không chờ database thật để viết branch tests.

## 12.2 Người C mock Người B

Fake retrieval service trả:

- Một visual `FrameCandidate` list.
- OCR duplicate cùng `frame_id`.
- Một ASR interval.
- Một summary video score.
- Một degraded branch.

Người C có thể hoàn thành fusion/dedup/diagnostics trước khi encoder chạy thật.

## 12.3 Contract test chung

Fakes và implementations thật phải chạy cùng một protocol conformance suite.
Không tạo fake có field khác implementation thật.

---

# 13. Git/PR rules chống lệch pha

## 13.1 Contract-first

- Merge A-00 trước implementation.
- Thay đổi domain field phải có review của cả ba người.
- Không rename enum/model trong feature PR không liên quan.
- Khi đổi contract, cập nhật model, JSON fixture và tests trong cùng PR.

## 13.2 Một task, một PR nhỏ

Tên đề xuất:

```text
online/a-02-sqlite-adapter
online/b-02-visual-branch
online/c-03-frame-fusion
```

Không gom adapters + retrieval + API vào một PR.

## 13.3 Không cùng sửa một file lớn

- Mỗi module có primary owner.
- Composition wiring nằm ở file riêng do C quản lý.
- Shared models chỉ sửa qua contract PR.
- Config defaults tập trung một nơi, không copy sang từng branch.

## 13.4 Đồng bộ hằng ngày

Mỗi ngày 10–15 phút, mỗi người báo đúng bốn dòng:

```text
Đã hoàn thành:
Đang làm:
Contract/input đang cần:
Blocker:
```

Nếu một interface thay đổi, thông báo trước khi merge, không để người khác phát
hiện qua import error.

---

# 14. Test strategy

## 14.1 Unit tests

- Không cần service thật.
- Test validation, conversion, ranking edge cases.
- Mỗi task owner viết tests cùng PR.

## 14.2 Contract tests

- Cùng input fixture phải tạo cùng domain output.
- Kiểm ID/type/score/provenance.
- Chạy cho fake và adapter thật có mock SDK.

## 14.3 Integration tests

- Dùng small consistent fixture.
- Test cross-DB JOIN.
- Test seven branches.
- Test partial failure và timeout.
- Không dùng production dataset cho mọi CI run.

## 14.4 End-to-end tests

Ít nhất:

1. t-KIS query chỉ visual.
2. t-KIS query có OCR/ASR/summary.
3. v-KIS manual text query dùng cùng pipeline.
4. Object hard/soft cases.
5. Optional branch failure nhưng core vẫn trả result.
6. Contract mismatch trả lỗi rõ.

## 14.5 Performance tests

Đo riêng:

- Encoder latency.
- Mỗi backend branch latency.
- Hydration batch latency.
- ASR mapping.
- Fusion/dedup.
- Total p50/p95.

Không tối ưu trước khi biết stage nào chậm.

---

# 15. Definition of Done chung

Một task chỉ được xem là xong khi:

- Input/output đúng contract.
- Không hardcode endpoint/path/secret/top-k trong logic.
- Có unit tests cho success, empty, invalid và failure.
- Có type hints.
- Không truyền SDK object ra ngoài adapter.
- Không nuốt exception thành empty success.
- Diagnostics đủ để biết branch/stage nào lỗi.
- Output deterministic với cùng input/config.
- Reviewer của component approve.
- Chạy test liên quan và ghi command/result trong PR.
- Limitations/open questions được ghi rõ.

Một milestone chỉ xong khi cả code, tests và integration contract đều xong; không
chỉ vì function đã được viết.

---

# 16. Decision gates còn phải họp/benchmark

Contract model trong tài liệu này đã được chọn. Các quyết định thuật toán sau
không nên chốt chỉ bằng cảm tính:

| Gate | Owner chuẩn bị | Nhóm cần quyết định |
|---|---|---|
| OQ-001 | A | Runtime IDs đã JOIN thật chưa |
| OQ-002 | C | Public API schema/pagination/top-k |
| OQ-003 | B | Top-k từng branch |
| OQ-004 | C | Gộp q0/q1/q2 |
| OQ-005 | C | ASR interval-to-frame policy |
| OQ-006/007 | C | Normalization/fusion/weights |
| OQ-008 | C | Summary boost/cap |
| OQ-009 | C | Object hard/soft default |
| OQ-010/011/012 | A + C | Position và image path |
| OQ-013–016 | B + C | TRAKE/DANTE |
| OQ-017/018 | C + B | VQA evidence/model |
| OQ-021 | A | Connection lifecycle/timeouts |
| OQ-022 | A + C | Missing metadata behavior |
| OQ-023 | A | Integration fixture |

Quy trình cho mỗi gate:

1. Owner viết 1 trang proposal hoặc benchmark nhỏ.
2. Nêu ít nhất hai lựa chọn và failure cases.
3. Cả nhóm chọn.
4. Ghi quyết định vào `06-DESIGN-DECISIONS.md`.
5. Cập nhật/xóa trạng thái tương ứng trong `08-OPEN-QUESTIONS.md`.
6. Mới merge implementation phụ thuộc quyết định đó.

---

# 17. Khi Offline chưa fix xong

Nhóm Online vẫn làm được:

- Domain models/ports/config.
- Fake adapters và fixtures.
- Query parser/bundle.
- Encoder interfaces và mocked tests.
- Seven branch orchestration trên fakes.
- Ranking/fusion interfaces và unit tests.
- API validation skeleton.

Chưa được tuyên bố sẵn sàng production cho đến khi:

- Offline P0/P1 liên quan đã fix.
- Contract validator không FAIL.
- Cross-DB sample JOIN thành công.
- Encoder dimensions/checkpoints khớp dữ liệu thật.
- Một end-to-end query chạy trên data thật.

Nếu dữ liệu Offline thật không khớp contract, A báo `CONTRACT_MISMATCH`; Online
không thêm workaround âm thầm để che lỗi producer.

---

# 18. Agenda buổi họp đề xuất

## 18.1 Phần nói mở đầu

> Nhóm mình sẽ không chia theo từng mode. Mình chia theo ba lớp để mỗi phần có
> một contract rõ và có thể mock phần trước. A phụ trách dữ liệu/adapters, B phụ
> trách query/retrieval, C phụ trách ranking/orchestration/API. Trước tiên cả ba
> review model contract; sau đó mỗi người code độc lập trên fakes và ghép ở từng
> vertical slice nhỏ.

## 18.2 Checklist 45–60 phút

1. 5 phút: xác nhận mục tiêu baseline.
2. 10 phút: gán A/B/C và reviewer.
3. 15 phút: review contract mục 3–5.
4. 10 phút: chọn task Wave 1.
5. 10 phút: thống nhất Git/PR/test rules.
6. 5 phút: liệt kê blocker cần hỏi nhóm Offline.

## 18.3 Kết quả phải có trước khi kết thúc họp

- [ ] Tên A/B/C.
- [ ] Người owner contract PR.
- [ ] Người owner fixture.
- [ ] Task đầu tiên của từng người.
- [ ] Reviewer của từng PR.
- [ ] Ngày ghép minimal vertical slice.
- [ ] Danh sách câu hỏi gửi nhóm Offline.

---

# 19. Task đầu tiên nên giao ngay

## Người A

```text
A-00 Domain contract + ports + serialization tests
A-02 SQLite read-only adapter skeleton
```

## Người B

```text
B-00 QueryBundle/parser dùng domain contract
B-01 Fake encoder
B-02 Visual branch chạy bằng fake ports
```

## Người C

```text
C-02 Normalizer interface + edge-case tests
C-03 Fusion skeleton bằng fake BranchResults
C-06 Dedup/near-frame deterministic tests
```

Sau khi A-00 merge, cả ba rebase/cập nhật branch và thay temporary fakes bằng
contract chính thức. Mục tiêu tích hợp đầu tiên là text → visual search → SQLite
hydration → ranked frame; không đợi đủ bảy branches mới ghép.

---

# 20. Các dấu hiệu cho thấy nhóm đang lệch pha

Dừng và sửa contract/process nếu thấy một trong các dấu hiệu:

- Ba người dùng ba tên khác nhau cho cùng branch.
- Người C phải đọc `_score`, Milvus `distance` hoặc SQLite row trực tiếp.
- Người B tự viết SQL để hydrate frame.
- Người A thêm fusion logic vào adapter.
- t-KIS và v-KIS có hai retrieval service khác nhau.
- Một PR thay shared model nhưng không cập nhật fixtures/tests.
- Empty result và backend error bị coi giống nhau.
- Raw BM25 và IP được cộng trực tiếp.
- Summary result tự sinh frame.
- Missing metadata bị bỏ qua không diagnostics.
- Một người liên tục phải sửa code của cả hai người còn lại ở cuối sprint.

Khi thấy dấu hiệu trên, không vá tại API cuối. Quay lại boundary model/port bị vi
phạm và sửa ở đúng owner.

---

# 21. Mốc sẵn sàng chuyển tiếp

## Ready for Online implementation

- Contract PR đã merge.
- Fakes có thể chạy tests của B/C.
- Các task không phụ thuộc data thật được mở.

## Ready for Online–Offline integration

- Offline fixes đã được nhóm Offline xác nhận.
- A chạy validator đạt PASS hoặc PARTIAL không ảnh hưởng core.
- Cross-DB IDs và dimensions khớp.

## Ready for KIS baseline demo

- Visual core chạy trên data thật.
- Seven branches có behavior success/empty/failure rõ.
- ASR/summary không bị xử lý sai level.
- Fusion/dedup deterministic.
- t-KIS/v-KIS dùng chung pipeline.
- Diagnostics chỉ ra được branch failures.

## Ready for advanced modes

- KIS baseline ổn định và có benchmark.
- TRAKE decision gates OQ-013–016 đã đóng.
- VQA decision gates OQ-012/OQ-017/OQ-018 đã đóng.

## Ready for competition rehearsal

- API/UI response contract ổn định.
- Startup/health/readiness chạy được.
- Có timeout/degradation policy.
- Có full rehearsal bằng quy trình người dùng thực tế.
- Không còn workaround thủ công không được ghi lại.
