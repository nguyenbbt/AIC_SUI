# 13 — Kế hoạch tiếp theo cho ba người: TRAKE/DANTE và VQA trước dữ liệu thật

## 1. Mục đích của tài liệu

Tài liệu này bắt đầu tại mốc:

- A, B và C đã được hợp nhất trên commit `5d536b5`.
- KIS đã code-ready với fake data và unit/integration tests lõi.
- Chưa có dữ liệu thật từ ban tổ chức.
- Chưa được phép tuyên bố runtime-ready hoặc competition-ready.

Không có dữ liệu thật **không chặn việc code TRAKE/DANTE và VQA**. Ba người sẽ:

1. Chốt contract và baseline thử nghiệm.
2. Code bằng fake ports và fixture đồng bộ.
3. Ghép end-to-end trên fixture.
4. Để việc kết nối, đo và tune bằng dữ liệu thật ở giai đoạn cuối.

Thứ tự mới:

```text
Chốt decision gates tối thiểu
→ Shared contracts + fakes
→ TRAKE/DANTE và VQA chạy song song
→ Advanced-mode integration
→ Hoàn thiện KIS/LLM rewrite còn thiếu
→ Dữ liệu thật + benchmark + tuning cuối
→ Optional Stable Diffusion/QUEST nếu còn thời gian
```

---

## 2. Ba người đang xây phần gì, nói đơn giản

| Người | Vai trò dễ hình dung | Trách nhiệm chính trong giai đoạn mới |
|---|---|---|
| A — Data & Infrastructure | Xây “ổ cắm” và dữ liệu giả đúng chuẩn | Shared models/ports, ordered frame access, image resolver interface, evidence reader, fake fixture |
| B — Query & Retrieval | Tìm các frame phù hợp với điều cần tìm | TRAKE/DANTE, event encoding/similarity, VQA retrieval rewrite, LLM query rewrite |
| C — Ranking, Orchestration & API | Ghép kết quả thành câu trả lời cuối | VQA evidence selection/orchestration, VLM interface, TRAKE output/API, diagnostics |

Không chia người theo database hay theo từng endpoint. Mỗi người tiếp tục giữ đúng layer
để tránh cuối cùng một người phải sửa lại toàn hệ thống.

---

## 3. Điều kiện quan trọng trước khi code

### 3.1 Không cần dữ liệu thật để chốt baseline thử nghiệm

Ta có thể chọn một cấu hình `experimental` để code và test. Khi có dữ liệu thật, cấu
hình đó mới được benchmark và có thể thay đổi.

Không được gọi cấu hình thử nghiệm là production decision.

### 3.2 Cần một buổi họp decision gate ngắn

Trước khi merge thuật toán phụ thuộc các OQ, cả ba phải ghi quyết định thử nghiệm vào
`docs/06-DESIGN-DECISIONS.md` và cập nhật `docs/08-OPEN-QUESTIONS.md`.

Các câu cần chốt để code TRAKE:

- `OQ-013`: phạm vi candidate cho DANTE.
- `OQ-014`: temporal distance dùng timestamp, shot gap hay keyframe gap.
- `OQ-015`: lambda mặc định, range và cách cấu hình.
- `OQ-016`: output trả một sequence hay top-k sequences, mỗi event chứa gì.

Các câu cần chốt để code VQA:

- `OQ-012`: interface resolve image path; lúc chưa có data thật dùng fake resolver.
- `OQ-017`: evidence budget.
- `OQ-018`: VLM port, prompt contract, answer type và no-answer behavior.

Không cần biết giá trị tối ưu cuối cùng. Cần chọn baseline có tên/version, configurable
và có thể thay thế sau benchmark.

### 3.3 LLM rewrite chưa hoàn thành trong code hiện tại

Code hiện tại nhận q0 và các paraphrase do caller truyền vào; chưa có adapter tự gọi
LLM để tạo q1/q2. Phần này thuộc Người B.

Trước khi gọi LLM thật phải chốt thêm:

- Provider/model.
- Prompt và output schema.
- Timeout/retry.
- Cache có hay không.
- Khi LLM lỗi: fallback bắt buộc là q0-only, không làm fail visual core.
- Secret/API key chỉ đi qua environment hoặc secret manager.

Nếu nhóm chưa chọn provider, B vẫn code được `QueryRewritePort`, fake rewriter và
fallback; actual LLM adapter làm sau.

---

## 4. Quy tắc Git để ba người không lệch pha

### 4.1 Base branch chung

Sau khi commit `5d536b5` được push, cả ba tạo nhánh từ cùng một remote commit.

```powershell
git fetch origin
git switch feature/online-phase-Knguyen
git pull --ff-only origin feature/online-phase-Knguyen
```

Mỗi task dùng một branch riêng. Không code trực tiếp trên branch tích hợp.

### 4.2 Thứ tự merge

```text
ADV-00 shared contract PR của A
→ ADV-01 shared fake fixture PR của A
→ PR TRAKE của B và PR VQA của C có thể chạy song song
→ PR integration/API của C
→ PR real adapters/tuning khi có dữ liệu
```

### 4.3 File ownership

| Khu vực | Primary owner |
|---|---|
| `online/domain/`, `online/ports/`, shared fakes | A; B và C bắt buộc review |
| `query_understanding/`, `online/retrieval/`, `online/trake/` | B |
| `online/ranking/`, `online/modes/`, `online/vqa/`, `retrieval_api/` | C |
| `retrieval_api/composition.py` | C; A/B review phần wiring của mình |

Không sửa shared model trong PR thuật toán riêng. Nếu cần đổi model, tách thành một
contract PR nhỏ và yêu cầu cả ba approve trước.

### 4.4 Một PR chỉ có một mục tiêu

Tên branch đề xuất:

```text
feature/advanced-contracts
feature/advanced-fake-fixture
feature/trake-dante-core
feature/trake-service
feature/vqa-retrieval-rewrite
feature/vqa-evidence-orchestrator
feature/advanced-api-routing
```

---

## 5. Wave 0 — Chốt contract và baseline thử nghiệm

### Mục tiêu

B và C có thể code độc lập mà không tự tạo hai kiểu model khác nhau.

### Người A — `A-ADV-00: Advanced shared contracts`

Người A chủ trì PR contract, nhưng model phải được B/C mô tả input/output trước.

Đầu ra cần có:

- TRAKE query chứa danh sách event có thứ tự.
- Event ID ổn định trong phạm vi query.
- TRAKE candidate/sequence result không chứa SDK object.
- VQA question, answer type và evidence reference.
- VQA answer result có answer, confidence/fallback status và evidence IDs.
- Ports cho ordered frame/visual embedding access.
- Port resolve ảnh, nhưng chưa hardcode đường dẫn thật.
- Port hydrate evidence OCR/ASR/summary theo candidate đã chọn.
- Port gọi VLM, mock được trong tests.

Yêu cầu contract:

- Pydantic strict/frozen giống shared models hiện tại.
- Không dùng Milvus `pk` làm ID.
- ASR vẫn là interval-level evidence.
- DANTE không được transition giữa hai `video_id`.
- VQA evidence phải tham chiếu nguồn; không trả raw database row.
- Config thuật toán chứa policy name/version và trạng thái `experimental`.

Tests A phải viết:

- Valid model serialization.
- Unknown/extra field bị reject.
- Duplicate event ID bị reject.
- Event list rỗng bị reject.
- Invalid interval/time/order bị reject.
- Frozen mappings/tuples không bị mutate.
- Fake implementations conform đúng ports.

Reviewer bắt buộc: B và C.

### Người B — `B-ADV-00: TRAKE/VQA contract proposal`

Trước khi A code model, B gửi cho A:

- Input tối thiểu của DANTE.
- Similarity matrix shape.
- Backpointer/output cần giữ để debug.
- Input/output của VQA retrieval rewrite.
- Input/output của generic query rewrite q0/q1/q2.

B không sửa domain model trong PR thuật toán.

### Người C — `C-ADV-00: VQA/API contract proposal`

Trước khi A code model, C gửi cho A:

- Evidence item tối thiểu VLM cần.
- Budget được biểu diễn ở đâu.
- No-answer và VLM failure cần status nào.
- TRAKE sequence cần serialize ra API như thế nào.
- Diagnostics nào phải có cho TRAKE và VQA.

C không chốt public API cuối nếu `OQ-002` chưa đóng; có thể dùng internal unstable
schema có version.

### Definition of Done Wave 0

- Cả ba approve shared contract PR.
- Không còn temporary dict riêng giữa B và C.
- Fakes dùng cùng model với implementation thật.
- OQ-013–018 có baseline experimental được nhóm ghi nhận.

---

## 6. Wave 1 — Ba người code song song trên fake data

## 6.1 Người A — `A-ADV-01: Advanced fixture và fake ports`

### Việc cần làm

Tạo một fixture nhỏ nhưng đủ bắt lỗi:

- Ít nhất hai video.
- Mỗi video có ordered frames và timestamp tăng dần.
- Ít nhất ba TRAKE events.
- Có frame cùng score để kiểm tra tie-break.
- Có một trường hợp không tồn tại valid sequence.
- Có OCR text ở frame.
- Có ASR interval bao phủ một hoặc nhiều frame.
- Có summary ở video level.
- Có object detections.
- Có fake image references cho VQA.
- Có trường hợp thiếu ảnh/evidence để test degradation.

Fake ports phải cung cấp:

- Ordered visual candidates theo từng video/event.
- Metadata/timestamp cho DANTE.
- Image resolution/reference cho VQA.
- OCR/ASR/summary evidence hydration.
- Health/failure toggles để B/C test timeout và unavailable behavior.

### Không làm trong wave này

- Không đoán path dữ liệu BTC.
- Không sửa database thật.
- Không load checkpoint/model thật.
- Không đưa fusion/ranking vào adapter.

### Tests

- Fake data có canonical IDs.
- Không có cross-video ID collision ngoài điều contract cho phép.
- Ordered frame output deterministic.
- Missing evidence phân biệt với backend failure.
- Fake resolver không trả local secret/path ngoài fixture root.

### Bàn giao

- `FakeOrderedFrameReaderPort` hoặc tên đã chốt.
- `FakeEvidenceReaderPort`.
- `FakeImageResolverPort`.
- `FakeVLMPort` có success, no-answer và failure modes.
- Shared fixture chỉ có một nguồn sự thật.

Reviewer: B và C.

---

## 6.2 Người B — `B-ADV-01: TRAKE/DANTE core`

### Luồng cần code

```text
ordered event texts
→ PE-Core text encoding interface
→ event-to-frame visual similarities
→ group candidates theo video
→ DANTE dynamic programming riêng từng video
→ backtracking
→ ranked event sequences
```

### Thành phần B sở hữu

- TRAKE query parser/builder.
- Event encoder orchestration dùng PE-Core text encoder hiện có.
- Candidate generation theo policy `OQ-013` đã được nhóm chọn.
- Similarity matrix theo từng video.
- DANTE recurrence.
- Temporal penalty theo `OQ-014`.
- Lambda config/validation theo `OQ-015`.
- Backpointer và deterministic tie-break.
- Sequence output theo `OQ-016`.
- TRAKE service trả domain result, không trả API JSON.

### Invariants bắt buộc

- Một DANTE run chỉ chứa một `video_id`.
- Không được transition từ video A sang video B.
- Event order không bị đảo.
- Same input/config cho same output.
- Không trộn OCR, ASR, summary, Stable Diffusion hoặc QUEST vào baseline matrix.
- Không gọi Milvus/SQLite SDK trực tiếp; chỉ dùng ports.
- Không hardcode lambda/top-M trong thuật toán; lấy từ validated config.

### Tests B phải có

1. Hai events có valid ordered sequence.
2. Ba events có valid ordered sequence.
3. Không có valid sequence.
4. Hai video không bao giờ bị nối chung.
5. Temporal penalty thay đổi winner đúng dự kiến.
6. Lambda bằng boundary values.
7. Equal scores có deterministic tie-break.
8. Empty candidate event.
9. Một video thiếu event candidate.
10. Backtracking trả đúng frame cho từng event.
11. Config NaN/Infinity/negative bị reject.
12. Encoder/port failure surfaced đúng error contract.

### Bàn giao cho C

- Một `TRAKE service` nhận validated query và trả validated sequence results.
- Provenance cho từng event/frame.
- Diagnostics tối thiểu: candidate count/video, matrix shape, DP latency, policy version.

Reviewer bắt buộc: C. A review phần port/ID.

---

## 6.3 Người C — `C-ADV-01: VQA evidence selector và orchestrator`

### Luồng cần code

```text
question
→ VQA retrieval request của B
→ KIS retrieval/ranking reuse
→ chọn evidence theo budget
→ resolve image references qua port của A
→ hydrate OCR/ASR/summary evidence qua port của A
→ gọi VLM port
→ evidence-only answer
→ answer + evidence references + diagnostics
```

### Thành phần C sở hữu

- Evidence budget config theo `OQ-017`.
- Candidate-to-evidence selection.
- Dedup ảnh/evidence.
- Token/image budget enforcement.
- VLM request domain model.
- VLM orchestration theo `OQ-018`.
- Evidence-only prompt contract.
- No-answer/fallback behavior.
- VLM timeout/error mapping.
- VQA response và diagnostics nội bộ.

### Invariants bắt buộc

- Không gọi VLM trên toàn dataset.
- Không gửi nhiều evidence hơn budget.
- Không tự đoán image path.
- Không để VLM failure thành empty success.
- Answer phải tham chiếu evidence đã gửi.
- Nếu không đủ evidence, dùng explicit insufficient/no-answer status.
- Không expose prompt secret, API key, raw exception hoặc local absolute path.

### Tests C phải có

1. Visual-only evidence.
2. Visual + OCR evidence.
3. Visual + ASR interval evidence.
4. Summary chỉ là support signal.
5. Evidence vượt budget bị cắt deterministic.
6. Duplicate frame không gửi ảnh hai lần.
7. Missing image reference.
8. Missing OCR/ASR metadata.
9. VLM timeout.
10. VLM malformed response.
11. No-answer response.
12. Evidence IDs trong answer đều tồn tại trong request gửi VLM.

### C có thể làm khi B chưa xong

C dùng fake retrieval service trả đúng shared contract. Khi B merge VQA retrieval
handoff, chỉ thay composition wiring; không viết lại evidence selector.

Reviewer bắt buộc: B. A review phần image/evidence ports.

---

## 7. Wave 2 — Bổ sung query rewrite và ghép advanced modes

## 7.1 Người B — `B-ADV-02: Query rewrite`

### Phần KIS LLM rewrite

- q0 luôn là original query.
- q1/q2 là paraphrase/rewrite, không thay q0.
- Output phải là structured list, không parse text tùy tiện.
- Trim/dedup q1/q2.
- Timeout hoặc provider lỗi → q0-only degraded result.
- Cache chỉ làm nếu đã chốt key/TTL.
- Prompt/version/model xuất hiện trong bounded diagnostics, không chứa secret.

### Phần VQA retrieval rewrite

- Rewrite câu hỏi thành mô tả evidence cần tìm.
- Không cố trả lời câu hỏi trong bước rewrite.
- Reuse KIS query bundle/branches.
- Trả retrieval request cho orchestrator của C.

### Khi chưa có LLM provider

Hoàn thành:

- `QueryRewritePort`.
- Fake deterministic rewriter.
- No-op q0-only implementation.
- Provider error/timeout tests.

Actual provider adapter là PR riêng sau khi nhóm chọn model.

---

## 7.2 Người C — `C-ADV-02: TRAKE/VQA routing và API nội bộ`

### TRAKE

- Nhận validated TRAKE request.
- Gọi service của B.
- Serialize sequence theo shared contract.
- Giới hạn output/top-k theo config đã chốt.
- Trả diagnostics khi không có valid sequence.

### VQA

- Route VQA request sang orchestrator.
- Map VLM/retrieval/evidence errors thành error contract an toàn.
- Response có evidence references.
- Health/readiness có trạng thái VLM adapter nếu VLM được bật.
- API còn mang nhãn internal/unstable cho đến khi `OQ-002` đóng.

### Tests

- Mode routing không gọi sai service.
- TRAKE timeout.
- VQA retrieval degraded nhưng còn evidence dùng được.
- VLM unavailable.
- Safe public errors.
- Lifespan drain trước khi close ports/executors.

---

## 7.3 Người A — `A-ADV-02: Composition support và conformance tests`

- Cung cấp factory/fake wiring cho advanced ports.
- Viết conformance suite dùng được cho fake và adapter thật sau này.
- Kiểm tra shared IDs/provenance qua A → B → C.
- Bảo đảm lifecycle không đóng resource khi request đang chạy.
- Không tạo actual image path policy trước khi `OQ-012` được chốt.

---

## 8. Wave 3 — Fake end-to-end integration

### Integration trace 1: TRAKE

```text
3 ordered events
→ fake PE-Core event vectors
→ two-video candidate fixture
→ per-video DANTE
→ best valid sequence
→ API/domain response
```

Phải chứng minh:

- Event order đúng.
- Không cross-video transition.
- Backpointer đúng.
- Tie-break deterministic.
- Policy/config xuất hiện trong diagnostics.

### Integration trace 2: VQA

```text
question
→ fake retrieval rewrite
→ fake KIS candidates
→ evidence budget
→ fake image/OCR/ASR/summary hydration
→ fake VLM
→ answer + evidence references
```

Phải chứng minh:

- Budget không bị vượt.
- Evidence-only policy được giữ.
- Missing optional evidence degrade rõ.
- VLM failure không bị nuốt.
- Output không chứa local path/secret/raw exception.

### Người chịu trách nhiệm

| Trace | Primary | Support |
|---|---|---|
| TRAKE end-to-end | B | A + C |
| VQA end-to-end | C | A + B |
| Shared fixture/conformance | A | B + C |

Wave 3 chỉ hoàn thành khi cả hai trace chạy bằng test tự động, không chỉ demo thủ công.

---

## 9. Wave 4 — Khi dữ liệu thật xuất hiện

Đây mới là lúc thay fake ports bằng actual resources và tune.

## 9.1 Người A

- Chạy Offline contract validator read-only.
- Xác minh OQ-001 trên record thật.
- Xác minh vector dimension/norm/checkpoint.
- Xây actual ordered-frame/embedding reader theo schema thật.
- Chốt actual image resolver theo OQ-012.
- Chạy cross-DB JOIN samples.
- Xác minh lifecycle/timeouts theo OQ-021.
- Báo `CONTRACT_MISMATCH` cho nhóm Offline; không sửa ID trong Online.

## 9.2 Người B

- Chạy PE-Core/Vietnamese encoders thật.
- Kiểm tra seven KIS branches trên data thật.
- Đo top-k/latency cho OQ-003.
- Chạy TRAKE similarity/DANTE trên video thật.
- Tune candidate scope/lambda; không đổi contract nếu chỉ thay config.
- Đánh giá LLM rewrite bằng query set thật.

## 9.3 Người C

- Benchmark KIS aggregation/normalization/fusion/summary/object policies.
- Chạy TRAKE ranking/API trace thật.
- Chạy VQA với actual evidence và VLM adapter.
- Đo total p50/p95 và stage latencies.
- Chốt API schema OQ-002.
- Kiểm tra competition output adapter và full rehearsal.

### Điều kiện rời Wave 4

- KIS có ít nhất một full real-data vertical slice.
- TRAKE có ít nhất một multi-event query thật, không cross-video.
- VQA có ít nhất một answer thật kèm evidence references.
- Required readiness checks healthy.
- Open questions liên quan đã được ghi thành design decisions.
- Không còn experimental default bị gọi nhầm là approved production config.

---

## 10. Stable Diffusion và QUEST

Hai phần này chưa làm ngay.

Chỉ mở sau khi:

- KIS fake + real vertical slice ổn định.
- TRAKE/DANTE fake integration xong.
- VQA fake integration xong.
- Nhóm còn đủ thời gian/GPU/budget.
- `OQ-019` hoặc `OQ-020` được chốt.

Nếu được mở:

| Phần | Primary đề xuất | Reviewer |
|---|---|---|
| Stable Diffusion query-to-image branch | B | A cho model/runtime, C cho fusion/degradation |
| QUEST rewrite/external exemplar branch | B | A cho external adapter, C cho routing/fusion |

Hai branch phải có manual feature flag trước; không được làm fail visual KIS core.

---

## 11. Task giao ngay cho từng người

### Người A bắt đầu

```text
A-ADV-00: Shared TRAKE/VQA contracts và ports
A-ADV-01: Advanced fake fixture + fake image/evidence/VLM ports
```

### Người B bắt đầu

```text
B-ADV-00: Viết proposal input/output TRAKE + VQA rewrite cho A
B-ADV-01: DANTE pure algorithm + tests trên shared fixture
```

### Người C bắt đầu

```text
C-ADV-00: Viết proposal VQA evidence/result + TRAKE API output cho A
C-ADV-01: VQA evidence budget/selector + fake VLM orchestration
```

Ba task này chạy song song. Chỉ phần import shared contract phải chờ PR A-ADV-00 merge;
B và C vẫn có thể viết test vectors/cases trước.

---

## 12. Lịch merge đề xuất

Không ước lượng theo ngày nếu nhóm chưa biết lịch cá nhân. Dùng thứ tự dependency:

| Mốc | Merge khi nào | Người chính |
|---|---|---|
| M0 | Decision gates experimental được ghi nhận | Cả ba |
| M1 | Shared contracts + conformance tests pass | A |
| M2 | Shared fake fixture pass | A |
| M3 | DANTE pure tests pass | B |
| M4 | VQA evidence/orchestrator fake tests pass | C |
| M5 | VQA retrieval rewrite handoff pass | B |
| M6 | TRAKE/VQA API routing tests pass | C |
| M7 | Hai fake end-to-end traces pass | Cả ba |
| M8 | Actual data adapters/vertical slices pass | Cả ba, khi có data |
| M9 | Benchmark và production decisions được chốt | Cả ba |

---

## 13. Definition of Done cho mỗi PR

Một task chỉ xong khi:

- Dùng shared strict contracts.
- Không truyền SDK objects qua layer.
- Không hardcode path, secret, endpoint, dimension, lambda, top-M hoặc evidence budget.
- Có success, empty, invalid và failure tests.
- Cùng input/config cho output deterministic.
- Provenance không bị mất.
- Error khác empty success.
- Diagnostics đủ biết stage nào lỗi.
- Không vi phạm owner boundary.
- Reviewer bắt buộc đã approve.
- Ghi chính xác test command và kết quả trong PR.
- Experimental limitation được ghi rõ.

---

## 14. Cách báo cáo trong họp hằng ngày

Mỗi người chỉ cần báo bốn dòng:

```text
Đã hoàn thành:
Đang làm:
Contract/input đang cần:
Blocker:
```

Ví dụ Người B:

```text
Đã hoàn thành: DANTE recurrence cho một video và test hai events.
Đang làm: backtracking và deterministic tie-break.
Contract/input đang cần: SequenceResult v1 từ PR A-ADV-00.
Blocker: nhóm chưa chốt output top-1 hay top-k tại OQ-016.
```

---

## 15. Kết luận

Ta không chờ dữ liệu thật mới code advanced modes.

Mốc hợp lý tiếp theo là:

1. Chốt baseline experimental cho OQ-013–018.
2. A merge shared contracts/fakes.
3. B code TRAKE/DANTE.
4. C code VQA.
5. B bổ sung query rewrite; C ghép routing/API.
6. Cả ba chạy fake end-to-end.
7. Khi có dữ liệu thật, thay adapters, benchmark và tune ở bước cuối.

Kết quả trước dữ liệu thật được gọi là **code-complete trên contracts/fakes**, không phải
runtime-ready. Cách này giúp nhóm tiến nhanh mà vẫn không phải viết lại toàn bộ khi dữ
liệu thật xuất hiện.
