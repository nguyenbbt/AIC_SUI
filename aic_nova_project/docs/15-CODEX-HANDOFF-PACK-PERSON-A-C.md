# 15 — Gói handoff cho Codex của Người A và Người C

## 1. Vì sao cần gói này

Codex của Người A và Người C không có lịch sử cuộc trò chuyện của Người B. Không
nên gửi toàn bộ chat cũ rồi mong Codex tự suy ra trạng thái hiện tại.

Mỗi Codex chỉ cần bốn thứ:

1. Một commit nền chính xác.
2. Tài liệu source of truth trong repository.
3. Một task nhỏ với input/output, file ownership và tests rõ ràng.
4. Một mẫu báo cáo để Người B kiểm tra và ghép code.

File này cung cấp nguyên văn các nội dung cần copy-paste.

---

## 2. Việc Người B phải làm trước khi gửi task

Commit hiện tại chứa toàn bộ A+B+C KIS và các quyết định TRAKE/VQA:

```text
499fae4 docs(online): resolve DANTE and VQA design gates
```

Người B phải push branch tích hợp trước:

```powershell
cd C:\Users\Nguyen\AIC_SUI\aic_nova_project
git status --short
git push origin feature/online-phase-Knguyen
```

Chỉ gửi task khi `git status --short` không có thay đổi chưa commit và remote đã
có commit `499fae4`.

Gửi cho cả A và C đường dẫn/tên các tài liệu sau trong repository:

```text
AGENTS.md
docs/06-DESIGN-DECISIONS.md
docs/08-OPEN-QUESTIONS.md
docs/14-TRAKE-DANTE-VQA-CODING-ASSIGNMENT.md
references/AIO_DANTE+QUEST.pdf
```

Không cần gửi toàn bộ transcript chat.

---

## 3. Thứ tự phối hợp để không tạo contract cạnh tranh

```text
A code shared contracts/ports
→ C review contract PR
→ A sửa review và merge contract
→ C bắt đầu VQA implementation trên contract đã merge

Trong lúc A làm contract:
B code DANTE pure DP/test oracle
C đọc/review contract specification, chưa tạo shared model riêng
```

C chờ một contract wave ngắn là chủ ý kiến trúc, không phải lãng phí thời gian.
Nếu C tự tạo `VQAEvidence` hoặc `VLMResponse` trước A, nhóm sẽ có hai schemas và
phải sửa lại sau.

---

# 4. Nội dung gửi cho Người A

## 4.1 Tin nhắn ngắn gửi trực tiếp cho Người A

Copy nguyên văn:

```text
Bạn phụ trách Person A — Data & Infrastructure cho advanced modes.

Hãy cập nhật từ branch feature/online-phase-Knguyen và xác nhận HEAD có commit
499fae4. Task đầu tiên là shared TRAKE/VQA contracts và ports, chưa làm database
adapter thật, chưa làm DANTE, chưa làm VQA orchestration.

Mở repository bằng Codex, sau đó gửi Codex prompt Person A trong
docs/15-CODEX-HANDOFF-PACK-PERSON-A-C.md. Khi xong hãy push một feature branch
riêng và gửi commit hash + test result cho tôi. Không merge trực tiếp.
```

## 4.2 Lệnh Người A dùng để lấy đúng code nền

```powershell
git fetch origin
git switch -c feature/a-advanced-contracts origin/feature/online-phase-Knguyen
git rev-parse --short HEAD
git status --short
```

Kết quả `git rev-parse --short HEAD` phải là `499fae4` hoặc descendant được
Người B xác nhận.

## 4.3 Prompt đầu tiên gửi cho Codex của Người A

Copy toàn bộ block sau:

```text
Bạn đang làm việc trong repository AIC Nova multimedia video retrieval.

ROLE
Bạn là Person A — Data & Infrastructure. Người B là integration owner. Chỉ làm
shared domain contracts và ports cho TRAKE/VQA trong milestone này.

BASELINE
- Branch hiện tại phải được tạo từ origin/feature/online-phase-Knguyen.
- Expected base commit: 499fae4 hoặc descendant được owner xác nhận.
- KIS A+B+C đã code-ready; không refactor KIS ngoài nhu cầu contract bắt buộc.
- Chưa có dữ liệu thật. Dùng strict contracts và test doubles, không đoán schema/path.

MANDATORY READING
1. Đọc toàn bộ AGENTS.md và tuân thủ nó.
2. Đọc docs/06-DESIGN-DECISIONS.md, đặc biệt DD-026 đến DD-031.
3. Đọc docs/08-OPEN-QUESTIONS.md, xác nhận OQ-013–018 đã RESOLVED và OQ-012 còn open.
4. Đọc toàn bộ docs/14-TRAKE-DANTE-VQA-CODING-ASSIGNMENT.md.
5. Inspect source patterns hiện tại trong online/domain, online/ports, online/testing
   và tests/online/contract trước khi sửa.

TASK — ADVANCED CONTRACT FOUNDATION
Thực hiện A-ADV-01, A-ADV-02 và A-ADV-03 như một contract-foundation milestone:

A. Shared TRAKE domain models
- Ordered TRAKE events/query.
- Per-event frame match.
- Per-video DANTE result với một best sequence.
- TRAKE diagnostics/policy references.
- Strict validation: event IDs unique, minimum two events, same video per sequence,
  strictly increasing local ordered indices, finite scores, frozen models.

B. Shared VQA domain models
- VQA question and answer types.
- Frozen VQAEvidenceBudget defaults exactly DD-030.
- Image/OCR/ASR/summary evidence models with stable evidence/source IDs.
- VLM request and structured VLM response exactly DD-031.
- VQA result/diagnostics.
- Preserve ASR interval ID/start/end; summary stays video-level.

C. Shared ports
- Full ordered visual corpus port for DANTE, exposed by video and batch.
- Image resolver interface only; no actual path policy while OQ-012 is open.
- OCR/ASR/summary evidence hydration ports.
- Mockable VLMPort.
- Ports must not expose Milvus/Elasticsearch/SQLite/Gemini SDK objects.

DECISIONS THAT MUST NOT CHANGE
- DANTE full ordered keyframes per video; no top-M/threshold/summary prefilter.
- Temporal distance is local ordered-keyframe index gap.
- Lambda default 0.001, valid range [0.001, 0.01].
- Output is top-k videos, one best N-frame sequence per video.
- AIC Nova uses PE-Core because the current visual index is PE-Core. Do not
  introduce BEiT-3 without migration/re-index.
- VQA budget/model/prompt must match DD-030/DD-031.

FILES IN SCOPE
- online/domain/trake.py
- online/domain/vqa.py
- online/domain/__init__.py only for exports
- online/ports/visual_corpus.py
- online/ports/evidence.py
- online/ports/images.py
- online/ports/vlm.py
- online/ports/__init__.py only for exports
- tests/online/contract/test_advanced_models.py
- tests/online/contract/test_advanced_ports.py

OUT OF SCOPE
- Do not implement DANTE recurrence/similarity/service.
- Do not implement VQA evidence selection/orchestration/API.
- Do not call/load Gemini, PE-Core or any real model.
- Do not implement actual Milvus/ES/SQLite adapters.
- Do not decide OQ-012 image paths.
- Do not modify retrieval_api or online/modes.
- Do not install dependencies without owner approval.
- Do not push/merge unless the human owner asks.

TEST REQUIREMENTS
Add success, empty, invalid, wrong-level, cross-video, non-increasing-order,
NaN/Infinity, extra-field, immutability, serialization and protocol-conformance tests.

Run at minimum:
python -m pytest tests/online/contract -q
python -m pytest tests/online/contract tests/online/adapters tests/online/retrieval tests/online/integration tests/online/ranking tests/online/modes -q
git diff --check

If FastAPI/runtime dependencies are missing, do not install silently. Report the
exact missing dependency and still run every unaffected suite.

WORKING METHOD
1. Inspect first and state exact planned files.
2. Implement the smallest cohesive contract change.
3. Do not invent fields outside docs/14 without reporting CONTRACT_CHANGE_REQUEST.
4. Preserve unrelated/user changes.
5. At the end report exact files, symbols, tests/results, limitations and git diff.

STOP CONDITION
The task is complete only when contracts/ports/tests pass. Fixture data and real
adapters are separate future PRs.
```

## 4.4 Người A phải gửi lại cho Người B

Yêu cầu A trả đúng mẫu:

```text
Branch:
Commit hash:
Base commit:
Files changed:
Models/ports added:
Tests run + exact result:
Contract changes requested (nếu có):
Known limitations:
```

Không nhận câu trả lời chỉ nói “đã xong” mà thiếu commit/test result.

---

# 5. Nội dung gửi cho Người C trong lúc A đang làm contract

## 5.1 Tin nhắn ngắn gửi trực tiếp cho Người C

Copy nguyên văn:

```text
Bạn phụ trách Person C — VQA orchestration và advanced API.

Hiện Person A đang làm shared TRAKE/VQA contracts. Trước khi contract đó merge,
bạn chưa tạo VQA domain models riêng. Việc trước mắt là đọc quyết định DD-026–031,
đọc docs/14 và review contract PR của A theo checklist trong docs/15.

Sau khi contract PR merge, tôi sẽ gửi commit nền mới và prompt coding C-ADV-01/02.
```

## 5.2 Prompt review gửi cho Codex của Người C

Copy toàn bộ block sau khi A đã push PR/branch nhưng trước khi merge:

```text
Bạn là Person C — Ranking, Orchestration & API reviewer cho shared advanced
contracts do Person A triển khai.

Đọc toàn bộ AGENTS.md, DD-026–DD-031 trong docs/06-DESIGN-DECISIONS.md và toàn bộ
docs/14-TRAKE-DANTE-VQA-CODING-ASSIGNMENT.md.

Đây là REVIEW-ONLY task. Không sửa code, không tạo competing models, không merge.

Review diff của Person A và báo findings theo severity, file/symbol và failure case.
Phải kiểm tra:

TRAKE
- Query giữ ordered events và unique event IDs.
- Sequence bắt buộc same video, đúng N events, local indices strictly increasing.
- Result biểu diễn top-k videos/one sequence per video.
- Diagnostics/config biểu diễn lambda [0.001, 0.01] và policy version.
- Không có Milvus pk/SDK object.

VQA
- Budget biểu diễn đầy đủ DD-030 và immutable.
- Evidence types giữ đúng frame/video/ASR interval levels.
- Stable evidence IDs đủ để validate VLM evidence references.
- VLM structured response đúng DD-031.
- answered/insufficient behavior validate được.
- Không chứa local path/secret trong public domain models.

PORTS
- C có thể implement evidence selector/orchestrator mà không gọi database SDK.
- VLMPort mockable.
- Image resolver chưa hardcode OQ-012.
- Missing evidence và backend failure phân biệt được.

TESTS
- Có invalid/cross-video/order/non-finite/extra-field/frozen cases.

Output chỉ gồm:
1. Findings theo P0/P1/P2 với file/symbol.
2. Questions/contract changes thực sự blocking C implementation.
3. Verdict: READY_FOR_C hoặc NEEDS_CHANGES.
```

---

# 6. Nội dung gửi cho Người C sau khi contract của A đã merge

## 6.1 Lệnh Người C lấy đúng code mới

Người B thay `<CONTRACT_COMMIT>` bằng commit sau khi merge A:

```powershell
git fetch origin
git switch -c feature/c-vqa-foundation origin/feature/online-phase-Knguyen
git rev-parse --short HEAD
git status --short
```

Không cho C tiếp tục nếu HEAD chưa chứa `<CONTRACT_COMMIT>`.

## 6.2 Prompt coding đầu tiên gửi cho Codex của Người C

Copy toàn bộ block, thay placeholder commit:

```text
Bạn đang làm việc trong repository AIC Nova multimedia video retrieval.

ROLE
Bạn là Person C — Ranking, Orchestration & API. Người B là integration owner.
Task này chỉ làm VQA evidence budget/selection và VLM request/response validation.

BASELINE
- Branch được tạo từ origin/feature/online-phase-Knguyen.
- Required contract commit: <CONTRACT_COMMIT>.
- Shared VQA models/ports của Person A đã merge; phải reuse, không tạo schema khác.
- Chưa có data/model thật; dùng shared fakes/test builders.

MANDATORY READING
1. Đọc toàn bộ AGENTS.md.
2. Đọc DD-030 và DD-031 trong docs/06-DESIGN-DECISIONS.md.
3. Đọc toàn bộ docs/14-TRAKE-DANTE-VQA-CODING-ASSIGNMENT.md.
4. Inspect shared VQA models/ports và current KIS ranking/provenance patterns.

TASK
Thực hiện C-ADV-01 và C-ADV-02:

A. Evidence budget/selector
- Enforce max 3 videos, max 3 primary frames/video, max 8 primary frames.
- Add ordered neighbors only while total images <= 12.
- Select one primary frame per selected video first, then fill by final score.
- Dedup canonical frame IDs.
- OCR <= 2,000 chars total.
- ASR <= 4,000 chars, only intervals intersecting ±5 sec windows.
- Summary <= 800 chars/video and <= 2,400 total.
- Combined OCR+ASR+summary <= 8,000 chars.
- Summary-only video must never enter evidence.
- Deterministic truncation/order and bounded diagnostics.

B. VLM request builder/response validator
- Build evidence-only request using shared models.
- Structured response only; no regex/free-form fallback.
- Answer types: short_text, yes_no, number, list.
- Status: answered or insufficient_evidence.
- Confidence: low/medium/high.
- Validate response evidence IDs are subset of request.
- answered requires non-empty answer and valid evidence references.
- Do not expose prompt secret/local path/API key.

FILES IN SCOPE
- online/vqa/__init__.py
- online/vqa/budget.py
- online/vqa/evidence_selector.py
- online/vqa/vlm_request.py or one equivalently focused file
- tests/online/vqa/test_budget.py
- tests/online/vqa/test_evidence_selector.py
- tests/online/vqa/test_vlm_contract.py

OUT OF SCOPE
- Do not edit shared domain/ports. If insufficient, report CONTRACT_CHANGE_REQUEST
  with exact field/type/reason and stop that subpart.
- Do not implement actual Gemini client/network call.
- Do not implement full VQA orchestrator yet.
- Do not edit retrieval_api/composition/routes yet.
- Do not implement DANTE.
- Do not decide OQ-012 actual image paths.
- Do not install dependencies without owner approval.
- Do not push/merge unless the human owner asks.

TEST REQUIREMENTS
- Video diversity and all budget boundaries.
- Duplicate frames/near-frame boundaries.
- OCR/ASR/summary truncation and ASR ±5 sec overlap.
- Summary-only video rejection.
- Missing optional evidence diagnostics.
- Valid/invalid VLM structured responses.
- Unknown evidence IDs and prompt/path secret safety.
- Same input/config gives deterministic output.

Run at minimum:
python -m pytest tests/online/vqa tests/online/contract -q
python -m pytest tests/online/contract tests/online/adapters tests/online/retrieval tests/online/integration tests/online/ranking tests/online/modes tests/online/vqa -q
git diff --check

WORKING METHOD
1. Inspect shared contracts and report any blocker before inventing alternatives.
2. State exact planned files.
3. Implement small pure/testable components.
4. Preserve existing KIS behavior and provenance.
5. Report exact files/symbols/tests/results/limitations.

STOP CONDITION
Complete when C-ADV-01/02 tests and unaffected Online regression pass. Full
orchestrator/API/actual Gemini are future tasks.
```

## 6.3 Người C phải gửi lại cho Người B

```text
Branch:
Commit hash:
Base/contract commit:
Files changed:
Selectors/validators added:
Tests run + exact result:
Contract change requests:
Known limitations:
```

---

# 7. Prompt tiếp theo cho Người A sau contract merge

Chỉ gửi sau khi contract PR đã merge và cả B/C review xong.

```text
Bạn là Person A. Shared advanced contracts/ports đã merge.

Task duy nhất: A-ADV-04 Shared advanced fixture theo
docs/14-TRAKE-DANTE-VQA-CODING-ASSIGNMENT.md.

Tạo shared fake fixture có:
- 2 videos, >=6 ordered frames/video.
- 3 TRAKE events.
- Video 1 có best correct-order sequence.
- Video 2 có high individual similarity nhưng wrong temporal order.
- Equal-score tie case.
- T_v < N no-sequence case.
- OCR, ASR interval, summary, object and fake image references.
- Missing image/ASR cases.
- FakeVLM success, insufficient, malformed and timeout modes.

Vectors nhỏ, finite, L2-normalized và expected cosine/DP score tính tay được.
Không load model/database thật. Không implement DANTE/VQA orchestration.

Files chỉ trong online/testing, tests/online/fixtures và related contract tests.
Run relevant contract/fake tests, unaffected Online suite và git diff --check.
Report commit hash, exact tests và fixture invariants.
```

---

# 8. Prompt tiếp theo cho Người C sau B handoff

Chỉ gửi sau khi:

- Shared fixture của A merge.
- VQA retrieval rewrite/handoff của B merge.
- C-ADV-01/02 merge.

Task tiếp theo của C là:

```text
C-ADV-03 VQA orchestrator
C-ADV-04 TRAKE mode adapter
C-ADV-05 advanced internal API routing/composition
```

Không nên đưa ba task này vào prompt đầu tiên. Chia ít nhất hai PR:

1. VQA orchestrator.
2. TRAKE/VQA routing + composition.

Prompt chi tiết đã nằm trong sections C-ADV-03 đến C-ADV-05 của `docs/14`.

---

# 9. Checklist Người B dùng để nhận code

Trước khi merge PR của A hoặc C:

- [ ] Base commit đúng.
- [ ] Branch chỉ chứa task được giao.
- [ ] Không có shared model cạnh tranh.
- [ ] Không sửa owner layer không được phép.
- [ ] Không hardcode path/secret/model key.
- [ ] Không đổi DD-026–DD-031.
- [ ] Có success/empty/invalid/failure tests.
- [ ] Full unaffected Online suite pass.
- [ ] `git diff --check` sạch.
- [ ] Có commit hash và exact test result.
- [ ] Known limitations được ghi rõ.

Nếu Codex báo “xong” nhưng thiếu một mục trên, trả lại đúng mục còn thiếu; không
tự merge rồi nhờ Người B sửa sau.

---

# 10. Cách gửi status hằng ngày

Cả A và C dùng đúng mẫu:

```text
Đã hoàn thành:
Đang làm:
Commit/branch hiện tại:
Contract/input đang cần:
Tests hiện tại:
Blocker:
```

Người B lưu các status này trong nhóm chat hoặc issue, không cần chuyển toàn bộ
conversation của Codex giữa ba người.

---

# 11. Tóm tắt thứ tự gửi

```text
1. Người B push commit 499fae4.
2. Gửi Person A prompt section 4.3.
3. Khi A push PR, gửi Person C review prompt section 5.2.
4. Merge A contract sau khi C verdict READY_FOR_C.
5. Gửi Person C coding prompt section 6.2 với contract commit thật.
6. Gửi Person A fixture prompt section 7.
7. Người B code DANTE song song.
8. Sau các handoff merge, mới giao C orchestrator/API.
```

Làm theo thứ tự này giúp mỗi Codex có đủ context từ repository và giảm tối đa
việc Người B phải sửa lại code của A/C ở cuối.
