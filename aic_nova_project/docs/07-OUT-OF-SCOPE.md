# 07 — OUT OF SCOPE

## 1. Mục đích

File này ngăn Codex mở rộng task ngoài phạm vi hoặc tự xây các module chưa được yêu cầu.

---

# 2. Ngoài phạm vi baseline hiện tại

## 2.1 Conversational KIS / clarification loop

Chưa làm:

```text
retrieve
→ detect ambiguity
→ ask user
→ re-filter
```

## 2.2 Full autonomous Agent loop

Chưa làm ReAct loop:

```text
reason
→ call retrieval
→ inspect result
→ choose another tool
→ repeat
```

## 2.3 LLM auto object parsing

Baseline lấy object constraints từ UI.

Không tự biến query text thành hard object filter.

## 2.4 Learned fusion

Chưa train:

- Learning-to-rank.
- Neural fusion.
- Learned branch weights.

Baseline dùng deterministic fusion.

## 2.5 Training/fine-tuning new encoders

Không train lại:

- PE-Core.
- Vietnamese embedding model.
- OCR.
- ASR.
- Object detector.
- VLM.

## 2.6 Thay database

Không thay:

- Milvus.
- Elasticsearch.
- SQLite.

## 2.7 Thay Offline schema

Online team không tự sửa schema Offline để giải quyết một vấn đề local nếu chưa thống nhất với Offline team.

## 2.8 Full production platform

Chưa làm:

- Multi-tenancy.
- Authentication.
- Billing.
- Horizontal autoscaling.
- Kubernetes.
- Production observability stack.
- Long-term feedback warehouse.

## 2.9 Data regeneration

Không chạy lại toàn bộ dataset trong task Online thông thường.

## 2.10 Unapproved semantic branches

Không tự thêm model/corpus branch mới ngoài:

- Visual.
- OCR.
- ASR.
- Summary.
- Optional SD/QUEST.

## 2.11 OCR region-level retrieval

Hiện retrieval canonical dùng keyframe-level concatenated OCR/embedding.

Region-level search chưa thuộc baseline.

## 2.12 Localized visual cell embeddings

Paper SOICT có localized search theo grid/cell, nhưng Offline schema hiện không có localized visual collection.

Chưa làm nếu không có quyết định và dữ liệu tương ứng.

## 2.13 Automatic submission client

Client gửi kết quả trực tiếp lên hệ thống chấm chưa thuộc retrieval core milestone đầu.

---

# 3. Có thể làm sau baseline

- Agent router.
- KISC.
- Learned fusion.
- Query-dependent branch routing.
- Auto object suggestion.
- Localized visual search.
- Region-level OCR.
- Feedback learning.
- Advanced temporal multimodal fusion.
- Cross-encoder reranker.
- Query-adaptive top-k.
- Distributed serving.
