# 00 — READ ME FIRST

## 1. Mục đích của bộ tài liệu

Thư mục `docs/` là nguồn ngữ cảnh chuẩn để Codex hiểu toàn bộ hệ thống trước khi phân tích, thiết kế hoặc viết code cho Phase Online.

Bộ tài liệu này phải giúp Codex phân biệt rõ:

1. Điều đã được xác nhận từ source code hiện tại.
2. Điều đã được nhóm thống nhất như một quyết định thiết kế.
3. Điều vẫn còn là câu hỏi mở.
4. Thành phần tùy chọn, chưa thuộc baseline.
5. Thành phần nằm ngoài phạm vi giai đoạn hiện tại.

Codex không được trộn các thiết kế cũ trong lịch sử thảo luận với kiến trúc hiện hành.

---

## 2. Thứ tự đọc bắt buộc

Codex phải đọc theo đúng thứ tự:

1. `AGENTS.md`
2. `docs/00-READ-ME-FIRST.md`
3. `docs/01-SYSTEM-OVERVIEW.md`
4. `docs/02-OFFLINE-PIPELINE-ACTUAL.md`
5. `docs/03-DATABASE-SCHEMA-CURRENT.md`
6. `docs/04-OFFLINE-ONLINE-CONTRACT.md`
7. `docs/05-ONLINE-PIPELINE-TARGET.md`
8. `docs/06-DESIGN-DECISIONS.md`
9. `docs/07-OUT-OF-SCOPE.md`
10. `docs/08-OPEN-QUESTIONS.md`
11. `docs/09-IMPLEMENTATION-PLAN.md`

Không được bắt đầu code trước khi đọc hết các file trên.

---

## 3. Thứ tự ưu tiên nguồn thông tin

Khi hai nguồn mâu thuẫn, dùng thứ tự ưu tiên sau:

1. Yêu cầu trực tiếp mới nhất của người dùng.
2. `docs/06-DESIGN-DECISIONS.md`.
3. `docs/04-OFFLINE-ONLINE-CONTRACT.md`.
4. `docs/03-DATABASE-SCHEMA-CURRENT.md`.
5. Source code hiện tại trên branch đang làm việc.
6. README của từng module.
7. Paper tham khảo.
8. Lịch sử chat hoặc thiết kế cũ.

Source code cho biết hệ thống **đang làm gì**.

Tài liệu quyết định cho biết hệ thống **phải làm gì**.

Nếu hai phần khác nhau, Codex phải báo `CONTRACT_MISMATCH`; không được tự chọn một bên rồi sửa code.

---

## 4. Nhãn bắt buộc khi báo cáo

Mọi kết luận phải được gắn một trong các nhãn:

- `CONFIRMED_CODE`: được xác nhận trực tiếp từ source code hiện tại.
- `CONFIRMED_DESIGN`: quyết định đã được nhóm chốt.
- `NEED_RUNTIME_VERIFICATION`: cần database, model, artifact hoặc service thật để kiểm tra.
- `OPEN_QUESTION`: chưa có quyết định cuối.
- `OPTIONAL`: nhánh có thể bật, nhưng không bắt buộc cho baseline.
- `OUT_OF_SCOPE`: chưa làm trong giai đoạn hiện tại.
- `CONTRACT_MISMATCH`: code, schema và tài liệu đang không khớp.

Không được đổi một `OPEN_QUESTION` thành quyết định nếu người dùng chưa xác nhận.

---

## 5. Cổng hiểu hệ thống trước khi code

Trước khi được phép tạo hoặc sửa source code, Codex phải hoàn thành bốn báo cáo:

### Gate A — Repository Map

Phải xác định:

- Cấu trúc thư mục.
- Entry point.
- CLI.
- Config.
- Tests.
- Output artifacts.
- Code kết nối Milvus, Elasticsearch và SQLite.
- Phần Online đã có thật và phần mới chỉ là scaffold.

### Gate B — Offline Trace

Phải trace được một video từ đầu đến cuối:

```text
Video
→ shot/keyframe
→ visual embedding
→ ASR + summary
→ OCR
→ object detection
→ text embedding
→ Module 7 indexing
→ Milvus + Elasticsearch + SQLite
```

### Gate C — Database Contract Audit

Phải kiểm tra:

- Bốn Milvus collections.
- Ba Elasticsearch indexes.
- Hai SQLite tables.
- Vector dimension.
- L2 normalization.
- HNSW/IP.
- `frame_id`.
- `video_id + interval_id`.
- OCR semantic indexing.
- Rollback/reset.

### Gate D — Online Understanding

Phải giải thích đúng:

- Textual KIS.
- Video KIS (`v-KIS`): thí sinh xem clip do BTC trình chiếu, tự viết truy vấn text
  và dùng chung pipeline text-to-keyframe với Textual KIS.
- TRAKE.
- VQA.
- Candidate levels.
- ASR interval mapping.
- Summary score.
- Object constraints từ UI.
- Normalize, fusion và dedup.

Chỉ khi người dùng xác nhận kết quả của cả bốn gate, Codex mới được lập kế hoạch code.

---

## 6. Quy tắc read-only trong giai đoạn hiểu hệ thống

Trong giai đoạn phân tích, Codex không được:

- Sửa file.
- Tạo code.
- Cài dependency.
- Tải model.
- Chạy model GPU.
- Khởi tạo hoặc reset database.
- Chạy `--reset-all`.
- Xóa artifact.
- Refactor.
- Đổi schema.
- Đổi model.
- Đổi tên collection/index/table.
- Commit hoặc push.

Codex được phép:

- Đọc file.
- Tìm symbol.
- Lập bản đồ dependency.
- Đọc test.
- Đọc Docker Compose.
- Đối chiếu code với docs.
- Đề xuất bước kiểm chứng không phá hủy.

---

## 7. Những điểm Codex phải nhớ ngay từ đầu

- Phase Offline đã tạo dữ liệu tìm kiếm; Phase Online không chạy lại toàn bộ preprocessing.
- Hệ thống hiện dùng Milvus, Elasticsearch và SQLite.
- Milvus hiện hành có cả `ocr_features`.
- `frame_id` là khóa keyframe xuyên database.
- Summary là tín hiệu bổ trợ, không được dùng để prefilter cứng.
- Object constraints do người dùng chọn trong UI ở baseline.
- TRAKE baseline dùng DANTE trên visual-semantic scores.
- VQA retrieval trước, VLM trả lời sau.
- Stable Diffusion và QUEST là optional.
- Agent tự hỏi lại người dùng chưa thuộc baseline hiện tại.

---

## 8. Kết quả mong đợi sau khi đọc bộ tài liệu

Codex phải có khả năng trả lời chính xác:

1. Mỗi module Offline nhận gì và sinh gì?
2. Mỗi artifact được đưa vào database nào?
3. Mỗi retrieval branch truy vấn database nào?
4. Kết quả branch đang ở cấp frame, interval hay video?
5. Làm sao chuyển ASR interval và video score về frame candidates?
6. Khóa JOIN nào được dùng?
7. Những phần nào đã có code?
8. Những phần nào là target design?
9. Những quyết định nào không được tự thay đổi?
10. Những open question nào chặn từng milestone?
