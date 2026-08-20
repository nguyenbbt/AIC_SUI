# AIC 2026 — Kiểm tra tuân thủ định dạng nộp bài vòng sơ tuyển

## 1. Phạm vi kiểm tra

```text
Nguồn quy định: Hướng dẫn nộp bài sơ tuyển do BTC cung cấp
Git base: origin/main
Base commit: 41e0ef30f4200baf2a9e35de7e4073f2f9689015
Working branch: codex/submission-compliance (chưa commit/push)
Ngày kiểm tra: 2026-08-20
```

Phạm vi code:

- `retrieval_api/submission.py`
- `retrieval_api/search_engine.py`
- `retrieval_api/demo.py`
- `ui/src/App.jsx`
- `ui/src/api.js`
- test submission/API và production UI build

## 2. Kết luận trước khi sửa

`CONTRACT_MISMATCH`: main chỉ tạo logical JSON:

```text
aic-kis-logical-submission.json
aic-trake-logical-submission.json
aic-vqa-logical-submission.json
```

Các file này không được BTC chấp nhận vì BTC bắt buộc:

```text
team...zip
└── submission/
    ├── query-1-kis.csv
    ├── query-2-kis.csv
    ├── query-3-qa.csv
    └── query-4-trake.csv
```

Main trước khi sửa cũng thiếu giới hạn answer 100 ký tự, kiểm tra suffix tên
query, CSV escaping, ZIP hierarchy và exact TRAKE event count ở bước export.

## 3. Ma trận yêu cầu BTC và trạng thái sau sửa

| Yêu cầu BTC | Trạng thái | Cách đáp ứng |
|---|---|---|
| Một CSV cho mỗi query | `CONFIRMED_CODE` | Mỗi item trong package trở thành một CSV |
| Tên CSV khớp tên query | `CONFIRMED_CODE` | `query-X-kind.txt` → `query-X-kind.csv` |
| KIS suffix `-kis` | `CONFIRMED_CODE` | Filename validator buộc suffix khớp mode |
| Q&A suffix `-qa` | `CONFIRMED_CODE` | Filename validator buộc suffix khớp mode |
| TRAKE suffix `-trake` | `CONFIRMED_CODE` | Filename validator buộc suffix khớp mode |
| UTF-8 | `CONFIRMED_CODE` | Encode bằng UTF-8 |
| Không BOM | `CONFIRMED_CODE` | ZIP self-check từ chối UTF-8 BOM |
| Delimiter dấu phẩy | `CONFIRMED_CODE` | Python `csv.writer(delimiter=",")` |
| Không header | `CONFIRMED_CODE` | Serializer chỉ ghi data rows |
| CRLF hoặc LF | `CONFIRMED_CODE` | Serializer dùng LF |
| Tối đa 100 dòng/file | `CONFIRMED_CODE` | Pydantic `max_length=100` và KIS UI cap 100 |
| Video không có `.mp4` | `CONFIRMED_CODE` | Model từ chối `.mp4` và path/video name không an toàn |
| Frame ID là integer | `CONFIRMED_CODE` | Strict integer `>= 0` |
| Giá trị frame là frame gốc | `CONFIRMED_CODE` | Xuất `source_frame_idx`, không xuất canonical JOIN ID |
| KIS: 2 cột | `CONFIRMED_CODE` | `video_id,frame_id` |
| Q&A: 3 cột | `CONFIRMED_CODE` | `video_id,frame_id,answer` |
| Answer tối đa 100 ký tự | `CONFIRMED_CODE` | Model + UI `maxLength=100` |
| Answer có comma/quote/newline | `CONFIRMED_CODE` | Standard CSV quote và double-quote escaping |
| Khoảng trắng answer được giữ | `CONFIRMED_CODE` | Validator không trim answer |
| TRAKE đúng N event | `CONFIRMED_CODE` | `event_count` phải bằng số frame mọi row |
| TRAKE đúng thứ tự event | `CONFIRMED_CODE` | Lấy `sequence` DANTE theo event order |
| ZIP có thư mục `submission/` | `CONFIRMED_CODE` | Packager ghi explicit directory entry và CSV bên dưới |
| ZIP chỉ có file hợp lệ | `CONFIRMED_CODE` | Packager self-check exact member set |
| Một tài khoản đội upload | `OPERATIONAL` | Không tự động hóa; captain upload thủ công |
| Tối đa 3 lần/gói | `OPERATIONAL` | Captain phải theo dõi; code không tự upload |
| Lần cuối dùng xếp hạng | `OPERATIONAL` | Phải có checklist trước upload |

## 4. Mapping identity quan trọng nhất

Trong hệ thống có hai giá trị dễ nhầm:

```text
Internal frame_id:    L21_V001_00003_050
BTC Frame Idx:        95
```

CSV phải ghi:

```csv
L21_V001,95
```

Không được ghi:

```csv
L21_V001,L21_V001_00003_050
```

Code lấy số `95` trực tiếp từ `source_frame_idx` do Offline publish. Không tính
lại bằng `timestamp * fps` và không parse canonical internal ID.

## 5. Luồng sử dụng trên UI

### KIS

1. Nhập đúng tên file query BTC, ví dụ `query-1-kis.txt`.
2. Chạy search.
3. Chọn từ 1 đến 100 frame.
4. Bấm **Lưu query vào gói**.

### Q&A

1. Nhập tên `query-X-qa.txt`.
2. Chạy VQA.
3. Chọn một evidence frame được VLM trích dẫn.
4. Kiểm tra/chỉnh answer; UI hiển thị bộ đếm `N/100`.
5. Bấm **Lưu kết quả query hiện tại**.

### TRAKE

1. Nhập tên `query-X-trake.txt`.
2. Nhập đúng N event theo thứ tự.
3. Chạy TRAKE.
4. Lưu kết quả; UI và backend cùng kiểm tra mọi row có đúng N frame.

### Tạo ZIP

1. Kiểm tra bảng **Gói nộp bài** có đủ tất cả query của đợt.
2. Đặt tên ZIP, ví dụ `team_ABC_round1`.
3. Bấm **Tải ZIP đúng chuẩn BTC**.
4. Không tự nén lại file ZIP do hệ thống tạo.

UI cũng xóa kết quả/selection cũ khi đổi mode, sửa nội dung query hoặc chạy lại
query. Mục đích là ngăn việc vô tình lưu candidate của câu cũ vào file câu mới.

## 6. Checklist bắt buộc trước mỗi upload

- [ ] ZIP mở được bằng Windows Explorer/7-Zip.
- [ ] Root ZIP chứa đúng thư mục `submission/`.
- [ ] `submission/` có đủ một CSV cho mỗi query BTC phát trong gói.
- [ ] Không có JSON, XLS/XLSX hoặc file thừa.
- [ ] Tên từng CSV khớp query sau khi đổi `.txt` thành `.csv`.
- [ ] Mở mỗi CSV bằng Notepad thấy text UTF-8 bình thường.
- [ ] Không có header.
- [ ] Mỗi CSV có 1–100 record theo CSV parser.
- [ ] Video ID không có `.mp4`.
- [ ] Frame ID là integer từ `source_frame_idx`.
- [ ] Q&A answer không quá 100 ký tự.
- [ ] TRAKE có đúng N frame cho N event và đúng thứ tự.
- [ ] Captain xác nhận số lần nộp còn lại.
- [ ] Chỉ dùng tài khoản đội BTC cấp.
- [ ] Cả nhóm xác nhận đây có phải lần nộp cuối muốn dùng xếp hạng hay không.

## 7. Những gì code không thể tự bảo đảm

`NEED_RUNTIME_VERIFICATION`:

- Dataset thật phải giữ `source_frame_idx` đúng frame đã decode.
- Phải chạy một real query mỗi mode và mở ZIP tải từ trình duyệt thật.
- Phải upload một gói thử nếu BTC có chức năng validation/dry-run.
- Chất lượng candidate/answer là bài toán retrieval, không phải format.
- Full monorepo test không thể collect trong Python hiện tại vì các module Offline
  cần môi trường riêng (`torchvision`, `google-genai`, `pymilvus`,
  `open_clip_torch` và package paths). Đây không phải lỗi từ submission patch,
  nhưng cần được chạy trong đúng Offline runtime trước cuộc thi.

`OPERATIONAL`:

- Theo dõi tối đa ba lần upload cho mỗi gói.
- Đăng nhập đúng vòng thi và đúng tài khoản đội.
- Không để nhiều thành viên đồng thời upload.
- Lưu SHA-256 và bản sao ZIP đã upload để tránh nhầm lần cuối.

## 8. Readiness decision

`CONFIRMED_CODE`: serializer và API package đáp ứng contract file được BTC mô
tả và có automated tests cho các biên quan trọng.

Verification đã chạy:

```text
python -m pytest -p no:cacheprovider --import-mode=importlib tests/online -q
529 passed

python -m pytest -p no:cacheprovider --import-mode=importlib data_pipeline/shot_keyframe/tests -q
25 passed

npm run build
Vite production build passed

git diff --check
passed
```

`NEED_RUNTIME_VERIFICATION`: chưa được gọi là competition-ready cho đến khi UI
được chạy với Online runtime thật, tạo một ZIP có đủ ba mode, mở/parse lại ZIP
đó và rehearsal quy trình upload bằng tài khoản đội.
