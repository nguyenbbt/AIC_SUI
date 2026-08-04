# Chạy toàn bộ Offline Pipeline bằng Modal GPU và Docker

Tài liệu này hướng dẫn chạy một lần toàn bộ Phase Offline từ video gốc đến
database đã index. PowerShell runner tự upload input, gọi lần lượt Module 1–6
trên Modal A10G, tải artifact về local và chạy Module 7 bằng Docker.

## Kiến trúc thực thi

| Stage | Nơi chạy | Output chính |
|---|---|---|
| Module 1 | Modal GPU A10G | `metadata`, `keyframes` |
| Module 2 | Modal GPU A10G | CLIP `ViT-B-32::openai` Parquet |
| Module 3 | Modal GPU A10G | audio, transcript và summary |
| Module 4 | Modal GPU A10G | OCR JSON |
| Module 5 | Modal GPU A10G | object detection JSON |
| Module 6 | Modal GPU A10G | text embedding Parquet |
| Artifact sync | Modal Volume → local | `data/processed` |
| Module 7 | Docker local | Milvus, Elasticsearch và SQLite |

GPU jobs là batch compute nên chạy trên Modal. Database có state và cần truy
cập lâu dài nên chạy bằng Docker local. `docker-compose.yml` không được dùng để
chạy Module 1–6.

## Điều kiện trước khi chạy

1. Đặt video `.mp4`, `.mkv`, `.avi` hoặc `.webm` trong `data/raw_videos`.
2. Nếu có caption, đặt `.srt` hoặc `.vtt` trong `data/captions`; tên file phải
   trùng `video_id`. Không có caption thì Module 3 tự chạy ASR.
3. Kích hoạt virtualenv và kiểm tra Modal profile:

   ```powershell
   & .\venv\Scripts\Activate.ps1
   modal profile current
   ```

4. Workspace Modal phải có spend limit lớn hơn chi phí dự kiến. Credit còn lại
   và spend limit là hai giá trị độc lập.
5. Docker Desktop phải ở Linux containers và Engine phải chạy:

   ```powershell
   docker version
   docker compose config --quiet
   ```

Module 3 mặc định dùng local model `Qwen/Qwen2.5-7B-Instruct` trên A10G.
Model 7B tốn nhiều thời gian tải và GPU hơn 1.5B, nhưng tuân thủ contract
tóm tắt tiếng Việt và dữ kiện ổn định hơn. One-click runner không cần
`GEMINI_API_KEY` và không gửi transcript tới API LLM bên ngoài.

## Chạy one-click

Kiểm tra toàn bộ command plan mà không upload, tạo GPU hay thay đổi database:

```powershell
.\scripts\run_offline_pipeline.ps1 -DryRun
```

Chạy pipeline với cơ chế resume an toàn:

```powershell
.\scripts\run_offline_pipeline.ps1
```

Nếu đây là lần đầu index sau khi chuyển từ PE-Core sang CLIP ViT-B/32, dùng:

```powershell
.\scripts\run_offline_pipeline.ps1 -ResetIndex
```

`-ResetIndex` truyền `--reset-all` vào Module 7 và xóa/tạo lại schema của cả ba
database. Chỉ dùng khi dữ liệu database hiện tại có thể tái tạo từ artifact.
Lệnh không chạy `docker compose down -v`, nên named volume Docker không bị xóa.

## Tham số của runner

| Tham số | Hành vi |
|---|---|
| `-VolumeName <name>` | Đổi Modal Volume; mặc định `aic-nova-offline-data` |
| `-Force` | Tạo lại output Module 2–6 thay vì resume |
| `-ResetIndex` | Xóa và tạo lại schema Module 7 trước khi index |
| `-SkipIndex` | Chỉ chạy GPU và pull artifact, không chạy Docker Module 7 |
| `-DryRun` | In toàn bộ lệnh, không thay đổi Modal, local data hay Docker |

Ví dụ ép tạo lại artifact nhưng chưa index:

```powershell
.\scripts\run_offline_pipeline.ps1 -Force -SkipIndex
```

## Những việc runner tự động thực hiện

1. Kiểm tra `modal`, `docker`, runner Python và video đầu vào.
2. Tạo Modal Volume nếu chưa tồn tại.
3. Upload `data/raw_videos` và caption tùy chọn.
4. Chạy Module 1–6 tuần tự; lỗi ở một module sẽ dừng ngay pipeline.
5. Commit output sau mỗi Modal job thành công.
6. Tải `/processed` về `data/processed` với chế độ overwrite.
7. Tạo `.env` từ `.env.example` nếu chưa có.
8. Build/start etcd, MinIO, Milvus và Elasticsearch.
9. Chạy indexing bằng image đã khóa Elasticsearch Python client ở major 8.

Các module có resume sẽ bỏ qua artifact hợp lệ khi không truyền `-Force`.
Vì mỗi module commit độc lập, có thể chạy lại cùng một lệnh sau lỗi mà không
mất output của các stage đã hoàn tất.

## Artifact sau khi hoàn tất

```text
data/processed/
├── metadata/
├── keyframes/
├── audio/
├── transcripts/
├── summaries/
├── ocr/
├── object_detection/
└── embeddings/
    ├── visual/
    ├── text_asr/
    ├── text_summary/
    └── text_ocr/
```

SQLite được lưu tại `data/metadata.db`. Milvus, MinIO và Elasticsearch sử dụng
named volume Docker được khai báo trong `docker-compose.yml`.

## Theo dõi và kiểm chứng

Trong lúc Modal chạy:

```powershell
modal app list
modal volume ls aic-nova-offline-data /processed
m-gpux billing usage --days 1 --all
```

Sau khi Module 7 hoàn tất:

```powershell
docker compose ps
docker compose logs --tail 100 elasticsearch milvus-standalone
Get-ChildItem .\data\processed\embeddings\visual\*.parquet
Test-Path .\data\metadata.db
```

Chạy regression repository khi cần xác nhận mã nguồn:

```powershell
.\venv\Scripts\python.exe scripts\run_all_tests.py -q
```

## Dừng database và kiểm soát chi phí

Modal function tự dừng sau từng module. Docker database được giữ chạy để hệ
thống online có thể truy cập. Dừng container nhưng giữ dữ liệu:

```powershell
docker compose down
```

Không thêm `-v` trừ khi thực sự muốn xóa toàn bộ named volume database.

Kiểm tra không còn tài nguyên Modal ngoài dự kiến:

```powershell
modal app list
m-gpux sessions list
m-gpux billing usage --days 1 --all
```

## Xử lý lỗi thường gặp

### Workspace exceeded its spend limit

GPU chưa được tạo. Mở Modal Dashboard, chọn đúng workspace và tăng spend limit.
Sau đó chạy lại runner; không cần upload thủ công lại artifact đã có.

### Pipeline dừng ở một Module Modal

Đọc lỗi cuối trong terminal, sửa nguyên nhân rồi chạy lại cùng command. Không
dùng `-Force` nếu muốn tận dụng resume. Kiểm tra output remote bằng:

```powershell
modal volume ls aic-nova-offline-data /processed
```

### Local chưa thấy artifact

Runner chỉ pull sau khi Module 6 thành công. Có thể tải thủ công:

```powershell
modal volume get --force `
  aic-nova-offline-data `
  /processed `
  .\data\processed
```

### Elasticsearch báo `compatible-with=9`

Rebuild image indexing để nhận dependency `<9`:

```powershell
docker compose build --no-cache indexing
docker compose run --rm indexing python -m src.indexing.cli --force
```

### Muốn xem lệnh chính xác trước khi trả phí

```powershell
.\scripts\run_offline_pipeline.ps1 -DryRun -Force -ResetIndex
```

Dry-run không gọi API tạo tài nguyên, không upload và không thay đổi database.
