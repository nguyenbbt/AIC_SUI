# Modal backup → Module 7 → Online runbook

## Mục tiêu

Dùng trực tiếp `E:\DATA AIC\modal_backup_safe` làm storage root của Phase
Online mà không sao chép 293.427 keyframe hoặc 873 video. Các artifact M1–M6
được chiếu vào contract `processed/` bằng Windows directory junction.

```text
E:\DATA AIC\modal_backup_safe
├── processed\                 # materialized view bằng junction
├── object_detection\         # output M5 ổn định sau khi tải xong
├── metadata.db                # M7 tạo
├── databases\                 # Docker/M7 tạo
└── .migration\backup-online-layout.json
```

`scripts/prepare_backup_online_root.ps1` không copy, move hoặc xóa artifact.
Nếu đường dẫn đích đã tồn tại nhưng không phải junction đúng target, script
fail closed và không ghi đè.

## Trạng thái đã xác minh ngày 2026-08-21

- 873 metadata JSON.
- 873 thư mục keyframe, tổng cộng 293.427 WebP.
- 873 visual embedding Parquet.
- 873 cleaned transcript và 873 summary JSON.
- 873 OCR JSON.
- 873 Parquet cho từng nhóm `text_asr`, `text_ocr`, `text_summary`.
- 873 raw video tại `E:\DATA AIC\raw_videos`.
- Object Detection chưa hoàn tất nên `processed\object_detection`,
  `metadata.db` và READY manifest chưa tồn tại.
- `.env` chỉ được cập nhật:

```dotenv
AIC_LOCAL_DATA_ROOT=E:\DATA AIC\modal_backup_safe
```

Do chưa có M5 hoàn chỉnh và READY manifest, M7/Online phải tiếp tục bị chặn.

## Kiểm tra hoặc dựng lại materialized view

Dry-run:

```powershell
.\scripts\prepare_backup_online_root.ps1 `
  -BackupRoot "E:\DATA AIC\modal_backup_safe" `
  -VideosRoot "E:\DATA AIC\raw_videos" `
  -ExpectedVideoCount 873 `
  -ExpectedKeyframeCount 293427 `
  -UpdateEnv `
  -DryRun
```

Materialize thật, idempotent:

```powershell
.\scripts\prepare_backup_online_root.ps1 `
  -BackupRoot "E:\DATA AIC\modal_backup_safe" `
  -VideosRoot "E:\DATA AIC\raw_videos" `
  -ExpectedVideoCount 873 `
  -ExpectedKeyframeCount 293427 `
  -UpdateEnv
```

## Sau khi Object Detection trên Modal hoàn tất

Tải M5 vào thư mục ổn định. Truyền backup root làm destination để tránh tạo
thêm cấp `object_detection\object_detection`:

```powershell
$profile = "TEN_PROFILE_MODAL_DANG_CHAY"
$volume = "aic-nova-btc-object-data"
$backupRoot = "E:\DATA AIC\modal_backup_safe"

modal profile activate $profile
modal volume get --force `
  $volume `
  /processed/object_detection `
  $backupRoot
```

Nếu tải bị ngắt, chạy lại đúng lệnh trên để resume/ghi bổ sung. Không tự tạo
junction Object Detection trước khi gate dưới đây PASS:

```powershell
.\scripts\prepare_backup_online_root.ps1 `
  -BackupRoot "E:\DATA AIC\modal_backup_safe" `
  -VideosRoot "E:\DATA AIC\raw_videos" `
  -ExpectedVideoCount 873 `
  -ExpectedKeyframeCount 293427 `
  -RequireObjects `
  -UpdateEnv
```

`-RequireObjects` parse đủ 873 JSON, kiểm tra `video_id`, `frames`, `frame_id`
và `objects`, đối chiếu tập ID với metadata rồi mới tạo:

```text
processed\object_detection → ..\object_detection
```

## Chạy Module 7 bằng Docker local

Chỉ chạy sau thông báo `M5 GATE: PASS`. Không dùng `--reset-all` mặc định để
tránh xóa database đã index nếu đây là một lần resume.

```powershell
$storageRoot = "E:\DATA AIC\modal_backup_safe"
$env:AIC_LOCAL_DATA_ROOT = $storageRoot.Replace("\", "/")

New-Item -ItemType Directory -Force `
  "$storageRoot\databases\etcd", `
  "$storageRoot\databases\minio", `
  "$storageRoot\databases\milvus", `
  "$storageRoot\databases\elasticsearch" | Out-Null

docker compose config --quiet
docker compose down --remove-orphans
docker compose up -d --build --wait `
  etcd minio milvus-standalone elasticsearch
docker compose build indexing

docker compose run --rm indexing `
  python -m src.indexing.cli --force

docker compose run --rm indexing `
  python -m src.indexing.publish_cli `
  --data-dir /workspace/data/processed `
  --dataset-id aic2026-btc-full-v1 `
  --manifest-path /workspace/data/processed/dataset-manifest.json `
  --building-manifest-path /workspace/data/processed/dataset-manifest.building.json

.\venv\Scripts\python.exe -m scripts.btc_storage_manager `
  validate-stage `
  --candidate-root $storageRoot
```

Không chạy `docker compose down -v`: database state đang nằm trên SSD và phải
được giữ để Phase Online sử dụng.

## Bật toàn bộ Phase Online

Sau khi publisher tạo READY manifest và `validate-stage` PASS:

```powershell
.\scripts\run_online_stack.ps1 `
  -Action Start `
  -StorageRoot "E:\DATA AIC\modal_backup_safe"
```

Kiểm tra hoặc dừng:

```powershell
.\scripts\run_online_stack.ps1 -Action Status `
  -StorageRoot "E:\DATA AIC\modal_backup_safe"

.\scripts\run_online_stack.ps1 -Action Stop `
  -StorageRoot "E:\DATA AIC\modal_backup_safe"
```

Online chỉ đọc database của M7; không đọc trực tiếp JSON/Parquet để thực hiện
business retrieval. Các junction chỉ phục vụ M7, validator, media resolver và
debugging đúng contract.

## Gate lỗi thường gặp

- `M5 GATE: WAITING`: chưa đủ 873 Object Detection JSON; không chạy M7.
- `Refusing to overwrite existing non-matching path`: có thư mục thật hoặc
  junction sai tại đích; dừng và kiểm tra, không xóa tự động.
- `Configured AIC_LOCAL_DATA_ROOT is unavailable`: SSD E: mất kết nối; runner
  fail closed, không fallback về D:.
- Thiếu READY manifest hoặc `metadata.db`: M7 chưa publish thành công; không bật
  Online.
- Docker Desktop lỗi: sửa Docker rồi chạy lại riêng M7; không chạy lại GPU M1–M6.

