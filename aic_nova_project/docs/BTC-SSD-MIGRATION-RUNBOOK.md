# BTC SSD migration và Offline/Online runbook

Tài liệu này là hướng dẫn chuẩn để giữ toàn bộ raw data, processed artifacts,
SQLite và database state trên SSD `E:`. Không chạy cleanup legacy hoặc upload
Modal có phí nếu gate tương ứng chưa được phê duyệt.

## Storage contract

```dotenv
AIC_LOCAL_DATA_ROOT=E:/DATA AIC
AIC_ONLINE_DATA_ROOT=E:/DATA AIC/processed
AIC_ONLINE_SQLITE_PATH=E:/DATA AIC/metadata.db
AIC_ONLINE_DATASET_MANIFEST_PATH=E:/DATA AIC/processed/dataset-manifest.json
```

Runner áp dụng precedence sau:

```text
-StorageRoot > process environment > .env > <project>/data
```

Fallback về `<project>/data` chỉ xảy ra khi không có cấu hình explicit. Nếu đã
cấu hình E: nhưng ổ không tồn tại, runner dừng ngay và không fallback sang D:.
Không copy `.env.example` đè lên `.env` và không in toàn bộ `.env` ra log.

## Trạng thái migration đã xác minh ngày 2026-08-15

- Rollback Docker legacy: `PASS`, `audit_scope=FULL`.
- Dataset rollback: 2 videos, 720 metadata, 2.999 objects.
- Bốn legacy named volumes vẫn tồn tại và không bị đổi tên/xóa.
- Raw BTC đã chuyển bằng same-volume rename vào `E:\DATA AIC\raw_videos`.
- 14 batch, 873 video, 82.984.040.674 bytes.
- 82,984 GB thập phân, tương đương 77,285 GiB.
- SHA-256 aggregate trước/sau migration trùng nhau.
- Journal nằm tại `E:\DATA AIC\.migration\raw-layout.json`.
- Inventory trước/sau nằm trong `E:\DATA AIC\.migration`.

## Config gates không làm thay đổi dữ liệu

```powershell
docker compose config --quiet
docker compose -f docker-compose.yml `
  -f docker-compose.rollback.yml config --quiet
```

Compose chính bind toàn bộ state sang `${AIC_LOCAL_DATA_ROOT}`. Rollback override
dùng đúng bốn external volume legacy và bind dataset D: vào `/workspace/data`.
Legacy, candidate và canonical luôn chạy tuần tự vì dùng chung container names và
host ports.

## Kiểm tra rollback legacy

Xem trước mà không gọi Docker:

```powershell
.\scripts\test_legacy_rollback.ps1 -DryRun
```

Chạy smoke thật, không xóa volume:

```powershell
.\scripts\test_legacy_rollback.ps1
```

Script thực hiện `down` không `-v`, start rollback stack, kiểm tra READY manifest,
fingerprint, full joins và `audit_scope=FULL`, sau đó lại `down` không `-v` và
xác minh đủ bốn volume.

## Raw migration có journal

Lệnh sau có thể resume nếu move bị ngắt. Nó fail closed nếu một batch xuất hiện ở
cả source và destination hoặc biến mất khỏi cả hai nơi:

```powershell
.\venv\Scripts\python.exe -m scripts.btc_storage_manager `
  migrate-raw `
  --storage-root "E:\DATA AIC" `
  --expected-batches 14
```

Không chạy lại mù và không xóa journal. Công cụ tái sử dụng pre-inventory đã ghi,
di chuyển batch còn ở source, rồi hash lại toàn bộ destination.

## Vertical slice một video

Dry-run không tạo hardlink, không upload, không chạy GPU và không đổi Docker:

```powershell
.\scripts\run_offline_pipeline.ps1 `
  -DryRun `
  -StorageRoot "E:\DATA AIC" `
  -DatasetId "btc-slice-L21-V001" `
  -SliceVideoId "L21_V001" `
  -VolumeName "aic-nova-btc-slice-data"
```

Sau khi phê duyệt chi phí upload/GPU riêng, bỏ `-DryRun` và thêm
`-ApprovePaidUpload`:

```powershell
.\scripts\run_offline_pipeline.ps1 `
  -StorageRoot "E:\DATA AIC" `
  -DatasetId "btc-slice-L21-V001" `
  -SliceVideoId "L21_V001" `
  -VolumeName "aic-nova-btc-slice-data" `
  -ApprovePaidUpload
```

Slice input được tạo trong `E:\DATA AIC\.staging\inputs` bằng hardlink nếu hệ
thống file hỗ trợ. Output được pull vào
`E:\DATA AIC\.staging\dataset-btc-slice-L21-V001\processed`. Runner không publish
thẳng vào canonical `processed`.

`AIC_MODAL_DATA_VOLUME` được runner đặt theo `-VolumeName`, nên Modal module và
Modal CLI luôn mount cùng một volume. Remote verifier CPU so khớp count, bytes và
aggregate SHA-256 trước khi module GPU đầu tiên chạy. Nếu volume có file thừa,
verifier dừng pipeline.

Nếu M1–M6 và artifact pull đã xong nhưng Docker Desktop chưa chạy hoặc indexing
validation lỗi, không gọi lại GPU. Khởi động Docker Desktop rồi resume từ staging:

```powershell
.\scripts\run_offline_pipeline.ps1 `
  -StorageRoot "E:\DATA AIC" `
  -DatasetId "btc-slice-L21-V001" `
  -IndexStagedOnly
```

`-IndexStagedOnly` không gọi Modal, không upload và không pull lại artifact. Nó
chỉ chạy candidate Docker indexing, READY publish và full contract validation.

## Promotion candidate

Chỉ promotion sau khi candidate indexing và full contract validation đều PASS:

```powershell
.\scripts\promote_btc_candidate.ps1 `
  -StorageRoot "E:\DATA AIC" `
  -CandidateRoot "E:\DATA AIC\.staging\dataset-btc-slice-L21-V001"
```

Script `down` candidate không `-v`, dùng journal và same-volume rename cho
`processed`, `metadata.db`, `databases`, start canonical E: và validate lại. Nếu
validation lỗi, script đảo journal rồi start rollback stack D:.

Vertical slice chỉ xác minh kiến trúc. Nó không mở cleanup gate mặc định.

## Full 873-video run

Full upload, Modal storage và GPU cần một phê duyệt chi phí riêng:

```powershell
.\scripts\run_offline_pipeline.ps1 `
  -StorageRoot "E:\DATA AIC" `
  -DatasetId "aic2026-btc-full-v1" `
  -VolumeName "aic-nova-btc-data" `
  -ApprovePaidUpload
```

Các chế độ upload:

- Mặc định: resume theo relative path, size và SHA-256.
- `-SkipUpload`: không gửi file nhưng vẫn bắt buộc remote inventory PASS.
- `-ForceUpload`: coi toàn bộ local inventory là changed và gửi lại toàn bộ.
- Có changed bytes nhưng thiếu `-ApprovePaidUpload`: dừng trước upload.

Trước full run, free-space policy bảo thủ là khoảng 251,85 GiB free. Đây là
ngưỡng dung lượng tăng thêm/staging, không phải tổng footprint raw.

## Cleanup legacy — mặc định bị khóa

Chỉ mở sau full 873-video canonical PASS, trừ khi có xác nhận rõ rằng slice một
video được chấp nhận làm dataset thay thế. Trước tiên chạy không có confirmation
để xem chính xác targets; script sẽ dừng ở destructive gate:

```powershell
.\scripts\cleanup_legacy_test_data.ps1 `
  -StorageRoot "E:\DATA AIC" `
  -ExpectedTestVideoIds "<test-video-id-1>","<test-video-id-2>"
```

Sau khi kiểm tra IDs và targets, mới chạy lại với:

```powershell
.\scripts\cleanup_legacy_test_data.ps1 `
  -StorageRoot "E:\DATA AIC" `
  -ExpectedTestVideoIds "<test-video-id-1>","<test-video-id-2>" `
  -ConfirmDestructiveCleanup
```

Modal test volume chỉ được xóa khi có inventory file chứng minh đúng hai test
IDs, đúng environment và có thêm `-DeleteModalTestVolume`. Script dùng bốn lệnh
`docker volume rm` explicit; không dùng `down -v`, không prune image và không
prune build cache.

## Regression gates

```powershell
.\venv\Scripts\python.exe -m pytest `
  -p no:cacheprovider --import-mode=importlib `
  tests\test_btc_storage_manager.py `
  tests\test_docker_contracts.py `
  tests\test_storage_migration_scripts.py `
  tests\test_modal_runner_contract.py `
  tests\test_offline_one_click_script.py `
  tests\test_online_one_click_script.py `
  feature_extraction\asr_transcript\tests\test_pipeline.py -q

.\venv\Scripts\python.exe -m scripts.run_all_tests
```

Không commit/push, xóa legacy data hoặc chạy full paid workload chỉ vì các test
code PASS; mỗi action đó vẫn có gate riêng.
