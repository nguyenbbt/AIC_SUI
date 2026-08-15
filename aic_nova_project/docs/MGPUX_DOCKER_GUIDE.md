# Chạy AI Challenge Offline với Docker và M-GPUX

Để chạy toàn bộ Module 1–7 bằng một lệnh, xem
[`OFFLINE_MODAL_DOCKER_PIPELINE.md`](OFFLINE_MODAL_DOCKER_PIPELINE.md).

Tài liệu này áp dụng cho cấu hình đã kiểm tra ngày 2026-08-04:

- M-GPUX CLI `2.9.4` trong `venv`.
- VS Code extension `puxpux.m-gpux@2.10.0`.
- Modal SDK `1.5.2`.
- Docker client `29.6.1`.
- Modal profile đang hoạt động: `nguyenkhoanguyen2006`.

Không lưu Modal token hoặc mật khẩu thật vào repository. Extension và CLI dùng
chung profile trong `~/.modal.toml`.

## 1. Mở terminal đúng môi trường

Từ repository root trong PowerShell:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
& .\venv\Scripts\Activate.ps1
```

Hai biến UTF-8 tránh lỗi `UnicodeEncodeError` khi M-GPUX in bảng có ký tự
Unicode trên Windows. File `.vscode/settings.json` đã đặt sẵn hai biến này cho
terminal mới của workspace.

Kiểm tra CLI và profile:

```powershell
m-gpux info
m-gpux account list
modal profile current
```

Trong VS Code, mở biểu tượng GPU ở Activity Bar hoặc chạy
`M-GPUX: Show Info`. Workspace đã recommend extension `puxpux.m-gpux` trong
`.vscode/extensions.json`.

## 2. Khởi động và kiểm tra Docker Desktop

Docker client đã được cài nhưng Docker Desktop engine phải đang chạy. Mở Docker
Desktop và chờ trạng thái **Engine running**, sau đó kiểm tra:

```powershell
docker version
docker compose config --quiet
```

Nếu `docker version` báo thiếu pipe `dockerDesktopLinuxEngine`, Docker Desktop
chưa chạy hoặc đang ở Windows containers. Chuyển sang Linux containers rồi thử
lại.

## 3. Chạy database và Module 7 bằng Docker local

Tạo file `.env` từ `.env.example` và đổi credential MinIO khi cần:

```powershell
Copy-Item .env.example .env
docker compose up -d --build etcd minio milvus-standalone elasticsearch
docker compose ps
```

Chép hoặc sinh artifact M1-M6 vào `data/processed`, rồi chạy indexing:

```powershell
docker compose run --rm indexing
```

Xem log và dừng stack:

```powershell
docker compose logs -f indexing
docker compose down
```

Không thêm `-v` khi chạy `docker compose down` nếu muốn giữ các named volume
database. `docker compose down -v` sẽ xoá dữ liệu database local.

## 4. Kiểm tra Compose bằng M-GPUX

Luôn chạy check trước; hai lệnh này không deploy và không tạo GPU billing:

```powershell
m-gpux compose check --file .\docker-compose.yml
m-gpux compose sandbox check --file .\docker-compose.yml
```

Stack này có năm service phụ thuộc nhau, vì vậy **Sandbox mode** phù hợp hơn
standard subprocess mode: mỗi service có image riêng, dependency readiness và
tunnel riêng.

Cấu hình repository đã chuẩn bị cho M-GPUX:

- Elasticsearch dùng `indexing/elasticsearch.Dockerfile`; không dùng
  `dockerfile_inline` vì M-GPUX 2.9.4 không build inline Dockerfile.
- `etcd:2379` và `minio:9000` được khai báo bằng `expose` để M-GPUX tạo tunnel.
- `MILVUS_URI` và `ES_URI` nằm trong environment để generator rewrite hostname
  sang tunnel URL.
- Module 7 đọc environment qua `build_parser()`.
- Artifact data dùng `/workspace/data`, tương ứng shared Modal Volume.

## 5. Chuẩn bị data cho M-GPUX Sandbox

M-GPUX dùng Modal Volume ổn định sau cho workspace này:

```text
m-gpux-compose-aic-nova-project-c536a3b21a
```

Upload artifact local trước khi deploy indexing stack:

```powershell
modal volume create m-gpux-compose-aic-nova-project-c536a3b21a
modal volume put m-gpux-compose-aic-nova-project-c536a3b21a .\data /data --force
```

Sau upload, `data/processed` ở local xuất hiện tại
`/workspace/data/processed` trong Sandbox.

## 6. Deploy M-GPUX Sandbox

Lệnh sau tạo tài nguyên Modal và có thể phát sinh chi phí. Chỉ chạy khi static
check ở bước 4 đã sạch:

```powershell
m-gpux compose sandbox up --file .\docker-compose.yml
```

M-GPUX sinh file root `modal_runner.py`; file này đã được `.gitignore` vì là
artifact tạm theo máy/profile. Runner batch M1-M7 do repository quản lý nằm ở
`scripts/offline_modal_runner.py`, nên không bị ghi đè.

Theo dõi và thao tác service:

```powershell
m-gpux compose sandbox ps
m-gpux compose sandbox logs indexing
m-gpux compose sandbox exec indexing
```

Dừng tài nguyên sau khi xong để tránh tiếp tục tính phí:

```powershell
m-gpux compose sandbox down
m-gpux sessions list
m-gpux stop --all
m-gpux billing usage --days 1 --all
```

Lưu ý: M-GPUX Sandbox 2.9.4 mount một shared workspace Volume nhưng không ánh
xạ đầy đủ từng named volume của Docker Compose. Artifact và SQLite ở
`/workspace/data` được giữ; state nội bộ của các database sandbox nên được xem
là tạm thời cho một batch indexing. Dùng Docker local nếu cần database state
lâu dài.

## 7. Chạy từng Module Offline trực tiếp trên Modal

Với các job GPU M1-M6, dùng runner allowlist của repository thay vì Compose:

```powershell
modal run scripts/offline_modal_runner.py `
  --module module1 `
  --arguments="--input /data/raw_videos --output /data/processed"
```

Chọn `module2` đến `module7` và truyền CLI args tương ứng. Runner mount Volume
`aic-nova-offline-data` tại `/data`, không khởi động web service giả và không
dùng `shell=True`.

## 8. Checklist xử lý sự cố

```powershell
# Docker engine
docker version
docker compose config --quiet

# Modal/M-GPUX
modal profile current
m-gpux account list
m-gpux compose sandbox check --file .\docker-compose.yml

# Full regression repository
python scripts/run_all_tests.py -q
```

- Extension không hiện: `Ctrl+Shift+P` → `Developer: Reload Window`.
- CLI và extension lệch minor version vẫn dùng cùng `~/.modal.toml`; ưu tiên
  nâng CLI/extension đồng bộ trước một deployment quan trọng.
- Không chạy được Docker: mở Docker Desktop, chọn Linux containers, kiểm tra
  WSL2 integration.
- Không tìm thấy data trong Sandbox: upload lại Volume theo bước 5 và xác nhận
  remote path là `/data`, tương ứng mount `/workspace/data`.
