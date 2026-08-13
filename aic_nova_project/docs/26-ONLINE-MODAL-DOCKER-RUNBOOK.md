# 26 - Chạy Online bằng Docker local và Modal GPU

## 1. Phạm vi và trạng thái

Runbook này dùng kiến trúc sau:

```text
React/Vite :5173
  -> FastAPI :8000
     -> Milvus :19530 + Elasticsearch :9200 + SQLite (local Docker/filesystem)
     -> private Modal Function (L4 GPU)
        -> OpenCLIP ViT-B-32::openai
        -> dangvantuan/vietnamese-embedding@4ab46e...
```

Trạng thái được kiểm chứng ngày 2026-08-12:

| Mode | Trạng thái | Ghi chú |
|---|---|---|
| KIS_TEXT | `CONFIRMED_CODE`; runtime gate đã pass trên dataset hiện tại | Full contract `PASS`, API ready, đủ 7 branch success, UI proxy trả keyframe thật |
| KIS_VIDEO | `CONFIRMED_CODE` | Dùng cùng text pipeline với KIS_TEXT; chưa chạy một truy vấn riêng trong lượt kiểm chứng này |
| TRAKE | `CONFIRMED_CODE`, `NEED_RUNTIME_VERIFICATION` | Dùng cùng Modal OpenCLIP encoder; cần bật flag và chạy real query trước khi tuyên bố ready |
| VQA | `CONFIRMED_CODE`, `NEED_RUNTIME_VERIFICATION` | Cần thêm Qwen/Qwen3.5-4B vLLM service; không được bật VQA khi service chưa sẵn sàng |

Modal app là private Function, không tạo public web endpoint. Client local xác thực
bằng Modal profile hiện hành. App giới hạn một container L4, scale về 0 sau 300
giây idle và cache model trong Modal Volume `aic-nova-online-model-cache`.

## 2. Quy tắc an toàn dữ liệu và chi phí

- Không chạy lại Offline chỉ để khởi động Online.
- Không chạy service `indexing` nếu validator chưa chứng minh dữ liệu thiếu hoặc sai.
- Không dùng `docker compose down -v`; tùy chọn `-v` xóa named volumes.
- `docker compose stop` giữ nguyên Milvus/Elasticsearch/MinIO volumes.
- Không ghi Modal token hoặc API key vào Git. Dùng Modal profile và environment variables.
- `modal deploy` không tự chạy truy vấn GPU. GPU bắt đầu tính phí khi encoder được gọi.
- Deployment có thể giữ nguyên vì container tự scale về 0. Chỉ dùng
  `modal app stop aic-nova-online-encoders` khi muốn xóa deployment và chấp nhận
  phải deploy lại lần sau.

## 3. Chuẩn bị một lần

Chạy từ thư mục chứa `online/`, `retrieval_api/`, `ui/` và `docker-compose.yml`:

```powershell
cd "D:\Project\AI Challenge 2026\aic_nova_project"

.\venv\Scripts\python.exe -m pip install -r online\requirements-runtime.txt
.\venv\Scripts\python.exe -m pip install -r online\requirements-modal.txt
Set-Location ui
npm install
Set-Location ..
```

Kiểm tra đúng Modal account/profile:

```powershell
.\venv\Scripts\modal.exe profile current
```

Kết quả của máy đã kiểm chứng là `nguyenkhoanguyen2006`. Nếu không đúng, dừng
lại và đổi profile trước khi deploy để tránh tính phí nhầm workspace.

## 4. Deploy hai query encoder lên Modal

Windows PowerShell cần UTF-8 để Modal CLI không lỗi `charmap` khi in progress:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

.\venv\Scripts\modal.exe deploy .\scripts\online_modal_encoders.py
```

Kiểm tra deployment:

```powershell
.\venv\Scripts\modal.exe app list
```

App cần xuất hiện với tên `aic-nova-online-encoders`. Source pin các model sau:

```text
ViT-B-32::openai                         -> 512 dimensions
dangvantuan/vietnamese-embedding@4ab46e -> 768 dimensions
```

Không đổi các model này nếu chưa chạy lại Offline embedding/indexing theo một
contract migration riêng.

## 5. Khởi động database Docker mà không reindex

Nếu Docker Desktop chưa chạy và máy không có lệnh `docker desktop start`:

```powershell
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
```

Sau khi `docker info` thành công, chỉ khởi động bốn service hạ tầng:

```powershell
docker compose up -d --wait etcd minio milvus-standalone elasticsearch
docker compose ps -a
```

Không thêm `indexing` vào lệnh trên. Với dữ liệu đã kiểm chứng, các counts đúng là:

```text
videos=2, metadata=720, objects=2999
visual=720, OCR=718, ASR=32, summary=2
```

Nếu counts khác manifest, dừng tại bước validator và điều tra. Không tự reset
volume hoặc chạy `--force`.

## 6. Chạy full contract validator với Modal encoder

Tạo smoke vectors ở thư mục tạm. Công cụ gọi một batch cho OpenCLIP và một batch
cho Vietnamese encoder; cùng text vector được dùng để kiểm tra OCR/ASR/summary:

```powershell
$smokePath = Join-Path $env:TEMP "aic-nova-modal-smoke-vectors.json"

.\venv\Scripts\python.exe -m scripts.generate_online_modal_smoke_vectors `
  --output $smokePath

.\venv\Scripts\python.exe -m online.validate_contract `
  --fail-on-partial `
  --encoder-smoke-json $smokePath
```

Chỉ tiếp tục khi JSON cuối có:

```text
status=PASS
audit_scope=FULL
checks_skipped=[]
encoder.visual_features=PASS
encoder.ocr_features=PASS
encoder.asr_features=PASS
encoder.summary_features=PASS
```

Validator là read-only.

## 7. Chạy KIS_TEXT thật

### Terminal 1 - FastAPI

```powershell
cd "D:\Project\AI Challenge 2026\aic_nova_project"

$env:AIC_ONLINE_ENCODER_BACKEND = "modal"
$env:AIC_ONLINE_MODAL_ENCODER_APP = "aic-nova-online-encoders"
$env:AIC_ONLINE_MODAL_ENCODER_FUNCTION = "encode"
$env:AIC_ONLINE_MODAL_ENVIRONMENT = ""
$env:AIC_ONLINE_MODAL_ENCODER_CACHE_SIZE = "256"
$env:AIC_ONLINE_RETRIEVAL_TIMEOUT_SEC = "180"
$env:AIC_ONLINE_TRAKE_ENABLED = "false"
$env:AIC_ONLINE_VQA_ENABLED = "false"

.\venv\Scripts\python.exe -m uvicorn retrieval_api.main:app `
  --host 127.0.0.1 `
  --port 8000
```

Giữ terminal này mở. Cold start đầu tiên có thể lâu vì Modal tải model; các lần
sau dùng model Volume và text cache.

### Terminal 2 - React UI

```powershell
cd "D:\Project\AI Challenge 2026\aic_nova_project\ui"
npm run dev -- --host 127.0.0.1
```

Mở `http://127.0.0.1:5173`. Vite chuyển `/api/*` sang FastAPI `:8000`.

### Kiểm tra readiness và truy vấn đủ 7 branch

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

Kết quả bắt buộc có `status=ready` và `kis.readiness=ready`.

```powershell
$body = @{
  query = "siêu bão cấp 5 ở Tây Thái Bình Dương"
  mode = "kis_text"
  query_id = "manual-kis-001"
  enabled_branches = @(
    "visual_dense", "ocr_dense", "ocr_bm25",
    "asr_dense", "asr_bm25", "summary_dense", "summary_bm25"
  )
  include_diagnostics = $true
} | ConvertTo-Json -Depth 5

$result = Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/search `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([Text.Encoding]::UTF8.GetBytes($body)) `
  -TimeoutSec 240

$result.candidates[0]
$result.diagnostics.branches
```

Mỗi branch phải là `success`, candidate phải có canonical `frame_id`, ví dụ hậu
tố `_00024_015`, và `/media/keyframes/{frame_id}` phải trả ảnh thật.

## 8. Bật và kiểm tra TRAKE

TRAKE dùng OpenCLIP Modal đã deploy, không cần model thứ ba. Dừng FastAPI bằng
`Ctrl+C`, giữ nguyên Docker và UI, rồi chạy lại Terminal 1 với:

```powershell
$env:AIC_ONLINE_TRAKE_ENABLED = "true"
$env:AIC_ONLINE_VQA_ENABLED = "false"

.\venv\Scripts\python.exe -m uvicorn retrieval_api.main:app `
  --host 127.0.0.1 `
  --port 8000
```

Readiness phải có `trake.enabled=true` và `trake.readiness=ready`.

```powershell
$body = @{
  query_id = "trake-real-001"
  event_texts = @(
    "cơn bão hình thành trên biển",
    "bản đồ dự báo đường đi của bão"
  )
  top_k_videos = 5
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/trake `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([Text.Encoding]::UTF8.GetBytes($body)) `
  -TimeoutSec 240
```

Chỉ ghi nhận TRAKE runtime-ready sau khi result giữ đúng thứ tự event, mọi
transition nằm trong cùng `video_id`, và các `frame_id` mở được bằng `/media/*`.

## 9. Chuẩn bị và kiểm tra VQA

### 9.1 Runtime bắt buộc

VQA hiện dùng `Qwen/Qwen3.5-4B`, revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, qua OpenAI-compatible vLLM tại
`http://localhost:8001/v1`. Adapter chỉ gửi evidence đã retrieve, tối đa 12 ảnh;
không gửi toàn bộ dataset.

`OPEN_QUESTION`: repository chưa phê duyệt một vLLM Docker image tag/digest cụ
thể. Không dùng `latest` trong competition. Chọn và ghi lại một tag/digest đã
được đội benchmark, thay `<PINNED_VLLM_TAG_OR_DIGEST>` bên dưới.

Máy Docker cần NVIDIA GPU runtime. Ví dụ cấu hình local:

```powershell
docker run --rm --gpus all `
  --name aic-nova-qwen-vlm `
  --ipc=host `
  -p 127.0.0.1:8001:8000 `
  -v aic_nova_qwen_hf_cache:/root/.cache/huggingface `
  -v aic_nova_qwen_vllm_cache:/root/.cache/vllm `
  vllm/vllm-openai:<PINNED_VLLM_TAG_OR_DIGEST> `
  Qwen/Qwen3.5-4B `
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a `
  --served-model-name Qwen/Qwen3.5-4B `
  --limit-mm-per-prompt '{"image":12}'
```

Nếu máy local không có NVIDIA GPU, cần triển khai một private Modal vLLM service
và một kết nối authenticated/restricted tương ứng trước; runner Modal encoder
trong mục 4 không phải VLM và không thể trả lời VQA.

Kiểm tra vLLM:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/v1/models
```

### 9.2 Bật VQA trong FastAPI

Dừng FastAPI, sau đó đặt:

```powershell
$env:AIC_ONLINE_ENCODER_BACKEND = "modal"
$env:AIC_ONLINE_VQA_ENABLED = "true"
$env:AIC_ONLINE_QWEN_VLM_AUTO_CONFIGURE = "true"
$env:AIC_ONLINE_QWEN_VLM_BASE_URL = "http://127.0.0.1:8001/v1"
$env:AIC_ONLINE_QWEN_VLM_MODEL = "Qwen/Qwen3.5-4B"
$env:AIC_ONLINE_QWEN_VLM_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
$env:AIC_ONLINE_QWEN_VLM_TIMEOUT_SEC = "15"
$env:AIC_ONLINE_QWEN_VLM_MAX_IMAGE_LONG_EDGE = "768"

.\venv\Scripts\python.exe -m uvicorn retrieval_api.main:app `
  --host 127.0.0.1 `
  --port 8000
```

Readiness phải có `vqa.enabled=true` và `vqa.readiness=ready`.

```powershell
$body = @{
  question_id = "vqa-real-001"
  question = "Bản đồ đang mô tả hiện tượng thời tiết nào?"
  answer_type = "short_text"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/vqa `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([Text.Encoding]::UTF8.GetBytes($body)) `
  -TimeoutSec 240
```

Chỉ ghi nhận VQA runtime-ready khi response có structured status, answer chỉ
dựa trên evidence, `evidence_ids` là subset của request, và UI cho người dùng
chọn một frame được VLM trích dẫn.

## 10. Dừng hệ thống mà giữ dữ liệu

Trong terminal API và UI, dùng `Ctrl+C`. Sau đó có thể dừng container:

```powershell
docker compose stop elasticsearch milvus-standalone minio etcd
```

Named volumes vẫn còn. Modal GPU tự scale về 0 sau 300 giây không có request.

## 11. Xử lý lỗi thường gặp

### `Workspace ... has exceeded its spend limit`

Kiểm tra `modal profile current`, spend limit và workspace trên dashboard. Đổi
profile chỉ khi người sở hữu workspace cho phép. Không retry vòng lặp vô hạn.

### `'charmap' codec can't encode characters`

Đặt `PYTHONUTF8=1` và `PYTHONIOENCODING=utf-8` trong terminal đang chạy Modal CLI.

### Docker daemon/pipe không tồn tại

Khởi động Docker Desktop, đợi `docker info` thành công rồi mới chạy Compose.

### Validator `FAIL` nhưng mọi database check đều pass

Nếu chỉ các check `encoder.*=NOT_RUN`, tạo smoke JSON theo mục 6. Nếu count,
schema, dimension hoặc JOIN fail, không reindex tự động; lưu report và điều tra
đúng resource trước.

### `/health/ready` chưa ready khi cold start

Xem log FastAPI và Modal app logs. Giữ timeout development ở 180 giây. Không
tăng timeout để che `DimensionMismatchError`, model identity mismatch hoặc DB
contract mismatch.

## 12. Tài liệu chính thức

- [Modal deployed functions](https://modal.com/docs/guide/trigger-deployed-functions)
- [Modal autoscaling và scale-to-zero](https://modal.com/docs/guide/scale)
- [Modal GPU](https://modal.com/docs/guide/gpu)
- [Docker Compose up](https://docs.docker.com/reference/cli/docker/compose/up/)
- [Docker volumes](https://docs.docker.com/reference/compose-file/volumes/)
- [vLLM Docker deployment](https://docs.vllm.ai/en/latest/deployment/docker/)
- [vLLM serve arguments](https://docs.vllm.ai/en/latest/cli/serve/)
- [Qwen/Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B)

## 13. Script PowerShell một lệnh

Runner `scripts/run_online_stack.ps1` quản lý toàn bộ Online stack bằng ba thao tác
`Start`, `Stop` và `Status`. Runner không chạy lại Offline, không ghi database, không
reset index và không xóa Docker volume.

### 13.1 Chạy đủ KIS, TRAKE và VQA

Nếu Qwen vLLM đã chạy tại `http://127.0.0.1:8001/v1`:

```powershell
.\scripts\run_online_stack.ps1
```

Lệnh mặc định thực hiện tuần tự:

1. Kiểm tra Python, Modal, Docker, Node/UI và READY manifest.
2. Khởi động Docker Desktop nếu cần, rồi bật etcd, MinIO, Milvus và Elasticsearch.
3. Xác nhận Modal profile `nguyenkhoanguyen2006` và deploy query encoders.
4. Kiểm tra Qwen qua `/v1/models`.
5. Tạo Modal smoke vectors và chạy full read-only contract validator.
6. Bật FastAPI với KIS, TRAKE, VQA; bật Vite UI; chờ cả API và UI proxy `ready`.

Nếu Qwen chưa chạy, truyền một vLLM image đã pin tag hoặc digest để runner tự tạo
container GPU:

```powershell
.\scripts\run_online_stack.ps1 `
  -QwenVllmImage "vllm/vllm-openai:<PINNED_TAG_OR_DIGEST>"
```

Runner từ chối tag `latest`. Repository chưa phê duyệt một vLLM tag/digest mặc định,
vì vậy không được thay placeholder bằng một phiên bản chưa benchmark. Nếu endpoint
không sẵn sàng và không có `-QwenVllmImage`, runner dừng trước khi bật API thay vì báo
sai rằng VQA đã hoạt động.

### 13.2 Chạy ngay KIS và TRAKE, tạm không bật VQA

```powershell
.\scripts\run_online_stack.ps1 -WithoutVQA
```

Chế độ này vẫn dùng Modal GPU cho OpenCLIP và Vietnamese query encoders. Nó chỉ tắt
VQA vì VQA cần Qwen VLM riêng.

### 13.3 Xem trạng thái và dừng an toàn

```powershell
.\scripts\run_online_stack.ps1 -Action Status
.\scripts\run_online_stack.ps1 -Action Stop
```

`Stop` chỉ dừng tiến trình FastAPI/Vite có command line thuộc đúng project, Qwen
container nếu chính runner đã bật nó, và bốn container hạ tầng. Named volumes cùng
Modal deployment được giữ nguyên; Modal GPU tự scale về 0 khi idle.

Log và state tạm nằm tại:

```text
%TEMP%\aic-nova-online-runtime
```

Các tùy chọn `-SkipModalDeploy` và `-SkipContractValidation` chỉ dành cho vòng lặp
phát triển đã được kiểm chứng. Lần chạy đầu hoặc sau khi đổi model/data không nên bỏ
qua hai gate này.
