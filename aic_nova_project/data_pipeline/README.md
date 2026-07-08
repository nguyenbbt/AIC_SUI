# Shot Detection & Keyframe Extraction Module

Module xử lý tiền kỳ cho hệ thống trợ lý ảo truy xuất video (AI Challenge 2026).
Nhiệm vụ chính: Tách video thành các shots bằng TransNetV2, trích xuất chính xác 3 keyframes cho mỗi shot tại các vị trí [0.15, 0.50, 0.85] và lưu dưới định dạng WebP kèm metadata chi tiết (JSON và Parquet).

## Cấu trúc thư mục
- `src/shot_keyframe/`: Mã nguồn chính của module.
- `tests/`: Các unit test và end-to-end test giả lập video.
- `requirements.txt`: Các thư viện cần thiết.
- `download_weights.py`: Script tự động tải weights của TransNetV2.
- `Dockerfile`: File để build Docker image tự chứa (không cần mạng lúc runtime).

## Yêu cầu hệ thống
- Python 3.10+
- (Tùy chọn) GPU hỗ trợ CUDA để chạy inference TransNetV2 nhanh hơn.

## Hướng dẫn sử dụng Local (Không dùng Docker)

1. Cài đặt thư viện:
   ```bash
   pip install -r requirements.txt
   ```

2. Chạy pipeline:
   ```bash
   python -m data_pipeline.shot_keyframe.cli --input /path/to/raw_videos --output /path/to/output --workers 4
   ```
   **Các tham số:**
   - `--input`: Thư mục chứa video gốc (tìm đệ quy các file `.mp4`, `.mkv`, `.avi`, `.webm`).
   - `--output`: Thư mục đích.
   - `--workers`: Số tiến trình chạy song song (mặc định: 1).
   - `--device`: `cpu` hoặc `cuda` (mặc định sẽ tự động nhận diện).
   - `--quality`: Chất lượng ảnh WebP (mặc định: 90).

## Hướng dẫn build & chạy bằng Docker

Môi trường Docker đảm bảo khả năng chạy offline hoàn toàn (cần tải file weights trước khi build).

1. Tải pretrained weights:
   ```bash
   python download_weights.py
   ```
   *Lệnh này sẽ tạo thư mục `weights/` và lưu file `transnetv2-pytorch-weights.pth` vào đó.*

2. Build Docker Image:
   ```bash
   docker build -t shot-keyframe .
   ```

3. Chạy container:
   **Chạy bằng CPU:**
   ```bash
   docker run -v /absolute/path/to/raw_videos:/data/raw_videos -v /absolute/path/to/output:/data/output shot-keyframe --input /data/raw_videos --output /data/output --workers 4
   ```

   **Chạy bằng GPU (nếu có NVIDIA Docker):**
   ```bash
   docker run --gpus all -v /absolute/path/to/raw_videos:/data/raw_videos -v /absolute/path/to/output:/data/output shot-keyframe --input /data/raw_videos --output /data/output --device cuda
   ```

## Cơ chế Resume
Hệ thống có cơ chế kiểm tra `metadata/{video_id}.json`. Nếu file này tồn tại và hợp lệ, hệ thống sẽ bỏ qua bước xử lý lại video đó, giúp bạn an toàn chạy lại lệnh khi bị gián đoạn.

## Testing
Bạn có thể tự chạy kiểm thử (bao gồm unit test và giả lập video end-to-end) bằng lệnh:
```bash
pytest tests/ -v
```
