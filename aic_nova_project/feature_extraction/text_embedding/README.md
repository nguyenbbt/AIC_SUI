# Module 6: Vietnamese Text Embedding

Module này có chức năng chuyển đổi các dữ liệu văn bản từ ASR, Video Summary, và OCR thành các vector nhúng (embeddings) sử dụng mô hình ngôn ngữ tiếng Việt bản địa. Các vector này sẽ được sử dụng cho tìm kiếm ngữ nghĩa (semantic search) thông qua Vector DB.

## Tính năng nổi bật

- **Sử dụng tiếng Việt bản địa**: Giữ nguyên sắc thái ngữ nghĩa của tiếng Việt, không cần dịch sang tiếng Anh. Mặc định sử dụng mô hình `dangvantuan/vietnamese-embedding`.
- **Hỗ trợ Chunking & Mean-Pooling**: Xử lý hiệu quả các văn bản dài (như Video Summary) bằng cách chia nhỏ (chunking), tính embedding cho từng phần, sau đó gộp lại (mean-pooling) để không làm mất thông tin quan trọng.
- **L2-Normalization**: Tất cả các vector đầu ra đều được chuẩn hóa L2 (norm = 1.0) để tối ưu cho Cosine Similarity.
- **Idempotency (Tính luỹ đẳng)**: Tự động bỏ qua các file đã được xử lý thành công trước đó để tiết kiệm thời gian khi chạy lại pipeline.
- **Xử lý Batch và OOM**: Sử dụng `sentence-transformers` hỗ trợ batch inference hiệu quả trên GPU/CPU.
- **100% Offline (Docker)**: Scripts tải trước weights vào trong Docker Image, đảm bảo quá trình chạy runtime không cần internet.

## Cấu trúc lưu trữ

Kết quả đầu ra được lưu dưới dạng file `.parquet` để tiết kiệm dung lượng và tăng tốc độ đọc cho Milvus:
- `data/embeddings/text_asr/{video_id}.parquet`
- `data/embeddings/text_summary/{video_id}.parquet`
- `data/embeddings/text_ocr/{video_id}.parquet`

## Cài đặt & Sử dụng

### Dùng Docker (Khuyên dùng)

1. **Build image** (sẽ tự động tải model weights về lưu trong image):
```bash
docker build -t text_embedding -f feature_extraction/text_embedding/Dockerfile .
```

2. **Chạy container**:
```bash
docker run --gpus all -v /path/to/data:/data text_embedding \
  --asr-dir /data/processed/transcripts \
  --summary-dir /data/processed/summaries \
  --ocr-dir /data/processed/ocr \
  --output-dir /data/processed/embeddings \
  --batch-size 128
```

### Chạy Local

1. Cài đặt thư viện:
```bash
pip install -r feature_extraction/text_embedding/requirements.txt
```

2. Tải model offline (tuỳ chọn):
```bash
python scripts/download_text_models.py
```

3. Chạy pipeline:
```bash
python -m src.text_embedding.cli \
  --asr-dir data/processed/transcripts \
  --summary-dir data/processed/summaries \
  --ocr-dir data/processed/ocr \
  --output-dir data/processed/embeddings \
  --batch-size 128
```

### CLI Arguments

| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `--asr-dir` | Thư mục chứa JSON của ASR đã clean | |
| `--summary-dir`| Thư mục chứa JSON của Video Summary | |
| `--ocr-dir` | Thư mục chứa JSON của OCR | |
| `--output-dir` | Thư mục gốc để lưu kết quả Parquet | **Bắt buộc** |
| `--model-name` | Tên mô hình HuggingFace | `dangvantuan/vietnamese-embedding` |
| `--batch-size` | Số lượng mẫu chạy trong một batch | `128` |
| `--max-length` | Số lượng token tối đa (cho truncation/chunking)| `256` |
| `--device` | Thiết bị (`cuda` hoặc `cpu`). Tự nhận diện nếu bỏ trống | `None` |
| `--force` | Ghi đè file nếu đã tồn tại | `False` |
