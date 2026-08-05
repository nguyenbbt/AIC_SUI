# Module 7: Multi-DB Indexing & Ingestion

Module chốt chặn cuối cùng của luồng Offline. Đọc toàn bộ output từ Module 1–6 và định tuyến dữ liệu vào **3 Database** chuyên biệt theo kiến trúc Polyglot Persistence:

| Database | Vai trò | Dữ liệu |
|---|---|---|
| **Milvus** | Dense vector search | Visual embeddings, ASR text embeddings, Summary embeddings |
| **Elasticsearch** | Full-text search (tiếng Việt) | OCR text, ASR transcript, Video summary |
| **SQLite** | Relational metadata & Object counting | Frame metadata, Object detection results |

## Tính năng nổi bật

- **Dynamic Dimension Detection**: Tự động đọc số chiều vector từ file Parquet đầu tiên — không hard-code dimension.
- **Per-Video Transaction & Rollback**: Mỗi video được xử lý độc lập với cơ chế Delete-then-Insert (idempotent upsert). Nếu bất kỳ DB nào fail giữa chừng, các DB đã insert thành công sẽ được rollback tự động.
- **Graceful Degradation**: Frame không có OCR text hay Object Detection sẽ tự động bị bỏ qua mà không ảnh hưởng đến các DB khác.
- **Vietnamese Full-text Search**: Elasticsearch sử dụng custom analyzer với `icu_tokenizer` + `icu_folding` để hỗ trợ tìm kiếm tiếng Việt không dấu (`nguoi` → `người`).
- **Batch Processing**: Insert dữ liệu theo batch (mặc định 500 records/batch) để tránh tràn RAM.

## Cài đặt & Sử dụng

### Khởi chạy Infrastructure (Docker Compose)

```bash
# Khởi chạy Milvus + Elasticsearch
docker compose up -d milvus-standalone elasticsearch

# Chờ các service healthy (khoảng 30-60 giây)
docker compose ps
```

### Chạy Indexing

**Qua Docker Compose** (khuyên dùng):
```bash
docker compose up indexing
```

**Chạy trực tiếp (Local)**:
```bash
python -m src.indexing.cli \
  --data-dir data/processed \
  --milvus-uri http://localhost:19530 \
  --es-uri http://localhost:9200 \
  --db-uri sqlite:///data/metadata.db \
  --batch-size 500
```

### CLI Arguments

| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `--data-dir` | **(Bắt buộc)** Thư mục gốc chứa toàn bộ dữ liệu đã xử lý | |
| `--milvus-uri` | URI kết nối Milvus | `http://localhost:19530` |
| `--es-uri` | URI kết nối Elasticsearch | `http://localhost:9200` |
| `--db-uri` | URI kết nối SQLite | `sqlite:///data/metadata.db` |
| `--batch-size` | Số lượng record insert trong mỗi batch | `500` |
| `--reset-all` | Xoá trắng toàn bộ 3 DB và tạo lại schema | `False` |
| `--force` | Ép buộc xử lý lại toàn bộ video | `False` |

## Primary Key Strategy

| Đơn vị | Primary Key | Dùng bởi |
|---|---|---|
| Frame | `frame_id` (string) | Milvus (Visual), ES (OCR), SQLite (Metadata, Objects) |
| ASR Interval | `video_id` + `interval_id` | Milvus (ASR), ES (ASR) |
| Video Summary | `video_id` | Milvus (Summary), ES (Summary) |

## Kiểm thử

```bash
# Chạy từ thư mục gốc project
PYTHONPATH=indexing pytest indexing/tests/ -v
```

### Test bao gồm:
1. **Data Loader Tests** (11 tests): Parse JSON/Parquet, dynamic dimension detection, empty text filtering.
2. **Orchestrator Rollback Tests** (4 tests): Giả lập ES failure → Milvus rollback, SQLite failure → cả Milvus + ES rollback, success case, graceful degradation.
