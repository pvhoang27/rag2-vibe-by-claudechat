# 🔮 RAG Chatbot — Local LLM + ChromaDB + RAGAS

Chatbot hỏi-đáp dựa trên tài liệu, sử dụng **Ollama (local LLM hoàn toàn miễn phí)** + **ChromaDB** + **LangChain** + đánh giá chuyên nghiệp bằng **RAGAS**.

---

## 📐 Kiến trúc

```
User Question
     │
     ▼
[Embedding Model]  ──────────────────────────────────────
     │                                                   │
     ▼                                                   │
[ChromaDB Vector Store] ← similarity search             │
     │                                                   │
     ▼                                                   │
[Top-K Chunks (Context)]                                │
     │                                                   │
     └──────────► [Ollama LLM] ◄──── RAG Prompt ────────┘
                       │
                       ▼
                  [Answer + Sources]
```

**Stack:**
| Layer | Technology |
|---|---|
| LLM | Ollama · `llama3.2:3b` (chạy local, miễn phí) |
| Embedding | Ollama · `nomic-embed-text` |
| Vector DB | ChromaDB (local, persistent) |
| Framework | LangChain + FastAPI |
| Evaluation | RAGAS framework |
| Package Manager | `uv` |

---

## 🗂 Cấu trúc project

```
rag_chatbot/
├── app/
│   ├── api/
│   │   ├── chat.py          # Endpoint hỏi đáp
│   │   ├── ingest.py        # Endpoint upload & ingest tài liệu
│   │   └── evaluation.py    # Endpoint chạy RAGAS evaluation
│   ├── core/
│   │   ├── config.py        # Cấu hình từ .env (Pydantic-Settings)
│   │   ├── dependencies.py  # Dependency injection
│   │   └── logger.py        # Structured logging (Loguru)
│   ├── models/
│   │   └── schemas.py       # Pydantic v2 request/response schemas
│   ├── services/
│   │   ├── vector_store.py  # ChromaDB management
│   │   ├── ingestion.py     # Document loading & chunking
│   │   ├── rag_chain.py     # LangChain RAG pipeline
│   │   └── evaluation.py    # RAGAS evaluation orchestrator
│   └── main.py              # FastAPI app entry-point
├── data/
│   └── raw/                 # Đặt tài liệu của bạn vào đây
├── frontend/
│   └── index.html           # Giao diện chat (mở bằng browser)
├── scripts/
│   ├── generate_sample_data.py  # Tạo dữ liệu mẫu để test
│   └── run_evaluation.py        # CLI đánh giá với rich output
├── tests/
│   └── test_api.py          # Unit + integration tests
├── .env.example             # Template cấu hình
├── pyproject.toml           # Dependencies (uv)
└── README.md
```

---

## 🚀 Hướng dẫn chạy từ A–Z

### Bước 0 — Yêu cầu hệ thống

| Yêu cầu | Tối thiểu | Ghi chú |
|---|---|---|
| RAM | 4 GB | `llama3.2:3b` cần ~2GB RAM |
| Disk | 5 GB | Cho model Ollama |
| OS | Windows 10/11 | Đã test trên Windows |
| Python | 3.10+ | |

---

### Bước 1 — Cài đặt Ollama

Ollama cho phép chạy LLM hoàn toàn local, miễn phí.

1. Tải Ollama tại: **https://ollama.com/download**
2. Cài đặt → mở **Ollama** (để nó chạy ở background)
3. Mở **PowerShell** hoặc **Command Prompt**, chạy:

```powershell
# Pull model LLM (nhỏ, phù hợp máy yếu ~2GB)
ollama pull llama3.2:3b

# Pull model embedding (rất nhẹ ~270MB)
ollama pull nomic-embed-text

# Kiểm tra Ollama đang chạy
ollama list
```

> 💡 **Máy yếu hơn nữa?** Dùng `ollama pull llama3.2:1b` (chỉ ~700MB) rồi cập nhật `OLLAMA_LLM_MODEL=llama3.2:1b` trong `.env`

---

### Bước 2 — Cài đặt `uv`

```powershell
# Cài uv (package manager Python thế hệ mới, nhanh hơn pip nhiều lần)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Khởi động lại terminal, kiểm tra
uv --version
```

---

### Bước 3 — Clone / tạo project

```powershell
# Di chuyển vào thư mục project
cd rag_chatbot

# Tạo virtual environment và cài dependencies
uv sync

# Kích hoạt môi trường ảo
.venv\Scripts\activate
```

> `uv sync` sẽ đọc `pyproject.toml` và cài tất cả dependencies tự động.

---

### Bước 4 — Cấu hình `.env`

```powershell
# Copy file template
copy .env.example .env
```

Mở `.env` bằng Notepad, chỉnh nếu cần (mặc định đã chạy được):

```env
OLLAMA_LLM_MODEL=llama3.2:3b      # đổi thành llama3.2:1b nếu máy yếu
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434
```

---

### Bước 5 — Tạo dữ liệu mẫu (vì bạn chưa có data)

```powershell
uv run python scripts/generate_sample_data.py
```

Script này sẽ tạo 4 file `.txt` về chủ đề AI/ML vào `data/raw/`. Bạn cũng có thể tự thêm file PDF, DOCX của mình vào thư mục đó.

---

### Bước 6 — Chạy server

```powershell
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Bạn sẽ thấy log:
```
Starting RAG Chatbot v0.1.0 [development]
LLM  : llama3.2:3b  @  http://localhost:11434
Embed: nomic-embed-text
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

### Bước 7 — Ingest tài liệu

Mở trình duyệt vào **http://localhost:8000/docs** (Swagger UI), hoặc dùng lệnh curl:

```powershell
# Ingest toàn bộ thư mục data/raw/
curl -X POST http://localhost:8000/ingest/directory
```

Hoặc upload file trực tiếp:
```powershell
curl -X POST http://localhost:8000/ingest/file -F "file=@data/raw/gioi_thieu_ai.txt"
```

---

### Bước 8 — Mở giao diện chat

Mở file `frontend/index.html` bằng trình duyệt (Chrome/Edge).  
Không cần server riêng cho frontend — nó gọi thẳng API backend.

---

### Bước 9 — Thử đặt câu hỏi

Trong giao diện hoặc qua API:

```powershell
curl -X POST http://localhost:8000/chat/query `
  -H "Content-Type: application/json" `
  -d "{\"question\": \"RAG là gì và tại sao cần dùng nó?\"}"
```

---

### Bước 10 — Chạy đánh giá (RAGAS)

```powershell
uv run python scripts/run_evaluation.py
```

Hoặc qua API:
```powershell
curl -X POST http://localhost:8000/eval/run `
  -H "Content-Type: application/json" `
  -d @scripts/eval_payload.json
```

Kết quả được lưu tại `data/eval_results/`.

---

### Bước 11 — Chạy tests

```powershell
uv run pytest tests/ -v
```

---

## 📊 Về đánh giá RAGAS

Project sử dụng **RAGAS** — framework đánh giá RAG chuyên nghiệp, không dùng rule-based if-else.

| Metric | Mô tả | Mục tiêu |
|---|---|---|
| **Faithfulness** | Câu trả lời có căn cứ trong context không? Phát hiện hallucination. | ≥ 0.8 |
| **Answer Relevancy** | Câu trả lời có đúng trọng tâm câu hỏi không? | ≥ 0.8 |
| **Context Recall** | Context retrieve có bao phủ đủ thông tin trong ground truth không? | ≥ 0.7 |
| **Context Precision** | Context có chính xác, ít đoạn nhiễu không? | ≥ 0.7 |

Mỗi metric được tính bởi chính LLM local (Ollama), đảm bảo đánh giá ngữ nghĩa thực sự thay vì string matching.

**Thang điểm:**
- `≥ 0.80` → 🟢 Excellent
- `0.60–0.79` → 🟡 Good  
- `0.40–0.59` → 🟠 Fair
- `< 0.40` → 🔴 Poor

---

## 🔧 Troubleshooting

| Lỗi | Nguyên nhân | Cách fix |
|---|---|---|
| `connection refused :11434` | Ollama chưa chạy | Mở Ollama app |
| `model not found` | Chưa pull model | `ollama pull llama3.2:3b` |
| `Out of memory` | RAM không đủ | Dùng `llama3.2:1b` |
| `No documents found` | Chưa ingest | Chạy `POST /ingest/directory` |
| `chromadb error` | Conflict version | `uv sync --reinstall` |

---

## 🌐 API Endpoints

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |
| POST | `/ingest/file` | Upload & ingest file |
| POST | `/ingest/directory` | Ingest thư mục `data/raw` |
| POST | `/chat/query` | Hỏi đáp RAG |
| GET | `/chat/collection` | Thông tin collection |
| DELETE | `/chat/collection` | Reset collection |
| POST | `/eval/run` | Chạy RAGAS evaluation |
