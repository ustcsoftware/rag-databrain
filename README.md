# rag-databrain

企业 RAG + 知识库 Agent。选型见 [docs/architecture.md](docs/architecture.md)。

## 跑起来

```powershell
cd "d:\personal code\rag-databrain"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
copy .env.example .env
# 填 OPENAI_API_KEY；国内中转再填 OPENAI_BASE_URL
uvicorn rag_databrain.main:app --reload
```

1. `POST /ingest` 上传 `.txt` / `.md` / `.pdf`
2. `POST /ask` `{"question":"..."}` → `{answer, sources}`

```powershell
curl.exe -F "file=@readme.md" http://127.0.0.1:8000/ingest
curl.exe -H "Content-Type: application/json" -d "{\"question\":\"这个项目是做什么的\"}" http://127.0.0.1:8000/ask
```

OpenAPI：http://127.0.0.1:8000/docs
