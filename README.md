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
2. `POST /ask` `{"question":"...","history":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}` → `{answer, sources}`（`history` 可省略）
3. `POST /ask/stream` 同一提问，SSE 按 token 推（`curl -N` 关缓冲）

```powershell
curl.exe -F "file=@readme.md" http://127.0.0.1:8000/ingest
curl.exe -H "Content-Type: application/json" -d "{\"question\":\"这个项目是做什么的\"}" http://127.0.0.1:8000/ask
curl.exe -N -H "Content-Type: application/json" -d "{\"question\":\"这个项目是做什么的\"}" http://127.0.0.1:8000/ask/stream
```

OpenAPI：http://127.0.0.1:8000/docs

精排不在默认安装里。启用：`pip install -e ".[rerank]"`，`.env` 设 `RERANK_MODEL=BAAI/bge-reranker-v2-m3`（对照换 `Qwen/Qwen3-Reranker-0.6B`）。不设则停在关键词和向量的融合名次。
