# rag-databrain

企业 RAG + 知识库 Agent。选型见 [docs/architecture.md](docs/architecture.md)。

## 跑起来

本机用 `py` 启动器（Python 3.13）。

```powershell
cd "d:\personal code\rag-databrain"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
uvicorn rag_databrain.main:app --reload
```

- `GET /health` → `{"ok": true}`
- `POST /ask` `{"question":"你好"}` → stub 回答（尚未接模型 / 向量库）
- OpenAPI：http://127.0.0.1:8000/docs
