# rag-databrain

企业 RAG：FastAPI + LangGraph（retrieve → generate）。切片在 `data/kb.sqlite`，模型和向量走 OpenAI 兼容接口。

## 跑

`uvicorn rag_databrain.main:app --reload`。`.env` 要有 `OPENAI_API_KEY`，中转再加 `OPENAI_BASE_URL`。

## 非协商

- `/ask` 一次性 JSON；流式只走 `POST /ask/stream`。不要改成 GET（浏览器 `EventSource` 不能 POST，前端用 `fetch` + `ReadableStream`）。
- 流式不要 `graph.invoke`：invoke 等 generate 整节点结束，出不了 token。检索走 `_hits` / `retrieve()`，生成 `stream=True`。
- 已经开始 SSE 之后失败发 `{"type":"error"}`，不要指望再改 HTTP 状态码。
- `/ask` 和 `/ask/stream` 共用同一套 prompt（`graph._messages`），不要两边各写一份。
- 进模型的资料按 `MAX_CONTEXT_CHARS` 从高相关切片往里装，超窗丢掉后面的；不要把检索到的全部原文硬塞。
- 换模型只改 `.env`，不要为中转再包 SDK。存储只经过 `store.py` / `storage.py`。

活文档：`docs/architecture.md`。SSE 注意点：`d:\personal code\knowledge-docs\workflow\rag-databrain-流式输出SSE.md`。
