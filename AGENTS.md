# rag-databrain

企业 RAG：FastAPI + LangGraph。切片在 `data/kb.sqlite`，模型和向量走 OpenAI 兼容接口。检索是工具，有步数上限的循环。

## 跑

`uvicorn rag_databrain.main:app --reload`。`.env` 要有 `OPENAI_API_KEY`，中转再加 `OPENAI_BASE_URL`。

## 非协商

- `/ask` 一次性 JSON；流式只走 `POST /ask/stream`。不要改成 GET（浏览器 `EventSource` 不能 POST，前端用 `fetch` + `ReadableStream`）。
- 流式不要 `graph.invoke`：invoke 等整段循环结束，出不了 token。token 只在最后一轮 `stream=True`。
- 已经开始 SSE 之后失败发 `{"type":"error"}`，不要指望再改 HTTP 状态码。
- `/ask` 和 `/ask/stream` 都走 `stream_ask`，不要两边各写一套循环。
- 进模型的资料按 `MAX_CONTEXT_CHARS` 从高相关切片往里装，超窗丢掉后面的；不要把检索到的全部原文硬塞。
- 工作记忆只走请求里的 `history`（user/assistant）。最近 N 轮、最多占窗口 30%。不要把聊天记录写进知识库表。
- Agent 循环在 `stream_ask`：模型调 `retrieve`，最后一轮禁止工具。不要退回「检索一次就生成」。相同 query 不重复搜。
- 作答必须有本轮 retrieve 到的资料。没有证据不要把模型正文发给用户，只回不知道。消歧反问除外。对话不能当资料。
- 用户问题先消歧：能补成唯一问句就改写，第一次 `retrieve` 用这句，生成仍用原话；几种读法会搜到不同资料就只反问一个缺口，本轮不检索。消歧结果解析失败就走原来的循环。不要上多查询或 HyDE。
- 换模型只改 `.env`，不要为中转再包 SDK。存储只经过 `store.py` / `storage.py`。
- 解析先拆成段落/表格/图片块再切。表格保留行列，跨页先拼再切；不要把版面压成纯文本滑窗。

活文档：`docs/architecture.md`。SSE 注意点：`d:\personal code\knowledge-docs\workflow\rag-databrain-流式输出SSE.md`。
