# 架构选型

活文档。改栈只改这里，并同步 `d:\personal code\knowledge-docs\workflow\rag-databrain-架构选型.md`。

## 一句话

FastAPI 对外，LangGraph 编排（检索 → 生成）。文档和切片写入 `data/kb.sqlite`，用 OpenAI 兼容接口做 embedding 和回答。

## 已定

| 日期 | 层 | 选择 | 原因 | 替代方案（未选） |
|---|---|---|---|---|
| 2026-10-20 | 语言 | Python 3.11+ | RAG/Agent 生态 |  |
| 2026-10-20 | API | FastAPI | async + OpenAPI | Flask / Django |
| 2026-10-20 | Agent | LangGraph | 有状态图，后续 HITL | CrewAI / 纯 LCEL |
| 2026-10-20 | 布局 | `src/rag_databrain` + `pyproject.toml` | 最小可安装包 | 空目录分层 |
| 2026-10-20 | 存储 | SQLite `documents` + `chunks` | 标准库，同名覆盖；换 Postgres 只改本层 | jsonl |
| 2026-10-20 | 检索 | 全表余弦扫描 | chunk 少时够用 | 现在不上 Chroma |
| 2026-10-20 | 解析 | pypdf + 500 字切分、80 字重叠 | 够 txt/md/pdf | LlamaIndex |
| 2026-10-20 | LLM / embedding | OpenAI 兼容 `base_url` | 换中转/Azure/vLLM 只改环境变量 | 绑定某一家 SDK |
| 2026-10-21 | 流式 | `POST /ask/stream` SSE，token 来自 OpenAI `stream=True` | `graph.invoke` 等整节点，出不了 token | LangGraph `astream_events`（本仓库 generate 不是 LC 模型） |

## 后置

| 层 | 倾向 | 触发条件 |
|---|---|---|
| 向量库 | Chroma → Qdrant（已有 Postgres 则 pgvector） | chunk 上万，或要 metadata 过滤 |
| 解析 | LlamaIndex | PDF/表格开始砸人 |
| 会话 / 元数据 | Postgres | 要落库的文档清单或对话历史 |
| 鉴权 | FastAPI 依赖 | 第二个用户 |

## 请求路径

`POST /ingest` 切分 + embedding 写入 `data/kb.sqlite`。

`POST /ask` → FastAPI → LangGraph（retrieve → generate）→ embedding 检索 + chat。

`POST /ask/stream` → 同一套检索和 prompt → OpenAI `stream=True` → SSE（`sources` / `delta` / `done` / `error`）。浏览器 EventSource 只能 GET，前端用 `fetch`。
