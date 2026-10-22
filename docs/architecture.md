# 架构选型

活文档。改栈只改这里，并同步 `d:\personal code\knowledge-docs\workflow\rag-databrain-架构选型.md`。面试题见 [选型面试题.md](选型面试题.md)。切分见 [切分问题.md](切分问题.md)。经典实践见 [面试经典实践.md](面试经典实践.md)。

## 一句话

FastAPI 对外，LangGraph 编排。检索是工具，模型循环调用直到作答或达到步数上限。文档写入 `data/kb.sqlite`，模型和向量走 OpenAI 兼容接口。

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
| 2026-10-21 | 上下文 | 检索多捞、按 `MAX_CONTEXT_CHARS` 从高相关往里装 | 中转窗口常小于 128k，不能把切片全塞进去 | tiktoken 按模型精确数 token |
| 2026-10-21 | 工作记忆 | 请求带 history：滑动窗口 + 预算上限 | 单轮问答接不上「它 / 再详细点」 | 服务端会话表、长期记忆、对话摘要 |
| 2026-10-21 | 循环 | `retrieve` 当工具，模型决定再搜或作答；最后一轮禁止工具 | 直线检索→生成不是 Agent | LangGraph 条件边拆节点（和 stream 重复） |
| 2026-10-22 | 消歧 | 检索前一次判断：改写后的问句去检索，或只反问一个缺口 | 真歧义时猜一个意思会答错；生成仍用原话 | 多查询、HyDE、缺槽位表 |
| 2026-10-22 | 扫描件 | 文本层平均每页不足 40 字则渲染，RapidOCR | 图片页和扫描件没有可检索的字 | 本机 Tesseract（要另装中文语言包） |

## 后置

| 层 | 倾向 | 触发条件 |
|---|---|---|
| 向量库 | Chroma → Qdrant（已有 Postgres 则 pgvector） | chunk 上万，或要 metadata 过滤 |
| 解析 | LlamaIndex | 双栏、表格、长乱码文本层开始砸人 |
| 会话 / 元数据 | Postgres | 要落库的文档清单或对话历史 |
| 鉴权 | FastAPI 依赖 | 第二个用户 |

## 请求路径

`POST /ingest` 切分 + embedding 写入 `data/kb.sqlite`。

`POST /ask` → 可选 history → 改写或反问 → 反问则本轮结束；否则模型循环调 retrieve（第一次用改写问句，有上限）→ 最后一轮作答。

`POST /ask/stream` → 同一套检索和 prompt → OpenAI `stream=True` → SSE（`sources` / `delta` / `done` / `error`）。浏览器 EventSource 只能 GET，前端用 `fetch`。
