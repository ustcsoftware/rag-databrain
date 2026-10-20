# 架构选型

活文档。改栈只改这里，并同步 `d:\personal code\knowledge-docs\workflow\rag-databrain-架构选型.md`。

## 一句话

FastAPI 对外，LangGraph 编排 Agent。向量库、LLM、入库、鉴权后置。

## 已定

| 日期 | 层 | 选择 | 原因 | 替代方案（未选） |
|---|---|---|---|---|
| 2026-10-20 | 语言 | Python 3.11+ | RAG/Agent 生态 |  |
| 2026-10-20 | API | FastAPI | async + OpenAPI | Flask / Django |
| 2026-10-20 | Agent | LangGraph | 有状态图，后续 HITL | CrewAI / 纯 LCEL |
| 2026-10-20 | 布局 | `src/rag_databrain` + `pyproject.toml` | 最小可安装包 | 空目录分层 |

## 后置（有真实文档或流量再开一行）

| 层 | 倾向 | 触发条件 |
|---|---|---|
| 向量库 | 本地 Chroma → 企业 Qdrant | 第一条文档要检索 |
| 解析 | pypdf + 固定 chunk | PDF/表格开始砸人再考虑 LlamaIndex |
| LLM | OpenAI 兼容 `base_url` | `/ask` 不再 stub |
| 会话 / 元数据 | Postgres | 要落库 |
| 鉴权 | FastAPI 依赖 | 第二个用户 |

## 请求路径

`POST /ask` → FastAPI → LangGraph。v0 只有 stub 节点。
