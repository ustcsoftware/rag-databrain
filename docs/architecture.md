# 架构选型

活文档。改栈只改这里，并同步 `d:\personal code\knowledge-docs\workflow\rag-databrain-架构选型.md`。面试题见 [选型面试题.md](选型面试题.md)。切分、名词、评测见 [rag.md](rag.md)。经典实践见 [面试经典实践.md](面试经典实践.md)。日志和业务库查询见 [日志与库查询.md](日志与库查询.md)。

## 一句话

FastAPI 对外，LangGraph 编排。检索是工具，模型在 loop 里调用直到作答或达到步数上限。文档写入 PostgreSQL，模型和向量走 OpenAI 兼容接口。

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
| 2026-10-21 | loop | `retrieve` 当工具，模型决定再搜或作答；最后一轮禁止工具 | 直线检索→生成不是 Agent | LangGraph 条件边拆节点（和 stream 重复） |
| 2026-10-22 | 消歧 | 检索前一次判断：改写后的问句去检索，或只反问一个缺口 | 真歧义时猜一个意思会答错；生成仍用原话 | 多查询、HyDE、缺槽位表 |
| 2026-10-22 | 扫描件 | 文本层平均每页不足 40 字则渲染，RapidOCR | 图片页和扫描件没有可检索的字 | 本机 Tesseract（要另装中文语言包） |
| 2026-10-23 | 解析 | 版面块 + 结构切分；PDF 用 pdfplumber，扫描件用 OCR 框 | 表格行列和标题路径不能在压成纯文本之后补 | LlamaIndex（版面启发式不够时） |
| 2026-10-23 | 落地 | 没有本轮 retrieve 证据就不把模型输出当答案 | 只改 prompt 拦不住自身知识 | 引用校验 / NLI 对每句找证据 |
| 2026-10-23 | 同步 | 文档哈希跳过、切片复用向量、整份替换删旧版 | 同名每次重嵌又贵又可能留下旧切片 | 任务队列做后台增量；改名追踪 |
| 2026-10-23 | 检索 | 型号和错误码硬约束，BM25 + 余弦，RRF，可选 Cross-Encoder，同源封顶 | 向量近邻会把另一型号、另一个错误码排到前面 | 只取向量前几条；现在不上 OpenSearch / pgvector |
| 2026-10-23 | 会话 | 请求内复制 history，图不挂 checkpointer | 进程内会话表在多 worker 下会串话或丢话 | 全局对话列表；所有人共用一个 thread_id |
| 2026-10-23 | 存储 | PostgreSQL `documents` + `chunks` | 向量索引和 Agent 的 JSON 状态都长在 Postgres 上；MySQL 的向量类型晚，检查点也没有现成实现 | MySQL；继续用 SQLite 文件 |
| 2026-10-24 | 向量库 | 不用 Milvus。向量留在 Postgres，上万再加 pgvector | 和切片同一笔事务；现在全表余弦够用 | Milvus、Qdrant、Chroma |
| 2026-10-24 | 沙箱 | 只放行检索；检索只读切片；未知工具仍有一条结果 | 文档问答没有进程可关。沙箱先是工具边界，和 loop 分开 | 现在不上进程沙箱、执行前审批 |

## 后置

| 层 | 倾向 | 触发条件 |
|---|---|---|
| 向量索引 | 在这份 Postgres 上加 pgvector。全文可换 OpenSearch。融合仍是 RRF | 切片上万，或要在 SQL 里按条件过滤 |
| Milvus | 独立向量服务。文档和向量拆成两笔提交，存储层要双写 | 单机 pgvector 的 HNSW 放不进这台库的内存，或向量查询把库打满。默认 `text-embedding-3-small` 是 1536 维，一条约 6KB；加上图，一百万条约十几 GB，32GB 内存的库还留在 Postgres。到千万级（向量加图过百 GB）再引入 |
| 解析 | LlamaIndex / 版面模型 | 双栏无沟、复杂版式、长乱码文本层，启发式开始砸人 |
| 会话 | Postgres 检查点表 | 对话要跨进程保留 |
| 鉴权 | FastAPI 依赖；过滤放在候选参与排序之前 | 第二个用户 |
| 精排 | Qwen3-Reranker-4B，或少量候选做 LLM 证据核验 | 同一标注集上 BGE 与 Qwen3-0.6B 仍排错复杂问句 |
| 评测 | 固定库、查询、切片和生成模型，比召回、排序、证据覆盖、答案对错、P95 和费用。做法见 [rag.md](rag.md) | 要说明精排比「只向量」多对了多少 |
| 进程沙箱 | 和审批一起，限制会起进程或改文件的工具。文件系统和子进程走同一个执行世界 | 出现检索以外、会执行命令或写数据的工具 |
| 日志与业务行 | 只读 MCP。角色先裁工具，路由再决定本轮渠道。步骤见 [日志与库查询.md](日志与库查询.md) | 问答要查线上日志或业务库 |

## 请求路径

`POST /ingest` 版面解析 + 结构切分 + embedding 写入 PostgreSQL（`DATABASE_URL`）。同名再传：哈希相同则跳过，否则只嵌新切片并整份换掉旧版。`DELETE /documents/{source}` 删文档及切片。旧的 SQLite 文件不再读取，要重新上传。检索仍把切片拉回内存算余弦，切片上万再在这个库上加 pgvector。不用 Milvus；引入条件见上面后置表。

`POST /ask` → 可选 history → 改写或反问 → 反问则本轮结束；否则模型在 loop 里调 retrieve（第一次用改写问句，有上限）。loop 决定还搜不搜；沙箱只放行检索，检索只读切片，未知工具仍有一条结果。retrieve 先按型号和错误码收窄，BM25 与余弦各取一批，RRF 融合；配置了精排模型才做交叉编码，同一来源封顶后交给生成。有检索证据才作答，否则不知道。

`POST /ask/stream` → 同一套检索和 prompt → OpenAI `stream=True` → SSE（`sources` / `delta` / `done` / `error`）。浏览器 EventSource 只能 GET，前端用 `fetch`。
