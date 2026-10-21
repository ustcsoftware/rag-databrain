import os
from typing import Iterator, TypedDict

from langgraph.graph import END, START, StateGraph

from rag_databrain.embedder import openai_client
from rag_databrain.store import retrieve

_SYSTEM = "根据资料回答。资料没有的就说不知道，不要编。"
_EMPTY = "知识库为空，或没有相关内容。"
# 多捞几条，真正进模型的由窗口裁；4 条 500 字几乎从不触顶。
_RETRIEVE_K = 8


class AgentState(TypedDict):
    question: str
    history: list
    chunks: list[str]
    sources: list[str]
    answer: str


def _window() -> int:
    return int(os.environ.get("MAX_CONTEXT_CHARS", "8000"))


def _turns_chars(turns: list[dict]) -> int:
    return sum(len(t["content"]) for t in turns)


def _working_memory(raw: list, question: str) -> list[dict]:
    # 只留 user/assistant；当前问题不要在 history 里再出现一次。
    turns = []
    for t in raw:
        if not isinstance(t, dict):
            continue
        role, content = t.get("role"), (t.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            turns.append({"role": role, "content": content})
    if turns and turns[-1]["role"] == "user" and turns[-1]["content"] == question:
        turns = turns[:-1]

    n = int(os.environ.get("HISTORY_MAX_TURNS", "6"))
    turns = turns[-max(n, 0) :]

    # 对话最多占窗口的一块，从最早的整句丢，避免闲聊挤掉资料。
    budget = int(_window() * float(os.environ.get("HISTORY_BUDGET_RATIO", "0.3")))
    while len(turns) > 1 and _turns_chars(turns) > budget:
        turns = turns[1:]
    if turns and _turns_chars(turns) > budget:
        last = turns[-1]
        turns = [{"role": last["role"], "content": last["content"][: max(budget, 0)]}]
    return turns


def _standalone_question(question: str, history: list[dict]) -> str:
    # 检索必须用独立问题，「再详细点」会打偏向量；生成仍用原句。失败就退回原句。
    if not history:
        return question

    lines = [f"{t['role']}: {t['content']}" for t in history[-4:]]
    try:
        resp = openai_client().chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            max_tokens=128,
            messages=[
                {
                    "role": "system",
                    "content": "把用户最后一句改写成独立问题，保留原意，不要回答。只输出改写后的问题。",
                },
                {
                    "role": "user",
                    "content": "对话：\n" + "\n".join(lines) + f"\n最后一句：{question}",
                },
            ],
        )
    except Exception:
        return question

    text = (resp.choices[0].message.content or "").strip()
    return text or question


def _fit_context(chunks: list[str], question: str, occupied: int = 0) -> list[str]:
    # 按相关度从前往后装；超窗截最后一条。ponytail: 按字符估，要对齐某家 tokenizer 再上 tiktoken。
    budget = _window() - len(_SYSTEM) - len(question) - occupied - 20
    if budget <= 0:
        return chunks[:1] if chunks else []

    out: list[str] = []
    used = 0
    for c in chunks:
        gap = 2 if out else 0
        room = budget - used - gap
        if room <= 0:
            break
        if len(c) <= room:
            out.append(c)
            used += gap + len(c)
            continue
        out.append(c[:room])
        break
    return out


def _prepare(question: str, raw_history: list) -> tuple[list[str], list[str], list[dict]]:
    # /ask 和 /ask/stream 共用：工作记忆 → 改写检索 → 资料填剩余窗口。
    history = _working_memory(raw_history, question)
    query = _standalone_question(question, history)
    rows = retrieve(query, k=_RETRIEVE_K)
    chunks = _fit_context(
        [h["text"] for h in rows], question, occupied=_turns_chars(history)
    )
    sources = list(dict.fromkeys(h["source"] for h in rows[: len(chunks)]))
    return chunks, sources, history


def _messages(question: str, chunks: list[str], history: list[dict]) -> list[dict]:
    # invoke 和 stream 必须同一套 prompt，否则两边回答会漂。
    context = "\n\n".join(chunks)
    return [
        {"role": "system", "content": _SYSTEM},
        *history,
        {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
    ]


def retrieve_node(state: AgentState) -> dict:
    chunks, sources, history = _prepare(state["question"], state.get("history") or [])
    return {"chunks": chunks, "sources": sources, "history": history}


def generate_node(state: AgentState) -> dict:
    if not state["chunks"]:
        return {"answer": _EMPTY}

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    resp = openai_client().chat.completions.create(
        model=model,
        messages=_messages(state["question"], state["chunks"], state.get("history") or []),
    )
    return {"answer": resp.choices[0].message.content or ""}


def stream_ask(question: str, history: list | None = None) -> Iterator[tuple[str, object]]:
    # 先把改写和检索跑完再写 SSE 头：缺 key 仍是 503，和 /ask 一致。
    chunks, sources, history = _prepare(question, history or [])
    if not chunks:
        yield "sources", sources
        yield "delta", _EMPTY
        return

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    stream = openai_client().chat.completions.create(
        model=model,
        messages=_messages(question, chunks, history),
        stream=True,
    )
    try:
        yield "sources", sources
        for chunk in stream:
            # 部分中转会夹空 choices，不能直接取 [0]。
            if not chunk.choices:
                continue
            text = chunk.choices[0].delta.content
            if text:
                yield "delta", text
    finally:
        stream.close()


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


graph = build_graph()
