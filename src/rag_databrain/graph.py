import json
import os
from typing import Iterator, TypedDict

from langgraph.graph import END, START, StateGraph

from rag_databrain.embedder import openai_client
from rag_databrain.store import retrieve

_SYSTEM = (
    "需要知识库时先调用 retrieve，query 写成独立完整问句。"
    "资料不够可以再检索。资料没有的就说不知道，不要编。"
    "确定作答时直接输出，不要调工具。"
)
_EMPTY = "知识库为空，或没有相关内容。"
_RETRIEVE_K = 8
_RETRIEVE_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": "从知识库检索资料。query 必须是独立完整的问句。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


class AgentState(TypedDict):
    question: str
    history: list
    chunks: list[str]
    sources: list[str]
    answer: str


def _window() -> int:
    return int(os.environ.get("MAX_CONTEXT_CHARS", "8000"))


def _model() -> str:
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def _max_steps() -> int:
    # 至少 2：一轮给工具、最后一轮必须作答。
    return max(int(os.environ.get("MAX_AGENT_STEPS", "4")), 2)


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

    budget = int(_window() * float(os.environ.get("HISTORY_BUDGET_RATIO", "0.3")))
    while len(turns) > 1 and _turns_chars(turns) > budget:
        turns = turns[1:]
    if turns and _turns_chars(turns) > budget:
        last = turns[-1]
        turns = [{"role": last["role"], "content": last["content"][: max(budget, 0)]}]
    return turns


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


def _tool_query(tc, fallback: str) -> str:
    # 中转有时给 dict、有时给 JSON 字符串。
    raw = tc.function.arguments
    if isinstance(raw, dict):
        q = raw.get("query")
    else:
        try:
            q = json.loads(raw or "{}").get("query")
        except json.JSONDecodeError:
            q = None
    return (q or fallback).strip()


def _asst_msg(msg) -> dict:
    # 下一轮要带上 tool_calls，SDK 对象不能直接丢进 messages。
    d: dict = {"role": "assistant", "content": msg.content or ""}
    if not msg.tool_calls:
        return d
    d["tool_calls"] = [
        {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments
                if isinstance(tc.function.arguments, str)
                else json.dumps(tc.function.arguments, ensure_ascii=False),
            },
        }
        for tc in msg.tool_calls
    ]
    return d


def _retrieve_obs(query: str, question: str, occupied: int) -> tuple[str, list[str]]:
    rows = retrieve(query, k=_RETRIEVE_K)
    chunks = _fit_context([h["text"] for h in rows], question, occupied=occupied)
    sources = list(dict.fromkeys(h["source"] for h in rows[: len(chunks)]))
    if not chunks:
        return "知识库没有相关内容。", []
    return "\n\n".join(chunks), sources


def stream_ask(question: str, history: list | None = None) -> Iterator[tuple[str, object]]:
    # 工具循环先跑完再吐 token：最后一轮禁止 retrieve，避免死循环。
    history = _working_memory(history or [], question)
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM},
        *history,
        {"role": "user", "content": question},
    ]
    sources: list[str] = []
    seen: set[str] = set()
    occupied = _turns_chars(history)
    client = openai_client()
    steps = _max_steps()

    for step in range(steps):
        last = step == steps - 1
        kwargs = {"model": _model(), "messages": messages}
        if not last:
            kwargs["tools"] = [_RETRIEVE_TOOL]

        if last:
            stream = client.chat.completions.create(**kwargs, stream=True)
            try:
                yield "sources", sources
                got = False
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    text = chunk.choices[0].delta.content
                    if text:
                        got = True
                        yield "delta", text
                if not got:
                    yield "delta", _EMPTY
            finally:
                stream.close()
            return

        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            yield "sources", sources
            yield "delta", (msg.content or "").strip() or _EMPTY
            return

        messages.append(_asst_msg(msg))
        for tc in msg.tool_calls:
            if tc.function.name != "retrieve":
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": "未知工具。"}
                )
                continue

            query = _tool_query(tc, question)
            if query in seen:
                # 同一问句不打第二次 embedding。
                obs = "该问句已检索过，请基于已有资料作答。"
                extra: list[str] = []
            else:
                seen.add(query)
                obs, extra = _retrieve_obs(query, question, occupied)
                sources = list(dict.fromkeys(sources + extra))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs})


def agent_node(state: AgentState) -> dict:
    # /ask 把循环收成一次 invoke；stream 走同一套 stream_ask。
    parts: list[str] = []
    sources: list[str] = []
    for kind, payload in stream_ask(state["question"], state.get("history") or []):
        if kind == "sources":
            sources = payload
        else:
            parts.append(str(payload))
    return {"answer": "".join(parts), "sources": sources}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_edge(START, "agent")
    g.add_edge("agent", END)
    return g.compile()


graph = build_graph()
