import json
import os
from typing import Iterator, TypedDict

from langgraph.graph import END, START, StateGraph

from rag_databrain.embedder import openai_client
from rag_databrain.store import retrieve

_SYSTEM = (
    "只根据 retrieve 返回的资料作答。对话只用来理解指代，不能当资料。"
    "需要资料时先 retrieve，query 写成独立完整问句。资料不够可以再检索。"
    "资料没有的就说不知道，不要用自身知识补充规定、数字或日期。"
    "确定作答时直接输出，不要调工具。"
)
# 检索前单独判断。真歧义如果交给循环，模型会猜一个意思直接去搜。
_ROUTE = (
    "你只判断这一轮怎么处理，不回答问题。只输出 JSON。"
    "能从对话补成唯一问句，或问句已经完整："
    '{"action":"search","query":"不看历史也能检索的完整问句"}。'
    "几种读法会搜到不同资料，对话也补不出唯一含义："
    '{"action":"clarify","question":"只问一个缺口"}。'
    "不要替用户选定。问候、致谢、与知识库无关时 action 用 search，query 用用户原话。"
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


def _parse_route(raw: str) -> tuple[str, str] | None:
    # 中转偶尔包一层 markdown，或在 JSON 前后加一句。解析失败就当没判断过。
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    action = data.get("action")
    if action == "clarify":
        question = data.get("question")
        if isinstance(question, str) and question.strip():
            return "clarify", question.strip()
        return None
    if action in ("search", "retrieve"):
        query = data.get("query")
        if isinstance(query, str) and query.strip():
            return "search", query.strip()
    return None


def _route(client, question: str, history: list[dict]) -> tuple[str, str] | None:
    resp = client.chat.completions.create(
        model=_model(),
        messages=[
            {"role": "system", "content": _ROUTE},
            *history,
            {"role": "user", "content": question},
        ],
    )
    content = resp.choices[0].message.content
    if not isinstance(content, str):
        return None
    return _parse_route(content)


def _system(pin: str | None) -> str:
    if not pin:
        return _SYSTEM
    # query 当数据嵌进去，避免改写结果里的标点把指令截断。
    return (
        _SYSTEM
        + "本轮第一次 retrieve 的 query 必须用这句，不要改成别的意思，也不要反问："
        + json.dumps(pin, ensure_ascii=False)
    )


def _retrieve_obs(query: str, question: str, occupied: int) -> tuple[str, list[str]]:
    rows = retrieve(query, k=_RETRIEVE_K)
    chunks = _fit_context([h["text"] for h in rows], question, occupied=occupied)
    sources = list(dict.fromkeys(h["source"] for h in rows[: len(chunks)]))
    if not chunks:
        return "知识库没有相关内容。", []
    return "\n\n".join(chunks), sources


def stream_ask(question: str, history: list | None = None) -> Iterator[tuple[str, object]]:
    # 先消歧：真歧义只反问。否则进工具循环，第一次检索用改写问句。最后一轮禁止 retrieve。
    history = _working_memory(history or [], question)
    client = openai_client()
    routed = _route(client, question, history)
    if routed is not None and routed[0] == "clarify":
        yield "sources", []
        yield "delta", routed[1]
        return

    pin = None if routed is None else routed[1]
    system = _system(pin)
    messages: list[dict] = [
        {"role": "system", "content": system},
        *history,
        {"role": "user", "content": question},
    ]
    sources: list[str] = []
    seen: set[str] = set()
    occupied = _turns_chars(history) + max(len(system) - len(_SYSTEM), 0)
    steps = _max_steps()

    for step in range(steps):
        last = step == steps - 1
        kwargs = {"model": _model(), "messages": messages}
        if not last:
            kwargs["tools"] = [_RETRIEVE_TOOL]

        if last:
            if not sources:
                # 没有检索证据就不要让模型用自身知识编一段。
                yield "sources", []
                yield "delta", _EMPTY
                return
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
            text = (msg.content or "").strip()
            yield "delta", text if sources else _EMPTY
            return

        messages.append(_asst_msg(msg))
        if pin:
            for call in messages[-1].get("tool_calls") or []:
                if call["function"]["name"] == "retrieve":
                    call["function"]["arguments"] = json.dumps(
                        {"query": pin}, ensure_ascii=False
                    )
                    break
        for tc in msg.tool_calls:
            if tc.function.name != "retrieve":
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": "未知工具。"}
                )
                continue

            if pin is not None:
                query = pin
                pin = None
            else:
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
