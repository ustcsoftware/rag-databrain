import os
from typing import Iterator, TypedDict

from langgraph.graph import END, START, StateGraph

from rag_databrain.embedder import openai_client
from rag_databrain.store import retrieve

_SYSTEM = "根据资料回答。资料没有的就说不知道，不要编。"
_EMPTY = "知识库为空，或没有相关内容。"


class AgentState(TypedDict):
    question: str
    chunks: list[str]
    sources: list[str]
    answer: str


def _hits(question: str) -> tuple[list[str], list[str]]:
    # /ask 和 /ask/stream 共用这一次检索，避免 handler 再抄一遍。
    rows = retrieve(question)
    chunks = [h["text"] for h in rows]
    sources = list(dict.fromkeys(h["source"] for h in rows))
    return chunks, sources


def _messages(question: str, chunks: list[str]) -> list[dict]:
    # invoke 和 stream 必须同一套 prompt，否则两边回答会漂。
    context = "\n\n".join(chunks)
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
    ]


def retrieve_node(state: AgentState) -> dict:
    chunks, sources = _hits(state["question"])
    return {"chunks": chunks, "sources": sources}


def generate_node(state: AgentState) -> dict:
    if not state["chunks"]:
        return {"answer": _EMPTY}

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    resp = openai_client().chat.completions.create(
        model=model,
        messages=_messages(state["question"], state["chunks"]),
    )
    return {"answer": resp.choices[0].message.content or ""}


def stream_ask(question: str) -> Iterator[tuple[str, object]]:
    # 先检索再打开 chat stream；第一发 yield 之前失败，handler 还能回 503。
    chunks, sources = _hits(question)
    if not chunks:
        yield "sources", sources
        yield "delta", _EMPTY
        return

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    stream = openai_client().chat.completions.create(
        model=model,
        messages=_messages(question, chunks),
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
