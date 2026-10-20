import os
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from rag_databrain.store import openai_client, retrieve


class AgentState(TypedDict):
    question: str
    chunks: list[str]
    sources: list[str]
    answer: str


def retrieve_node(state: AgentState) -> dict:
    hits = retrieve(state["question"])
    return {
        "chunks": [h["text"] for h in hits],
        "sources": list(dict.fromkeys(h["source"] for h in hits)),
    }


def generate_node(state: AgentState) -> dict:
    if not state["chunks"]:
        return {"answer": "知识库为空，或没有相关内容。"}

    context = "\n\n".join(state["chunks"])
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    resp = openai_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "根据资料回答。资料没有的就说不知道，不要编。",
            },
            {
                "role": "user",
                "content": f"资料：\n{context}\n\n问题：{state['question']}",
            },
        ],
    )
    return {"answer": resp.choices[0].message.content or ""}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


graph = build_graph()
