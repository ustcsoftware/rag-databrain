from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict):
    question: str
    answer: str


def stub_reply(state: AgentState) -> dict:
    # 证明图能跑；接 LLM / 检索时替换这个节点，不要另起一套编排。
    return {"answer": f"(stub) {state['question']}"}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("stub_reply", stub_reply)
    g.add_edge(START, "stub_reply")
    g.add_edge("stub_reply", END)
    return g.compile()


graph = build_graph()
