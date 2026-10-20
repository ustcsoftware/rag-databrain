from fastapi import FastAPI
from pydantic import BaseModel

from rag_databrain.graph import graph

app = FastAPI(title="rag-databrain")


class AskIn(BaseModel):
    question: str


class AskOut(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask", response_model=AskOut)
def ask(body: AskIn):
    result = graph.invoke({"question": body.question, "answer": ""})
    return AskOut(answer=result["answer"])
