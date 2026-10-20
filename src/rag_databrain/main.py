from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

load_dotenv()

from rag_databrain.graph import graph
from rag_databrain.store import ingest, read_file

app = FastAPI(title="rag-databrain")

MAX_UPLOAD = 10 * 1024 * 1024


class AskIn(BaseModel):
    question: str


class AskOut(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ingest")
def ingest_file(file: UploadFile):
    raw = file.file.read()
    name = file.filename or "upload"

    if len(raw) > MAX_UPLOAD:
        raise HTTPException(400, "文件超过 10MB")

    try:
        text = read_file(name, raw)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    if not text.strip():
        raise HTTPException(400, "没有可提取的文本")

    try:
        n = ingest(name, text)
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e

    return {"source": name, "chunks": n}


@app.post("/ask", response_model=AskOut)
def ask(body: AskIn):
    try:
        result = graph.invoke(
            {"question": body.question, "chunks": [], "sources": [], "answer": ""}
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e

    return AskOut(answer=result["answer"], sources=result["sources"])
