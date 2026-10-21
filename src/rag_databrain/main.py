import json

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

load_dotenv()

from rag_databrain.graph import graph, stream_ask
from rag_databrain.parse import read_file
from rag_databrain.store import ingest

app = FastAPI(title="rag-databrain")

MAX_UPLOAD = 10 * 1024 * 1024

# 关掉反向代理缓冲，否则 token 会攒成一整段才到浏览器。
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


class HistoryTurn(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def role_ok(cls, v: str) -> str:
        # 不让客户端注入 system，工作记忆只认对话双方。
        if v not in ("user", "assistant"):
            raise ValueError("history.role 只允许 user 或 assistant")
        return v


class AskIn(BaseModel):
    question: str
    history: list[HistoryTurn] = Field(default_factory=list)


class AskOut(BaseModel):
    answer: str
    sources: list[str]


def _sse(payload: dict) -> str:
    # json.dumps 会把正文里的换行转义掉，SSE 一帧不会被拆行。
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_item(kind: str, payload: object) -> str:
    # graph 只产语义事件，SSE 帧格式留在 handler。
    if kind == "sources":
        return _sse({"type": "sources", "sources": payload})
    return _sse({"type": "delta", "text": payload})


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

    try:
        n = ingest(name, text)
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e

    return {"source": name, "chunks": n}


@app.post("/ask", response_model=AskOut)
def ask(body: AskIn):
    try:
        result = graph.invoke(
            {
                "question": body.question,
                "history": [t.model_dump() for t in body.history],
                "chunks": [],
                "sources": [],
                "answer": "",
            }
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e

    return AskOut(answer=result["answer"], sources=result["sources"])


@app.post("/ask/stream")
def ask_stream(body: AskIn):
    # 先把工具循环跑完再写 SSE 头：缺 key 仍是 503；token 只出最后一轮。
    gen = stream_ask(body.question, [t.model_dump() for t in body.history])
    try:
        first = next(gen)
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e

    def events():
        yield _sse_item(*first)
        try:
            for kind, payload in gen:
                yield _sse_item(kind, payload)
            yield _sse({"type": "done"})
        except Exception as e:
            # 头已经发出，只能用事件报错，改不了 HTTP 状态码。
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
