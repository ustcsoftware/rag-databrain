import hashlib
from typing import TypedDict

from rag_databrain.embedder import embed_texts
from rag_databrain.parse import chunk_blocks
from rag_databrain.storage import (
    delete_document,
    get_document,
    load_chunks,
    load_chunks_for_source,
    replace_document,
)


class IngestResult(TypedDict):
    source: str
    chunks: int
    version: int
    unchanged: bool
    embedded: int


def ingest(source: str, blocks: list[dict]) -> IngestResult:
    # 哈希没变就跳过；变了只给新切片打向量，旧切片复用。落库仍整份替换，避免残留。
    texts = chunk_blocks(blocks)
    if not texts:
        return {"source": source, "chunks": 0, "version": 0, "unchanged": True, "embedded": 0}

    digest = _content_hash(texts)
    prev = get_document(source)
    if prev and prev["content_hash"] == digest:
        return {
            "source": source,
            "chunks": prev["n_chunks"],
            "version": prev["version"],
            "unchanged": True,
            "embedded": 0,
        }

    reused = {c["text"]: c["embedding"] for c in load_chunks_for_source(source)}
    need = [t for t in dict.fromkeys(texts) if t not in reused]
    fresh = dict(zip(need, embed_texts(need))) if need else {}
    rows = [{"text": t, "embedding": reused.get(t) or fresh[t]} for t in texts]
    version = 1 if prev is None else prev["version"] + 1
    n = replace_document(source, rows, digest, version)
    return {
        "source": source,
        "chunks": n,
        "version": version,
        "unchanged": False,
        "embedded": len(need),
    }


def remove(source: str) -> bool:
    return delete_document(source)


def retrieve(question: str, k: int = 4) -> list[dict]:
    # 空库直接返回，避免为「没有文档」再打一次 embedding。
    rows = load_chunks()
    if not rows:
        return []

    q = embed_texts([question])[0]
    ranked = sorted(rows, key=lambda r: _cosine(q, r["embedding"]), reverse=True)
    return ranked[: min(k, len(ranked))]


def _content_hash(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        raw = t.encode("utf-8")
        h.update(len(raw).to_bytes(4, "big"))
        h.update(raw)
    return h.hexdigest()


def _cosine(a: list[float], b: list[float]) -> float:
    # ponytail: 全量扫描，chunk 上万或要过滤时换 Chroma / Qdrant
    den = (sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5)
    if den == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / den
