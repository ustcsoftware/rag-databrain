from rag_databrain.embedder import embed_texts
from rag_databrain.parse import chunk_blocks
from rag_databrain.storage import load_chunks, replace_document


def ingest(source: str, blocks: list[dict]) -> int:
    # 流水线只串三层，解析/向量/落库都不在本函数里实现。
    chunks = chunk_blocks(blocks)
    if not chunks:
        return 0

    embs = embed_texts(chunks)

    return replace_document(
        source,
        [{"text": c, "embedding": e} for c, e in zip(chunks, embs)],
    )


def retrieve(question: str, k: int = 4) -> list[dict]:
    # 空库直接返回，避免为「没有文档」再打一次 embedding。
    rows = load_chunks()
    if not rows:
        return []

    q = embed_texts([question])[0]
    ranked = sorted(rows, key=lambda r: _cosine(q, r["embedding"]), reverse=True)
    return ranked[: min(k, len(ranked))]


def _cosine(a: list[float], b: list[float]) -> float:
    # ponytail: 全量扫描，chunk 上万或要过滤时换 Chroma / Qdrant
    den = (sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5)
    if den == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / den
