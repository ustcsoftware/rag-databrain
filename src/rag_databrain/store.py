from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from rag_databrain.embedder import embed_texts
from rag_databrain.storage import load_chunks, replace_document


def chunk_text(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + size])
        i += size - overlap
    return chunks


def read_file(name: str, raw: bytes) -> str:
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        pages = PdfReader(BytesIO(raw)).pages
        return "\n".join((p.extract_text() or "") for p in pages)
    if suffix in {".txt", ".md"}:
        for enc in ("utf-8", "gbk"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        raise ValueError("文本编码无法识别")
    raise ValueError("只支持 .txt .md .pdf")


def ingest(source: str, text: str) -> int:
    chunks = chunk_text(text)
    if not chunks:
        return 0

    embs = embed_texts(chunks)

    return replace_document(
        source,
        [{"text": c, "embedding": e} for c, e in zip(chunks, embs)],
    )


def retrieve(question: str, k: int = 4) -> list[dict]:
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
