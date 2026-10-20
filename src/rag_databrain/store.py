import os
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from pypdf import PdfReader

from rag_databrain.storage import load_chunks, replace_document


def openai_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("缺少 OPENAI_API_KEY")
    kwargs = {"api_key": key}
    base = os.environ.get("OPENAI_BASE_URL", "").strip()
    if base:
        kwargs["base_url"] = base
    return OpenAI(**kwargs)


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


def embed(texts: list[str]) -> list[list[float]]:
    client = openai_client()
    model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    out = []
    for i in range(0, len(texts), 64):
        resp = client.embeddings.create(model=model, input=texts[i : i + 64])
        out.extend(d.embedding for d in resp.data)
    return out


def ingest(source: str, text: str) -> int:
    chunks = chunk_text(text)
    if not chunks:
        return 0

    embs = embed(chunks)

    return replace_document(
        source,
        [{"text": c, "embedding": e} for c, e in zip(chunks, embs)],
    )


def retrieve(question: str, k: int = 4) -> list[dict]:
    rows = load_chunks()
    if not rows:
        return []

    q = embed([question])[0]
    ranked = sorted(rows, key=lambda r: _cosine(q, r["embedding"]), reverse=True)
    return ranked[: min(k, len(ranked))]


def _cosine(a: list[float], b: list[float]) -> float:
    # ponytail: 全量扫描，chunk 上万或要过滤时换 Chroma / Qdrant
    den = (sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5)
    if den == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / den
