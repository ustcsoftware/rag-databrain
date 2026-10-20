import json
import os
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
KB_PATH = ROOT / "data" / "kb.jsonl"


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

    rows = _load()
    rows = [r for r in rows if r["source"] != source]
    for i, (chunk, vec) in enumerate(zip(chunks, embs)):
        rows.append({"id": f"{source}:{i}", "source": source, "text": chunk, "embedding": vec})
    _save(rows)
    return len(chunks)


def retrieve(question: str, k: int = 4) -> list[dict]:
    rows = _load()
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


def _load() -> list[dict]:
    if not KB_PATH.exists():
        return []
    rows = []
    for line in KB_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _save(rows: list[dict]) -> None:
    KB_PATH.parent.mkdir(parents=True, exist_ok=True)
    KB_PATH.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
