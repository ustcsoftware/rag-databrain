import logging
import math
import os
import re

log = logging.getLogger(__name__)

# 型号、错误码先按原文卡住。ponytail: 多码 AND 滤空（E07 和 E08 对照）就退回原池。
_CODE = re.compile(r"[A-Za-z]{1,6}\s*-?\s*\d{2,}")
_MODEL = re.compile(r"[A-Za-z0-9]{1,12}\s*型")
_TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")
_RRF_K = 60
_encoder = None
_encoder_key = None


def retrieve_ranked(question: str, chunks: list[dict], qvec: list[float], k: int) -> list[dict]:
    # 召回池和最终条数分开。k 是交给生成的证据数，不是两路各自的 Top-K。
    if not chunks or k <= 0:
        return []

    fused = rank_orders(question, chunks, qvec)["reranked"]
    picked = _diversify(fused, k, _env_int("RERANK_PER_SOURCE", 3))
    for doc in fused:
        doc["final_rank"] = None
    for i, doc in enumerate(picked, 1):
        doc["final_rank"] = i
    if fused and os.environ.get("RERANK_TRACE", "").strip() == "1":
        # 默认日志级别看不到 debug。追查错误回答时打开，看证据丢在哪一步。
        log.warning("retrieve pool %s", [_trace(d) for d in fused])
    elif fused:
        log.debug("retrieve pool %s", [_trace(d) for d in fused])
    return picked


def rank_orders(question: str, chunks: list[dict], qvec: list[float]) -> dict[str, list[dict]]:
    # 评测要四条名单。hybrid 在精排排序之前拍一份，避免和 reranked 变成同一顺序。
    if not chunks:
        empty: list[dict] = []
        return {"dense": empty, "bm25": empty, "hybrid": empty, "reranked": empty}

    pool = _constrain(question, chunks)
    bm25_hits = _bm25_top(question, pool, _env_int("BM25_K", 30))
    dense_hits = _dense_top(qvec, pool, _env_int("DENSE_K", 30))
    bm25_rank = {d["id"]: i for i, d in enumerate(bm25_hits, 1)}
    dense_rank = {d["id"]: i for i, d in enumerate(dense_hits, 1)}
    fused = _rrf([bm25_hits, dense_hits])[: _env_int("RERANK_CANDIDATES", 30)]
    for i, doc in enumerate(fused, 1):
        doc["bm25_rank"] = bm25_rank.get(doc["id"])
        doc["dense_rank"] = dense_rank.get(doc["id"])
        doc["rrf_rank"] = i
        doc["rerank_score"] = None

    hybrid = list(fused)
    scores = _score(question, [_clip(d["text"]) for d in fused])
    if scores is not None:
        for doc, score in zip(fused, scores):
            doc["rerank_score"] = float(score)
        fused.sort(key=lambda d: d["rerank_score"], reverse=True)
    return {"dense": dense_hits, "bm25": bm25_hits, "hybrid": hybrid, "reranked": fused}


def _trace(doc: dict) -> dict:
    return {
        "chunk_id": doc["id"],
        "bm25_rank": doc.get("bm25_rank"),
        "dense_rank": doc.get("dense_rank"),
        "rrf_rank": doc.get("rrf_rank"),
        "rerank_score": doc.get("rerank_score"),
        "final_rank": doc.get("final_rank"),
    }


def _constrain(question: str, chunks: list[dict]) -> list[dict]:
    keys = []
    seen = set()
    for raw in _CODE.findall(question) + _MODEL.findall(question):
        key = _flat(raw)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    if not keys:
        return chunks
    hit = [c for c in chunks if all(_has(c["text"], k) for k in keys)]
    return hit or chunks


def _flat(text: str) -> str:
    return re.sub(r"[\s-]+", "", text.lower())


def _has(text: str, key: str) -> bool:
    flat = _flat(text)
    if key.endswith("型"):
        return key in flat
    return _bounded(flat, key)


def _bounded(text: str, key: str) -> bool:
    # 「E07」不能命中「E075」。
    start = 0
    while True:
        i = text.find(key, start)
        if i < 0:
            return False
        end = i + len(key)
        before_ok = i == 0 or not text[i - 1].isdigit()
        after_ok = end == len(text) or not text[end].isdigit()
        if before_ok and after_ok:
            return True
        start = i + 1


def _tokenize(text: str) -> list[str]:
    # 英文和错误码整词；中文用二字，避免再装分词器。
    out = []
    for m in _TOKEN.finditer(text.lower()):
        tok = m.group()
        if tok[0].isascii():
            out.append(tok)
            continue
        if len(tok) == 1:
            out.append(tok)
        else:
            out.extend(tok[i : i + 2] for i in range(len(tok) - 1))
    return out


def _bm25_top(question: str, chunks: list[dict], n: int) -> list[dict]:
    # ponytail: 每次请求重算词频，切片上万再缓存或换 OpenSearch。
    if n <= 0 or not chunks:
        return []
    q_tokens = _tokenize(question)
    docs = [_tokenize(c["text"]) for c in chunks]
    if not q_tokens:
        return [_view(c) for c in chunks[:n]]

    n_docs = len(docs)
    avgdl = sum(len(d) for d in docs) / n_docs or 1.0
    df: dict[str, int] = {}
    for doc in docs:
        for tok in set(doc):
            df[tok] = df.get(tok, 0) + 1

    k1, b = 1.5, 0.75
    scores = []
    for doc in docs:
        tf: dict[str, int] = {}
        for tok in doc:
            tf[tok] = tf.get(tok, 0) + 1
        dl = len(doc) or 1
        score = 0.0
        for tok in q_tokens:
            freq = tf.get(tok, 0)
            if not freq:
                continue
            n_t = df[tok]
            idf = math.log((n_docs - n_t + 0.5) / (n_t + 0.5) + 1)
            score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / avgdl))
        scores.append(score)
    order = [i for i in sorted(range(n_docs), key=lambda i: scores[i], reverse=True) if scores[i] > 0]
    return [_view(chunks[i]) for i in order[:n]]


def _dense_top(qvec: list[float], chunks: list[dict], n: int) -> list[dict]:
    if n <= 0 or not chunks:
        return []
    ranked = sorted(chunks, key=lambda c: _cosine(qvec, c["embedding"]), reverse=True)
    return [_view(c) for c in ranked[:n]]


def _cosine(a: list[float], b: list[float]) -> float:
    den = (sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5)
    if den == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / den


def _rrf(lists: list[list[dict]]) -> list[dict]:
    acc: dict[str, float] = {}
    meta: dict[str, dict] = {}
    for items in lists:
        for rank, doc in enumerate(items, 1):
            acc[doc["id"]] = acc.get(doc["id"], 0.0) + 1.0 / (_RRF_K + rank)
            meta.setdefault(doc["id"], doc)
    return [meta[i] for i in sorted(acc, key=lambda i: acc[i], reverse=True)]


def _diversify(ranked: list[dict], k: int, per_source: int) -> list[dict]:
    # 相关性高的相邻切片会占满名额。封顶之后不够再按原序补回。
    if per_source <= 0:
        per_source = k
    picked: list[dict] = []
    skipped: list[dict] = []
    counts: dict[str, int] = {}
    seen: set[str] = set()
    for doc in ranked:
        if doc["text"] in seen:
            continue
        seen.add(doc["text"])
        if counts.get(doc["source"], 0) >= per_source:
            skipped.append(doc)
            continue
        picked.append(doc)
        counts[doc["source"]] = counts.get(doc["source"], 0) + 1
        if len(picked) >= k:
            return picked
    for doc in skipped:
        picked.append(doc)
        if len(picked) >= k:
            break
    return picked


def _score(query: str, texts: list[str]) -> list[float] | None:
    # 空模型表示停在 RRF。BGE 与 Qwen3-Reranker 都用 CrossEncoder，换名即可。
    # 分数只用来排序，不是答案正确的概率。
    model = os.environ.get("RERANK_MODEL", "").strip()
    if not model or not texts:
        return None
    pairs = [(query, text) for text in texts]
    scores = _cross_encoder(model).predict(pairs, batch_size=16, show_progress_bar=False)
    try:
        return [float(s) for s in scores]
    except TypeError:
        return [float(scores)]


def _cross_encoder(model: str):
    global _encoder, _encoder_key
    key = (model, _env_int("RERANK_MAX_LENGTH", 512))
    if _encoder is not None and _encoder_key == key:
        return _encoder
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise RuntimeError('精排需要 pip install -e ".[rerank]"') from e
    _encoder = CrossEncoder(model, max_length=key[1])
    _encoder_key = key
    return _encoder


def _clip(text: str) -> str:
    n = _env_int("RERANK_MAX_CHARS", 2000)
    return text if len(text) <= n else text[:n]


def _view(doc: dict) -> dict:
    return {"id": doc["id"], "source": doc["source"], "text": doc["text"]}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)
