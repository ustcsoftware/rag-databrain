"""固定当前库，对比向量、BM25、混合、精排。指标和 K 见 docs/rag.md。

用法：
  python -m rag_databrain.eval --chunks
  python -m rag_databrain.eval 题集.jsonl

--chunks 只打印切片 id 和开头 80 字，用来填写 relevant_ids。

一行一道题：
{"question":"A型变频器 E07 怎么处理","kind":"model","relevant_ids":["手册.pdf:2"],"facts":["先断电"]}

公开集如果只标到文档、没有切片 id，改用 relevant_sources，值要和切片的 source 一致。
这时召回按文档算：该文档第一块切片落在前 K 里才算命中。

kind：fact 事实，model 精确型号，cross 跨段落，ambiguous 歧义，unanswerable 无答案。
歧义和无答案不进召回分母。relevant_ids 是 文档名:序号，序号从 0，和入库切分顺序一致。
facts 是人能核对的短句，用来算 Evidence Recall。
答案正确率、引用准确率、单次费用不在这里算。
"""

import json
import math
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from rag_databrain.embedder import embed_texts
from rag_databrain.rank import _diversify, rank_orders
from rag_databrain.storage import get_document, load_chunks

_RECALL_K = 20
_MRR_K = 5
_NDCG_K = 5
_SKIP = {"ambiguous", "unanswerable"}
_MODES = ("dense", "bm25", "hybrid", "rerank")


def recall_at(ranked_ids: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def mrr_at(ranked_ids: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    for i, doc_id in enumerate(ranked_ids[:k], 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at(ranked_ids: list[str], relevant: set[str], k: int) -> float | None:
    # 相关记 1。IDCG 把同样条数的相关切片排到最前。
    if not relevant:
        return None
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k], 1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
    return dcg / idcg if idcg else 0.0


def evidence_recall(texts: list[str], facts: list[str]) -> float | None:
    if not facts:
        return None
    blob = "\n".join(texts)
    return sum(1 for fact in facts if fact and fact in blob) / len(facts)


def p95(values: list[float]) -> float | None:
    # 第 ceil(0.95*N) 个。N=20 时是第 19 个。
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in ("-h", "--help"):
        print(__doc__.strip())
        return

    chunks = load_chunks()
    if not chunks:
        raise SystemExit("知识库是空的。先入库，再评测。")

    if args[0] == "--chunks":
        _print_chunks(chunks)
        return

    path = Path(args[0])
    if not path.is_file():
        raise SystemExit(f"没有题集 {path}。格式见 python -m rag_databrain.eval --help")

    questions = _load(path)
    _print_corpus(chunks)
    _score(questions, chunks)


def _print_chunks(chunks: list[dict]) -> None:
    for chunk in chunks:
        text = chunk["text"].replace("\n", " ")
        if len(text) > 80:
            text = text[:80] + "…"
        print(f"{chunk['id']}\t{text}")


def _load(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path} 第 {i} 行不是 JSON：{e}") from e
        if not isinstance(row, dict) or not str(row.get("question") or "").strip():
            raise SystemExit(f"{path} 第 {i} 行没有 question")
        rows.append(row)
    if not rows:
        raise SystemExit(f"{path} 里没有题")
    return rows


def _print_corpus(chunks: list[dict]) -> None:
    sources = sorted({c["source"] for c in chunks})
    model = os.environ.get("OPENAI_EMBEDDING_MODEL", "").strip() or "text-embedding-3-small"
    rerank = os.environ.get("RERANK_MODEL", "").strip() or "（未配置）"
    print(f"切片 {len(chunks)}，文档 {len(sources)}。嵌入 {model}。精排 {rerank}。")
    print(
        "BM25_K={bm25} DENSE_K={dense} RERANK_CANDIDATES={cand} FINAL_K={final} RERANK_PER_SOURCE={per}".format(
            bm25=os.environ.get("BM25_K") or "30",
            dense=os.environ.get("DENSE_K") or "30",
            cand=os.environ.get("RERANK_CANDIDATES") or "30",
            final=os.environ.get("FINAL_K") or "6",
            per=os.environ.get("RERANK_PER_SOURCE") or "3",
        )
    )
    for source in sources:
        doc = get_document(source)
        if doc is None:
            print(f"{source}  没有文档行")
            continue
        print(f"{source}  version={doc['version']}  hash={doc['content_hash']}")

    for name, default in (("BM25_K", 30), ("DENSE_K", 30), ("RERANK_CANDIDATES", 30)):
        raw = os.environ.get(name, "").strip()
        n = int(raw) if raw else default
        if n < _RECALL_K:
            print(f"{name}={n} 小于 {_RECALL_K}，Recall@{_RECALL_K} 只在截断后的名单上算。")
    print()


def _as_labels(ranked: list[dict], by_source: bool) -> list[str]:
    # 文档级只认每个文档第一次出现的切片。后面的重复块不再算一条相关。
    if not by_source:
        return [d["id"] for d in ranked]
    seen = set()
    labels = []
    for doc in ranked:
        src = doc["source"]
        if src in seen:
            labels.append("")
            continue
        seen.add(src)
        labels.append(src)
    return labels


def _resolve(row: dict, chunks: list[dict]) -> tuple[set[str], list[str], bool]:
    # 库里已经没有的 id 仍留在分母里，否则召回会被算高。
    by_source = bool(row.get("relevant_sources"))
    known = {c["source"] for c in chunks} if by_source else {c["id"] for c in chunks}
    keys = row.get("relevant_sources") if by_source else row.get("relevant_ids")
    found = set()
    notes = []
    for doc_id in keys or []:
        if not doc_id:
            continue
        found.add(doc_id)
        if doc_id not in known:
            label = "文档不在库里" if by_source else "id 不在库里"
            notes.append(f"{label}: {doc_id}")
    return found, notes, by_source


def _score(questions: list[dict], chunks: list[dict]) -> None:
    evidence_k = int(os.environ.get("FINAL_K", "6") or "6")
    per_source = int(os.environ.get("RERANK_PER_SOURCE", "3") or "3")
    rerank_on = bool(os.environ.get("RERANK_MODEL", "").strip())
    modes = [m for m in _MODES if m != "rerank" or rerank_on]
    buckets = {m: {"recall": [], "mrr": [], "ndcg": [], "evidence": []} for m in modes}
    by_kind: dict[str, dict] = {}
    latencies: list[float] = []
    skip = {"ambiguous": 0, "unanswerable": 0, "nolabel": 0}
    notes: list[str] = []
    doc_level = False

    for n, row in enumerate(questions, 1):
        kind = (row.get("kind") or "fact").strip()
        if kind in _SKIP:
            skip[kind] = skip.get(kind, 0) + 1
            continue

        relevant, row_notes, by_source = _resolve(row, chunks)
        doc_level = doc_level or by_source
        notes.extend(f"第 {n} 题 {note}" for note in row_notes)
        facts = [f for f in (row.get("facts") or []) if f]
        if not relevant:
            skip["nolabel"] += 1
            continue

        started = time.perf_counter()
        qvec = embed_texts([row["question"]])[0]
        orders = rank_orders(row["question"], chunks, qvec)
        latencies.append(time.perf_counter() - started)

        lists = {
            "dense": orders["dense"],
            "bm25": orders["bm25"],
            "hybrid": orders["hybrid"],
            "rerank": orders["reranked"],
        }
        for mode in modes:
            ranked = lists[mode]
            labels = _as_labels(ranked, by_source)
            # 精排的召回看融合候选。没进这 20 条，精排没有机会。
            recall_labels = _as_labels(orders["hybrid"], by_source) if mode == "rerank" else labels
            rec = recall_at(recall_labels, relevant, _RECALL_K)
            mrr = mrr_at(labels, relevant, _MRR_K)
            nd = ndcg_at(labels, relevant, _NDCG_K)
            shown = _diversify(ranked, evidence_k, per_source)
            ev = evidence_recall([d["text"] for d in shown], facts)
            buckets[mode]["recall"].append(rec)
            buckets[mode]["mrr"].append(mrr)
            buckets[mode]["ndcg"].append(nd)
            if ev is not None:
                buckets[mode]["evidence"].append(ev)
            if mode == "hybrid":
                slot = by_kind.setdefault(kind, {"recall": [], "mrr": [], "ndcg": [], "n": 0})
                slot["n"] += 1
                slot["recall"].append(rec)
                slot["mrr"].append(mrr)
                slot["ndcg"].append(nd)

    scored = len(buckets["hybrid"]["recall"])
    print(
        f"题 {len(questions)}，参与检索 {scored}，"
        f"歧义 {skip['ambiguous']}，无答案 {skip['unanswerable']}，"
        f"没有相关切片 {skip['nolabel']}。"
    )
    print()
    if doc_level:
        print("Recall、MRR、nDCG 按文档算：标准文档的第一块切片落在前 K 里算命中。")
    ev_head = f"Evidence@{evidence_k}"
    print(f"{'模式':<10}{'Recall@20':>12}{'MRR@5':>10}{'nDCG@5':>10}{ev_head:>14}")
    for mode in modes:
        b = buckets[mode]
        print(
            f"{mode:<10}{_fmt(b['recall']):>12}{_fmt(b['mrr']):>10}"
            f"{_fmt(b['ndcg']):>10}{_fmt(b['evidence']):>14}"
        )
    if rerank_on:
        print("rerank 的 Recall@20 用精排前的融合候选，和 hybrid 相同。")
    else:
        print("精排列未跑：没有 RERANK_MODEL。")
    if buckets["hybrid"]["evidence"] and len(buckets["hybrid"]["evidence"]) != scored:
        print(
            f"Evidence 只平均了写了 facts 的题"
            f"（{len(buckets['hybrid']['evidence'])}/{scored}）。"
        )

    if by_kind:
        print()
        print("hybrid 按题型")
        for kind, slot in by_kind.items():
            print(
                f"{kind:<14} n={slot['n']:<4}"
                f" Recall@20={_fmt(slot['recall'])}"
                f"  MRR@5={_fmt(slot['mrr'])}"
                f"  nDCG@5={_fmt(slot['ndcg'])}"
            )

    p = p95(latencies)
    if p is not None:
        tail = "含嵌入和精排" if rerank_on else "含嵌入，不含精排"
        print()
        print(f"检索 P95 {p * 1000:.0f} ms（{tail}，不含生成）")
    print("答案正确率、引用准确率、单次费用：没有标准答案和单价，未算。")
    for note in notes[:20]:
        print(note)
    if len(notes) > 20:
        print(f"另有 {len(notes) - 20} 条标注对不上库。")


def _fmt(values: list[float]) -> str:
    if not values:
        return "—"
    return f"{sum(values) / len(values):.3f}"


if __name__ == "__main__":
    main()
