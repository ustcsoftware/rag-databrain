"""MultiHop-RAG 单独打分，不写入业务库。

用法：python -m rag_databrain.multihop data/multihop-hf

目录里要有 corpus.json 和 MultiHopRAG.json。
来源：https://huggingface.co/datasets/yixuantt/MultiHopRAG
609 篇新闻都参与检索。只留下标准文档会把召回算高。
"""

import hashlib
import json
import os
import sys
from array import array
from pathlib import Path

from dotenv import load_dotenv

from rag_databrain.embedder import embed_texts
from rag_databrain.eval import _score
from rag_databrain.parse import chunk_blocks


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = sys.argv[1:] if argv is None else argv
    root = Path(args[0] if args else "data/multihop-hf")
    corpus_path = root / "corpus.json"
    question_path = root / "MultiHopRAG.json"
    if not corpus_path.is_file() or not question_path.is_file():
        raise SystemExit(
            f"没有找到 {corpus_path} 和 {question_path}。"
            "从 https://huggingface.co/datasets/yixuantt/MultiHopRAG 下载这两个文件。"
        )

    questions = _questions(json.loads(question_path.read_text(encoding="utf-8")))
    chunks = _chunks(json.loads(corpus_path.read_text(encoding="utf-8")))
    docs = len({c["source"] for c in chunks})
    print(f"文档 {docs}，切片 {len(chunks)}，题 {len(questions)}。这份索引不写入业务库。")
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise SystemExit("缺少 OPENAI_API_KEY。向量要现算，不能拿业务库里的切片代替。")

    _embed(chunks, root)
    # ponytail: 每题重算 BM25。一万多切片乘两千题会慢，要再快就按切片缓存词频。
    _score(questions, chunks)


def _questions(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        kind = row.get("question_type") or "fact"
        if kind == "null_query":
            out.append(
                {"question": row["query"], "kind": "unanswerable", "relevant_sources": [], "facts": []}
            )
            continue
        sources = []
        facts = []
        for ev in row.get("evidence_list") or []:
            title = ev.get("title") or ""
            if title and title not in sources:
                sources.append(title)
            fact = ev.get("fact") or ""
            if fact and fact not in facts:
                facts.append(fact)
        out.append(
            {
                "question": row["query"],
                "kind": kind,
                "relevant_sources": sources,
                "facts": facts,
            }
        )
    return out


def _chunks(docs: list[dict]) -> list[dict]:
    chunks = []
    for doc in docs:
        body = (doc.get("body") or "").strip()
        title = (doc.get("title") or "").strip()
        if not body or not title:
            continue
        parts = chunk_blocks([{"kind": "paragraph", "text": body, "path": title}])
        for i, text in enumerate(parts):
            chunks.append({"id": str(len(chunks)), "source": title, "idx": i, "text": text})
    return chunks


def _embed(chunks: list[dict], root: Path) -> None:
    digest = hashlib.sha256("\n".join(c["text"] for c in chunks).encode("utf-8")).hexdigest()
    meta_path = root / "chunk-embeddings.meta"
    bin_path = root / "chunk-embeddings.bin"
    if meta_path.is_file() and bin_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("sha256") == digest and meta.get("n") == len(chunks):
            dim = int(meta["dim"])
            raw = array("f")
            raw.frombytes(bin_path.read_bytes())
            if len(raw) != len(chunks) * dim:
                raise SystemExit("嵌入缓存长度对不上，删掉 chunk-embeddings.bin 再跑。")
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = list(raw[i * dim : (i + 1) * dim])
            print("用已缓存的切片嵌入。")
            return

    vectors = embed_texts([c["text"] for c in chunks])
    if len(vectors) != len(chunks) or not vectors[0]:
        raise SystemExit("嵌入条数和切片对不上。")
    dim = len(vectors[0])
    raw = array("f")
    for chunk, vec in zip(chunks, vectors):
        vals = [float(x) for x in vec]
        chunk["embedding"] = vals
        raw.extend(vals)
    bin_path.write_bytes(raw.tobytes())
    meta_path.write_text(
        json.dumps({"sha256": digest, "n": len(chunks), "dim": dim}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
