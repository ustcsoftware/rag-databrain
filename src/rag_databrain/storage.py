import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "kb.sqlite"


class Chunk(TypedDict):
    id: str
    source: str
    idx: int
    text: str
    embedding: list[float]


class Document(TypedDict):
    source: str
    n_chunks: int
    updated_at: str


def _connect() -> sqlite3.Connection:
    # 每次短连接，避免 FastAPI 和以后的 Celery worker 抢同一连接。
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
          source TEXT PRIMARY KEY,
          n_chunks INTEGER NOT NULL,
          updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
          id TEXT PRIMARY KEY,
          source TEXT NOT NULL,
          idx INTEGER NOT NULL,
          text TEXT NOT NULL,
          embedding TEXT NOT NULL,
          FOREIGN KEY (source) REFERENCES documents(source) ON DELETE CASCADE
        );
        """
    )
    return conn


def replace_document(source: str, chunks: list[dict]) -> int:
    # 同名必须先删后插，否则旧切片会继续被检索到。
    conn = _connect()
    try:
        conn.execute("DELETE FROM documents WHERE source = ?", (source,))

        conn.execute(
            "INSERT INTO documents (source, n_chunks, updated_at) VALUES (?, ?, ?)",
            (source, len(chunks), datetime.now(timezone.utc).isoformat()),
        )
        for i, c in enumerate(chunks):
            conn.execute(
                "INSERT INTO chunks (id, source, idx, text, embedding) VALUES (?, ?, ?, ?, ?)",
                (
                    f"{source}:{i}",
                    source,
                    i,
                    c["text"],
                    json.dumps(c["embedding"], separators=(",", ":")),
                ),
            )
        conn.commit()
        return len(chunks)
    finally:
        conn.close()


def load_chunks() -> list[Chunk]:
    # 检索要全表；过滤和向量库以后换实现，不在这里加 WHERE。
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, source, idx, text, embedding FROM chunks ORDER BY source, idx"
        ).fetchall()
        return [
            {
                "id": r["id"],
                "source": r["source"],
                "idx": r["idx"],
                "text": r["text"],
                "embedding": json.loads(r["embedding"]),
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_document(source: str) -> Document | None:
    # 没有该文档返回 None，不要用空 dict 冒充存在。
    conn = _connect()
    try:
        r = conn.execute(
            "SELECT source, n_chunks, updated_at FROM documents WHERE source = ?",
            (source,),
        ).fetchone()
        if r is None:
            return None
        return {
            "source": r["source"],
            "n_chunks": r["n_chunks"],
            "updated_at": r["updated_at"],
        }
    finally:
        conn.close()
