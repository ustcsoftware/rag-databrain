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
    content_hash: str
    version: int


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
          updated_at TEXT NOT NULL,
          content_hash TEXT NOT NULL DEFAULT '',
          version INTEGER NOT NULL DEFAULT 1
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
    _ensure_columns(conn)
    return conn


_chunks: list[Chunk] | None = None


def _ensure_columns(conn: sqlite3.Connection) -> None:
    # 旧库只有 source/n_chunks/updated_at，CREATE IF NOT EXISTS 不会加列。
    cols = {r[1] for r in conn.execute("PRAGMA table_info(documents)")}
    if "content_hash" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''")
    if "version" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN version INTEGER NOT NULL DEFAULT 1")


def replace_document(source: str, chunks: list[dict], content_hash: str, version: int) -> int:
    # 同名必须先删后插，否则旧切片会继续被检索到。
    conn = _connect()
    try:
        conn.execute("DELETE FROM documents WHERE source = ?", (source,))

        conn.execute(
            "INSERT INTO documents (source, n_chunks, updated_at, content_hash, version) VALUES (?, ?, ?, ?, ?)",
            (source, len(chunks), datetime.now(timezone.utc).isoformat(), content_hash, version),
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
        _invalidate()
        return len(chunks)
    finally:
        conn.close()


def delete_document(source: str) -> bool:
    conn = _connect()
    try:
        cur = conn.execute("DELETE FROM documents WHERE source = ?", (source,))
        conn.commit()
        gone = cur.rowcount > 0
        if gone:
            _invalidate()
        return gone
    finally:
        conn.close()


def load_chunks_for_source(source: str) -> list[Chunk]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, source, idx, text, embedding FROM chunks WHERE source = ? ORDER BY idx",
            (source,),
        ).fetchall()
        return [_row_chunk(r) for r in rows]
    finally:
        conn.close()


def load_chunks() -> list[Chunk]:
    # 检索要全表；过滤和向量库以后换实现，不在这里加 WHERE。
    global _chunks
    if _chunks is not None:
        return _chunks
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, source, idx, text, embedding FROM chunks ORDER BY source, idx"
        ).fetchall()
        _chunks = [_row_chunk(r) for r in rows]
        return _chunks
    finally:
        conn.close()


def _row_chunk(r: sqlite3.Row) -> Chunk:
    return {
        "id": r["id"],
        "source": r["source"],
        "idx": r["idx"],
        "text": r["text"],
        "embedding": json.loads(r["embedding"]),
    }


def _invalidate() -> None:
    global _chunks
    _chunks = None


def get_document(source: str) -> Document | None:
    # 没有该文档返回 None，不要用空 dict 冒充存在。
    conn = _connect()
    try:
        r = conn.execute(
            "SELECT source, n_chunks, updated_at, content_hash, version FROM documents WHERE source = ?",
            (source,),
        ).fetchone()
        if r is None:
            return None
        return {
            "source": r["source"],
            "n_chunks": r["n_chunks"],
            "updated_at": r["updated_at"],
            "content_hash": r["content_hash"] or "",
            "version": int(r["version"] or 1),
        }
    finally:
        conn.close()
