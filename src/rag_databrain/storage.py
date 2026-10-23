import os
from datetime import datetime, timezone
from typing import TypedDict

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


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


def _connect() -> psycopg.Connection:
    # 每次短连接，避免 FastAPI 和以后的 worker 抢同一连接。
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("需要 DATABASE_URL")
    conn = psycopg.connect(url, row_factory=dict_row)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
          source TEXT PRIMARY KEY,
          n_chunks INTEGER NOT NULL,
          updated_at TEXT NOT NULL,
          content_hash TEXT NOT NULL DEFAULT '',
          version INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
          id TEXT PRIMARY KEY,
          source TEXT NOT NULL REFERENCES documents(source) ON DELETE CASCADE,
          idx INTEGER NOT NULL,
          text TEXT NOT NULL,
          embedding JSONB NOT NULL
        )
        """
    )
    # 建表要先提交。调用方失败回滚时，只丢掉这次写入，表还在。
    conn.commit()
    return conn


_chunks: list[Chunk] | None = None


def replace_document(source: str, chunks: list[dict], content_hash: str, version: int) -> int:
    # 同名整份替换。删和写在同一事务里，失败则仍是上一版。
    conn = _connect()
    try:
        conn.execute("DELETE FROM documents WHERE source = %s", (source,))
        conn.execute(
            "INSERT INTO documents (source, n_chunks, updated_at, content_hash, version) VALUES (%s, %s, %s, %s, %s)",
            (source, len(chunks), datetime.now(timezone.utc).isoformat(), content_hash, version),
        )
        for i, c in enumerate(chunks):
            conn.execute(
                "INSERT INTO chunks (id, source, idx, text, embedding) VALUES (%s, %s, %s, %s, %s)",
                (f"{source}:{i}", source, i, c["text"], Jsonb(c["embedding"])),
            )
        conn.commit()
        _invalidate()
        return len(chunks)
    finally:
        conn.close()


def delete_document(source: str) -> bool:
    conn = _connect()
    try:
        cur = conn.execute("DELETE FROM documents WHERE source = %s", (source,))
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
            "SELECT id, source, idx, text, embedding FROM chunks WHERE source = %s ORDER BY idx",
            (source,),
        ).fetchall()
        return [_row_chunk(r) for r in rows]
    finally:
        conn.close()


def load_chunks() -> list[Chunk]:
    # ponytail: 全表拉回内存算余弦。切片上万再改 pgvector，不换库。
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


def _row_chunk(r) -> Chunk:
    return {
        "id": r["id"],
        "source": r["source"],
        "idx": r["idx"],
        "text": r["text"],
        "embedding": r["embedding"],
    }


def _invalidate() -> None:
    global _chunks
    _chunks = None


def get_document(source: str) -> Document | None:
    # 没有该文档返回 None，不要用空 dict 冒充存在。
    conn = _connect()
    try:
        r = conn.execute(
            "SELECT source, n_chunks, updated_at, content_hash, version FROM documents WHERE source = %s",
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
