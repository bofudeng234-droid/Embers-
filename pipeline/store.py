"""
SQLite + sqlite-vec 存储层 · 单文件,零运维。
schema:
  videos(id, url, title, caption, hashtags, duration_sec, watched_at, notes, embedding_text)
  vec_videos(rowid, embedding)  -- sqlite-vec 虚拟表
"""
from __future__ import annotations

import os
import sqlite3
import struct
from pathlib import Path
from typing import Iterable

import sqlite_vec
from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(os.getenv("EMBERS_DB_PATH", "./data/embers.db"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "2048"))  # 默认匹配智谱 embedding-3


def connect() -> sqlite3.Connection:
    """打开数据库连接并加载 sqlite-vec 扩展。"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """首次运行时创建表。已存在则跳过。"""
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS videos (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT,
            caption TEXT,
            hashtags TEXT,
            duration_sec INTEGER,
            watched_at TEXT,
            notes TEXT,
            embedding_text TEXT
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS vec_videos USING vec0(
            embedding float[{EMBEDDING_DIM}]
        );
        """
    )
    conn.commit()


def serialize_vector(v: list[float]) -> bytes:
    """sqlite-vec 用紧凑 float32 字节序列。"""
    return struct.pack(f"{len(v)}f", *v)


def upsert_video(
    conn: sqlite3.Connection,
    row: dict,
    embedding: list[float],
) -> int:
    """写入一条视频 + 它的 embedding。返回 rowid。"""
    cursor = conn.execute(
        """
        INSERT INTO videos (id, url, title, caption, hashtags, duration_sec, watched_at, notes, embedding_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            url=excluded.url, title=excluded.title, caption=excluded.caption,
            hashtags=excluded.hashtags, duration_sec=excluded.duration_sec,
            watched_at=excluded.watched_at, notes=excluded.notes,
            embedding_text=excluded.embedding_text
        """,
        (
            row["id"], row["url"], row.get("title"), row.get("caption"),
            row.get("hashtags"), row.get("duration_sec"),
            row.get("watched_at"), row.get("notes"),
            row.get("embedding_text"),
        ),
    )

    # vec_videos 的 rowid 需要跟 videos 的逻辑 id 对齐;最简法:同步重建 vec 行
    # 真实工程会做 upsert,但 v0.1 demo 我们假设单次入库
    rowid = conn.execute("SELECT rowid FROM videos WHERE id = ?", (row["id"],)).fetchone()[0]
    conn.execute(
        "INSERT OR REPLACE INTO vec_videos(rowid, embedding) VALUES (?, ?)",
        (rowid, serialize_vector(embedding)),
    )
    conn.commit()
    return rowid


def search(conn: sqlite3.Connection, query_vec: list[float], top_k: int = 5) -> list[dict]:
    """向量相似度检索,返回 top_k 条视频(含距离)。"""
    rows = conn.execute(
        """
        SELECT v.id, v.url, v.title, v.caption, v.hashtags, v.notes, v.watched_at, vec.distance
        FROM vec_videos AS vec
        JOIN videos AS v ON v.rowid = vec.rowid
        WHERE vec.embedding MATCH ?
          AND k = ?
        ORDER BY vec.distance
        """,
        (serialize_vector(query_vec), top_k),
    ).fetchall()

    return [
        {
            "id": r[0], "url": r[1], "title": r[2], "caption": r[3],
            "hashtags": r[4], "notes": r[5], "watched_at": r[6],
            "distance": r[7],
        }
        for r in rows
    ]


def count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]


if __name__ == "__main__":
    conn = connect()
    init_schema(conn)
    print(f"DB ready at {DB_PATH}. Rows: {count(conn)}")
