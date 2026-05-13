"""
SQLite + sqlite-vec 存储层 · v0.3 双轨向量

v0.3 schema:
  videos(id, url, title, caption, duration_sec, watched_at,
         transcript, frame_desc, embedding_text)
  vec_videos(rowid, embedding[文本 2048d])   -- 智谱 text-embedding-3
  vec_images(rowid, embedding[视觉 512d])    -- Chinese-CLIP
"""
from __future__ import annotations

import os
import sqlite3
import struct
from pathlib import Path

import sqlite_vec
from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(os.getenv("EMBERS_DB_PATH", "./data/embers.db"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "2048"))  # 智谱 embedding-3
CLIP_DIM = int(os.getenv("CLIP_DIM", "512"))              # Chinese-CLIP base


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS videos (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT,
            caption TEXT,
            duration_sec INTEGER,
            watched_at TEXT,
            transcript TEXT,
            frame_desc TEXT,
            embedding_text TEXT
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS vec_videos USING vec0(
            embedding float[{EMBEDDING_DIM}]
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS vec_images USING vec0(
            embedding float[{CLIP_DIM}]
        );
        """
    )
    conn.commit()


def serialize_vector(v: list[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


def upsert_video(
    conn: sqlite3.Connection,
    row: dict,
    text_embedding: list[float],
    image_embedding: list[float] | None = None,
) -> int:
    """写入一条视频 + 文本向量 + (可选)视觉向量。"""
    conn.execute(
        """
        INSERT INTO videos (id, url, title, caption, duration_sec, watched_at,
                            transcript, frame_desc, embedding_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            url=excluded.url, title=excluded.title, caption=excluded.caption,
            duration_sec=excluded.duration_sec, watched_at=excluded.watched_at,
            transcript=excluded.transcript, frame_desc=excluded.frame_desc,
            embedding_text=excluded.embedding_text
        """,
        (
            row["id"], row["url"],
            row.get("title"), row.get("caption"),
            row.get("duration_sec"), row.get("watched_at"),
            row.get("transcript"), row.get("frame_desc"),
            row.get("embedding_text"),
        ),
    )

    rowid = conn.execute("SELECT rowid FROM videos WHERE id = ?", (row["id"],)).fetchone()[0]

    # 文本向量(必有)
    conn.execute(
        "INSERT OR REPLACE INTO vec_videos(rowid, embedding) VALUES (?, ?)",
        (rowid, serialize_vector(text_embedding)),
    )

    # 视觉向量(可选 · v0.3 新增)
    if image_embedding is not None:
        conn.execute(
            "INSERT OR REPLACE INTO vec_images(rowid, embedding) VALUES (?, ?)",
            (rowid, serialize_vector(image_embedding)),
        )

    conn.commit()
    return rowid


def search_text(conn: sqlite3.Connection, query_vec: list[float], top_k: int = 5) -> list[dict]:
    """文本向量库检索。"""
    return _search(conn, "vec_videos", query_vec, top_k)


def search_image(conn: sqlite3.Connection, query_vec: list[float], top_k: int = 5) -> list[dict]:
    """视觉向量库检索。"""
    return _search(conn, "vec_images", query_vec, top_k)


def _search(conn, vec_table: str, query_vec: list[float], top_k: int) -> list[dict]:
    rows = conn.execute(
        f"""
        SELECT v.id, v.url, v.title, v.caption, v.transcript, v.frame_desc,
               v.watched_at, v.duration_sec, vec.distance
        FROM {vec_table} AS vec
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
            "transcript": r[4], "frame_desc": r[5],
            "watched_at": r[6], "duration_sec": r[7],
            "distance": r[8],
        }
        for r in rows
    ]


# 兼容旧接口
search = search_text


def count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]


def count_with_image(conn: sqlite3.Connection) -> int:
    """有视觉向量的视频数(v0.3 新增)。"""
    try:
        rows = conn.execute("SELECT COUNT(*) FROM vec_images").fetchone()
        return rows[0] if rows else 0
    except Exception:
        return 0


if __name__ == "__main__":
    conn = connect()
    init_schema(conn)
    print(f"DB ready at {DB_PATH}")
    print(f"  视频总数: {count(conn)}")
    print(f"  含视觉向量: {count_with_image(conn)}")
    print(f"  text dim: {EMBEDDING_DIM} · image dim: {CLIP_DIM}")
