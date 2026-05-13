"""
SQLite + sqlite-vec 存储层 · 单文件,零运维。
v0.2 schema:
  videos(id, url, title, caption, duration_sec, watched_at,
         transcript, frame_desc, embedding_text)
  vec_videos(rowid, embedding)  -- sqlite-vec 虚拟表
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
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "2048"))


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
            frame_desc TEXT,       -- JSON array of strings
            embedding_text TEXT
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS vec_videos USING vec0(
            embedding float[{EMBEDDING_DIM}]
        );
        """
    )
    conn.commit()


def serialize_vector(v: list[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


def upsert_video(conn: sqlite3.Connection, row: dict, embedding: list[float]) -> int:
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
    conn.execute(
        "INSERT OR REPLACE INTO vec_videos(rowid, embedding) VALUES (?, ?)",
        (rowid, serialize_vector(embedding)),
    )
    conn.commit()
    return rowid


def search(conn: sqlite3.Connection, query_vec: list[float], top_k: int = 5) -> list[dict]:
    rows = conn.execute(
        """
        SELECT v.id, v.url, v.title, v.caption, v.transcript, v.frame_desc,
               v.watched_at, v.duration_sec, vec.distance
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
            "transcript": r[4], "frame_desc": r[5],
            "watched_at": r[6], "duration_sec": r[7],
            "distance": r[8],
        }
        for r in rows
    ]


def count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]


if __name__ == "__main__":
    conn = connect()
    init_schema(conn)
    print(f"DB ready at {DB_PATH}. Rows: {count(conn)}")
