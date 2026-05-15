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
CLAP_DIM = int(os.getenv("CLAP_DIM", "512"))              # LAION-CLAP


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

        CREATE VIRTUAL TABLE IF NOT EXISTS vec_audios USING vec0(
            embedding float[{CLAP_DIM}]
        );

        -- v0.7 memory_anchors:每条视频派生出多条 anchor(highlight/vibe/core/domain/hook),
        -- 每条 anchor 单独 embed 进 vec_anchors。检索时 anchor 命中 → 通过 video_id 合并。
        -- 用 anchors.rowid 跟 vec_anchors.rowid 对齐(都是 sqlite 自增 INTEGER PRIMARY KEY)。
        CREATE TABLE IF NOT EXISTS anchors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            text TEXT NOT NULL,
            FOREIGN KEY (video_id) REFERENCES videos(id)
        );
        CREATE INDEX IF NOT EXISTS idx_anchors_video ON anchors(video_id);

        CREATE VIRTUAL TABLE IF NOT EXISTS vec_anchors USING vec0(
            embedding float[{EMBEDDING_DIM}]
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
    audio_embedding: list[float] | None = None,
) -> int:
    """写入一条视频 + 文本向量 + (可选)视觉向量 + (可选)音频向量。"""
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

    # sqlite-vec 的 vec0 虚拟表对 INSERT OR REPLACE 支持有 bug,
    # 已存在的 rowid 会触发 UNIQUE constraint failed。
    # 用先 DELETE 再 INSERT 的方式实现 upsert。

    # 文本向量(必有)
    conn.execute("DELETE FROM vec_videos WHERE rowid = ?", (rowid,))
    conn.execute(
        "INSERT INTO vec_videos(rowid, embedding) VALUES (?, ?)",
        (rowid, serialize_vector(text_embedding)),
    )

    # 视觉向量(可选 · v0.3 新增)
    if image_embedding is not None:
        conn.execute("DELETE FROM vec_images WHERE rowid = ?", (rowid,))
        conn.execute(
            "INSERT INTO vec_images(rowid, embedding) VALUES (?, ?)",
            (rowid, serialize_vector(image_embedding)),
        )

    # 音频向量(可选 · v0.4 新增) · image 帖子没 mp4,无音频向量 → 清掉旧的避免脏数据
    if audio_embedding is not None:
        conn.execute("DELETE FROM vec_audios WHERE rowid = ?", (rowid,))
        conn.execute(
            "INSERT INTO vec_audios(rowid, embedding) VALUES (?, ?)",
            (rowid, serialize_vector(audio_embedding)),
        )
    else:
        # 之前可能误判为视频留下了音频向量,清掉
        conn.execute("DELETE FROM vec_audios WHERE rowid = ?", (rowid,))

    conn.commit()
    return rowid


def upsert_anchors(
    conn: sqlite3.Connection,
    video_id: str,
    anchor_rows: list[tuple[str, str, str]],
    embed_fn,
) -> int:
    """为一条视频写入多个 anchor + 它们的 embedding。

    流程:
      1. 删除该 video_id 旧的 anchors + vec_anchors(避免重跑产生孤儿)
      2. 对每条 anchor 调 embed_fn 拿向量
      3. 插入 anchors 行(自增 id) → 用 id 同步插入 vec_anchors

    返回写入的 anchor 数。embed_fn 失败的 anchor 跳过。
    """
    # 1. 找出旧的 anchor id,从 vec_anchors 删
    old_ids = [r[0] for r in conn.execute(
        "SELECT id FROM anchors WHERE video_id = ?", (video_id,)
    ).fetchall()]
    if old_ids:
        for aid in old_ids:
            conn.execute("DELETE FROM vec_anchors WHERE rowid = ?", (aid,))
        conn.execute("DELETE FROM anchors WHERE video_id = ?", (video_id,))

    # 2. 逐条插入
    n_written = 0
    for vid, kind, text in anchor_rows:
        try:
            vec = embed_fn(text)
        except Exception as e:
            print(f"  [anchors] embed 失败,跳过 ({kind}): {e}")
            continue
        cur = conn.execute(
            "INSERT INTO anchors(video_id, kind, text) VALUES (?, ?, ?)",
            (vid, kind, text),
        )
        anchor_id = cur.lastrowid
        conn.execute(
            "INSERT INTO vec_anchors(rowid, embedding) VALUES (?, ?)",
            (anchor_id, serialize_vector(vec)),
        )
        n_written += 1
    conn.commit()
    return n_written


def search_anchor(
    conn: sqlite3.Connection,
    query_vec: list[float],
    top_k: int = 10,
) -> list[dict]:
    """anchor 路检索 · 同一视频多个 anchor 命中只保留 distance 最小那条。

    返回字段跟其它 search_* 对齐: id/url/title/caption/transcript/frame_desc/
    duration_sec/watched_at/distance,额外多 anchor_kind / anchor_text。
    """
    # vec_anchors 查 top_k * 3,因为同一视频会重复
    rows = conn.execute(
        f"""
        SELECT a.video_id, a.kind, a.text, vec.distance
        FROM vec_anchors AS vec
        JOIN anchors AS a ON a.id = vec.rowid
        WHERE vec.embedding MATCH ?
          AND k = ?
        ORDER BY vec.distance
        """,
        (serialize_vector(query_vec), top_k * 3),
    ).fetchall()

    # 同一 video_id 保留距离最小那条
    best_by_video: dict[str, tuple[float, str, str]] = {}
    for vid, kind, text, dist in rows:
        prev = best_by_video.get(vid)
        if prev is None or dist < prev[0]:
            best_by_video[vid] = (dist, kind, text)

    # 排序 + 拼上完整 video 字段(供前端展示)
    sorted_vids = sorted(best_by_video.keys(), key=lambda v: best_by_video[v][0])[:top_k]
    if not sorted_vids:
        return []
    placeholders = ",".join("?" * len(sorted_vids))
    video_rows = conn.execute(
        f"""SELECT id, url, title, caption, transcript, frame_desc, duration_sec, watched_at
            FROM videos WHERE id IN ({placeholders})""",
        sorted_vids,
    ).fetchall()
    by_id = {r[0]: r for r in video_rows}

    out: list[dict] = []
    for vid in sorted_vids:
        v = by_id.get(vid)
        if not v:
            continue
        dist, kind, text = best_by_video[vid]
        out.append({
            "id": v[0], "url": v[1], "title": v[2], "caption": v[3],
            "transcript": v[4], "frame_desc": v[5],
            "duration_sec": v[6], "watched_at": v[7],
            "distance": dist,
            "anchor_kind": kind,
            "anchor_text": text,
        })
    return out


def count_anchors(conn: sqlite3.Connection) -> int:
    try:
        rows = conn.execute("SELECT COUNT(*) FROM anchors").fetchone()
        return rows[0] if rows else 0
    except Exception:
        return 0


def search_text(conn: sqlite3.Connection, query_vec: list[float], top_k: int = 5) -> list[dict]:
    """文本向量库检索。"""
    return _search(conn, "vec_videos", query_vec, top_k)


def search_image(conn: sqlite3.Connection, query_vec: list[float], top_k: int = 5) -> list[dict]:
    """视觉向量库检索。"""
    return _search(conn, "vec_images", query_vec, top_k)


def search_audio(conn: sqlite3.Connection, query_vec: list[float], top_k: int = 5) -> list[dict]:
    """音频向量库检索(v0.4)。"""
    return _search(conn, "vec_audios", query_vec, top_k)


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


def count_with_audio(conn: sqlite3.Connection) -> int:
    """有音频向量的视频数(v0.4 新增)。"""
    try:
        rows = conn.execute("SELECT COUNT(*) FROM vec_audios").fetchone()
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
