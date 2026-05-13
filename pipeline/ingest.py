"""
v0.2 多模态 pipeline 编排器

每条 URL 经过:
  1. yt-dlp 下载视频文件 + 抓 caption / title / 时长
  2. ffmpeg 抽 4 个关键帧
  3. faster-whisper 转写音频
  4. GLM-4V 描述每一帧
  5. 合成 embedding 输入文本(caption + transcript + frame_desc)
  6. 智谱 embedding-3 → 2048d 向量
  7. 写入 SQLite + sqlite-vec

用法:
  python -m pipeline.ingest data/videos.csv

容错策略:
  · 单条视频 download/transcribe/vlm 任一失败,不阻断其他视频
  · 至少 caption 拿到就能入库(向量质量降级)
  · 全部失败才跳过这一条
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from pipeline import store, embed
from pipeline.download import download_video
from pipeline.frames import extract_frames, get_duration
from pipeline.transcribe import transcribe
from pipeline.vision import describe_frames


def process_one(url: str, video_id: str, tmp_root: Path) -> dict[str, Any] | None:
    """处理单条 URL,返回准备好的 row dict(含 embedding_text + frame_desc JSON)。
    任何环节失败都返回 None。
    """
    print(f"\n[{video_id}] {url}")
    t0 = time.time()

    # 临时工作目录(单视频隔离,处理完整体删)
    work_dir = tmp_root / video_id
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ============ 1. 下载 ============
        print(f"  · 下载...")
        meta = download_video(url, work_dir)
        if not meta:
            print(f"  ✗ 下载失败,跳过")
            return None
        video_path = meta["video_path"]
        duration = meta.get("duration")
        print(f"  ✓ 下载完成 · {duration}s · caption: {(meta.get('caption') or '')[:50]}")

        # ============ 2. 抽帧 ============
        print(f"  · 抽帧 4 张...")
        frame_paths = extract_frames(video_path, work_dir, n_frames=4, duration=duration)
        print(f"  ✓ 抽到 {len(frame_paths)} 帧")

        # ============ 3. 转写 ============
        print(f"  · Whisper 转写...")
        transcript = transcribe(video_path)
        print(f"  ✓ 转写完成 · {len(transcript)} 字 · {transcript[:60]}{'...' if len(transcript) > 60 else ''}")

        # ============ 4. VLM 描述 ============
        print(f"  · VLM 描述 {len(frame_paths)} 帧...")
        frame_descs = describe_frames(frame_paths) if frame_paths else []
        # 过滤空描述
        frame_descs = [d for d in frame_descs if d]
        for i, d in enumerate(frame_descs, 1):
            print(f"    画面{i}: {d}")

        # ============ 5. 准备 row ============
        watched_at = datetime.now(timezone.utc).isoformat()
        upload_date = meta.get("upload_date")  # "20260512"
        if upload_date and len(upload_date) == 8:
            watched_at = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}T00:00:00"

        row = {
            "id": video_id,
            "url": url,
            "title": meta.get("title"),
            "caption": meta.get("caption"),
            "duration_sec": int(duration) if duration else None,
            "watched_at": watched_at,
            "transcript": transcript,
            "frame_desc": json.dumps(frame_descs, ensure_ascii=False) if frame_descs else None,
        }

        # 合成 embedding 输入文本(注意:embed.build_embedding_text 接受 dict 形式)
        row_for_embed = {**row, "frame_desc": frame_descs}
        embedding_text = embed.build_embedding_text(row_for_embed)
        row["embedding_text"] = embedding_text

        # ============ 6. Embed ============
        print(f"  · embedding ({len(embedding_text)} 字)...")
        vec = embed.embed(embedding_text)

        elapsed = time.time() - t0
        print(f"  ✅ 完成 · 共 {elapsed:.1f}s")

        return {"row": row, "vec": vec}

    except Exception as e:
        print(f"  ✗ 异常: {e}")
        traceback.print_exc(limit=2)
        return None
    finally:
        # 清理临时文件(节省硬盘)
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    if "url" not in df.columns:
        raise ValueError("CSV 必须有 url 列")
    if "id" not in df.columns:
        df["id"] = [f"v{i+1:03d}" for i in range(len(df))]

    print(f"读到 {len(df)} 条 URL,开始处理...\n")

    conn = store.connect()
    store.init_schema(conn)

    tmp_root = Path(tempfile.mkdtemp(prefix="embers_"))
    print(f"临时工作区: {tmp_root}\n")

    succeeded = 0
    failed = 0
    try:
        for _, csv_row in df.iterrows():
            video_id = str(csv_row["id"])
            url = str(csv_row["url"]).strip()
            if not url or url == "nan":
                continue

            result = process_one(url, video_id, tmp_root)
            if result is None:
                failed += 1
                continue

            store.upsert_video(conn, result["row"], result["vec"])
            succeeded += 1

        total = store.count(conn)
        print(f"\n{'='*60}")
        print(f"✅ 入库完成")
        print(f"   成功: {succeeded} 条 · 失败: {failed} 条")
        print(f"   DB 总条数: {total}")
        print(f"   DB 位置:  {store.DB_PATH}")
        print(f"{'='*60}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/videos.csv"
    main(csv)
