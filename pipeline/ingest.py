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
from pipeline import clip_embed
from pipeline.download import download_video
from pipeline.vision import describe_frames

# v0.4 音频路 · 容错导入(CLAP / Whisper 装不上时 ingest 仍能跑,只是缺音频信号)
try:
    from pipeline import clap_embed
    from pipeline.transcribe import transcribe
    AUDIO_AVAILABLE = True
except Exception as _e:
    print(f"[ingest] 音频路不可用(将跳过): {_e}")
    AUDIO_AVAILABLE = False


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
        # ============ 1. 打开页面 + 抽帧 + 抓 caption(一站式) ============
        print(f"  · Playwright 打开页面 + 截图抽帧...")
        meta = download_video(url, work_dir, video_id=video_id)
        if not meta:
            print(f"  ✗ 页面处理失败,跳过")
            return None
        duration = meta.get("duration")
        frame_paths = [Path(p) for p in meta.get("frame_paths", [])]
        print(f"  ✓ 完成 · {duration}s · {len(frame_paths)} 帧 · caption: {(meta.get('caption') or '')[:50]}")

        # ============ 2a. CLAP 音频向量(v0.4) ============
        transcript = ""
        audio_vec = None
        video_path = meta.get("video_path")
        if AUDIO_AVAILABLE and video_path:
            try:
                print(f"  · CLAP 音频编码...")
                audio_vec = clap_embed.audio_to_vec(video_path)
                print(f"  ✓ 音频向量 · {len(audio_vec)}d")
            except Exception as e:
                print(f"  ✗ CLAP 失败,跳过音频向量: {e}")
                audio_vec = None

            # 2b. Whisper 转写(独立 try,跟 CLAP 互不影响)
            try:
                print(f"  · Whisper 转写...")
                transcript = transcribe(video_path)
                if transcript:
                    print(f"  ✓ 转写 · {len(transcript)} 字 · {transcript[:50]}{'...' if len(transcript) > 50 else ''}")
                else:
                    print(f"  · (转写为空,可能没人声)")
            except Exception as e:
                print(f"  ✗ Whisper 失败,跳过转写: {e}")
                transcript = ""
        elif not video_path:
            print(f"  · (没有 mp4,跳过音频路)")

        # ============ 3a. CLIP 视觉向量(真多模态)============
        image_vec = None
        if frame_paths:
            try:
                print(f"  · CLIP 视觉编码 {len(frame_paths)} 帧...")
                frame_vecs = clip_embed.images_to_vecs(frame_paths)
                image_vec = clip_embed.average_image_vecs(frame_vecs)
                print(f"  ✓ 视觉向量就绪 · {len(image_vec)}d")
            except Exception as e:
                print(f"  ✗ CLIP 编码失败,跳过视觉向量: {e}")
                image_vec = None

        # ============ 3b. VLM 文字描述帧(语义补充)============
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

        return {"row": row, "vec": vec, "image_vec": image_vec, "audio_vec": audio_vec}

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

            store.upsert_video(
                conn, result["row"], result["vec"],
                image_embedding=result.get("image_vec"),
                audio_embedding=result.get("audio_vec"),
            )
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
