"""
ingest 加速版 · 阶段 1

跟 pipeline/ingest.py 同样的产出,但:
  1. 砍掉 CLAP 音频向量(评测验证价值最低,权重 0.7,只懂氛围)
     —— Whisper transcript 保留(价值高:对白/台词是关键检索信号)
  2. 单视频内并发:
     - VLM 4 帧描述: 串行 4 次 API → ThreadPoolExecutor 并发
     - CLIP 编码 / Whisper 转写 与 VLM 并发(本地计算 || 网络等待)
     - 主 embedding 与 anchor 提炼并发
     - anchor 的 ~10 条向量并发 embed(而非串行)
  3. 视频之间仍串行(阶段 1 不引入多进程复杂度)

预期单条 ~30s → ~16-20s。

不碰 pipeline/ingest.py / store.py,可与旧版并存。
用法: python -m pipeline.ingest_fast data/videos.csv
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline import store, embed
from pipeline import clip_embed
from pipeline.download import download_video
from pipeline import vision

try:
    from pipeline.transcribe import transcribe
    WHISPER_AVAILABLE = True
except Exception as _e:
    print(f"[ingest_fast] Whisper 不可用(将跳过转写): {_e}")
    WHISPER_AVAILABLE = False

try:
    from pipeline import memory_anchors
    ANCHORS_AVAILABLE = True
except Exception as _e:
    print(f"[ingest_fast] anchors 不可用(将跳过): {_e}")
    ANCHORS_AVAILABLE = False

VLM_WORKERS = int(os.getenv("VLM_WORKERS", "4"))
ANCHOR_EMBED_WORKERS = int(os.getenv("ANCHOR_EMBED_WORKERS", "8"))


def _describe_frames_concurrent(frame_paths: list[Path]) -> list[str]:
    """4 帧 VLM 描述并发(纯 API,线程并发收益明确)。保持帧顺序。"""
    if not frame_paths:
        return []
    with ThreadPoolExecutor(max_workers=VLM_WORKERS) as ex:
        descs = list(ex.map(vision.describe_frame, frame_paths))
    return [d for d in descs if d]


def _clip_then_whisper(frame_paths: list[Path], video_path: str | None) -> tuple[list | None, str]:
    """CLIP 视觉编码 + Whisper 转写(本地计算,组内串行,跟 VLM 组并发)。"""
    image_vec = None
    if frame_paths:
        try:
            frame_vecs = clip_embed.images_to_vecs(frame_paths)
            image_vec = clip_embed.average_image_vecs(frame_vecs)
        except Exception as e:
            print(f"  ✗ CLIP 失败: {e}")
            image_vec = None

    transcript = ""
    if WHISPER_AVAILABLE and video_path:
        try:
            transcript = transcribe(video_path)
        except Exception as e:
            print(f"  ✗ Whisper 失败: {e}")
            transcript = ""
    return image_vec, transcript


def _store_anchors_fast(conn, video_id: str, anchors: dict) -> int:
    """anchor 入库 · 向量并发 embed 后批量写。复用 store 的 DELETE+INSERT 语义。"""
    rows = memory_anchors.to_anchor_rows(video_id, anchors)
    if not rows:
        return 0
    texts = [r[2] for r in rows]
    with ThreadPoolExecutor(max_workers=ANCHOR_EMBED_WORKERS) as ex:
        vecs = list(ex.map(embed.embed, texts))
    # 预算好的向量塞 dict,喂给 store.upsert_anchors(它按 text 查 embed_fn)
    vec_map: dict[str, list] = {}
    for t, v in zip(texts, vecs):
        vec_map[t] = v
    return store.upsert_anchors(conn, video_id, rows, lambda t: vec_map[t])


def process_one_fast(url: str, video_id: str, tmp_root: Path) -> dict[str, Any] | None:
    print(f"\n[{video_id}] {url}")
    t0 = time.time()
    work_dir = tmp_root / video_id
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. 下载 + 抽帧(序列,无法并发)
        meta = download_video(url, work_dir, video_id=video_id)
        if not meta:
            print(f"  ✗ 页面处理失败,跳过")
            return None
        duration = meta.get("duration")
        frame_paths = [Path(p) for p in meta.get("frame_paths", [])]
        video_path = meta.get("video_path")
        print(f"  ✓ 下载 · {duration}s · {len(frame_paths)} 帧 · {(meta.get('caption') or '')[:40]}")

        # 2. VLM(API) 与 CLIP+Whisper(本地) 两组并发
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_vlm = ex.submit(_describe_frames_concurrent, frame_paths)
            fut_local = ex.submit(_clip_then_whisper, frame_paths, video_path)
            frame_descs = fut_vlm.result()
            image_vec, transcript = fut_local.result()
        print(f"  ✓ VLM {len(frame_descs)} 描述 · CLIP {'ok' if image_vec else '-'} "
              f"· 转写 {len(transcript)} 字")

        # 3. 组装 row
        watched_at = datetime.now(timezone.utc).isoformat()
        upload_date = meta.get("upload_date")
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
        row_for_embed = {**row, "frame_desc": frame_descs}
        embedding_text = embed.build_embedding_text(row_for_embed)
        row["embedding_text"] = embedding_text

        # 4. 主 embedding 与 anchor 提炼并发
        anchors = None
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_vec = ex.submit(embed.embed, embedding_text)
            fut_anchor = (
                ex.submit(memory_anchors.extract, row) if ANCHORS_AVAILABLE else None
            )
            vec = fut_vec.result()
            if fut_anchor is not None:
                try:
                    anchors = fut_anchor.result()
                except Exception as e:
                    print(f"  ✗ anchor 提炼失败: {e}")
                    anchors = None

        elapsed = time.time() - t0
        print(f"  ✅ 处理完成 · {elapsed:.1f}s")
        return {"row": row, "vec": vec, "image_vec": image_vec, "anchors": anchors}

    except Exception as e:
        print(f"  ✗ 异常: {e}")
        traceback.print_exc(limit=2)
        return None
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    if "url" not in df.columns:
        raise ValueError("CSV 必须有 url 列")
    if "id" not in df.columns:
        df["id"] = [f"v{i+1:03d}" for i in range(len(df))]

    print(f"读到 {len(df)} 条 URL (ingest_fast · 砍 CLAP · 单视频内并发)\n")

    conn = store.connect()
    store.init_schema(conn)

    force = os.getenv("FORCE_REINGEST", "0").lower() in ("1", "true", "yes")
    existing_ids: set[str] = set()
    if not force:
        existing_ids = {r[0] for r in conn.execute("SELECT id FROM videos").fetchall()}
        if existing_ids:
            print(f"DB 已有 {len(existing_ids)} 条,自动跳过\n")

    tmp_root = Path(tempfile.mkdtemp(prefix="embers_fast_"))
    succeeded = skipped = failed = 0
    t_all = time.time()
    try:
        for _, csv_row in df.iterrows():
            video_id = str(csv_row["id"])
            url = str(csv_row["url"]).strip()
            if not url or url == "nan":
                continue
            if video_id in existing_ids:
                skipped += 1
                continue

            result = process_one_fast(url, video_id, tmp_root)
            if result is None:
                failed += 1
                continue

            store.upsert_video(
                conn, result["row"], result["vec"],
                image_embedding=result.get("image_vec"),
                audio_embedding=None,  # 砍 CLAP
            )
            if result.get("anchors"):
                try:
                    n = _store_anchors_fast(conn, result["row"]["id"], result["anchors"])
                    print(f"  · {n} anchors (vibe={result['anchors'].get('vibe')})")
                except Exception as e:
                    print(f"  ✗ anchors 入库失败: {e}")
            succeeded += 1

            if succeeded % 10 == 0:
                avg = (time.time() - t_all) / succeeded
                print(f"\n  ─── 进度: {succeeded} 入库 · 均 {avg:.1f}s/条 "
                      f"· 预计剩余 {avg * (len(df) - skipped - succeeded) / 60:.0f} 分钟 ───\n")

        total = store.count(conn)
        dt = time.time() - t_all
        print(f"\n{'='*60}")
        print(f"✅ ingest_fast 完成 · 用时 {dt/60:.1f} 分钟")
        print(f"   新增 {succeeded} · 跳过 {skipped} · 失败 {failed}")
        print(f"   均 {dt / max(succeeded,1):.1f}s/条 · DB 总 {total}")
        print(f"{'='*60}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/videos.csv"
    main(csv)
