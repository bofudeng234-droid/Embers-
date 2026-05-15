"""
ingest 并行版 · 阶段 2 · 多进程视频级并发

架构 (producer-consumer):
  - N 个 worker 进程: 各自独立 Playwright profile + 独立浏览器,
    各自跑完整 process_one_fast (download/CLIP/VLM/Whisper/embed/anchor)
  - 主进程 = producer(派发任务) + 单一 DB writer(串行写,无 SQLite 锁冲突)
  - worker 算完结果经 Queue 发回主进程写库

关键取舍:
  - worker 用独立临时 profile(不登录) → 部分视频 Whisper transcript 降级,
    但 frame 截图/caption/VLM/anchor 不受影响(检索主力信号都在)
  - 砍 CLAP(继承 ingest_fast)

预期: 4 worker → ~4x;  450 条 ~2.5h → ~35-45min

用法:
  python -m pipeline.ingest_parallel data/videos.csv --workers 4
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import pandas as pd

from pipeline import store


def _worker(idx: int, task_q: mp.Queue, result_q: mp.Queue) -> None:
    """子进程: 设独立 profile → 延迟 import → 循环处理任务。"""
    # 必须在 import pipeline.download 之前设好(download.py 的 PROFILE_DIR
    # 是模块级常量,import 时求值)。spawn 模式下子进程重新 import,OK。
    os.environ["PW_PROFILE_DIR"] = f"/tmp/embers_pw_w{idx}"
    os.environ["PW_HEADLESS"] = "1"

    import tempfile
    from pipeline.ingest_fast import process_one_fast

    tmp_root = Path(tempfile.mkdtemp(prefix=f"embers_w{idx}_"))
    try:
        while True:
            task = task_q.get()
            if task is None:  # sentinel
                break
            vid, url = task
            try:
                res = process_one_fast(url, vid, tmp_root)
            except Exception as e:
                print(f"  [w{idx}] {vid} 异常: {e}")
                res = None
            if res is None:
                result_q.put({"status": "fail", "video_id": vid})
            else:
                result_q.put({
                    "status": "ok",
                    "video_id": vid,
                    "row": res["row"],
                    "vec": res["vec"],
                    "image_vec": res.get("image_vec"),
                    "anchors": res.get("anchors"),
                })
    finally:
        import shutil
        shutil.rmtree(tmp_root, ignore_errors=True)
        result_q.put({"status": "worker_done", "worker": idx})


def main(csv_path: str, workers: int = 4) -> None:
    df = pd.read_csv(csv_path)
    if "url" not in df.columns:
        raise ValueError("CSV 必须有 url 列")
    if "id" not in df.columns:
        df["id"] = [f"v{i+1:03d}" for i in range(len(df))]

    conn = store.connect()
    store.init_schema(conn)

    force = os.getenv("FORCE_REINGEST", "0").lower() in ("1", "true", "yes")
    existing: set[str] = set()
    if not force:
        existing = {r[0] for r in conn.execute("SELECT id FROM videos").fetchall()}

    tasks: list[tuple[str, str]] = []
    for _, r in df.iterrows():
        vid = str(r["id"])
        url = str(r["url"]).strip()
        if not url or url == "nan" or vid in existing:
            continue
        tasks.append((vid, url))

    total = len(tasks)
    if total == 0:
        print("没有待处理的新视频(都已入库)")
        return

    print(f"并行 ingest · {total} 条待处理 · {workers} workers · 已跳过 {len(existing)}\n")

    task_q: mp.Queue = mp.Queue()
    result_q: mp.Queue = mp.Queue()
    for t in tasks:
        task_q.put(t)
    for _ in range(workers):
        task_q.put(None)  # 每个 worker 一个退出 sentinel

    procs = [
        mp.Process(target=_worker, args=(i, task_q, result_q), daemon=True)
        for i in range(workers)
    ]
    for p in procs:
        p.start()

    from pipeline.ingest_fast import _store_anchors_fast

    done_workers = 0
    succeeded = failed = 0
    t0 = time.time()
    while done_workers < workers:
        msg = result_q.get()
        st = msg["status"]
        if st == "worker_done":
            done_workers += 1
            continue
        if st == "fail":
            failed += 1
            print(f"  ✗ {msg['video_id']} 失败")
            continue
        # ok → 主进程串行写库
        try:
            store.upsert_video(
                conn, msg["row"], msg["vec"],
                image_embedding=msg.get("image_vec"),
                audio_embedding=None,
            )
            if msg.get("anchors"):
                _store_anchors_fast(conn, msg["row"]["id"], msg["anchors"])
            succeeded += 1
            done = succeeded + failed
            elapsed = time.time() - t0
            avg = elapsed / max(succeeded, 1)
            eta = avg * (total - done) / 60
            print(f"  ✓ [{done}/{total}] {msg['video_id']} "
                  f"· 均 {avg:.1f}s/条 · ETA {eta:.0f}min")
        except Exception as e:
            failed += 1
            print(f"  ✗ {msg['video_id']} 写库失败: {e}")

    for p in procs:
        p.join(timeout=10)

    dt = time.time() - t0
    print(f"\n{'='*60}")
    print(f"✅ 并行 ingest 完成 · 用时 {dt/60:.1f} 分钟")
    print(f"   新增 {succeeded} · 失败 {failed} · DB 总 {store.count(conn)}")
    print(f"   有效并行均速 {dt/max(succeeded,1):.1f}s/条 (×{workers} workers)")
    print(f"{'='*60}")


if __name__ == "__main__":
    # macOS Python 3.14 默认 spawn;显式声明更稳
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="data/videos.csv")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    main(args.csv, workers=args.workers)
