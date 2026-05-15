"""
backfill memory_anchors 到现有 DB 视频
不重跑 download / CLIP / VLM,只对已入库视频跑 anchor 提炼并入 vec_anchors。

用法:
  python -m pipeline.backfill_anchors            # 跑所有视频
  python -m pipeline.backfill_anchors --ids v007 # 只跑指定 id
  python -m pipeline.backfill_anchors --force    # 已有 anchor 的视频也重跑
"""
from __future__ import annotations

import argparse
import sys
import time

from pipeline import store, embed, memory_anchors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default=None, help="逗号分隔 id 列表(默认全部)")
    parser.add_argument("--force", action="store_true", help="已有 anchors 的也重跑")
    args = parser.parse_args()

    conn = store.connect()
    store.init_schema(conn)

    if args.ids:
        target_ids = [s.strip() for s in args.ids.split(",") if s.strip()]
    else:
        rows = conn.execute("SELECT id FROM videos ORDER BY id").fetchall()
        target_ids = [r[0] for r in rows]

    print(f"=== backfill memory_anchors · {len(target_ids)} 条候选 ===\n")
    t0 = time.time()
    n_ok = 0
    n_skip = 0
    n_fail = 0
    for vid in target_ids:
        row = conn.execute(
            "SELECT id, title, caption, transcript, frame_desc, duration_sec FROM videos WHERE id = ?",
            (vid,),
        ).fetchone()
        if not row:
            print(f"  [{vid}] 不存在,跳过")
            continue
        existing = conn.execute("SELECT COUNT(*) FROM anchors WHERE video_id = ?", (vid,)).fetchone()[0]
        if existing and not args.force:
            print(f"  [{vid}] 已有 {existing} 条 anchor,跳过(用 --force 覆盖)")
            n_skip += 1
            continue

        v = dict(zip(["id", "title", "caption", "transcript", "frame_desc", "duration_sec"], row))
        print(f"  [{vid}] {(v['title'] or '')[:40]}")
        anchors = memory_anchors.extract(v)
        if not anchors:
            n_fail += 1
            continue
        rows = memory_anchors.to_anchor_rows(vid, anchors)
        n = store.upsert_anchors(conn, vid, rows, embed.embed)
        print(f"        → 写入 {n} 条 anchors · vibe={anchors.get('vibe')} · domain={','.join(anchors.get('domain_tags') or [])}")
        n_ok += 1

    elapsed = time.time() - t0
    total_anchors = store.count_anchors(conn)
    print(f"\n=== 完成 · 用时 {elapsed:.1f}s ===")
    print(f"  视频: {n_ok} ok · {n_skip} skip · {n_fail} fail")
    print(f"  DB 当前 anchor 总数: {total_anchors}")


if __name__ == "__main__":
    main()
