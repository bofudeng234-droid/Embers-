"""
评测 RAG 召回质量 · 用 simulate_recall 生成的 (query, target_id) 评测集打 /search API

输出指标:
  recall@k:  target_id 在前 k 名内的比例
  MRR:       Mean Reciprocal Rank (1/rank,没命中算 0)
  按 memory_type 分桶看哪种记忆类型最难召回

用法:
  python -m pipeline.eval_recall                 # 读 data/eval_queries.csv,打本地 server
  python -m pipeline.eval_recall --csv X.csv     # 用别的评测集
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import requests


def run_one(server: str, query: str, top_k: int = 10) -> list[str]:
    """调 /search 拿 hit ids,失败返回空列表。"""
    try:
        r = requests.post(
            f"{server}/search",
            json={"query": query, "top_k": top_k, "expand": True, "rerank": False},
            timeout=30,
        )
        r.raise_for_status()
        return [h["id"] for h in r.json().get("hits", [])]
    except Exception as e:
        print(f"  ! /search 失败: {e}")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/eval_queries.csv")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"找不到 {csv_path}")
        sys.exit(1)

    cases = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("query") or "").strip()
            t = (row.get("target_id") or "").strip()
            if q and t:
                cases.append((q, t))

    if not cases:
        print("评测集为空 (CSV 里所有 query 都失败了?)")
        sys.exit(1)

    print(f"=== 跑 {len(cases)} 条评测 · server={args.server} ===\n")

    n_at = {1: 0, 3: 0, 5: 0, 10: 0}
    mrr_sum = 0.0
    misses = []

    for i, (query, target) in enumerate(cases, 1):
        hits = run_one(args.server, query, top_k=args.top_k)
        rank = hits.index(target) if target in hits else -1
        rr = 1.0 / (rank + 1) if rank >= 0 else 0.0
        mrr_sum += rr
        for k in n_at:
            if 0 <= rank < k:
                n_at[k] += 1
        flag = f"@{rank+1}" if rank >= 0 else "MISS"
        if args.verbose or rank == -1 or rank >= 5:
            print(f"  [{i:2}] {target} ← {query!r:38} {flag}")
        if rank == -1:
            misses.append((query, target, hits[:5]))

    n = len(cases)
    print()
    print(f"=== 评测结果 ({n} 条) ===")
    for k in [1, 3, 5, 10]:
        pct = n_at[k] / n * 100
        print(f"  recall@{k:<2} = {n_at[k]:>2}/{n} = {pct:5.1f}%")
    print(f"  MRR      = {mrr_sum / n:.3f}")

    if misses:
        print(f"\n=== MISS 详情 ({len(misses)} 条全没命中) ===")
        for q, t, top5 in misses:
            print(f"  query={q!r} · expected={t} · top5={top5}")


if __name__ == "__main__":
    main()
