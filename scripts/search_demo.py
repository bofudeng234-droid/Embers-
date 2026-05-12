"""
命令行 demo · 输入一句模糊回忆,返回 Top-K 视频。
用法:
  python scripts/search_demo.py "那个有猫在扫地机器人上的搞笑视频"
"""
from __future__ import annotations

import sys

from pipeline import store, embed


def main(query: str, top_k: int = 5) -> None:
    print(f"🔍 Query: {query}")
    conn = store.connect()
    store.init_schema(conn)

    q_vec = embed.embed(query)
    results = store.search(conn, q_vec, top_k=top_k)

    if not results:
        print("(没找到任何视频,你跑过 ingest 了吗?)")
        return

    print(f"\nTop {len(results)} 命中:\n")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['title']} · distance={r['distance']:.4f}")
        print(f"      URL: {r['url']}")
        print(f"      Notes: {r['notes']}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('用法: python scripts/search_demo.py "你的模糊回忆"')
        sys.exit(1)
    main(" ".join(sys.argv[1:]))
