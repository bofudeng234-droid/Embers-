"""
批量入库 · 从 data/videos.csv 读 → 批量 embed → 写 SQLite + sqlite-vec。
用法:
  python -m pipeline.ingest data/videos.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pipeline import store, embed


def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    print(f"读到 {len(df)} 行视频元数据,开始入库...")

    conn = store.connect()
    store.init_schema(conn)

    # 1. 给每行构造 embedding 文本
    df["embedding_text"] = df.apply(lambda r: embed.build_embedding_text(r.to_dict()), axis=1)

    # 2. 批量 embed(每 100 条一批,降低 round-trip)
    texts = df["embedding_text"].tolist()
    print("调用 OpenAI 批量 embedding...")
    vectors = embed.embed_batch(texts, batch_size=100)
    assert len(vectors) == len(df), "embedding 数量不匹配输入"

    # 3. 逐行写入
    print("写入 SQLite + sqlite-vec...")
    for (_, row), vec in tqdm(zip(df.iterrows(), vectors), total=len(df)):
        store.upsert_video(conn, row.to_dict(), vec)

    total = store.count(conn)
    print(f"\n✅ 入库完成,共 {total} 条视频。DB 位置: {store.DB_PATH}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/videos.csv"
    main(csv_path)
