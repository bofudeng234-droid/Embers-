"""
星空降维 · vec_videos 2048d → 3D 坐标 + 星系聚类

产出 star_coords(video_id, x, y, z, galaxy_id) 表 —— 纯衍生数据,
可随时全量重算覆盖,不碰检索层(vec_videos / videos / vec_anchors)。

降维: UMAP(cosine, 3D) · 聚类: sklearn HDBSCAN(在 3D 结果上聚,
保证"星空里看到的簇 = galaxy_id" 视觉一致)。

用法:
  python -m pipeline.starfield            # 全量重算 star_coords
  python -m pipeline.starfield --summary  # 重算 + 打印星系分布抽样
"""
from __future__ import annotations

import argparse
import struct
import sys
import time

import numpy as np

from pipeline import store

COORD_SCALE = 50.0  # 归一化到 [-50, 50] 立方体,Three.js 好摆


def _load_vectors(conn) -> tuple[list[str], np.ndarray]:
    """读所有 video 的 2048d 文本向量。返回 (video_ids, matrix[N,2048])。"""
    rows = conn.execute(
        """
        SELECT v.id, vec.embedding
        FROM vec_videos AS vec
        JOIN videos AS v ON v.rowid = vec.rowid
        ORDER BY v.rowid
        """
    ).fetchall()
    ids: list[str] = []
    vecs: list[np.ndarray] = []
    dim = store.EMBEDDING_DIM
    for vid, blob in rows:
        if blob is None:
            continue
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.shape[0] != dim:
            # 兼容:某些 sqlite-vec 版本可能多/少头,尝试截断
            arr = arr[:dim] if arr.shape[0] > dim else None
            if arr is None:
                continue
        ids.append(vid)
        vecs.append(arr)
    if not vecs:
        return [], np.empty((0, dim), dtype=np.float32)
    return ids, np.vstack(vecs)


def _reduce_3d(mat: np.ndarray) -> np.ndarray:
    """UMAP 2048d → 3D(cosine 度量,最适合文本 embedding)。"""
    import umap
    n = mat.shape[0]
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=min(15, max(2, n - 1)),
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    emb = reducer.fit_transform(mat)
    # 归一化到 [-COORD_SCALE, COORD_SCALE]
    mn, mx = emb.min(axis=0), emb.max(axis=0)
    span = np.where((mx - mn) == 0, 1.0, mx - mn)
    norm = (emb - mn) / span * 2 - 1  # [-1,1]
    return (norm * COORD_SCALE).astype(np.float32)


def _cluster(coords3d: np.ndarray) -> np.ndarray:
    """在 3D 结果上 HDBSCAN 聚星系。返回 galaxy_id(噪声点 = -1)。"""
    from sklearn.cluster import HDBSCAN
    n = coords3d.shape[0]
    mcs = max(5, n // 60)  # 规模自适应:点越多,最小星系越大
    clusterer = HDBSCAN(min_cluster_size=mcs, min_samples=3)
    return clusterer.fit_predict(coords3d)


def _write(conn, ids: list[str], coords: np.ndarray, galaxies: np.ndarray) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS star_coords (
            video_id TEXT PRIMARY KEY,
            x REAL, y REAL, z REAL,
            galaxy_id INTEGER,
            FOREIGN KEY (video_id) REFERENCES videos(id)
        )
    """)
    conn.execute("DELETE FROM star_coords")  # 全量重算:整表覆盖
    conn.executemany(
        "INSERT INTO star_coords(video_id, x, y, z, galaxy_id) VALUES (?,?,?,?,?)",
        [
            (vid, float(c[0]), float(c[1]), float(c[2]), int(g))
            for vid, c, g in zip(ids, coords, galaxies)
        ],
    )
    conn.commit()


def rebuild(summary: bool = False) -> int:
    conn = store.connect()
    store.init_schema(conn)
    t0 = time.time()

    ids, mat = _load_vectors(conn)
    if len(ids) < 5:
        print(f"向量太少({len(ids)}),先 ingest 再重算星空")
        return 0
    print(f"读到 {len(ids)} 条向量 · {mat.shape[1]}d → UMAP 3D ...")

    coords = _reduce_3d(mat)
    galaxies = _cluster(coords)
    _write(conn, ids, coords, galaxies)

    n_gal = len(set(int(g) for g in galaxies if g >= 0))
    n_noise = int((galaxies < 0).sum())
    dt = time.time() - t0
    print(f"✓ star_coords 重算完成 · {len(ids)} 星 · {n_gal} 星系 "
          f"· {n_noise} 游离星 · 用时 {dt:.1f}s")

    if summary:
        print("\n=== 星系分布抽样 ===")
        from collections import defaultdict
        members = defaultdict(list)
        for vid, g in zip(ids, galaxies):
            members[int(g)].append(vid)
        for g in sorted(members, key=lambda k: -len(members[k])):
            sample_ids = members[g][:4]
            ph = ",".join("?" * len(sample_ids))
            titles = conn.execute(
                f"SELECT substr(title,1,28) FROM videos WHERE id IN ({ph})",
                sample_ids,
            ).fetchall()
            label = "游离星" if g < 0 else f"星系 {g}"
            ts = " | ".join(t[0] or "?" for t in titles)
            print(f"  {label} ({len(members[g])} 星): {ts}")
    return len(ids)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    sys.exit(0 if rebuild(summary=args.summary) else 1)
