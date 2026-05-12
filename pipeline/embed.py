"""
Embedding 生成 · OpenAI text-embedding-3-small (1536d) · 单条 + 批量。
"""
from __future__ import annotations

import os
from typing import Iterable

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client: OpenAI | None = None
MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def build_embedding_text(row: dict) -> str:
    """把视频的多个字段拼成一段语义文本,作为 embedding 的输入。
    顺序经过权重设计:notes(用户记忆描述)最重,caption 次之,hashtags 最弱。
    """
    parts: list[str] = []
    if row.get("notes"):
        parts.append(f"记忆: {row['notes']}")
    if row.get("title"):
        parts.append(f"标题: {row['title']}")
    if row.get("caption"):
        parts.append(f"内容: {row['caption']}")
    if row.get("hashtags"):
        parts.append(f"标签: {row['hashtags']}")
    return "\n".join(parts) if parts else (row.get("url") or "")


def embed(text: str) -> list[float]:
    """单条文本 → 向量。"""
    resp = client().embeddings.create(model=MODEL, input=text)
    return resp.data[0].embedding


def embed_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """批量 embed。OpenAI 单次支持多输入,效率更高。"""
    results: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        resp = client().embeddings.create(model=MODEL, input=chunk)
        # API 返回顺序与 input 一致
        results.extend([d.embedding for d in resp.data])
    return results


if __name__ == "__main__":
    sample = {
        "title": "猫在 Roomba 上",
        "caption": "猫咪骑在扫地机器人上来回滑行",
        "hashtags": "#搞笑 #猫",
        "notes": "记得它叫了一声很搞笑",
    }
    text = build_embedding_text(sample)
    print("Embedding input:\n", text)
    v = embed(text)
    print(f"\nVector length: {len(v)}  Sample first 5: {v[:5]}")
