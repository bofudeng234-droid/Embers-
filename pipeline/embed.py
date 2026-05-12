"""
Embedding 生成层 · 多 provider 切换 · 统一通过 OpenAI 兼容协议接入

支持的 provider(在 .env 用 EMBEDDING_PROVIDER 切换):
  zhipu   智谱 GLM embedding-3(默认,中文场景推荐)
  openai  OpenAI text-embedding-3-small
"""
from __future__ import annotations

import os
from typing import Iterable

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============ Provider 配置 ============
PROVIDERS = {
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "api_key_env": "ZHIPU_API_KEY",
        "default_model": "embedding-3",
        "native_dim": 2048,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "text-embedding-3-small",
        "native_dim": 1536,
    },
}

PROVIDER = os.getenv("EMBEDDING_PROVIDER", "zhipu").lower()
if PROVIDER not in PROVIDERS:
    raise ValueError(
        f"EMBEDDING_PROVIDER={PROVIDER!r} 不支持。可选: {list(PROVIDERS)}"
    )

_cfg = PROVIDERS[PROVIDER]
MODEL = os.getenv("EMBEDDING_MODEL", _cfg["default_model"])
NATIVE_DIM = _cfg["native_dim"]

_client: OpenAI | None = None


def client() -> OpenAI:
    """惰性初始化 OpenAI 兼容客户端。"""
    global _client
    if _client is None:
        api_key = os.getenv(_cfg["api_key_env"])
        if not api_key:
            raise RuntimeError(
                f"未找到 {_cfg['api_key_env']}。请在 .env 里配置 "
                f"{_cfg['api_key_env']}=...(provider={PROVIDER})"
            )
        _client = OpenAI(api_key=api_key, base_url=_cfg["base_url"])
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


def embed_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """批量 embed。单次请求多 input,降低 round-trip 开销。
    智谱 embedding-3 单次最多 64 条,OpenAI 最多 2048 条,我们用 64 兼容两边。
    """
    results: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        resp = client().embeddings.create(model=MODEL, input=chunk)
        results.extend([d.embedding for d in resp.data])
    return results


if __name__ == "__main__":
    print(f"Provider: {PROVIDER}")
    print(f"Model:    {MODEL}")
    print(f"Native dim: {NATIVE_DIM}")

    sample = {
        "title": "猫在 Roomba 上",
        "caption": "猫咪骑在扫地机器人上来回滑行",
        "hashtags": "#搞笑 #猫",
        "notes": "记得它叫了一声很搞笑",
    }
    text = build_embedding_text(sample)
    print(f"\nEmbedding input:\n{text}")

    v = embed(text)
    print(f"\n✅ Vector length: {len(v)}")
    print(f"   Sample first 5: {v[:5]}")
    assert len(v) == NATIVE_DIM, f"模型返回维度 {len(v)} 不等于声明的 {NATIVE_DIM}"
    print(f"   维度匹配 OK")
