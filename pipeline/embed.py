"""
Embedding 生成层 · 多 provider 切换

v0.2 关键变化:
  embedding 输入文本 = caption(yt-dlp metadata)+ transcript(Whisper)+ frame_desc ×N(VLM)
  全部客观信号,不依赖用户 notes。
"""
from __future__ import annotations

import json
import os
from typing import Iterable

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

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
    raise ValueError(f"EMBEDDING_PROVIDER={PROVIDER!r} 不支持。可选: {list(PROVIDERS)}")

_cfg = PROVIDERS[PROVIDER]
MODEL = os.getenv("EMBEDDING_MODEL", _cfg["default_model"])
NATIVE_DIM = _cfg["native_dim"]

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv(_cfg["api_key_env"])
        if not api_key:
            raise RuntimeError(
                f"未找到 {_cfg['api_key_env']}(EMBEDDING_PROVIDER={PROVIDER})"
            )
        _client = OpenAI(api_key=api_key, base_url=_cfg["base_url"])
    return _client


def build_embedding_text(row: dict) -> str:
    """把多路客观信号拼成 embedding 输入。
    顺序经过权重设计:transcript(口播内容)和 frame_desc(视觉内容)权重最高,
    caption 中等,title/hashtags 弱。
    """
    parts: list[str] = []

    if row.get("title"):
        parts.append(f"标题: {row['title']}")

    if row.get("caption") and row["caption"] != row.get("title"):
        cap = row["caption"][:400]  # 截断超长 caption
        parts.append(f"描述: {cap}")

    if row.get("transcript"):
        tr = row["transcript"][:800]  # 转写文本截断,避免单条太长
        parts.append(f"音频: {tr}")

    # frame_desc 是 JSON 字符串(从 DB 读)或 list[str](入库前)
    fd = row.get("frame_desc")
    if fd:
        if isinstance(fd, str):
            try:
                fd = json.loads(fd)
            except Exception:
                fd = [fd]
        if isinstance(fd, list) and fd:
            for i, d in enumerate(fd, 1):
                if d:
                    parts.append(f"画面{i}: {d}")

    return "\n".join(parts) if parts else (row.get("url") or "")


def embed(text: str) -> list[float]:
    """单条文本 → 向量。"""
    resp = client().embeddings.create(model=MODEL, input=text)
    return resp.data[0].embedding


def embed_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
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
        "caption": "搞笑萌宠日常,小猫第一次坐上扫地机器人",
        "transcript": "你看这只猫,它居然敢站在扫地机器人上面来回滑。哈哈哈太搞笑了。",
        "frame_desc": [
            "一只橘猫静止站在白色扫地机器人上",
            "扫地机器人启动,猫姿势紧张",
            "猫开始张嘴叫喊",
            "扫地机器人撞到家具,猫被甩出"
        ],
    }
    text = build_embedding_text(sample)
    print(f"\nEmbedding input:\n{text}\n")

    v = embed(text)
    print(f"✅ Vector length: {len(v)}")
    print(f"   Sample first 5: {v[:5]}")
    assert len(v) == NATIVE_DIM, f"维度 {len(v)} != {NATIVE_DIM}"
    print(f"   维度匹配 OK")
