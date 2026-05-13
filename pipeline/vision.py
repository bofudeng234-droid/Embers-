"""
VLM(GLM-4V-Flash)多帧场景描述

为每一帧生成一句中文描述(≤30 字),作为 embedding 输入的"画面"信号。
模型走智谱开放平台 / OpenAI 兼容协议,跟 embedding 用同一个 API key。
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

VLM_PROVIDER = os.getenv("VLM_PROVIDER", "zhipu")
VLM_PROVIDERS = {
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "api_key_env": "ZHIPU_API_KEY",
        "default_model": "glm-4v-flash",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
}

VLM_MODEL = os.getenv("VLM_MODEL")  # 不设则用 provider 默认

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        cfg = VLM_PROVIDERS[VLM_PROVIDER]
        api_key = os.getenv(cfg["api_key_env"])
        if not api_key:
            raise RuntimeError(
                f"未找到 {cfg['api_key_env']}(VLM_PROVIDER={VLM_PROVIDER})"
            )
        _client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
    return _client


def _model_name() -> str:
    return VLM_MODEL or VLM_PROVIDERS[VLM_PROVIDER]["default_model"]


def _encode_image(path: Path) -> str:
    """图片 → base64 data URL。"""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{b64}"


def describe_frame(frame_path: str | Path) -> str:
    """单帧 → 一句中文描述(≤30 字)。失败返回空串。"""
    frame_path = Path(frame_path)
    if not frame_path.exists():
        return ""

    try:
        resp = client().chat.completions.create(
            model=_model_name(),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _encode_image(frame_path)}},
                    {"type": "text", "text": (
                        "用一句简洁中文(≤30 字)描述这一帧的核心内容和氛围。"
                        "重点描述主体、动作、场景。"
                        "只输出描述,不要前缀、不要引号、不要 markdown。"
                    )},
                ],
            }],
            max_tokens=80,
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip().strip('"').strip("'")
    except Exception as e:
        print(f"  [vlm] 失败 ({frame_path.name}): {e}")
        return ""


def describe_frames(frame_paths: list[str | Path]) -> list[str]:
    """批量描述帧,串行调用(VLM 多模态接口通常不支持并发安全)。"""
    return [describe_frame(p) for p in frame_paths]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python -m pipeline.vision <frame.jpg> [<frame2.jpg> ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        print(f"\n描述 {path}:")
        desc = describe_frame(path)
        print(f"  → {desc}")
