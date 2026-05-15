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


FRAME_DESC_PROMPT = """用中文详细描述这一帧画面,60-100 字。

**必须包含以下信息(看到了就写,没有就跳过,不要硬编)**:
1. 主体: 人/动物/物品 — 具体到品种、品牌、型号(例: "蓝金渐层猫" 而不是 "猫"; "奔驰大G" 而不是 "SUV")
2. 动作/表情: 在做什么、表情情绪(例: "倒立、嘴角微笑")
3. 服饰外观: 颜色 + 款式 + 标志性细节(例: "白色瑜伽服 + 红色发带")
4. 场景: 地点类型 + 标志物 + 时间段(例: "公园湖边台阶,白天")
5. 字幕/文字: 画面上出现的所有文字内容(标题、字幕、品牌 logo、商品名)
6. 视觉风格: 主色调、画质感(例: "冷色调、复古滤镜")

**不要写**:
- "图中" "画面里" 等前缀
- 引号、markdown、句号外的标点结尾
- 解释性废话("这是一个有趣的场景"之类)

只输出描述本身。"""


def describe_frame(frame_path: str | Path) -> str:
    """单帧 → 一段中文详细描述(60-100 字)。失败返回空串。

    v0.8 prompt 升级:从 30 字简描升级到 60-100 字结构化描述,
    显式要求字幕/品牌/配色/品种等区分度信号 — 解决"黑色SUV开门"
    这种过于通用的描述无法区分多条豪车视频的问题。
    """
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
                    {"type": "text", "text": FRAME_DESC_PROMPT},
                ],
            }],
            max_tokens=250,
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
