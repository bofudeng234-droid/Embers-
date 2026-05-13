"""
Whisper 转写层(faster-whisper · CT2 量化,M 系列 Mac 友好)

模型选择:
  small  (~244MB 量化后) — 中文 hackathon 视频默认够用
  base   (~74MB)        — 更快但偶尔漏短句
  medium (~769MB)       — 更准但慢一倍
"""
from __future__ import annotations

import os
from pathlib import Path

from faster_whisper import WhisperModel

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "auto")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "zh")

_model: WhisperModel | None = None


def model() -> WhisperModel:
    """惰性加载模型(首次约 30 秒下载 + 加载,之后内存常驻)。"""
    global _model
    if _model is None:
        print(f"  [whisper] 首次加载模型 {WHISPER_MODEL_SIZE}...")
        _model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE,
        )
        print(f"  [whisper] 模型就绪")
    return _model


def transcribe(audio_or_video_path: str | Path) -> str:
    """转写,返回单段拼接的文本。失败返回空串。"""
    path = str(audio_or_video_path)
    if not Path(path).exists():
        return ""

    try:
        segments, _info = model().transcribe(
            path,
            language=WHISPER_LANGUAGE,
            beam_size=1,             # 速度优先;hackathon 不追求最佳精度
            vad_filter=True,         # 跳过静音段,显著加速
            vad_parameters={"min_silence_duration_ms": 500},
        )
        return " ".join(s.text.strip() for s in segments).strip()
    except Exception as e:
        print(f"  [whisper] 失败: {e}")
        return ""


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python -m pipeline.transcribe <video_or_audio_path>")
        sys.exit(1)
    p = sys.argv[1]
    print(f"测试转写: {p}")
    text = transcribe(p)
    if text:
        print("✅ 转写结果:")
        print(text[:500] + ("..." if len(text) > 500 else ""))
    else:
        print("❌ 转写为空")
