"""
LAION-CLAP 本地音频 embedding · 音频版 CLIP

  · 音频 → 512d 向量(audio encoder)
  · 文字 → 512d 向量(text encoder · 跟 CLIP-text 不同模型,但同一向量空间)

模型:laion/clap-htsat-unfused
  - ~500 MB 下载
  - 输出 512d
  - 支持音乐 / 语音 / 环境音 / 音效

跟 CLIP 互补:
  - CLIP 处理"画面"
  - CLAP 处理"声音"(乐曲风格、人声语气、音效氛围)
  - 两者向量空间不通,需要分别检索
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

MODEL_NAME = os.getenv("CLAP_MODEL", "laion/clap-htsat-unfused")
DEVICE_PREF = os.getenv("CLAP_DEVICE", "cpu").lower()
SAMPLE_RATE = 48000  # CLAP 要求 48kHz 输入

_model: ClapModel | None = None
_processor: ClapProcessor | None = None
_device: str = "cpu"


def _pick_device() -> str:
    if DEVICE_PREF == "mps" and torch.backends.mps.is_available():
        return "mps"
    if DEVICE_PREF == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load() -> tuple[ClapModel, ClapProcessor]:
    global _model, _processor, _device
    if _model is None:
        _device = _pick_device()
        print(f"  [clap] 加载 {MODEL_NAME} · device={_device}")
        _processor = ClapProcessor.from_pretrained(MODEL_NAME)
        _model = ClapModel.from_pretrained(MODEL_NAME).to(_device)
        _model.eval()
        for p in _model.parameters():
            p.requires_grad = False
        print(f"  [clap] 模型就绪")
    return _model, _processor


def _normalize(t: torch.Tensor) -> torch.Tensor:
    return t / t.norm(dim=-1, keepdim=True)


def _load_audio(path: str | Path) -> np.ndarray:
    """读音频文件 → 48kHz 单通道 float32 numpy。"""
    import librosa
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def _to_tensor(out, m, modality: str) -> torch.Tensor:
    """兼容 transformers 不同版本:get_*_features 可能返回 tensor 或 ModelOutput。"""
    if isinstance(out, torch.Tensor):
        return out

    proj_dim = getattr(m.config, "projection_dim", None)
    # CLAP 的投影层有几种命名: audio_projection / text_projection
    proj_attr = "audio_projection" if modality == "audio" else "text_projection"
    proj = getattr(m, proj_attr, None)
    proj_in = getattr(proj, "in_features", None) if proj else None
    proj_out = getattr(proj, "out_features", None) if proj else None

    # 优先用 embeds 字段(若有)
    embed_attr = f"{modality}_embeds"
    if hasattr(out, embed_attr) and getattr(out, embed_attr) is not None:
        return getattr(out, embed_attr)

    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        pool = out.pooler_output
        d = pool.shape[-1]
        if proj_dim is not None and d == proj_dim:
            return pool
        if proj_out is not None and d == proj_out:
            return pool
        if proj is not None and proj_in is not None and d == proj_in:
            return proj(pool)
        return pool

    raise TypeError(f"无法从 {type(out).__name__} 提取 {modality} 特征")


def audio_to_vec(path: str | Path) -> list[float]:
    """音频文件 → 单位长度 512d 向量。
    支持 mp3 / wav / m4a / mp4(后者会自动提取音轨)等。
    """
    m, p = load()
    audio = _load_audio(path)
    try:
        inputs = p(audio=audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(_device)
    except TypeError:
        inputs = p(audios=audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(_device)
    with torch.no_grad():
        raw = m.get_audio_features(**inputs)
        feats = _normalize(_to_tensor(raw, m, "audio"))
    return feats.squeeze(0).cpu().numpy().astype(np.float32).tolist()


def text_to_vec(text: str) -> list[float]:
    """文字 → 单位长度 512d 向量(对齐音频空间)。"""
    m, p = load()
    inputs = p(text=[text], return_tensors="pt", padding=True).to(_device)
    with torch.no_grad():
        raw = m.get_text_features(**inputs)
        feats = _normalize(_to_tensor(raw, m, "text"))
    return feats.squeeze(0).cpu().numpy().astype(np.float32).tolist()


def dim() -> int:
    """输出维度。"""
    return 512  # CLAP 默认 512


if __name__ == "__main__":
    import sys
    print(f"模型: {MODEL_NAME}")
    print(f"设备: {_pick_device()}\n")

    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"测试音频 → 向量: {path}")
        v = audio_to_vec(path)
        print(f"  shape: ({len(v)},)  · norm: {sum(x*x for x in v)**0.5:.4f}")
        print(f"  前 5 维: {v[:5]}")

    text = "悠扬的钢琴音乐配上女声哼唱"
    print(f"\n测试文本 → 向量: {text}")
    v = text_to_vec(text)
    print(f"  shape: ({len(v)},)  · norm: {sum(x*x for x in v)**0.5:.4f}")
    print(f"  前 5 维: {v[:5]}")
