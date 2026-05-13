"""
Chinese-CLIP 本地推理 · 真多模态 embedding

  · 图片 → 512d 向量(视觉编码器)
  · 中文 → 512d 向量(文本编码器)
  · 两者在同一个 512d 语义空间,可以直接 cosine 相似度比较

模型:OFA-Sys/chinese-clip-vit-base-patch16
  - ~400 MB 下载,首次启动需要等
  - 推理速度:CPU ~80-150ms / image,M 系列 Mac 用 MPS 可 ~30ms / image
  - 输出维度:512(投影层后)

设备配置:
  CLIP_DEVICE=cpu  (默认,最稳)
  CLIP_DEVICE=mps  (Apple Silicon,3-5x 提速,但偶有 op 兼容问题)
  CLIP_DEVICE=cuda (有 NVIDIA GPU 时用)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

MODEL_NAME = os.getenv("CLIP_MODEL", "OFA-Sys/chinese-clip-vit-base-patch16")
DEVICE_PREF = os.getenv("CLIP_DEVICE", "cpu").lower()

# 单例
_model: ChineseCLIPModel | None = None
_processor: ChineseCLIPProcessor | None = None
_device: str = "cpu"


def _pick_device() -> str:
    """挑选可用设备(CPU 默认,MPS 加速,CUDA)。"""
    if DEVICE_PREF == "mps" and torch.backends.mps.is_available():
        return "mps"
    if DEVICE_PREF == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load() -> tuple[ChineseCLIPModel, ChineseCLIPProcessor]:
    """惰性加载模型 + 处理器(首次约 5-15 秒,之后 cached)。"""
    global _model, _processor, _device
    if _model is None:
        _device = _pick_device()
        print(f"  [clip] 加载 {MODEL_NAME} · device={_device}")
        _processor = ChineseCLIPProcessor.from_pretrained(MODEL_NAME)
        _model = ChineseCLIPModel.from_pretrained(MODEL_NAME).to(_device)
        _model.eval()
        # 推理时禁用梯度,显著节省内存
        for p in _model.parameters():
            p.requires_grad = False
        print(f"  [clip] 模型就绪 · 输出维度 {_model.config.projection_dim}")
    return _model, _processor


def dim() -> int:
    """输出向量的维度(model.config.projection_dim)。"""
    m, _ = load()
    return int(m.config.projection_dim)


def image_to_vec(path: str | Path) -> list[float]:
    """图片 → 单位长度的 512d 向量。失败抛异常(让上层决定怎么处理)。"""
    m, p = load()
    image = Image.open(path).convert("RGB")
    inputs = p(images=image, return_tensors="pt").to(_device)
    with torch.no_grad():
        feats = m.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0).cpu().numpy().astype(np.float32).tolist()


def images_to_vecs(paths: list[str | Path]) -> list[list[float]]:
    """批量 · 4 张帧一起跑,避免每帧重复 model.forward 启动开销。"""
    if not paths:
        return []
    m, p = load()
    images = [Image.open(pp).convert("RGB") for pp in paths]
    inputs = p(images=images, return_tensors="pt").to(_device)
    with torch.no_grad():
        feats = m.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return [feats[i].cpu().numpy().astype(np.float32).tolist() for i in range(feats.shape[0])]


def text_to_vec(text: str) -> list[float]:
    """文本 → 单位长度的 512d 向量。与图片向量同一空间。"""
    m, p = load()
    inputs = p(text=[text], padding=True, return_tensors="pt").to(_device)
    with torch.no_grad():
        feats = m.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0).cpu().numpy().astype(np.float32).tolist()


def average_image_vecs(vecs: list[list[float]]) -> list[float]:
    """N 帧向量取平均 + 归一化 = 视频整体视觉指纹。
    简化版策略:不存每一帧,只存平均。
    """
    if not vecs:
        raise ValueError("空向量列表")
    arr = np.array(vecs, dtype=np.float32)
    mean = arr.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean = mean / norm
    return mean.astype(np.float32).tolist()


if __name__ == "__main__":
    import sys
    print(f"模型: {MODEL_NAME}")
    print(f"设备: {_pick_device()}\n")

    if len(sys.argv) > 1:
        # 测图片
        path = sys.argv[1]
        print(f"测试图片 → 向量: {path}")
        v = image_to_vec(path)
        print(f"  shape: ({len(v)},)  · norm: {sum(x*x for x in v)**0.5:.4f}")
        print(f"  前 5 维: {v[:5]}")

    # 测文本
    text = "猫骑在扫地机器人上"
    print(f"\n测试文本 → 向量: {text}")
    v = text_to_vec(text)
    print(f"  shape: ({len(v)},)  · norm: {sum(x*x for x in v)**0.5:.4f}")
    print(f"  前 5 维: {v[:5]}")
