"""
bge-reranker 精排层 · v0.8

作用:
  embedding 召回(快但粗) → reranker 精排(慢但准)
  对 top_k * 2 召回结果用 cross-encoder 重新打分,提升 recall@1。

模型: BAAI/bge-reranker-v2-m3
  - 多语言(中文 + 英文 + ...) · 568M 参数
  - 输入 [query, doc] pair → 输出相关性分数(higher = better)
  - 跟 embedding 的"语义相似"不同,reranker 是真"对比阅读"

性能(M 系列 Mac MPS):
  - 加载 ~1.1 GB 模型(首次)
  - 10 个 pair rerank ~100-300ms · 比 LLM rerank 快 10-50x

调用:
  from pipeline.reranker import rerank
  reranked_hits = rerank("我的 query", hits)
"""
from __future__ import annotations

import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
DEVICE_PREF = os.getenv("RERANKER_DEVICE", "auto").lower()

_tokenizer: AutoTokenizer | None = None
_model: AutoModelForSequenceClassification | None = None
_device: str = "cpu"


def _pick_device() -> str:
    if DEVICE_PREF in ("auto", "mps") and torch.backends.mps.is_available():
        return "mps"
    if DEVICE_PREF in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load() -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    global _tokenizer, _model, _device
    if _model is None:
        _device = _pick_device()
        print(f"  [reranker] 加载 {MODEL_NAME} · device={_device}")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(_device)
        _model.eval()
        for p in _model.parameters():
            p.requires_grad = False
        print(f"  [reranker] 模型就绪")
    return _tokenizer, _model


def _default_doc(hit: dict) -> str:
    """从一条命中拼出供 reranker 阅读的"文档文本"。
    顺序:title → caption → frame_desc → transcript;每段截断防止 token 溢出。
    """
    parts: list[str] = []
    if hit.get("title"):
        parts.append(str(hit["title"]))
    if hit.get("caption"):
        parts.append(str(hit["caption"])[:300])
    if hit.get("frame_desc"):
        fd = hit["frame_desc"]
        if isinstance(fd, str):
            parts.append(fd[:400])
    if hit.get("transcript"):
        parts.append(str(hit["transcript"])[:300])
    if hit.get("anchor_text"):  # anchor 路命中的具体钩子
        parts.append(f"[hook] {hit['anchor_text']}")
    return " | ".join(parts)


def score(query: str, docs: list[str], max_length: int = 512) -> list[float]:
    """对 (query, doc) pairs 打分。返回 docs 长度的 score 列表(越大越相关)。"""
    if not docs:
        return []
    tok, mdl = load()
    pairs = [[query, d] for d in docs]
    inputs = tok(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    ).to(_device)
    with torch.no_grad():
        scores = mdl(**inputs, return_dict=True).logits.view(-1).float().cpu().numpy().tolist()
    return scores


def rerank(
    query: str,
    hits: list[dict],
    doc_fn=None,
    top_k: int | None = None,
) -> list[dict]:
    """对 hits 列表用 cross-encoder 重新排序。

    - doc_fn(hit) → str (从 hit 提取供 reranker 阅读的文本,默认拼 title+caption+...)
    - top_k=None 返回全部,否则只取前 top_k
    - 每条 hit 注入 _rerank_score 字段
    """
    if not hits:
        return hits
    if doc_fn is None:
        doc_fn = _default_doc

    docs = [doc_fn(h) for h in hits]
    scores = score(query, docs)
    paired = list(zip(hits, scores))
    paired.sort(key=lambda x: -x[1])

    out: list[dict] = []
    for h, s in paired:
        h["_rerank_score"] = float(s)
        out.append(h)
    if top_k is not None:
        return out[:top_k]
    return out


if __name__ == "__main__":
    # 单测
    q = "公园里倒立的女生"
    candidates = [
        {"id": "v001", "title": "公园瑜伽倒立", "caption": "湖边台阶上做瑜伽"},
        {"id": "v002", "title": "奔驰大G停地库", "caption": "白色大G停车"},
        {"id": "v014", "title": "猫坐着睡", "caption": "小猫不趴下睡"},
    ]
    print(f"query: {q}")
    reranked = rerank(q, candidates)
    for h in reranked:
        print(f"  {h['id']} score={h['_rerank_score']:+.3f} {h['title']}")
