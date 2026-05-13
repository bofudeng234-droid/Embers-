"""
Embers · FastAPI 服务层

启动:
  cd /Users/bofu/embers
  source .venv/bin/activate
  uvicorn api.server:app --reload --host 127.0.0.1 --port 8000

接口:
  GET  /health           健康检查 + 当前 DB 视频条数
  GET  /videos           列出全部视频(分页,用于"余烬墙")
  POST /search           接收模糊回忆 query → 返回 top-K + 可选 LLM 解释
  GET  /                 静态前端(/web/index.html)
"""
from __future__ import annotations

import os
import time
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

from pipeline import store, embed
try:
    from pipeline import clip_embed
    CLIP_AVAILABLE = True
except Exception as _e:
    print(f"[server] CLIP 不可用(将只用文本检索): {_e}")
    CLIP_AVAILABLE = False

load_dotenv()

app = FastAPI(
    title="Embers / 余烬",
    description="找回你刷过的每一条视频。",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ LLM provider 配置(rerank 用) ============
LLM_PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "api_key_env": "MOONSHOT_API_KEY",
        "default_model": "moonshot-v1-8k",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "api_key_env": "ZHIPU_API_KEY",
        "default_model": "glm-4-flash",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").lower()
LLM_MODEL = os.getenv("LLM_MODEL")  # 不设则用 provider 默认

_llm_client: OpenAI | None = None


def llm_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        if LLM_PROVIDER not in LLM_PROVIDERS:
            raise RuntimeError(f"LLM_PROVIDER={LLM_PROVIDER} 不支持")
        cfg = LLM_PROVIDERS[LLM_PROVIDER]
        api_key = os.getenv(cfg["api_key_env"])
        if not api_key:
            raise RuntimeError(
                f"未找到 {cfg['api_key_env']}。请在 .env 里配置(provider={LLM_PROVIDER})"
            )
        _llm_client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
    return _llm_client


def llm_model_name() -> str:
    if LLM_MODEL:
        return LLM_MODEL
    return LLM_PROVIDERS[LLM_PROVIDER]["default_model"]


# ============ Models ============
class HealthResponse(BaseModel):
    status: str
    video_count: int
    video_count_with_image: int
    embedding_dim: int
    embedding_provider: str
    llm_provider: str
    clip_available: bool


class VideoRow(BaseModel):
    id: str
    url: str
    title: str | None = None
    caption: str | None = None
    transcript: str | None = None
    frame_desc: str | None = None
    duration_sec: int | None = None
    watched_at: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(5, ge=1, le=20)
    rerank: bool = Field(False, description="是否启用 LLM 解释(慢一点但更有说服力)")


class SearchHit(VideoRow):
    distance: float
    explanation: str | None = None


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    latency_ms: int


# ============ Helpers ============
def get_conn():
    conn = store.connect()
    store.init_schema(conn)
    return conn


# ============ Endpoints ============
@app.get("/health", response_model=HealthResponse)
def health():
    conn = get_conn()
    try:
        return HealthResponse(
            status="ok",
            video_count=store.count(conn),
            video_count_with_image=store.count_with_image(conn),
            embedding_dim=store.EMBEDDING_DIM,
            embedding_provider=embed.PROVIDER,
            llm_provider=LLM_PROVIDER,
            clip_available=CLIP_AVAILABLE,
        )
    finally:
        conn.close()


@app.get("/videos", response_model=list[VideoRow])
def list_videos(limit: int = 200, offset: int = 0):
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, url, title, caption, transcript, frame_desc, duration_sec, watched_at
            FROM videos
            ORDER BY watched_at DESC NULLS LAST
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        return [
            VideoRow(
                id=r[0], url=r[1], title=r[2], caption=r[3],
                transcript=r[4], frame_desc=r[5],
                duration_sec=r[6], watched_at=r[7],
            )
            for r in rows
        ]
    finally:
        conn.close()


def _rrf_fuse(text_hits: list[dict], image_hits: list[dict], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion · 把文本路和视觉路的命中按"倒数排名分"加权融合。
    经典 hybrid retrieval 算法,公式:score(d) = Σ 1 / (k + rank(d))
    """
    scores: dict[str, float] = {}
    payload: dict[str, dict] = {}

    for rank, h in enumerate(text_hits):
        vid = h["id"]
        scores[vid] = scores.get(vid, 0.0) + 1.0 / (k + rank)
        payload.setdefault(vid, h).setdefault("_source", set()).add("text")

    for rank, h in enumerate(image_hits):
        vid = h["id"]
        scores[vid] = scores.get(vid, 0.0) + 1.0 / (k + rank)
        d = payload.setdefault(vid, h)
        d.setdefault("_source", set()).add("image")
        # 用更小的 distance(更近的命中)
        if h.get("distance", 1e9) < d.get("distance", 1e9):
            d["distance"] = h["distance"]

    # 按融合得分降序
    sorted_ids = sorted(scores.keys(), key=lambda v: -scores[v])
    out: list[dict] = []
    for vid in sorted_ids:
        d = payload[vid]
        d["_score"] = scores[vid]
        d["_source"] = "+".join(sorted(d["_source"]))
        out.append(d)
    return out


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    t0 = time.time()
    q_text = req.query.strip()

    # ============ 双轨向量化 ============
    try:
        q_text_vec = embed.embed(q_text)
    except Exception as e:
        raise HTTPException(500, f"text embedding 失败: {e}")

    q_image_vec: list[float] | None = None
    if CLIP_AVAILABLE:
        try:
            q_image_vec = clip_embed.text_to_vec(q_text)
        except Exception as e:
            print(f"  [/search] CLIP 编码失败,降级到纯文本检索: {e}")
            q_image_vec = None

    # ============ 双轨检索 ============
    conn = get_conn()
    try:
        text_hits = store.search_text(conn, q_text_vec, top_k=req.top_k * 2)
        image_hits: list[dict] = []
        if q_image_vec is not None and store.count_with_image(conn) > 0:
            image_hits = store.search_image(conn, q_image_vec, top_k=req.top_k * 2)
    finally:
        conn.close()

    # ============ RRF 融合 ============
    if image_hits:
        raw_hits = _rrf_fuse(text_hits, image_hits)[:req.top_k]
    else:
        # 没有视觉向量(库还没 ingest CLIP 信号),回退纯文本
        raw_hits = text_hits[:req.top_k]

    explanations: dict[str, str] = {}
    if req.rerank and raw_hits:
        try:
            explanations = _llm_explain(q_text, raw_hits)
        except Exception:
            explanations = {}

    hits = [
        SearchHit(
            id=h["id"], url=h["url"], title=h["title"], caption=h["caption"],
            transcript=h.get("transcript"), frame_desc=h.get("frame_desc"),
            duration_sec=h.get("duration_sec"), watched_at=h.get("watched_at"),
            distance=h["distance"],
            explanation=explanations.get(h["id"]),
        )
        for h in raw_hits
    ]

    return SearchResponse(
        query=q_text, hits=hits,
        latency_ms=int((time.time() - t0) * 1000),
    )


def _llm_explain(query: str, hits: list[dict]) -> dict[str, str]:
    """用配置的 LLM(默认 DeepSeek)给 top hits 生成"我猜是这条因为..."的解释。"""
    top = hits[:3]
    candidates_text = "\n".join(
        f"[{h['id']}] 标题: {h.get('title') or '(无)'} | 笔记: {h.get('notes') or '(无)'} | caption: {h.get('caption') or '(无)'}"
        for h in top
    )

    prompt = f"""用户的模糊回忆: "{query}"

下面是从他观看历史里检索到的最相似的几条视频:

{candidates_text}

对每条视频,用一句话(≤25 字)解释"为什么这条可能是用户在找的"。
只输出 JSON,key 是视频 id,value 是解释字符串。不要任何 markdown 或多余文字。
示例: {{"v001": "提到了猫骑在扫地机器人上还叫了一声"}}
"""

    resp = llm_client().chat.completions.create(
        model=llm_model_name(),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.3,
    )

    text = resp.choices[0].message.content.strip()
    # 容错:LLM 偶尔会包 ```json
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        return {}


# ============ Static frontend ============
_WEB_DIR = Path(__file__).parent.parent / "web"
if _WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=_WEB_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(_WEB_DIR / "index.html")
