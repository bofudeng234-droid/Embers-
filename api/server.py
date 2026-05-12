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
    embedding_dim: int
    embedding_provider: str
    llm_provider: str


class VideoRow(BaseModel):
    id: str
    url: str
    title: str | None = None
    caption: str | None = None
    hashtags: str | None = None
    notes: str | None = None
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
            embedding_dim=store.EMBEDDING_DIM,
            embedding_provider=embed.PROVIDER,
            llm_provider=LLM_PROVIDER,
        )
    finally:
        conn.close()


@app.get("/videos", response_model=list[VideoRow])
def list_videos(limit: int = 200, offset: int = 0):
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, url, title, caption, hashtags, notes, watched_at
            FROM videos
            ORDER BY watched_at DESC NULLS LAST
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        return [
            VideoRow(
                id=r[0], url=r[1], title=r[2], caption=r[3],
                hashtags=r[4], notes=r[5], watched_at=r[6],
            )
            for r in rows
        ]
    finally:
        conn.close()


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    t0 = time.time()
    q_text = req.query.strip()

    try:
        q_vec = embed.embed(q_text)
    except Exception as e:
        raise HTTPException(500, f"embedding 失败: {e}")

    conn = get_conn()
    try:
        raw_hits = store.search(conn, q_vec, top_k=req.top_k)
    finally:
        conn.close()

    explanations: dict[str, str] = {}
    if req.rerank and raw_hits:
        try:
            explanations = _llm_explain(q_text, raw_hits)
        except Exception:
            explanations = {}

    hits = [
        SearchHit(
            id=h["id"], url=h["url"], title=h["title"], caption=h["caption"],
            hashtags=h["hashtags"], notes=h["notes"], watched_at=h["watched_at"],
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
