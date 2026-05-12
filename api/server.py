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
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from pipeline import store, embed

load_dotenv()

app = FastAPI(
    title="Embers / 余烬",
    description="找回你刷过的每一条视频。",
    version="0.1.0",
)

# Demo 阶段:允许任意来源访问。生产收紧到主页域名。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Models ============
class HealthResponse(BaseModel):
    status: str
    video_count: int
    embedding_dim: int


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
    """每次请求新连接,避免 SQLite 多线程坑。"""
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
        )
    finally:
        conn.close()


@app.get("/videos", response_model=list[VideoRow])
def list_videos(limit: int = 200, offset: int = 0):
    """列出所有视频,用于前端的"余烬墙"渲染。"""
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
    """模糊查询入口 · 文本 → embedding → 向量库 top-K → (可选)LLM rerank。"""
    import time
    t0 = time.time()

    # 1. query 改写(可选,目前直接用原文)
    q_text = req.query.strip()

    # 2. embed query
    try:
        q_vec = embed.embed(q_text)
    except Exception as e:
        raise HTTPException(500, f"embedding 失败: {e}")

    # 3. 向量检索
    conn = get_conn()
    try:
        raw_hits = store.search(conn, q_vec, top_k=req.top_k)
    finally:
        conn.close()

    # 4. (可选)LLM 解释
    explanations: dict[str, str] = {}
    if req.rerank and raw_hits:
        try:
            explanations = _llm_explain(q_text, raw_hits)
        except Exception:
            # 解释失败不阻断主搜索结果
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
        query=q_text,
        hits=hits,
        latency_ms=int((time.time() - t0) * 1000),
    )


def _llm_explain(query: str, hits: list[dict]) -> dict[str, str]:
    """用 Claude 对 top hits 生成"我猜是这条因为..."的解释。
    返回 {video_id: explanation}。
    """
    from anthropic import Anthropic
    client = Anthropic()

    top = hits[:3]
    candidates_text = "\n".join(
        f"[{h['id']}] 标题: {h['title']} | 笔记: {h['notes']} | caption: {h['caption']}"
        for h in top
    )

    prompt = f"""用户的模糊回忆: "{query}"

下面是从他观看历史里检索到的最相似的几条视频:

{candidates_text}

对每条视频,用一句话(≤25 字)解释"为什么这条可能是用户在找的"。
输出严格 JSON 格式: {{"视频id": "解释"}}, 不要其他文字。
"""

    resp = client.messages.create(
        model=os.getenv("RERANK_MODEL", "claude-sonnet-4-6"),
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )

    import json
    text = resp.content[0].text.strip()
    # 容错:Claude 偶尔会包 ```json
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
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
