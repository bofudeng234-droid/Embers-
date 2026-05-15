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
    print(f"[server] CLIP 不可用: {_e}")
    CLIP_AVAILABLE = False

try:
    from pipeline import clap_embed
    CLAP_AVAILABLE = True
except Exception as _e:
    print(f"[server] CLAP 不可用: {_e}")
    CLAP_AVAILABLE = False

try:
    from pipeline.query_expand import expand_query
    EXPAND_AVAILABLE = True
except Exception as _e:
    print(f"[server] query_expand 不可用: {_e}")
    EXPAND_AVAILABLE = False
    def expand_query(q: str) -> list[str]:
        return [q]

# v0.8 bge-reranker 精排 · 容错导入(模型加载失败时降级到纯 RRF 结果)
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() in ("1", "true", "yes")
if RERANKER_ENABLED:
    try:
        from pipeline import reranker as _reranker_module
        RERANKER_AVAILABLE = True
    except Exception as _e:
        print(f"[server] reranker 不可用: {_e}")
        RERANKER_AVAILABLE = False
else:
    RERANKER_AVAILABLE = False

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
    video_count_with_audio: int
    embedding_dim: int
    embedding_provider: str
    llm_provider: str
    clip_available: bool
    clap_available: bool


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
    expand: bool = Field(True, description="是否启用 LLM 查询改写(默认开,处理谐音 / 模糊描述)")


class SearchHit(VideoRow):
    distance: float
    explanation: str | None = None
    sources: list[str] = []  # 命中来源:text / image / audio,可多个
    path_distances: dict[str, float] = {}  # 每一路的真实 distance(用来识别"真匹配 vs 凑数")


class SearchResponse(BaseModel):
    query: str
    expanded_queries: list[str] = []  # LLM 改写后的多个候选(第一个=原 query)
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
            video_count_with_audio=store.count_with_audio(conn),
            embedding_dim=store.EMBEDDING_DIM,
            embedding_provider=embed.PROVIDER,
            llm_provider=LLM_PROVIDER,
            clip_available=CLIP_AVAILABLE,
            clap_available=CLAP_AVAILABLE,
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


PATH_WEIGHTS = {"text": 1.5, "image": 1.0, "audio": 0.7, "anchor": 1.3}
# 每路只取前 N 名进 fusion · 防止 distance 较大的"凑数命中"灌进总分
PATH_MAX_RANK = {"text": 5, "image": 5, "audio": 3, "anchor": 5}


def _rrf_fuse(*hit_lists_named: tuple[str, list[dict]], k: int = 60) -> list[dict]:
    """Distance-aware 加权融合(替代纯 RRF)。

    设计要点:
      1. score = Σ w_path * 1/(1 + distance_path)  —— distance 越低得分越高
      2. 每路只算前 N 名(PATH_MAX_RANK):
         避免某条视频在 audio 路 dist=1.2(远距离凑数)被无脑奖励。
      3. text 权重 1.5(信号最浓 — caption+transcript+frame_desc),
         audio 权重 0.7(CLAP 对具体内容识别弱)。
      4. 缺路自然是 0,不被惩罚 —— 图集帖没 audio 不再吃亏。
    """
    scores: dict[str, float] = {}
    payload: dict[str, dict] = {}
    path_distances: dict[str, dict[str, float]] = {}

    for source_name, hits in hit_lists_named:
        w = PATH_WEIGHTS.get(source_name, 1.0)
        max_rank = PATH_MAX_RANK.get(source_name, 5)
        for h in hits[:max_rank]:
            vid = h["id"]
            dist = float(h.get("distance", 1.0))
            scores[vid] = scores.get(vid, 0.0) + w / (1.0 + dist)
            d = payload.setdefault(vid, h)
            d.setdefault("_source", set()).add(source_name)
            path_distances.setdefault(vid, {})[source_name] = dist
            if dist < d.get("distance", 1e9):
                d["distance"] = dist

    sorted_ids = sorted(scores.keys(), key=lambda v: -scores[v])
    out: list[dict] = []
    for vid in sorted_ids:
        d = payload[vid]
        d["_score"] = scores[vid]
        d["_source"] = "+".join(sorted(d["_source"]))
        d["_path_distances"] = path_distances[vid]
        out.append(d)
    return out


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    t0 = time.time()
    q_text = req.query.strip()

    # ============ 0. LLM 查询改写(谐音 / 模糊描述展开) ============
    expanded = [q_text]
    if req.expand and EXPAND_AVAILABLE:
        try:
            expanded = expand_query(q_text)
        except Exception as e:
            print(f"  [/search] expand 失败,只用原 query: {e}")
            expanded = [q_text]

    # ============ 1. 每个改写查询都跑四路检索 ============
    # 同一路里多个 query 的命中合并,取每条视频的最小 distance
    merged_text: dict[str, dict] = {}
    merged_image: dict[str, dict] = {}
    merged_audio: dict[str, dict] = {}
    merged_anchor: dict[str, dict] = {}

    for q in expanded:
        try:
            q_text_vec = embed.embed(q)
        except Exception as e:
            print(f"  [/search] '{q[:30]}' text embed 失败: {e}")
            continue

        q_image_vec = None
        if CLIP_AVAILABLE:
            try:
                q_image_vec = clip_embed.text_to_vec(q)
            except Exception:
                pass

        q_audio_vec = None
        if CLAP_AVAILABLE:
            try:
                q_audio_vec = clap_embed.text_to_vec(q)
            except Exception:
                pass

        conn = get_conn()
        try:
            for h in store.search_text(conn, q_text_vec, top_k=req.top_k * 2):
                vid = h["id"]
                if vid not in merged_text or h["distance"] < merged_text[vid]["distance"]:
                    merged_text[vid] = h
            if q_image_vec is not None and store.count_with_image(conn) > 0:
                for h in store.search_image(conn, q_image_vec, top_k=req.top_k * 2):
                    vid = h["id"]
                    if vid not in merged_image or h["distance"] < merged_image[vid]["distance"]:
                        merged_image[vid] = h
            if q_audio_vec is not None and store.count_with_audio(conn) > 0:
                for h in store.search_audio(conn, q_audio_vec, top_k=req.top_k * 2):
                    vid = h["id"]
                    if vid not in merged_audio or h["distance"] < merged_audio[vid]["distance"]:
                        merged_audio[vid] = h
            # v0.7: anchor 路 · 用 q_text_vec(智谱 2048d,跟 vec_anchors 同空间)
            if store.count_anchors(conn) > 0:
                for h in store.search_anchor(conn, q_text_vec, top_k=req.top_k * 2):
                    vid = h["id"]
                    if vid not in merged_anchor or h["distance"] < merged_anchor[vid]["distance"]:
                        merged_anchor[vid] = h
        finally:
            conn.close()

    # 转回有序列表(按 distance 升序),供 RRF 排名用
    text_hits = sorted(merged_text.values(), key=lambda x: x["distance"])
    image_hits = sorted(merged_image.values(), key=lambda x: x["distance"])
    audio_hits = sorted(merged_audio.values(), key=lambda x: x["distance"])
    anchor_hits = sorted(merged_anchor.values(), key=lambda x: x["distance"])

    # ============ 2. RRF 多路融合 ============
    fuse_inputs: list[tuple[str, list[dict]]] = [("text", text_hits)]
    if image_hits:
        fuse_inputs.append(("image", image_hits))
    if audio_hits:
        fuse_inputs.append(("audio", audio_hits))
    if anchor_hits:
        fuse_inputs.append(("anchor", anchor_hits))

    # RRF 输出 top_k*3 进 reranker(给精排留挑选空间) → reranker 取 top_k
    rrf_candidates = _rrf_fuse(*fuse_inputs) if len(fuse_inputs) > 1 else text_hits
    rerank_pool_size = req.top_k * 3
    rrf_candidates = rrf_candidates[:rerank_pool_size]

    # v0.8: cross-encoder 精排(bge-reranker-v2-m3)
    # RRF 看的是 distance,reranker 看的是 (query, doc) 真实相关性 — 更准
    if RERANKER_AVAILABLE and len(rrf_candidates) > 1:
        try:
            raw_hits = _reranker_module.rerank(q_text, rrf_candidates, top_k=req.top_k)
        except Exception as e:
            print(f"  [/search] reranker 失败,降级 RRF: {e}")
            raw_hits = rrf_candidates[:req.top_k]
    else:
        raw_hits = rrf_candidates[:req.top_k]

    explanations: dict[str, str] = {}
    if req.rerank and raw_hits:
        try:
            explanations = _llm_explain(q_text, raw_hits)
        except Exception:
            explanations = {}

    def _parse_sources(s) -> list[str]:
        if not s:
            return ["text"]
        if isinstance(s, str):
            return s.split("+")
        if isinstance(s, (set, list, tuple)):
            return sorted(list(s))
        return ["text"]

    hits = [
        SearchHit(
            id=h["id"], url=h["url"], title=h["title"], caption=h["caption"],
            transcript=h.get("transcript"), frame_desc=h.get("frame_desc"),
            duration_sec=h.get("duration_sec"), watched_at=h.get("watched_at"),
            distance=h["distance"],
            explanation=explanations.get(h["id"]),
            sources=_parse_sources(h.get("_source")),
            path_distances=h.get("_path_distances", {}),
        )
        for h in raw_hits
    ]

    return SearchResponse(
        query=q_text,
        expanded_queries=expanded,
        hits=hits,
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
