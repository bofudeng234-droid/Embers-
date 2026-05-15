"""
记忆锚点提炼器 · Inverse-Search Indexing 的核心

理论:
  传统 RAG: 索引"内容是什么" → embedding 进库
  认知 RAG: 索引"用户以后会用什么搜回这个" → 预测的 query 进库

为每条视频提炼 5 类锚点:
  1. highlights    金句/亮点台词           → 锚点 ②
  2. vibe          情绪标签               → 锚点 ①
  3. core_message  一句话总结(默会内容救命) → 锚点 ⑤
  4. domain_tags   圈子标签              → 锚点 ⑥
  5. search_hooks  预测的检索 query       → 锚点 ①②⑤⑥ 的元预测(最核心)

每条 anchor 单独 embed,作为第 4 路检索加入 RAG。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = os.getenv("ANCHOR_MODEL", "deepseek-v4-flash")
BASE_URL = "https://api.deepseek.com/v1"


SYSTEM_PROMPT = """你是视频"记忆锚点"提炼器。任务是预测**用户以后会用什么模糊回忆的方式搜索回这条视频**,把这些预测当成索引字段。

# 5 类锚点字段

## 1. highlights (list, 0-3 条)
视频里最让人印象深的金句、亮点台词、或听起来魔性的话。
- 必须是1句话级别的"会被记住的话",不是大段转写
- 没台词的视频(纯画面/图集) → 留空 []

## 2. vibe (string)
视频给人的整体情绪/氛围,只能选下面之一:
搞笑 / 感动 / 燃 / 治愈 / 干货 / 烧脑 / 高级 / 离谱 / 萌 / 帅 / 浪漫 / 神秘 / 解压 / 怀旧 / 中性

## 3. core_message (string, 一句话)
用一句话告诉别人这视频在讲什么 — **默会专业内容必填!这是救命字段**。
- 把"Multi Agent Harness 工程实践"翻译成"让多个 AI 共同协作完成一个任务的方法"
- 把"扫黑除恶之广州队长篇"翻译成"广州队长扫黑除恶混剪"

## 4. domain_tags (list, 1-3 个)
圈子/领域标签。例: AI工程 / 足球 / NBA / 萌宠 / 美食 / 汽车 / 旅行 / 演唱会 / 国漫 / 时尚 / 剧情演绎 / 自动驾驶 / 编程 / 健身 / 城市风景 等。

## 5. search_hooks (list, 3-5 个) ⭐ 最关键字段
**模拟模糊记忆的真实搜索行为。** 想象用户 2-3 周后想找回这条视频,会用什么搜索词?

不同记忆类型都试一下,让 5 个 hooks 覆盖多种检索路径:
- 金句型: 那个说"XXX"的视频
- 画面型: 公园里穿白衣服倒立的女生
- 情绪型: 好可爱的猫坐着睡的
- 标志物型: 戴牛仔帽的鸵鸟
- 模糊语义型: 讲 ai 工程的那个

要求:
- 每条 4-25 字,像真用户在搜索框打字
- 不堆关键词,每条只用 1-2 个最深刻的特征
- 允许犹豫词 "那个"、"好像是"
- **不要照抄原文**,要重新表达
- 不要用 # hashtag

# 输出格式

严格 JSON 单对象,无 markdown 无解释:
{
  "highlights": ["...", "..."],
  "vibe": "...",
  "core_message": "...",
  "domain_tags": ["...", "..."],
  "search_hooks": ["...", "...", "...", "...", "..."]
}"""


USER_TEMPLATE = """视频内容:

标题: {title}
描述: {caption}
转写: {transcript}
画面描述: {frame_desc}
时长: {duration_sec}秒
帖子类型: {post_type}

按要求输出 JSON(严格 5 字段):"""


_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置")
        _client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)
    return _client


def _format_frame_desc(fd: str | None) -> str:
    if not fd:
        return "(无)"
    try:
        frames = json.loads(fd)
        if isinstance(frames, list):
            return " / ".join(frames)
    except Exception:
        pass
    return fd


def extract(video: dict, max_retries: int = 2) -> dict[str, Any] | None:
    """从视频内容提炼 anchor 五元组。失败返回 None。"""
    title = video.get("title") or ""
    caption = video.get("caption") or ""
    transcript = video.get("transcript") or ""
    frame_desc = _format_frame_desc(video.get("frame_desc"))
    duration = video.get("duration_sec")
    post_type = "图集" if not duration else "视频"

    user = USER_TEMPLATE.format(
        title=title or "(无)",
        caption=caption or "(无)",
        transcript=transcript or "(无)",
        frame_desc=frame_desc,
        duration_sec=duration if duration else "未知",
        post_type=post_type,
    )

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.3 + 0.1 * attempt,  # 重试时稍微提高随机性
                max_tokens=800,
                timeout=30.0,
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                last_err = "empty response"
                continue
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
            obj = json.loads(text)
            # 兜底清洗
            return {
                "highlights": _clean_list(obj.get("highlights"), max_n=3),
                "vibe": (obj.get("vibe") or "中性").strip(),
                "core_message": (obj.get("core_message") or "").strip(),
                "domain_tags": _clean_list(obj.get("domain_tags"), max_n=3),
                "search_hooks": _clean_list(obj.get("search_hooks"), max_n=5),
            }
        except Exception as e:
            last_err = f"{e}"
            continue

    print(f"  [anchors] 提炼失败: {last_err}")
    return None


def _clean_list(v: Any, max_n: int) -> list[str]:
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for x in v:
        if isinstance(x, str):
            s = x.strip().lstrip("#").strip()
            if s and s not in out:
                out.append(s)
        if len(out) >= max_n:
            break
    return out


def to_anchor_rows(video_id: str, anchors: dict) -> list[tuple[str, str, str]]:
    """把 anchors dict 展平成 (video_id, kind, text) 三元组列表,方便入库。"""
    rows: list[tuple[str, str, str]] = []
    for h in anchors.get("highlights", []) or []:
        rows.append((video_id, "highlight", h))
    if anchors.get("vibe"):
        rows.append((video_id, "vibe", anchors["vibe"]))
    if anchors.get("core_message"):
        rows.append((video_id, "core", anchors["core_message"]))
    for t in anchors.get("domain_tags", []) or []:
        rows.append((video_id, "domain", t))
    for h in anchors.get("search_hooks", []) or []:
        rows.append((video_id, "hook", h))
    return rows


if __name__ == "__main__":
    # 单测:从 DB 取一条视频跑提炼
    import argparse
    from pipeline import store
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="v007")
    args = parser.parse_args()
    conn = store.connect()
    row = conn.execute(
        "SELECT id, title, caption, transcript, frame_desc, duration_sec FROM videos WHERE id = ?",
        (args.id,),
    ).fetchone()
    if not row:
        print(f"{args.id} 不存在")
    else:
        v = dict(zip(["id", "title", "caption", "transcript", "frame_desc", "duration_sec"], row))
        print(f"=== {v['id']} · {(v['title'] or '')[:50]} ===\n")
        result = extract(v)
        if result:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print(f"\n=== 展平为 {len(to_anchor_rows(v['id'], result))} 条 anchor rows ===")
            for r in to_anchor_rows(v["id"], result):
                print(f"  [{r[1]}] {r[2]}")
        else:
            print("提炼失败")
