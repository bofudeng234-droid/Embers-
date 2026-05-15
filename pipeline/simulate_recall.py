"""
模拟"健忘用户"生成评测 query · 给定一条视频 → 输出一个模糊回忆型搜索词

设计原理:
  - 不真灌 100k 噪声 token(贵 + 同样 prompt-engineering 可达)
  - 用情境化 prompt 让 LLM "扮演" 3 周前刷过这条视频、现在记忆模糊的用户
  - LLM 提供商: DeepSeek-V3(中文强 + 便宜,比 Claude Opus 便宜 ~140x)

输出:每条视频生成一个 {"query": "...", "target_id": "vXXX", "notes": "..."}
  → 直接拼成评测集 CSV,用于 recall@k / MRR 跟踪

调用:
  python -m pipeline.simulate_recall          # 随机 5 条试水
  python -m pipeline.simulate_recall --n 50   # 全集 50 条
  python -m pipeline.simulate_recall --ids v007,v013  # 指定 id
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from pipeline import store

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# V4 Flash 是当前默认 · deepseek-chat 是 V4 Flash 的别名(将废弃)
MODEL = os.getenv("SIMULATE_MODEL", "deepseek-v4-flash")
BASE_URL = "https://api.deepseek.com/v1"


SYSTEM_PROMPT = """你扮演一个普通的抖音用户 — 注意,**不是 AI、不是工程师、就是一个爱刷视频的普通人**。
现在你坐下来,想找回一条 2-3 周前刷过的视频。

# 你的认知状态

- 中间你又刷了几百条其他视频,大部分细节都忘了
- 你**只记得 1-2 个**具体的东西,不是 5 个
- 你不在乎"完整描述",你只想在搜索框里输入个词找回那条视频
- 即使视频本身是专业内容,**你也不会用专业术语**(普通用户记不住"harness"、"蓝金渐层"这种)
- 你的输入会带打字痕迹: 简短、可能错字、可能犹豫、可能记错一个细节

# 必须随机扮演下面 6 种"记忆类型"之一

每次随机选一种,不要老是同一种:

- **A 金句型**: 只记得一句让你印象深的台词 → "那个说'依旧老夫老妻'的"
- **B 画面型**: 只记得一个亮眼画面 → "公园里穿白衣服倒立的女生"
- **C 情绪型**: 只记得当时的感受 → "好可爱的猫坐着睡的那个"
- **D BGM/声音型**: 只记得听到的声音 → "用 so far away 当背景的"
- **E 标志物型**: 只记得一个奇特物品/人 → "戴牛仔帽的那个鸵鸟"
- **F 模糊语义型**: 只记得视频大概在讲啥 → "讲 ai 工程的那个"

# 模拟搜索词的具体要求

1. **字数 4-20 字**,真用户搜索词都很短
2. **只用 1-2 个最深刻的特征**,严禁堆叠 4-5 个关键词
3. 像在打字框里随手打的,不像写文案,不要文绉绉
4. **允许并鼓励**:
   - 犹豫语气: "那个...", "好像是...", "忘了叫啥的"
   - 故意拼错一两个字 (真人打字会错)
   - 拟声词、感叹词、模糊形容词 ("好搞笑的"、"超可爱的")
   - 不完整: "讲那个的", "穿白衣服的"
   - 谐音 (如果是英文/外语内容)
5. **严禁**:
   - 照抄 caption 原文片语
   - 使用 # hashtag (搜索框里不会打)
   - 专业术语 (除非是普通生活里常见的)
   - 把视频内容总结成正经描述

# 输出格式

严格 JSON,不要 markdown:
{"query": "...", "memory_type": "A/B/C/D/E/F"}

如果视频内容太空泛、实在编不出合理的模糊搜索词,输出:
{"query": null, "memory_type": null, "reason": "..."}"""


USER_TEMPLATE = """下面是你前几周刷到过的那条视频。读完后,用你日常的口语模拟你现在的搜索词。

标题: {title}
caption: {caption}
转写: {transcript}
画面描述: {frame_desc}
时长: {duration_sec}秒

现在输出你的搜索词 (严格 JSON,无 markdown):"""


_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置 (.env)")
        _client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)
    return _client


def fetch_video(conn, video_id: str) -> dict | None:
    row = conn.execute(
        "SELECT id, title, caption, transcript, frame_desc, duration_sec FROM videos WHERE id = ?",
        (video_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "title": row[1] or "",
        "caption": row[2] or "",
        "transcript": row[3] or "",
        "frame_desc": row[4] or "",
        "duration_sec": row[5],
    }


def simulate_one(
    video: dict,
    temperature: float = 0.95,
    max_retries: int = 2,
    use_pro_thinking: bool = False,
) -> dict:
    """单条视频 → 一个 query。

    use_pro_thinking=True:
      切换到 deepseek-v4-pro + thinking 模式,模型会先"内心独白"模拟回忆挣扎,
      再输出最终 query。理论上比 Flash 单 turn 输出更接近真用户记忆漂移。
      注意:thinking 模式不支持 temperature。
    """
    fd = video.get("frame_desc") or ""
    try:
        frames = json.loads(fd) if fd else []
        fd_readable = " / ".join(frames) if isinstance(frames, list) else fd
    except Exception:
        fd_readable = fd

    user = USER_TEMPLATE.format(
        title=video.get("title") or "(无)",
        caption=video.get("caption") or "(无)",
        transcript=video.get("transcript") or "(无)",
        frame_desc=fd_readable or "(无)",
        duration_sec=video.get("duration_sec") if video.get("duration_sec") else "未知",
    )

    last_err = None
    last_thinking = None
    for attempt in range(max_retries + 1):
        try:
            if use_pro_thinking:
                resp = client().chat.completions.create(
                    model="deepseek-v4-pro",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=400,  # 给 thinking + 输出留空间
                    timeout=60.0,
                    reasoning_effort="medium",
                    extra_body={"thinking": {"type": "enabled"}},
                )
                # thinking 内容可能不存在(SDK 版本不一定支持 reasoning_content 字段)
                msg = resp.choices[0].message
                last_thinking = getattr(msg, "reasoning_content", None)
            else:
                t = max(0.5, temperature - 0.15 * attempt)
                resp = client().chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                    temperature=t,
                    max_tokens=200,
                    timeout=30.0,
                )

            text = (resp.choices[0].message.content or "").strip()
            if not text:
                last_err = "empty response"
                continue
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
            obj = json.loads(text)
            return {
                "target_id": video["id"],
                "query": obj.get("query"),
                "memory_type": obj.get("memory_type"),
                "reason": obj.get("reason"),
                "thinking": last_thinking,
            }
        except Exception as e:
            last_err = f"{e}"
            continue

    return {
        "target_id": video["id"],
        "query": None,
        "memory_type": None,
        "reason": f"调用失败(已重试 {max_retries} 次): {last_err}",
        "thinking": last_thinking,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="随机抽取多少条 (默认 5)")
    parser.add_argument("--ids", type=str, default=None, help="逗号分隔的视频 id 列表,会覆盖 --n")
    parser.add_argument("--per-video", type=int, default=1, help="每条视频生成 N 个不同 query")
    parser.add_argument("--all", action="store_true", help="对 DB 里所有视频跑(覆盖 --n)")
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--out", type=str, default=None, help="保存到 CSV 路径")
    parser.add_argument("--pro", action="store_true", help="用 deepseek-v4-pro + thinking 模式(模拟记忆挣扎)")
    parser.add_argument("--show-thinking", action="store_true", help="打印模型的 thinking 过程(仅 --pro)")
    args = parser.parse_args()

    conn = store.connect()
    store.init_schema(conn)

    rows = conn.execute("SELECT id FROM videos ORDER BY id").fetchall()
    all_ids = [r[0] for r in rows]
    if not all_ids:
        print("DB 里没视频,先跑 ingest")
        sys.exit(1)

    if args.ids:
        ids = [s.strip() for s in args.ids.split(",") if s.strip()]
    elif args.all:
        ids = all_ids
    else:
        ids = random.sample(all_ids, min(args.n, len(all_ids)))

    total = len(ids) * args.per_video
    model_label = "deepseek-v4-pro(thinking)" if args.pro else MODEL
    print(f"=== 模拟生成 {total} 条评测 query · {len(ids)} 视频 × {args.per_video} 次 · model={model_label} ===\n")

    results = []
    for vid in ids:
        v = fetch_video(conn, vid)
        if not v:
            print(f"  [{vid}] 不存在,跳过")
            continue
        title_short = (v["title"] or "")[:30]
        for k in range(args.per_video):
            out = simulate_one(
                v,
                temperature=args.temperature,
                use_pro_thinking=args.pro,
            )
            mtype = out.get("memory_type") or "?"
            suffix = f"#{k+1}" if args.per_video > 1 else ""
            if out["query"]:
                print(f"  [{vid}{suffix}] {title_short}")
                print(f"           → ({mtype}) {out['query']!r}")
                if args.show_thinking and out.get("thinking"):
                    th = out["thinking"][:300].replace("\n", " ")
                    print(f"           💭 {th}...")
                print()
            else:
                print(f"  [{vid}{suffix}] {title_short}")
                print(f"           → 拒绝: {out.get('reason')}\n")
            results.append(out)

    # 可选导出
    if args.out:
        import csv
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["query", "target_id", "reason"])
            for r in results:
                w.writerow([r.get("query") or "", r["target_id"], r.get("reason") or ""])
        print(f"\n→ 已写入 {args.out}")

    ok = sum(1 for r in results if r["query"])
    print(f"\n=== 完成: {ok}/{len(results)} 条成功生成 ===")


if __name__ == "__main__":
    main()
