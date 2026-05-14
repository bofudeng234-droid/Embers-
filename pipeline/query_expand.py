"""
LLM 查询预处理 · 把用户的模糊/谐音/碎片化记忆改写成多个语义可检索的 query

典型场景:
  - 谐音:"撒发儿啊"        → ["so far away", "Avicii Far Away", "电子舞曲"]
  - 抽象:"那个洗脑的英文歌" → ["catchy English song", "viral pop song", "网红 BGM"]
  - 碎片:"猫"              → ["猫", "猫咪"]  (无须改写,保留原始)
  - 详细:"两个蓝衣球员热身" → ["两个蓝衣球员热身"]  (已足够具体)

调用 DeepSeek(便宜 + 中文强)。失败时降级为 [原 query]。
"""
from __future__ import annotations

import json
import os
import re

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 复用 server.py 的 LLM provider 配置
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
}

PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").lower()
MODEL = os.getenv("QUERY_EXPAND_MODEL") or LLM_PROVIDERS.get(PROVIDER, {}).get("default_model", "deepseek-chat")

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        cfg = LLM_PROVIDERS[PROVIDER]
        api_key = os.getenv(cfg["api_key_env"])
        if not api_key:
            raise RuntimeError(f"{cfg['api_key_env']} 未配置")
        _client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
    return _client


SYSTEM_PROMPT = """你是视频搜索的"模糊记忆改写器"。用户输入的查询可能是:

1. 谐音(中文音译外语,典型场景是音乐):
   - "撒发儿啊" → so far away
   - "扣扣库" → cocoa puff / coco
   - "嗨皮波斯德 to you" → happy birthday to you

2. 模糊描述(用户记不清细节):
   - "那个洗脑的英文歌" → viral catchy English pop song
   - "讲创业的那个" → 创业故事 / startup founder talk

3. 已足够具体的描述(无需改写,保留原样):
   - "猫骑扫地机器人" → 保留
   - "两个蓝衣球员" → 保留

你的任务:
- 第一个永远是用户的原始 query(不要删掉)
- 其余 1-3 个是改写候选,优先覆盖:外语原词、歌曲/作者名、音乐风格/类型、关键名词扩展
- 中英文均可输出(CLIP/CLAP 在英文场景效果更好,所以谐音改写出英文是首选)
- 总数不超过 4 个

输出严格的 JSON,格式: {"queries": ["原query", "改写1", "改写2", ...]}
不要任何其他文字、注释、解释、markdown 标记。"""


def expand_query(query: str) -> list[str]:
    """LLM 改写 query。返回 1-4 个改写后的候选(第一个永远是原 query)。
    失败时降级为 [query]。
    """
    query = query.strip()
    if not query:
        return [query]

    try:
        c = client()
        resp = c.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"输入: {query!r}"},
            ],
            response_format={"type": "json_object"},
            max_tokens=300,
            temperature=0.3,
        )
        text = resp.choices[0].message.content.strip()
        # 容错:有的 LLM 偶尔会包 ```json
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
        obj = json.loads(text)
        queries = obj.get("queries") or []

        # 清洗:去空、去重、保证原 query 在最前
        cleaned: list[str] = []
        seen: set[str] = set()
        if query not in seen:
            cleaned.append(query)
            seen.add(query)
        for q in queries:
            if not isinstance(q, str):
                continue
            qs = q.strip()
            if qs and qs not in seen and len(cleaned) < 4:
                cleaned.append(qs)
                seen.add(qs)
        return cleaned or [query]
    except Exception as e:
        print(f"  [query_expand] 失败,降级到原 query: {e}")
        return [query]


if __name__ == "__main__":
    import sys
    tests = sys.argv[1:] or [
        "撒发儿啊",
        "那个洗脑的英文歌",
        "猫骑扫地机器人",
        "嗨皮波斯德",
        "讲 AI 创业的视频",
    ]
    print(f"Provider: {PROVIDER} · Model: {MODEL}\n")
    for t in tests:
        print(f"原 query: {t!r}")
        out = expand_query(t)
        for i, q in enumerate(out):
            tag = "(原)" if i == 0 else f"(改{i})"
            print(f"  {tag} {q}")
        print()
