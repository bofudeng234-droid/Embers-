# 余烬 / Embers

> 你刷过的每一条视频,都是一颗 ember。我们让它不灭。
> 第四代燧人 · Hackathon Case #001(抖音 Hackathon 2026,赛道三:AI 体验)

## 核心问题

抖音的观看历史只保留 ~2 天。用户经常会想起一条看过的视频,但只剩模糊的描述("那个有只猫在 Roomba 上的搞笑视频"),输入抖音搜索框搜不到——因为搜索靠精确文本匹配,而人的记忆是默会的、语义的。

## 解决方案

把每条看过的视频转成多模态 embedding,存进用户本地的向量库。当用户用任何模糊描述(文字/语音)发起回忆,通过向量相似度找回最匹配的视频。

**这相当于把抖音的观看历史从"原始视频列表"变成"语义记忆库",存储成本下降 ~170×,搜索精度从"关键词命中"变成"语义命中"。**

## 技术栈(48h 单兵交付)

| 层 | 工具 | 理由 |
|---|---|---|
| 视频转写 | OpenAI Whisper(或抖音 caption API) | 成熟、准、便宜 |
| 文本 embedding | OpenAI text-embedding-3-small(1536d) | 速度快、质量稳、~$0.02/1M tokens |
| 视觉 embedding(v0.2) | CLIP ViT-L/14 | 多模态联合检索 |
| 向量存储 | sqlite-vec(单文件 SQLite) | 零运维、本地化、隐私故事强 |
| LLM rerank | Claude Sonnet 4.6 | 模糊 query 改写 + 结果解释 |
| API | FastAPI | Python 单文件起步 |
| 前端 | 纯 HTML + Alpine.js + Framer Motion CDN | 不引重型框架,polish 速度 > 一切 |
| 部署(赛后) | Cloudflare Pages + Workers(已有账号) | 跟 fourth-fire.com 同栈 |

## 关键架构(QKV 类比)

```
观看的视频
   ↓
[caption + 转写 + 弱信号(停留/重看)] → embedding
   ↓
存进向量库:
   K = 视频的语义指纹(embedding)
   V = 视频 URL + 时间戳 + 元数据

用户的模糊回忆
   ↓
LLM 改写 → query embedding (Q)
   ↓
向量相似度搜索 → Top-K 候选
   ↓
LLM rerank + 给出"我猜是这条因为..."的解释
   ↓
返回 Top 3 + 跳到关键秒数
```

## 目录结构

```
embers/
├── README.md
├── requirements.txt
├── .env.example
├── pipeline/
│   ├── ingest.py        从 CSV 批量入库(本周末前跑 200 条)
│   ├── embed.py         调用 embedding API
│   ├── transcribe.py    Whisper 处理音频(v0.2)
│   └── store.py         SQLite + sqlite-vec 操作
├── api/
│   └── server.py        FastAPI · /search 接口
├── web/
│   ├── index.html       余烬墙 + 搜索栏(主 demo UI)
│   └── style.css
├── data/
│   ├── videos.csv       200 条视频元数据(用户手动收集)
│   └── embers.db        SQLite 向量库(运行时生成)
└── scripts/
    └── capture_helper.md  如何从抖音批量收集视频元数据
```

## 时间线

| 时间 | 任务 |
|---|---|
| 周二晚-周三上午 | 跑通最小 pipeline:1 条视频文本 → embedding → 存库 → query → top-1 |
| 周三 | 扩展到 200 条 batch 入库;基础搜索 API |
| 周四 | Web UI(余烬墙 + 搜索栏 + 命中动画) |
| 周五 | Demo 视频脚本 + 路演口稿;UI 抛光 |
| 周六-周日 | 48h 比赛现场 |

## 关键设计原则

1. **隐私第一**:所有数据本地处理,不上传任何视频内容到我们服务器
2. **不存视频,只存指纹**:核心叙事就是这个
3. **多模态 v0.1 用文本撑场**:周末没时间跑视觉 embedding,先把文本路径打磨光滑,视觉作为路演 roadmap
4. **demo 数据集 ≥ 200 条**:数量是说服力,200 条搜起来比 20 条说服力强 10 倍

## 跟 第四代燧人 品牌的关系

- 项目本身代号:**余烬 / Embers**
- 出品方:**第四代燧人**
- Demo 路演时露出:"出品:第四代燧人 · The 4th Fire — AI 是火,我们让你看过的火不熄。"
- 赛后内容:作为 fourth-fire.com 的 **Case #001** 完整记录回流到主站
