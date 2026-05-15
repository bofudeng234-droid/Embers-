"""
抖音推荐流批量抓 URL · v0.9 批量数据集生成

策略:
  - Playwright 打开 douyin.com 推荐流
  - 自动按方向键下切换下一条
  - 监听 location.pathname 变化(/video/{id} 或 /note/{id})
  - 去重 + 追加到 data/videos.csv (v0NN 自增 id)

默认 headless=False:
  - 抖音对 headless 检测更严,显示模式反爬更松
  - 你能看到抓取过程,异常时可手动干预

用法:
  python -m pipeline.crawl_douyin --n 30                 # 抓 30 条
  python -m pipeline.crawl_douyin --n 1000 --headless    # 抓 1000 条 headless
  python -m pipeline.crawl_douyin --n 50 --csv data/x.csv  # 写到别的 CSV
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PWTimeout,
)

from pipeline.download import PC_UA, PROFILE_DIR


def _extract_video_id(url: str) -> str | None:
    """从 douyin URL 抽 video/note id。"""
    m = re.search(r'/(?:video|note)/(\d+)', url)
    return m.group(1) if m else None


def crawl_recommend(
    n: int = 30,
    headless: bool = False,
    delay_min: float = 2.0,
    delay_max: float = 4.0,
) -> list[str]:
    """从抖音推荐流抓 N 条 URL。

    抖音推荐流的 location.href 始终是 /?recommend=1,真实视频 URL 只能通过
    点击"复制链接"按钮 → 读剪贴板 拿到。所以本函数:
      1. 模拟点击右侧"复制链接"按钮
      2. 用 navigator.clipboard.readText() 读出真实 URL
      3. 按方向键 ↓ 切下一条
    """
    print(f"  [crawl] 启动 Chromium · headless={headless} · target={n}")
    pw = sync_playwright().start()
    ctx = pw.chromium.launch_persistent_context(
        user_data_dir=str(PROFILE_DIR),
        headless=headless,
        user_agent=PC_UA,
        viewport={"width": 1366, "height": 768},
        locale="zh-CN",
        timezone_id="Asia/Shanghai",
        bypass_csp=True,
        permissions=["clipboard-read", "clipboard-write"],  # 关键:读剪贴板
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ],
    )
    ctx.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN', 'zh', 'en'] });
        Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
        window.chrome = { runtime: {} };
    """)

    page = ctx.new_page()
    urls: list[str] = []
    seen_ids: set[str] = set()

    try:
        print(f"  [crawl] 打开 douyin.com")
        try:
            page.goto("https://www.douyin.com/", wait_until="domcontentloaded", timeout=30000)
        except PWTimeout:
            print(f"  [crawl] ⚠ 页面加载超时,继续看看")

        page.wait_for_timeout(2000)

        # ============ 半自动启动:等用户进入推荐流 ============
        # 抖音 PC 默认是"精选"视频墙(grid),需手动进入推荐流(/?recommend=1)
        # 推荐流 URL 不变,真实视频 URL 通过点击"复制链接"按钮 + 读剪贴板获取
        print()
        print("=" * 60)
        print("  请在浏览器里:")
        print("  1. 等抖音页面加载完")
        print("  2. 点左侧 [推荐] 标签(进入推荐流,URL 会变成 ?recommend=1)")
        print("  3. 视频开始播放后,回到这个终端按 Enter")
        print("=" * 60)
        try:
            input("\n  → 准备好了按 Enter 开始自动抓 (Ctrl+C 取消): ")
        except (EOFError, KeyboardInterrupt):
            print("\n  [crawl] 取消")
            return []

        # 把焦点放到 video 元素上,让方向键能切视频
        try:
            page.evaluate("""() => {
                const v = document.querySelector('video');
                if (v) v.focus();
                else document.body.focus();
            }""")
        except Exception:
            pass

        print(f"  [crawl] 开始自动循环 (点复制链接 → 读剪贴板 → 按方向键)\n")

        stuck = 0
        t0 = time.time()

        while len(urls) < n:
            # ============ 1. 点"复制链接"按钮 ============
            clicked = False
            try:
                # 抖音 PC 的"复制链接"按钮:文本就是"复制链接"
                copy_btn = page.get_by_text("复制链接").first
                copy_btn.click(timeout=3000)
                clicked = True
            except Exception as e:
                # 备选:试找带 share/link 关键字的 button
                try:
                    page.locator('[data-e2e*="copy"], [data-e2e*="share"]').first.click(timeout=2000)
                    clicked = True
                except Exception:
                    pass

            if not clicked:
                stuck += 1
                if stuck % 3 == 1:
                    print(f"  [crawl] ⚠ 点不到'复制链接'按钮 (第 {stuck} 次)")
                if stuck >= 10:
                    print(f"  [crawl] 连续失败,放弃")
                    break
                page.wait_for_timeout(1500)
                continue

            # 等剪贴板写入
            page.wait_for_timeout(400)

            # ============ 2. 读剪贴板 ============
            try:
                clip = page.evaluate("() => navigator.clipboard.readText()")
            except Exception as e:
                print(f"  [crawl] ⚠ 剪贴板读失败: {e}")
                clip = ""

            url_found = None
            if clip:
                # 抖音"复制链接"通常输出: "标题 https://v.douyin.com/XXX/  复制此链接,打开抖音..."
                m = re.search(r'https?://[^\s]*douyin[^\s]*', clip)
                if m:
                    url_found = m.group(0).rstrip('.,;:')

            if url_found:
                # 短链 v.douyin.com/XXX 没有数字 id,用 hash 部分去重
                key = url_found
                m_id = _extract_video_id(url_found)
                if m_id:
                    key = m_id
                if key not in seen_ids:
                    seen_ids.add(key)
                    urls.append(url_found)
                    stuck = 0
                    elapsed = time.time() - t0
                    print(f"  [crawl] [{len(urls):>3}/{n}] +{elapsed:>5.0f}s · {url_found}")
                else:
                    stuck += 1
                    if stuck >= 15:
                        print(f"  [crawl] ⚠ 连续 {stuck} 次重复,可能卡在同一条")
                        if stuck >= 25:
                            break
            else:
                stuck += 1

            # ============ 3. 按方向键切下一条 ============
            try:
                page.keyboard.press("ArrowDown")
            except Exception as e:
                print(f"  [crawl] ⚠ ArrowDown 失败: {e}")

            wait_ms = random.randint(int(delay_min * 1000), int(delay_max * 1000))
            page.wait_for_timeout(wait_ms)

        elapsed = time.time() - t0
        avg = elapsed / max(len(urls), 1)
        print(f"\n  [crawl] ✓ 完成 · 抓到 {len(urls)} 条 · 用时 {elapsed:.0f}s · 平均 {avg:.1f}s/条")
        return urls

    finally:
        try:
            page.close()
        except Exception:
            pass
        try:
            ctx.close()
        except Exception:
            pass
        try:
            pw.stop()
        except Exception:
            pass


def append_to_csv(urls: list[str], csv_path: Path) -> int:
    """追加 URL 到 videos.csv, 跳过已存在 URL, 自增 v{N+i} id。"""
    existing_urls: set[str] = set()
    next_idx = 1

    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = (row.get("url") or "").strip()
                if url:
                    existing_urls.add(url)
                m = re.match(r'v(\d+)', row.get("id", "") or "")
                if m:
                    next_idx = max(next_idx, int(m.group(1)) + 1)

    new_rows: list[tuple[str, str]] = []
    for url in urls:
        if url in existing_urls:
            continue
        new_rows.append((f"v{next_idx:03d}", url))
        next_idx += 1

    if not new_rows:
        print(f"  [crawl] 没有新 URL(全部已在 CSV),未写入")
        return 0

    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["id", "url"])
        for vid, url in new_rows:
            writer.writerow([vid, url])

    print(f"  [crawl] 写入 {len(new_rows)} 条新 URL → {csv_path}")
    return len(new_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="抓多少条 (默认 30)")
    parser.add_argument("--csv", default="data/videos.csv", help="追加写入的 CSV 路径")
    parser.add_argument("--headless", action="store_true", help="无头模式 (默认显示浏览器)")
    parser.add_argument("--delay-min", type=float, default=2.0, help="每条之间最短等待秒数")
    parser.add_argument("--delay-max", type=float, default=4.0, help="每条之间最长等待秒数")
    args = parser.parse_args()

    urls = crawl_recommend(
        n=args.n,
        headless=args.headless,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
    )

    if urls:
        append_to_csv(urls, Path(args.csv))
    else:
        print("  [crawl] 一条都没抓到,检查是否被反爬 / 没登录")
        sys.exit(1)


if __name__ == "__main__":
    main()
