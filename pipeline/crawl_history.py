"""
抖音观看历史抓取 · 浏览自己账号页面,零封号风险

页面: https://www.douyin.com/user/self?showTab=record  (观看历史 tab)

两个模式:
  --explore   只 dump DOM 结构(第一次用,看清楚卡片选择器)
  (默认)      正式抓:滚动加载 + 抽每条视频链接 → 追加 videos.csv

复用 download.py 的持久化 profile(已登录的话直接用,没登录浏览器里扫码一次)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

from pipeline.download import PC_UA, PROFILE_DIR

HISTORY_URL = "https://www.douyin.com/user/self?from_tab_name=main&showTab=record"


def _launch(headless: bool):
    pw = sync_playwright().start()
    ctx = pw.chromium.launch_persistent_context(
        user_data_dir=str(PROFILE_DIR),
        headless=headless,
        user_agent=PC_UA,
        viewport={"width": 1366, "height": 900},
        locale="zh-CN",
        timezone_id="Asia/Shanghai",
        bypass_csp=True,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ],
    )
    ctx.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN','zh','en'] });
        window.chrome = { runtime: {} };
    """)
    return pw, ctx


def explore():
    """打开观看历史页,dump DOM 结构给人看。"""
    pw, ctx = _launch(headless=False)
    page = ctx.new_page()
    try:
        print(f"  [explore] 打开 {HISTORY_URL}")
        try:
            page.goto(HISTORY_URL, wait_until="domcontentloaded", timeout=30000)
        except PWTimeout:
            print("  [explore] ⚠ 加载超时,继续")
        page.wait_for_timeout(3000)

        print()
        print("=" * 60)
        print("  浏览器里确认:")
        print("  1. 已登录你的抖音账号(没登录就扫码登录,profile 会记住)")
        print("  2. 看到观看历史的视频列表")
        print("  3. 回终端按 Enter 开始 dump DOM")
        print("=" * 60)
        try:
            input("\n  → 准备好按 Enter: ")
        except (EOFError, KeyboardInterrupt):
            return

        info = page.evaluate(r"""
            () => {
                const out = {};
                out.url = location.href;
                out.title = document.title;
                // 所有 /video/ /note/ 链接
                const links = Array.from(document.querySelectorAll('a[href]'))
                    .map(a => a.href)
                    .filter(h => /\/(video|note)\/\d+/.test(h));
                out.video_links_count = links.length;
                out.video_links_sample = [...new Set(links)].slice(0, 10);
                // 常见容器的 data-e2e
                out.e2e = [...new Set(Array.from(document.querySelectorAll('[data-e2e]'))
                    .map(el => el.getAttribute('data-e2e')))].slice(0, 40);
                // 是否有"暂无"之类空状态
                const bodyText = (document.body.innerText || '').slice(0, 300);
                out.body_head = bodyText;
                return out;
            }
        """)
        print("\n===== DOM DUMP =====")
        print(json.dumps(info, ensure_ascii=False, indent=2))
        print("====================\n")

        # 滚一屏看看会不会加载更多 + 链接增长
        before = info["video_links_count"]
        page.mouse.wheel(0, 4000)
        page.wait_for_timeout(2500)
        after = page.evaluate(r"""
            () => Array.from(document.querySelectorAll('a[href]'))
                .map(a => a.href).filter(h => /\/(video|note)\/\d+/.test(h)).length
        """)
        print(f"  滚动前链接数: {before} · 滚一屏后: {after} "
              f"({'✓ 懒加载 OK' if after > before else '⚠ 没增长,可能要换滚动容器'})")

        input("\n  → 看完按 Enter 关闭浏览器: ")
    finally:
        page.close(); ctx.close(); pw.stop()


def _load_existing(csv_path: Path) -> tuple[set[str], set[str]]:
    """读 CSV 已有的 (video_ids, urls),用于抓取时跳过已写入的。"""
    ids: set[str] = set()
    urls: set[str] = set()
    if not csv_path.exists():
        return ids, urls
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            u = (row.get("url") or "").strip()
            if u:
                urls.add(u.split("?")[0])
                m = re.search(r"/(?:video|note)/(\d+)", u)
                if m:
                    ids.add(m.group(1))
    return ids, urls


def _probe_links(page) -> int:
    """页面当前能抓到多少 /video/ /note/ 链接(0 = 没登录/被登录墙挡)。"""
    return page.evaluate(r"""
        () => Array.from(document.querySelectorAll('a[href]'))
            .map(a => a.href).filter(h => /\/(video|note)\/\d+/.test(h)).length
    """)


def _open_history(auto: bool):
    """打开观看历史页。

    auto=True: 先 headless 试 → 登录探测失败(probe==0)自动降级 headed
    重试一次(抖音反爬策略会变,headed 真实窗口最难被识别)。
    返回 (pw, ctx, page) 或 None(auto 两种都没登录态 → 本轮放弃)。
    """
    def _try(headless: bool):
        pw, ctx = _launch(headless=headless)
        page = ctx.new_page()
        try:
            page.goto(HISTORY_URL, wait_until="domcontentloaded", timeout=30000)
        except PWTimeout:
            print("  [crawl] ⚠ 加载超时,继续")
        page.wait_for_timeout(6000 if auto else 3000)
        return pw, ctx, page

    if not auto:
        return _try(headless=False)

    # auto: headless 优先(无窗口干扰)
    pw, ctx, page = _try(headless=True)
    probe = _probe_links(page)
    if probe > 0:
        print(f"  [crawl] auto(headless) 探测到 {probe} 条,登录态 OK")
        return pw, ctx, page

    # 降级:关掉 headless,真实有头窗口重试(抖音最难识别有头)
    print("  [crawl] ⚠ headless 没探测到登录态,降级 headed 重试...")
    try:
        page.close(); ctx.close(); pw.stop()
    except Exception:
        pass
    pw, ctx, page = _try(headless=False)
    probe = _probe_links(page)
    if probe > 0:
        print(f"  [crawl] auto(headed 降级) 探测到 {probe} 条,登录态 OK")
        return pw, ctx, page

    print("  [crawl] ⚠ headed 仍没探测到登录态,本轮放弃")
    try:
        page.close(); ctx.close(); pw.stop()
    except Exception:
        pass
    return None


def crawl(n: int, csv_path: Path, scroll_pause: float = 2.0, auto: bool = False):
    """正式抓:滚动加载观看历史,抽视频链接。
    抓取时跳过 CSV 里已存在的(只收集 n 条**新** URL)。

    auto=True: daemon 无人值守 — headless 优先,失败自动降级 headed。
    """
    existing_ids, _existing_urls = _load_existing(csv_path)
    if existing_ids:
        print(f"  [crawl] CSV 已有 {len(existing_ids)} 条,抓取时会跳过它们")

    print(f"  [crawl] 打开观看历史 (auto={auto})")
    opened = _open_history(auto)
    if opened is None:
        return 0
    pw, ctx, page = opened
    collected: dict[str, str] = {}  # id -> url (只装新的)
    try:
        if not auto:
            print()
            print("=" * 60)
            print("  确认已登录 + 看到观看历史列表后,回终端按 Enter")
            print("=" * 60)
            try:
                input("\n  → 按 Enter 开始: ")
            except (EOFError, KeyboardInterrupt):
                return 0

        print(f"  [crawl] 开始滚动抓取,目标 {n} 条...\n")
        stuck = 0
        t0 = time.time()
        while len(collected) < n:
            links = page.evaluate(r"""
                () => Array.from(document.querySelectorAll('a[href]'))
                    .map(a => a.href)
                    .filter(h => /\/(video|note)\/\d+/.test(h))
            """)
            before = len(collected)
            skipped_existing = 0
            for link in links:
                m = re.search(r'/(?:video|note)/(\d+)', link)
                if m:
                    vid = m.group(1)
                    if vid in existing_ids:
                        skipped_existing += 1
                        continue  # 已在 CSV,不占 n 名额
                    if vid not in collected:
                        collected[vid] = link.split("?")[0]
            grew = len(collected) - before
            elapsed = time.time() - t0
            print(f"  [crawl] 新收集 {len(collected):>4}/{n} (+{grew}) "
                  f"· 跳过已有 {skipped_existing} · {elapsed:.0f}s")

            if grew == 0:
                stuck += 1
                if stuck >= 8:
                    print(f"  [crawl] 连续 {stuck} 次无新增,观看历史到底了")
                    break
            else:
                stuck = 0

            # 滚到真正的底部触发懒加载(关键:抖音观看历史是
            # data-e2e="scroll-list" 容器,滚到底 + footer 进视口才刷下一页)
            page.evaluate("""
                () => {
                    const sl = document.querySelector('[data-e2e="scroll-list"]');
                    if (sl) sl.scrollTo(0, sl.scrollHeight);
                    window.scrollTo(0, document.body.scrollHeight);
                    const footer = document.querySelector('[data-e2e="page-footer"]');
                    if (footer) footer.scrollIntoView({block: 'end'});
                    // 兜底:最后一个视频卡片也滚进视口
                    const cards = document.querySelectorAll('a[href*="/video/"], a[href*="/note/"]');
                    if (cards.length) cards[cards.length - 1].scrollIntoView({block: 'end'});
                }
            """)
            # 加载慢,stuck 时多等一会儿再判定
            wait = scroll_pause * (2.0 if stuck > 0 else 1.0)
            page.wait_for_timeout(int(wait * 1000))

        urls = list(collected.values())[:n]
        elapsed = time.time() - t0
        print(f"\n  [crawl] ✓ 抓到 {len(urls)} 条 · 用时 {elapsed:.0f}s")
        return _append_csv(urls, csv_path)
    finally:
        page.close(); ctx.close(); pw.stop()


def _append_csv(urls: list[str], csv_path: Path) -> int:
    existing: set[str] = set()
    next_idx = 1
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                u = (row.get("url") or "").strip()
                if u:
                    existing.add(u)
                m = re.match(r"v(\d+)", row.get("id", "") or "")
                if m:
                    next_idx = max(next_idx, int(m.group(1)) + 1)
    new_rows = []
    for u in urls:
        if u not in existing:
            new_rows.append((f"v{next_idx:03d}", u))
            next_idx += 1
    if not new_rows:
        print("  [crawl] 没有新 URL,CSV 未改")
        return 0
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["id", "url"])
        for vid, u in new_rows:
            w.writerow([vid, u])
    print(f"  [crawl] 写入 {len(new_rows)} 条新 URL → {csv_path}")
    return len(new_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore", action="store_true", help="只 dump DOM 结构(第一次用)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--csv", default="data/videos.csv")
    ap.add_argument("--scroll-pause", type=float, default=2.0)
    ap.add_argument("--auto", action="store_true",
                    help="无人值守(daemon 用):不等 Enter,headless,主 profile 登录态")
    args = ap.parse_args()
    if args.explore:
        explore()
    else:
        crawl(args.n, Path(args.csv), args.scroll_pause, auto=args.auto)


if __name__ == "__main__":
    main()
