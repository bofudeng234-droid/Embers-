"""
基于 Playwright 的抖音视频下载层(取代 yt-dlp)

策略:
  · 用真 Chromium 打开抖音页面 → TLS 指纹 / 签名层反爬完全绕过
  · 从 DOM 提取 <video> 元素的 src + og:* meta 数据
  · 用同一个 BrowserContext.request 下载视频文件(自动带浏览器 cookies/headers)
  · 浏览器全程复用(atexit 自动清理),100 条视频也只启动一次

输入:抖音视频 URL(短链 v.douyin.com/xxx/ / 长链 douyin.com/video/xxx 都可,带尾部追踪 OK)
输出:本地 mp4 文件 + caption / 时长 / 封面 等 metadata
"""
from __future__ import annotations

import atexit
import base64
import re
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import (
    sync_playwright,
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PWTimeout,
)


MOBILE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/16.0 Mobile/15E148 Safari/604.1"
)


# ============ URL 规范化 ============
def normalize_douyin_url(raw: str) -> str | None:
    """从用户粘贴的"脏 URL"(尾部签名/空格/中文标点)抽取规范 URL。"""
    if not raw:
        return None
    m = re.search(r'https?://[^\s一-鿿,，。！？]+', raw)
    if not m:
        return None
    url = m.group(0).rstrip(".,;:!?。，")

    # Playwright 自己会跟随重定向,这里只做基础清洗
    # iesdouyin.com/share/video/{id} → www.douyin.com/video/{id}
    m = re.search(r'iesdouyin\.com/share/video/(\d+)', url)
    if m:
        return f"https://www.douyin.com/video/{m.group(1)}"
    return url


# ============ 浏览器单例(进程内复用) ============
_pw: Playwright | None = None
_browser: Browser | None = None
_context: BrowserContext | None = None


def _ensure_browser() -> BrowserContext:
    """惰性初始化 Playwright + Chromium + Context。复用 = 速度。"""
    global _pw, _browser, _context
    if _context is None:
        print("  [pw] 启动 Chromium...")
        _pw = sync_playwright().start()
        _browser = _pw.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        _context = _browser.new_context(
            user_agent=MOBILE_UA,
            viewport={"width": 390, "height": 844},
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
            # 让 navigator.webdriver 不返回 true,反检测
            bypass_csp=True,
        )
        # 在所有页面加载前注入反检测脚本
        _context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN', 'zh', 'en'] });
        """)
        print("  [pw] 浏览器就绪")
    return _context


def _cleanup_browser() -> None:
    """进程退出时清理。atexit 自动调用。"""
    global _pw, _browser, _context
    try:
        if _context is not None:
            _context.close()
        if _browser is not None:
            _browser.close()
        if _pw is not None:
            _pw.stop()
    except Exception:
        pass
    _context = None
    _browser = None
    _pw = None


atexit.register(_cleanup_browser)


# ============ 核心下载逻辑 ============
def _wait_for_video(page: Page, timeout_ms: int = 20000) -> dict | None:
    """等 <video> 元素加载完元数据,返回 {src, duration, width, height}。"""
    try:
        page.wait_for_selector("video", timeout=timeout_ms)
        # 等 readyState >= 1(metadata loaded)就够拿 duration / src
        page.wait_for_function(
            "() => { const v = document.querySelector('video'); return v && v.readyState >= 1 && (v.src || v.currentSrc); }",
            timeout=timeout_ms,
        )
        return page.evaluate("""
            () => {
                const v = document.querySelector('video');
                const src = v.src || v.currentSrc || (v.querySelector('source') ? v.querySelector('source').src : null);
                return {
                    src: src,
                    duration: v.duration || null,
                    width: v.videoWidth || null,
                    height: v.videoHeight || null,
                };
            }
        """)
    except PWTimeout:
        return None


def _extract_metadata(page: Page) -> dict:
    """从 DOM 抽取 og:* / meta description 等 metadata。"""
    return page.evaluate("""
        () => {
            const get = (sel, attr) => {
                const el = document.querySelector(sel);
                return el ? (attr ? el.getAttribute(attr) : el.textContent) : null;
            };
            return {
                title: document.title,
                og_title: get('meta[property="og:title"]', 'content'),
                og_description: get('meta[property="og:description"]', 'content'),
                og_image: get('meta[property="og:image"]', 'content'),
                description: get('meta[name="description"]', 'content'),
            };
        }
    """)


def _fetch_blob_via_page(page: Page, blob_url: str) -> bytes | None:
    """blob: URL 不能 HTTP 直接 GET,用页面内 fetch + FileReader 取 base64。"""
    try:
        b64 = page.evaluate("""
            async (url) => {
                const r = await fetch(url);
                const b = await r.blob();
                return await new Promise((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onloadend = () => resolve(reader.result.split(',')[1]);
                    reader.onerror = reject;
                    reader.readAsDataURL(b);
                });
            }
        """, blob_url)
        return base64.b64decode(b64)
    except Exception as e:
        print(f"  [pw-download] blob 读取失败: {e}")
        return None


def download_video(url: str, output_dir: Path | str, video_id: str | None = None) -> dict[str, Any] | None:
    """主入口:URL → 下载视频文件 + 抓 metadata。失败返回 None,不抛异常。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized = normalize_douyin_url(url)
    if not normalized:
        print(f"  [pw-download] URL 无效: {url}")
        return None
    if normalized != url:
        print(f"  [pw-download] URL 规范化: → {normalized}")

    if not video_id:
        # 从 URL 抽 id 当文件名
        m = re.search(r'/video/(\d+)', normalized)
        video_id = m.group(1) if m else f"v_{int(time.time())}"

    ctx = _ensure_browser()
    page = ctx.new_page()
    t0 = time.time()

    try:
        # ============ 1. 打开页面 ============
        try:
            page.goto(normalized, wait_until="domcontentloaded", timeout=25000)
        except PWTimeout:
            print(f"  [pw-download] 页面加载超时")
            return None

        # 给前端一点时间让 video 元素挂载(抖音是 SPA)
        page.wait_for_timeout(1500)

        # ============ 2. 等 video 元素加载 ============
        video_info = _wait_for_video(page, timeout_ms=20000)
        if not video_info or not video_info.get("src"):
            print(f"  [pw-download] DOM 中未找到视频源")
            return None

        video_src = video_info["src"]
        duration = video_info.get("duration") or 0

        # ============ 3. 抓 metadata ============
        meta = _extract_metadata(page)
        title = meta.get("og_title") or meta.get("title") or ""
        caption = meta.get("og_description") or meta.get("description") or title

        # ============ 4. 下载视频文件 ============
        output_path = output_dir / f"{video_id}.mp4"

        if video_src.startswith("blob:"):
            # blob URL → 通过页面内 fetch
            print(f"  [pw-download] blob: URL,通过页面 fetch")
            data = _fetch_blob_via_page(page, video_src)
            if not data:
                return None
            output_path.write_bytes(data)
        else:
            # 直接 HTTP — 用 browser context 的 request(带 cookies/UA)
            try:
                resp = ctx.request.get(video_src, timeout=30000)
                if resp.status != 200:
                    print(f"  [pw-download] 视频源 HTTP {resp.status}")
                    return None
                output_path.write_bytes(resp.body())
            except Exception as e:
                print(f"  [pw-download] 视频下载失败: {e}")
                return None

        file_size = output_path.stat().st_size
        if file_size < 1024:
            print(f"  [pw-download] 下载文件过小({file_size} bytes),可能失败")
            return None

        elapsed = time.time() - t0
        print(f"  [pw-download] ✓ 视频 {file_size//1024} KB · {duration:.1f}s · 耗时 {elapsed:.1f}s")

        return {
            "video_path": str(output_path),
            "title": title,
            "caption": caption,
            "duration": int(duration) if duration else None,
            "id": video_id,
            "thumbnail": meta.get("og_image"),
            "width": video_info.get("width"),
            "height": video_info.get("height"),
            "upload_date": None,  # Playwright 路径不提供,留空
        }

    except Exception as e:
        print(f"  [pw-download] 异常: {e}")
        return None
    finally:
        page.close()


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://v.douyin.com/AvbkvZXTNNA/"
    print(f"测试 Playwright 下载: {url}")
    res = download_video(url, "./data/_test_download")
    if res:
        print("\n✅ 成功")
        for k, v in res.items():
            if k == "caption":
                v = (v or "")[:80] + "..." if v and len(v) > 80 else v
            print(f"  {k}: {v}")
    else:
        print("\n❌ 失败")
