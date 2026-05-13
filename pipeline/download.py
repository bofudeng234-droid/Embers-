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
import os
import re
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import (
    sync_playwright,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PWTimeout,
)


# 抖音 PC web 反而比 mobile web 更开放(mobile 总弹"打开 App 观看")
PC_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

HEADLESS = os.getenv("PW_HEADLESS", "1").lower() not in ("0", "false", "no")
DEBUG_DUMP = os.getenv("PW_DEBUG", "0").lower() in ("1", "true", "yes")
DEBUG_DUMP_DIR = Path("./data/_debug")


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


# ============ 浏览器单例(持久化 profile) ============
_pw: Playwright | None = None
_context: BrowserContext | None = None

# 持久化用户数据目录(cookies / localStorage / 登录态 都存这里)
# 第一次扫码登录抖音后,后续所有运行都直接复用,无需再登
PROFILE_DIR = Path(os.getenv(
    "PW_PROFILE_DIR",
    str(Path.home() / ".embers_pw_profile")
))


def _ensure_browser() -> BrowserContext:
    """惰性初始化 Chromium · 持久化 profile · 自动反检测。"""
    global _pw, _context
    if _context is None:
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  [pw] 启动 Chromium · headless={HEADLESS} · profile={PROFILE_DIR}")
        _pw = sync_playwright().start()
        _context = _pw.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=HEADLESS,
            user_agent=PC_UA,
            viewport={"width": 1366, "height": 768},
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
            bypass_csp=True,
            device_scale_factor=2,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        _context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN', 'zh', 'en'] });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            window.chrome = { runtime: {} };
        """)
        print("  [pw] 浏览器就绪")
    return _context


def _dump_debug(page: Page, label: str) -> None:
    """保存当前页面的截图 + HTML,辅助调试 '为什么没找到视频'。"""
    DEBUG_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        png = DEBUG_DUMP_DIR / f"{label}.png"
        html = DEBUG_DUMP_DIR / f"{label}.html"
        page.screenshot(path=str(png), full_page=True)
        html.write_text(page.content(), encoding="utf-8")
        print(f"  [pw-download] 调试快照已存: {png} / {html.name}")
    except Exception as e:
        print(f"  [pw-download] 调试 dump 失败: {e}")


def _cleanup_browser() -> None:
    """进程退出时清理。atexit 自动调用。"""
    global _pw, _context
    try:
        if _context is not None:
            _context.close()
        if _pw is not None:
            _pw.stop()
    except Exception:
        pass
    _context = None
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


FRAME_PCTS = [0.10, 0.35, 0.60, 0.85]


def _capture_video_frames(
    page: Page,
    output_dir: Path,
    video_id: str,
    duration: float,
    n_frames: int = 4,
) -> list[Path]:
    """直接对 <video> 元素 screenshot 抽帧 — 绕过 mp4 下载。
    抖音的 secsdk 拦 fetch 但拦不了浏览器内截图。
    只截 <video> 元素 bounding box,不包含周边 UI。
    """
    if not duration or duration < 1:
        return []

    pcts = FRAME_PCTS[:n_frames]
    timestamps = [max(0.5, duration * p) for p in pcts]
    frame_paths: list[Path] = []

    # 先暂停,避免播放中截图模糊
    try:
        page.evaluate("document.querySelector('video').pause()")
    except Exception:
        pass

    video_loc = page.locator("video").first

    for i, t in enumerate(timestamps):
        try:
            # seek 到目标时间
            page.evaluate(f"""
                () => {{
                    const v = document.querySelector('video');
                    if (v) v.currentTime = {t};
                }}
            """)
            # 等 seek 完成 + readyState 足够
            page.wait_for_function(
                f"""
                () => {{
                    const v = document.querySelector('video');
                    return v && Math.abs(v.currentTime - {t}) < 0.3 && v.readyState >= 2;
                }}
                """,
                timeout=5000,
            )
            # 让帧稳定一下(渲染管线)
            page.wait_for_timeout(250)

            frame_path = output_dir / f"{video_id}_f{i+1}.jpg"
            video_loc.screenshot(
                path=str(frame_path),
                type="jpeg",
                quality=85,
            )
            if frame_path.exists() and frame_path.stat().st_size > 1024:
                frame_paths.append(frame_path)
                print(f"  [pw-frames] 帧 {i+1} @ t={t:.1f}s · {frame_path.stat().st_size//1024} KB")
        except Exception as e:
            print(f"  [pw-frames] 帧 {i+1} @ t={t:.1f}s 失败: {e}")
            continue

    return frame_paths


def _fetch_via_page(page: Page, url: str) -> bytes | None:
    """通过页面自己的 fetch 下载任意 URL(http/https/blob)。
    页面上下文自动带 Referer、cookies、UA 等,绕过 403 类问题。
    返回二进制,失败返回 None。
    """
    try:
        b64 = page.evaluate("""
            async (url) => {
                const r = await fetch(url, { credentials: 'include' });
                if (!r.ok) throw new Error('HTTP ' + r.status);
                const b = await r.blob();
                return await new Promise((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onloadend = () => resolve(reader.result.split(',')[1]);
                    reader.onerror = reject;
                    reader.readAsDataURL(b);
                });
            }
        """, url)
        return base64.b64decode(b64)
    except Exception as e:
        print(f"  [pw-download] 页面 fetch 失败: {e}")
        return None


def _is_video_response(response) -> bool:
    """判断 HTTP 响应是不是抖音视频 mp4 / m4a 流。"""
    try:
        ct = (response.headers or {}).get("content-type", "").lower()
        url_lower = response.url.lower()
        # 抖音视频域名(CDN)
        if "douyinvod" in url_lower or "byteimg" in url_lower:
            if "video" in ct or "mp4" in ct or "octet-stream" in ct:
                return True
        # 兜底:URL 含 .mp4 / playback
        if ".mp4" in url_lower or "playback" in url_lower or "play_video" in url_lower:
            return True
    except Exception:
        pass
    return False


def download_video(url: str, output_dir: Path | str, video_id: str | None = None) -> dict[str, Any] | None:
    """主入口:URL → 抽帧 + 截 audio + 抓 metadata。失败返回 None,不抛异常。

    v0.4 关键:用 response interception 被动截获浏览器自己加载的 mp4 流。
    secsdk 拦 fetch 但拦不了 <video> 元素自身的加载——这是漏洞。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized = normalize_douyin_url(url)
    if not normalized:
        print(f"  [pw-download] URL 无效: {url}")
        return None
    if normalized != url:
        print(f"  [pw-download] URL 规范化: → {normalized}")

    if not video_id:
        m = re.search(r'/video/(\d+)', normalized)
        video_id = m.group(1) if m else f"v_{int(time.time())}"

    ctx = _ensure_browser()
    page = ctx.new_page()
    t0 = time.time()

    # ============ 0. 设置 response 拦截器(被动截获 mp4 流) ============
    captured_video: dict[str, Any] = {}

    def _on_response(response):
        if captured_video.get("bytes"):
            return  # 已经拿到了
        if _is_video_response(response):
            try:
                body = response.body()
                if body and len(body) > 100_000:  # 至少 100KB 才算视频
                    captured_video["bytes"] = body
                    captured_video["url"] = response.url
                    captured_video["size"] = len(body)
            except Exception:
                pass

    page.on("response", _on_response)

    try:
        # ============ 1. 打开页面 ============
        try:
            page.goto(normalized, wait_until="domcontentloaded", timeout=25000)
        except PWTimeout:
            print(f"  [pw-download] 页面加载超时")
            return None

        # 给前端一点时间让 video 元素挂载(抖音是 SPA)
        page.wait_for_timeout(3000)

        # 检测常见拦截:登录墙 / 二维码弹窗 / "打开 App" 提示
        wall = page.evaluate("""
            () => {
                const text = (document.body && document.body.innerText) || '';
                const hits = [];
                if (text.includes('打开APP') || text.includes('打开抖音')) hits.push('open_app');
                if (text.includes('登录') && text.length < 1500) hits.push('login_wall');
                if (text.includes('扫码下载')) hits.push('download_app');
                return hits.length ? hits.join(',') : null;
            }
        """)
        if wall:
            print(f"  [pw-download] 检测到拦截页面: {wall}")
            if DEBUG_DUMP:
                _dump_debug(page, f"{video_id}_wall")

        # ============ 2. 等 video 元素加载 ============
        video_info = _wait_for_video(page, timeout_ms=20000)
        if not video_info or not video_info.get("src"):
            print(f"  [pw-download] DOM 中未找到视频源")
            # 不管 DEBUG_DUMP,失败时一律存快照(辅助你排查)
            _dump_debug(page, f"{video_id}_no_video")
            return None

        video_src = video_info["src"]
        duration = video_info.get("duration") or 0

        # ============ 3. 抓 metadata ============
        meta = _extract_metadata(page)
        title = meta.get("og_title") or meta.get("title") or ""
        caption = meta.get("og_description") or meta.get("description") or title

        # ============ 3.5 等视频完整缓冲(让 response 拦截器收完 mp4) ============
        try:
            page.wait_for_function(
                """
                () => {
                    const v = document.querySelector('video');
                    if (!v || !v.duration) return false;
                    for (let i = 0; i < v.buffered.length; i++) {
                        if (v.buffered.end(i) >= v.duration - 0.5) return true;
                    }
                    return false;
                }
                """,
                timeout=15000,
            )
        except PWTimeout:
            print(f"  [pw-download] ⚠ 视频缓冲超时,继续(音频可能不完整)")

        # ============ 4. 截图抽帧(取代 mp4 下载) ============
        # 抖音 secsdk 拦 fetch 但拦不了浏览器内截图。
        # 直接对 <video> 元素 screenshot,只截视频内容区域。
        print(f"  [pw-download] 直接对 video 元素截图抽帧...")
        frame_paths = _capture_video_frames(
            page, output_dir, video_id, duration=duration or 10, n_frames=4
        )
        if not frame_paths:
            print(f"  [pw-download] 没抽到任何帧")
            _dump_debug(page, f"{video_id}_no_frames")
            return None

        # ============ 5. 保存被拦截的 mp4(若拿到) ============
        video_path: str | None = None
        if captured_video.get("bytes"):
            mp4 = output_dir / f"{video_id}.mp4"
            mp4.write_bytes(captured_video["bytes"])
            video_path = str(mp4)
            print(f"  [pw-download] ✓ 截获 mp4 流 · {captured_video['size']//1024} KB")
        else:
            print(f"  [pw-download] ⚠ 没拦到 mp4(无音频,纯视觉)")

        elapsed = time.time() - t0
        print(f"  [pw-download] ✓ 抽到 {len(frame_paths)} 帧 · 耗时 {elapsed:.1f}s")

        return {
            "video_path": video_path,
            "frame_paths": [str(p) for p in frame_paths],
            "title": title,
            "caption": caption,
            "duration": int(duration) if duration else None,
            "id": video_id,
            "thumbnail": meta.get("og_image"),
            "width": video_info.get("width"),
            "height": video_info.get("height"),
            "upload_date": None,
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
