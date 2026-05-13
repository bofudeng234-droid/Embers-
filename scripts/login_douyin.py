"""
一次性扫码登录抖音,保存登录态到持久化 profile

用法:
  python scripts/login_douyin.py

流程:
  1. 弹出可见 Chromium 窗口,自动打开 douyin.com
  2. 你点击右上角"登录" → 用抖音 APP 扫二维码
  3. 看到首页视频流 = 登录成功
  4. 回到终端窗口,按 Enter
  5. 脚本保存 cookies 到 ~/.embers_pw_profile/ 后退出

之后所有 ingest 都自动复用这个登录态,免再登。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 项目根目录加进 sys.path,以便从 scripts/ 跑能找到 pipeline 包
sys.path.insert(0, str(Path(__file__).parent.parent))

from playwright.sync_api import sync_playwright

from pipeline.download import PROFILE_DIR, PC_UA


def main() -> None:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"启动 Chromium(可见模式)· profile={PROFILE_DIR}\n")

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=False,
            user_agent=PC_UA,
            viewport={"width": 1366, "height": 768},
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
            bypass_csp=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN', 'zh', 'en'] });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            window.chrome = { runtime: {} };
        """)

        page = context.new_page()
        try:
            page.goto("https://www.douyin.com/", wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            print(f"页面加载超时(可以无视,继续登录):{e}")

        print("=" * 60)
        print("  浏览器已经打开抖音。请按以下步骤操作:")
        print("")
        print("  1. 点右上角 '登录' 按钮")
        print("  2. 用抖音 APP 扫二维码登录")
        print("  3. 看到首页视频流 = 登录成功")
        print("")
        print("  ⚠️ 完成后回到这个终端窗口,按 Enter 退出脚本")
        print("=" * 60)
        print()

        input("✏️  登录完成后按 Enter: ")

        # 验证 cookies 是否成功落地
        cookies = context.cookies()
        douyin_cookies = [c for c in cookies if "douyin" in c.get("domain", "")]

        print(f"\n✅ 退出前 cookie 数:")
        print(f"   全部域名:     {len(cookies)}")
        print(f"   douyin 域名:  {len(douyin_cookies)}")
        print(f"   profile 路径: {PROFILE_DIR}")

        # 找一下登录后才会有的关键 cookie
        important = ["sessionid", "sessionid_ss", "passport_csrf_token", "ttwid"]
        found = [c["name"] for c in cookies if c.get("name") in important]
        if found:
            print(f"   关键登录态 cookies:{', '.join(found)}")
        else:
            print(f"   ⚠️ 没找到 sessionid 等登录态 cookie,可能没真正登录成功")

        context.close()

    print("\n登录态已保存。现在可以跑:")
    print("    python -m pipeline.download \"https://v.douyin.com/AvbkvZXTNNA/\"")


if __name__ == "__main__":
    main()
