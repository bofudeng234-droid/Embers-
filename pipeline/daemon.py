"""
embers daemon · 每 12 小时自动:抓观看历史 → 增量并行入库

由 launchd 触发(每天 0:00 / 12:00 + 开机补跑)。本脚本只负责"一轮":

  1. 距上次成功 < MIN_INTERVAL_HOURS → 跳过(防开机/RunAtLoad 频繁触发)
  2. crawl_history --auto: 主 profile(已登录)抓新观看历史 URL → 追加 videos.csv
  3. 把主 profile 复制成 N 份 worker profile(带登录 cookies,排除 Cache)
  4. ingest_parallel: CLIP_DEVICE=mps + N worker 增量入库(已入库自动 skip)
  5. 更新 state 文件

state: ~/.embers_daemon_state.json   日志: ~/.embers_daemon.log
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY = str(PROJECT_ROOT / ".venv" / "bin" / "python")
CSV = str(PROJECT_ROOT / "data" / "videos.csv")

MAIN_PROFILE = Path.home() / ".embers_pw_profile"
STATE_FILE = Path.home() / ".embers_daemon_state.json"
LOG_FILE = Path.home() / ".embers_daemon.log"

MIN_INTERVAL_HOURS = float(os.getenv("EMBERS_DAEMON_MIN_HOURS", "8"))
WORKERS = int(os.getenv("EMBERS_DAEMON_WORKERS", "4"))
CRAWL_TARGET = int(os.getenv("EMBERS_DAEMON_CRAWL_N", "1000"))

# 复制 profile 时跳过这些大目录(登录态不在这,纯缓存)
PROFILE_SKIP = shutil.ignore_patterns(
    "Cache", "Code Cache", "GPUCache", "ShaderCache", "DawnCache",
    "GrShaderCache", "component_crx_cache", "*.log",
)


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _read_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _write_state(d: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2))
    except Exception as e:
        log(f"⚠ 写 state 失败: {e}")


def _hours_since_last_success(state: dict) -> float | None:
    ts = state.get("last_success")
    if not ts:
        return None
    try:
        last = datetime.fromisoformat(ts)
        return (datetime.now(timezone.utc) - last).total_seconds() / 3600
    except Exception:
        return None


def _derive_worker_profiles(n: int) -> bool:
    """把主 profile 复制成 n 份带登录态的 worker profile。"""
    if not MAIN_PROFILE.exists():
        log(f"⚠ 主 profile 不存在 {MAIN_PROFILE},worker 将无登录态")
        return False
    for i in range(n):
        dst = Path(f"/tmp/embers_pw_w{i}")
        shutil.rmtree(dst, ignore_errors=True)
        try:
            shutil.copytree(MAIN_PROFILE, dst, ignore=PROFILE_SKIP,
                            dirs_exist_ok=True)
        except Exception as e:
            log(f"⚠ copy worker profile {i} 失败: {e}")
            return False
    log(f"✓ 已派生 {n} 份带登录态 worker profile")
    return True


def run_cycle(force: bool = False) -> int:
    state = _read_state()
    hrs = _hours_since_last_success(state)
    if not force and hrs is not None and hrs < MIN_INTERVAL_HOURS:
        log(f"距上次成功仅 {hrs:.1f}h (<{MIN_INTERVAL_HOURS}h),跳过本轮")
        return 0

    log(f"=== daemon cycle 开始 (上次成功 {hrs if hrs is None else f'{hrs:.1f}h 前'}) ===")
    t0 = time.time()

    # 1. 抓观看历史(主 profile,已登录,无人值守)
    log("→ crawl_history --auto")
    r = subprocess.run(
        [PY, "-m", "pipeline.crawl_history", "--auto",
         "--n", str(CRAWL_TARGET), "--csv", CSV],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=3600,
    )
    crawl_tail = (r.stdout or "")[-500:] + (r.stderr or "")[-300:]
    log(f"crawl 结束 rc={r.returncode}\n{crawl_tail}")

    # 2. 派生带登录态的 worker profile
    _derive_worker_profiles(WORKERS)

    # 3. 并行增量入库(已入库自动 skip;新增 0 也无害,几秒退出)
    log(f"→ ingest_parallel (mps, {WORKERS} workers)")
    env = {**os.environ, "CLIP_DEVICE": "mps"}
    r2 = subprocess.run(
        [PY, "-m", "pipeline.ingest_parallel", CSV, "--workers", str(WORKERS)],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=18000, env=env,
    )
    ing_tail = (r2.stdout or "")[-800:] + (r2.stderr or "")[-300:]
    log(f"ingest 结束 rc={r2.returncode}\n{ing_tail}")

    # 4. 清理临时 worker profile(释放磁盘)
    for i in range(WORKERS):
        shutil.rmtree(Path(f"/tmp/embers_pw_w{i}"), ignore_errors=True)

    ok = (r2.returncode == 0)
    dt = (time.time() - t0) / 60
    if ok:
        state["last_success"] = datetime.now(timezone.utc).isoformat()
        state["last_cycle_minutes"] = round(dt, 1)
        _write_state(state)
        log(f"=== cycle 完成 · 用时 {dt:.1f} 分钟 ===")
        return 0
    else:
        log(f"=== cycle 失败 rc={r2.returncode} · 用时 {dt:.1f} 分钟,state 不更新 ===")
        return 1


if __name__ == "__main__":
    force = "--force" in sys.argv
    sys.exit(run_cycle(force=force))
