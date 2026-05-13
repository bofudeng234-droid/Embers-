"""
ffmpeg 关键帧抽取层

策略:在视频时长的 10% / 35% / 60% / 85% 各抽一帧。
为什么这四个点:第一帧通常黑屏 / logo,最后一帧通常字幕收尾,这四个点更代表视频"内容"。
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


FRAME_PCTS = [0.10, 0.35, 0.60, 0.85]


def _ensure_ffmpeg() -> None:
    """确保系统装了 ffmpeg。"""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError(
            "未找到 ffmpeg/ffprobe。请先安装:\n"
            "  brew install ffmpeg"
        )


def get_duration(video_path: str | Path) -> float | None:
    """用 ffprobe 取视频时长(秒)。"""
    _ensure_ffmpeg()
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json", str(video_path),
            ],
            capture_output=True, text=True, timeout=10, check=True,
        )
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception as e:
        print(f"  [frames] ffprobe 失败: {e}")
        return None


def extract_frames(
    video_path: str | Path,
    output_dir: Path | str,
    n_frames: int = 4,
    duration: float | None = None,
) -> list[Path]:
    """按 FRAME_PCTS 抽 N 帧 jpg。返回抽到的帧路径列表(可能少于 n_frames)。"""
    _ensure_ffmpeg()

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if duration is None:
        duration = get_duration(video_path)
    if not duration or duration < 1.0:
        return []

    # 选 N 个时间戳;n_frames > len(FRAME_PCTS) 时退化为等分
    if n_frames <= len(FRAME_PCTS):
        pcts = FRAME_PCTS[:n_frames]
    else:
        pcts = [(i + 1) / (n_frames + 1) for i in range(n_frames)]

    timestamps = [max(0.5, duration * p) for p in pcts]
    frame_paths: list[Path] = []

    stem = video_path.stem
    for i, t in enumerate(timestamps):
        out = output_dir / f"{stem}_f{i+1}.jpg"
        try:
            # -ss 在 -i 前面是 fast seek,精度稍差但快很多
            # -vframes 1 只抽一帧;-q:v 2 高质量;-y 覆盖
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", f"{t:.2f}",
                    "-i", str(video_path),
                    "-frames:v", "1",
                    "-q:v", "2",
                    "-loglevel", "error",
                    str(out),
                ],
                capture_output=True, timeout=15, check=True,
            )
            if out.exists() and out.stat().st_size > 0:
                frame_paths.append(out)
        except Exception as e:
            print(f"  [frames] 抽帧 t={t:.1f}s 失败: {e}")
            continue

    return frame_paths


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python -m pipeline.frames <video_path>")
        sys.exit(1)
    video = sys.argv[1]
    print(f"测试抽帧: {video}")
    dur = get_duration(video)
    print(f"时长: {dur} 秒")
    frames = extract_frames(video, "./data/_test_frames", n_frames=4)
    print(f"抽到 {len(frames)} 帧:")
    for f in frames:
        print(f"  {f} ({f.stat().st_size} bytes)")
