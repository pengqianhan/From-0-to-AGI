"""一条命令把一章视频从源码做成 MP4。

    uv run --extra video python -m video_kit.build chapters/01-linear-regression            # 1080p 全片
    uv run --extra video python -m video_kit.build chapters/01-linear-regression --preview  # 480p 样片

流程：解析 script.md → 逐镜合成旁白 → Manim 渲染（旁白随分镜插入）→ 用同一份时间轴
生成字幕 → ffmpeg 烧录字幕 → 交付检查（时长、音轨、峰值电平）→ 每镜抽一帧供人工检查。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from . import theme
from .script import parse_script
from .tts import synthesize_shots

REPO = Path(__file__).resolve().parent.parent


def _fmt_srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: Path, shot_starts: dict[str, float], audios) -> int:
    entries = []
    for a in audios:
        base = shot_starts[a.shot_id]
        for s in a.sentences:
            entries.append((base + s.start, base + s.end, s.display))
    entries.sort()
    lines = []
    for i, (st, en, text) in enumerate(entries, 1):
        lines += [str(i), f"{_fmt_srt_time(st)} --> {_fmt_srt_time(en)}", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return len(entries)


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
         str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)


def _max_volume_db(path: Path) -> float | None:
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-map", "0:a:0", "-af", "volumedetect",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    m = re.search(r"max_volume:\s*(-?[\d.]+) dB", r.stderr)
    return float(m.group(1)) if m else None


def build(chapter_dir: Path, preview: bool, scene_class: str) -> Path:
    chapter_dir = chapter_dir.resolve()
    video_dir = chapter_dir / "video"
    out = video_dir / "out"
    out.mkdir(parents=True, exist_ok=True)
    num = chapter_dir.name.split("-")[0]
    tag = f"ch{num}" + ("_preview" if preview else "")

    shots = parse_script(video_dir / "script.md")
    print(f"[build] {chapter_dir.name}：{len(shots)} 个分镜")
    audios = synthesize_shots(shots, out / "audio")
    timings = {a.shot_id: {"wav": a.wav, "duration": a.duration} for a in audios}
    timings_path = out / "timings.json"
    timings_path.write_text(json.dumps(timings, ensure_ascii=False, indent=1), encoding="utf-8")
    narration_total = sum(a.duration for a in audios)
    print(f"[build] 旁白总长 {narration_total / 60:.1f} 分钟")

    shot_log = out / f"{tag}_shots.json"
    env = dict(os.environ, VIDEO_TIMINGS=str(timings_path), VIDEO_SHOT_LOG=str(shot_log))
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    quality = ["-ql"] if preview else ["-qh", "--fps", "30"]
    media = out / "media"
    cmd = [sys.executable, "-m", "manim", "render", *quality, "--disable_caching",
           "--progress_bar", "none", "--media_dir", str(media), "-o", f"{tag}_raw",
           str(video_dir / "scenes.py"), scene_class]
    print("[build] 渲染：", " ".join(cmd[2:]))
    subprocess.run(cmd, check=True, env=env, cwd=video_dir)
    raw = next(media.glob(f"videos/scenes/*/{tag}_raw.mp4"))

    log = json.loads(shot_log.read_text(encoding="utf-8"))
    missing = [s.shot_id for s in shots if s.shot_id not in log["shots"]]
    extra = [k for k in log["shots"] if k not in timings]
    if missing or extra:
        raise SystemExit(f"[build] 分镜不一致：scenes.py 缺少 {missing}，多出 {extra}")

    srt = (out if preview else video_dir) / ("subtitles_preview.srt" if preview else "subtitles.srt")
    n = write_srt(srt, log["shots"], audios)
    print(f"[build] 字幕 {n} 条 → {srt.relative_to(REPO)}")

    final = out / f"{tag}.mp4"
    style = (f"FontName={theme.cjk_font()},FontSize=15,PrimaryColour=&H00FFFFFF,"
             "OutlineColour=&H00101418,BorderStyle=1,Outline=1.2,Shadow=0,MarginV=10")
    srt_arg = str(srt).replace("\\", "/").replace(":", r"\:").replace(",", r"\,")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
         "-vf", f"subtitles='{srt_arg}':force_style='{style}'",
         "-c:v", "libx264", "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "160k", str(final)],
        check=True,
    )

    # ── 交付检查 ──────────────────────────────────────────────────────────
    info = _probe(final)
    dur = float(info["format"]["duration"])
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    peak = _max_volume_db(final) if has_audio else None
    problems = []
    if not has_audio:
        problems.append("没有音轨")
    if peak is not None and peak >= -0.1:
        problems.append(f"音频可能削波（峰值 {peak} dB）")
    if not preview and not (300 <= dur <= 600):
        problems.append(f"时长 {dur / 60:.1f} 分钟，不在 5–10 分钟内")
    if log["total"] + 1 < narration_total:
        problems.append("视频比旁白短")

    frames = out / "frames"
    frames.mkdir(exist_ok=True)
    for sid, st in log["shots"].items():
        shot_dur = timings[sid]["duration"]
        # 中点一帧 + 结尾前一帧：很多版式问题要等一镜的元素全部出现后才看得到
        for suffix, t in (("", st + shot_dur / 2), ("_end", st + max(0.1, shot_dur - 0.4))):
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(final),
                 "-frames:v", "1", "-vf", "scale=960:-2",
                 str(frames / f"{tag}_{sid}{suffix}.png")],
                check=True,
            )

    report = {
        "video": str(final.relative_to(REPO)),
        "resolution": f"{vstream['width']}x{vstream['height']}",
        "duration_min": round(dur / 60, 2),
        "audio": has_audio,
        "peak_db": peak,
        "shots": len(shots),
        "problems": problems,
        "note": "旁白发音与语速未经人工试听",
    }
    (out / f"{tag}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("[build] 交付检查：", json.dumps(report, ensure_ascii=False))
    return final


def main() -> None:
    ap = argparse.ArgumentParser(description="渲染一章的讲解视频")
    ap.add_argument("chapter_dir", type=Path)
    ap.add_argument("--preview", action="store_true", help="480p 低清样片")
    ap.add_argument("--scene", default="ChapterScene")
    args = ap.parse_args()
    build(args.chapter_dir, args.preview, args.scene)


if __name__ == "__main__":
    main()
