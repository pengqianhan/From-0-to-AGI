"""NarratedScene：让动画和旁白按分镜对齐的 Manim 场景基类。

用法（章节的 video/scenes.py）：

    from video_kit.scene import NarratedScene, zh

    class ChapterScene(NarratedScene):
        def construct(self):
            with self.shot("S01"):
                self.play(Write(zh("一条直线")), run_time=self.fit(2))
            with self.shot("S02"):
                ...

`with self.shot(id)` 在这一镜开始时插入旁白音频；退出时如果动画比旁白短，
自动 wait 补齐，保证下一镜从旁白结束后开始。`self.fit(t)` 把期望的动画时长
压缩到本镜剩余时间以内，避免动画拖得比旁白还长。
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    FadeIn,
    FadeOut,
    Rectangle,
    Scene,
    Text,
    VGroup,
    VMobject,
)

from . import theme


def zh(text: str, size: float = 36, color: str = theme.FG, **kw) -> Text:
    """中文文字（统一字体）。"""
    return Text(text, font=theme.cjk_font(), font_size=size, color=color, **kw)


class NarratedScene(Scene):
    #: 例如 "第 1 章"
    chapter_label: str = ""
    #: 例如 "y = ax + b"
    chapter_title: str = ""

    def setup(self) -> None:
        self.camera.background_color = theme.BG
        path = os.environ.get("VIDEO_TIMINGS")
        self._timings: dict[str, dict] = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self._timings = json.load(f)
        self._shot_log: dict[str, float] = {}
        self._shot_end = 0.0

    # ── 分镜 ────────────────────────────────────────────────────────────────
    @contextmanager
    def shot(self, shot_id: str):
        info = self._timings.get(shot_id, {"wav": None, "duration": 3.0})
        start = self.renderer.time
        self._shot_log[shot_id] = start
        if info.get("wav"):
            self.add_sound(info["wav"])
        self._shot_end = start + float(info["duration"])
        yield float(info["duration"])
        remaining = self._shot_end - self.renderer.time
        if remaining > 0.02:
            self.wait(remaining)
        elif remaining < -0.5:
            print(f"[video_kit] 警告：{shot_id} 的动画比旁白长 {-remaining:.1f}s")

    def remaining(self) -> float:
        """本镜旁白还剩多少秒。"""
        return max(0.1, self._shot_end - self.renderer.time)

    def fit(self, desired: float, reserve: float = 0.0) -> float:
        """动画时长：不超过本镜剩余时间（减去 reserve 留给后续动画）。"""
        return max(0.1, min(desired, self.remaining() - reserve))

    def tear_down(self) -> None:
        log_path = os.environ.get("VIDEO_SHOT_LOG")
        if log_path:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"shots": self._shot_log, "total": self.renderer.time},
                    f, ensure_ascii=False, indent=1,
                )

    # ── 常用画面元素 ────────────────────────────────────────────────────────
    def chapter_card(self) -> VGroup:
        """片头标题卡。"""
        label = zh(self.chapter_label, 30, theme.MUTED)
        title = zh(self.chapter_title, 64, theme.FG)
        series = zh("From 0 to AGI", 24, theme.MUTED)
        card = VGroup(label, title, series).arrange(DOWN, buff=0.45)
        return card

    def heading(self, text: str) -> Text:
        """左上角的小标题（只创建，不管理；一般用 set_heading）。"""
        return zh(text, 30, theme.MUTED).to_corner(UP + LEFT, buff=0.4)

    def set_heading(self, text: str | None) -> list:
        """换标题：返回动画列表（旧标题淡出、新标题淡入），交给 self.play(*...)。

        全片同一时刻只有一个标题，避免新旧标题叠在一起。text=None 表示只清掉旧标题。
        """
        anims = []
        old = getattr(self, "_heading", None)
        if old is not None:
            anims.append(FadeOut(old))
        self._heading = self.heading(text) if text else None
        if self._heading is not None:
            anims.append(FadeIn(self._heading))
        return anims

    def demo_badge(self, text: str = "极小配置演示") -> VGroup:
        """第三、四部分用极小配置数据时，画面右上角必须带的标注。"""
        label = zh(text, 22, theme.BG)
        box = Rectangle(
            width=label.width + 0.4, height=label.height + 0.25,
            fill_color=theme.HIGHLIGHT, fill_opacity=1, stroke_width=0,
        )
        badge = VGroup(box, label)
        return badge.to_corner(UP + RIGHT, buff=0.3)

    def show_badge(self) -> VGroup:
        badge = self.demo_badge()
        self.play(FadeIn(badge), run_time=0.3)
        return badge



def polyline_in_axes(axes, points_xy, **style) -> VGroup:
    """把一串数据坐标画成折线，只保留落在坐标轴范围内的部分（超出范围处断开）。

    用来画等高线、参数轨迹等可能超出坐标范围的曲线，避免线条溢出到画面其他区域。
    """
    x0, x1 = axes.x_range[0], axes.x_range[1]
    y0, y1 = axes.y_range[0], axes.y_range[1]
    runs, cur = [], []
    for x, y in points_xy:
        if x0 <= x <= x1 and y0 <= y <= y1:
            cur.append(axes.c2p(x, y))
        else:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
    if len(cur) >= 2:
        runs.append(cur)
    group = VGroup()
    for run in runs:
        m = VMobject(**style)
        m.set_points_as_corners(np.array(run))
        group.add(m)
    return group


MONO_FONT = "Noto Sans Mono"


def code_block(source: str, size: float = 22, color: str = theme.FG,
               line_buff: float = 0.18) -> VGroup:
    """等宽字体的代码块，保留缩进（Manim 的 Text 会吞掉行首空格，这里换成不间断空格）。

    返回每行一个 Text 的 VGroup，左对齐，方便逐行高亮：code_block(src)[2] 是第 3 行。
    """
    lines = source.strip("\n").splitlines()
    group = VGroup()
    for line in lines:
        indent = len(line) - len(line.lstrip(" "))
        shown = " " * indent + line.lstrip(" ") if line.strip() else " "
        group.add(Text(shown, font=MONO_FONT, font_size=size, color=color))
    group.arrange(DOWN, aligned_edge=LEFT, buff=line_buff)
    return group
