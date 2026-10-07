"""NarratedScene: a Manim scene base class that aligns animation and narration shot by shot.

Usage (in the video/scenes.py of a chapter):

    from video_kit.scene import NarratedScene, zh

    class ChapterScene(NarratedScene):
        def construct(self):
            with self.shot("S01"):
                self.play(Write(zh("一条直线")), run_time=self.fit(2))
            with self.shot("S02"):
                ...

`with self.shot(id)` adds the narration audio at the start of the shot. At the exit, if the
animation is shorter than the narration, it waits for the remaining time. Thus the next shot
starts after the narration ends. `self.fit(t)` cuts the desired animation time to the time
that is left in the shot, so that the animation does not continue after the narration.
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
    """Chinese text (with the font of the course)."""
    return Text(text, font=theme.cjk_font(), font_size=size, color=color, **kw)


class NarratedScene(Scene):
    #: Chapter label on the title card, for example the Chinese text for "Chapter 1"
    chapter_label: str = ""
    #: Chapter title on the title card, for example "y = ax + b"
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

    # ── Shots ───────────────────────────────────────────────────────────────
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
            print(f"[video_kit] Warning: the animation of {shot_id} is {-remaining:.1f}s longer than the narration")

    def remaining(self) -> float:
        """Return the seconds of narration that are left in this shot."""
        return max(0.1, self._shot_end - self.renderer.time)

    def fit(self, desired: float, reserve: float = 0.0) -> float:
        """Return an animation time that is not more than the time left in this shot.

        `reserve` seconds stay free for the animations that come after.
        """
        return max(0.1, min(desired, self.remaining() - reserve))

    def tear_down(self) -> None:
        log_path = os.environ.get("VIDEO_SHOT_LOG")
        if log_path:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"shots": self._shot_log, "total": self.renderer.time},
                    f, ensure_ascii=False, indent=1,
                )

    # ── Common frame elements ───────────────────────────────────────────────
    def chapter_card(self) -> VGroup:
        """Title card for the opening."""
        label = zh(self.chapter_label, 30, theme.MUTED)
        title = zh(self.chapter_title, 64, theme.FG)
        series = zh("From 0 to AGI", 24, theme.MUTED)
        card = VGroup(label, title, series).arrange(DOWN, buff=0.45)
        return card

    def heading(self, text: str) -> Text:
        """Small heading at the top left. It only makes the heading; usually use set_heading."""
        return zh(text, 30, theme.MUTED).to_corner(UP + LEFT, buff=0.4)

    def set_heading(self, text: str | None) -> list:
        """Change the heading. Return a list of animations for self.play(*...).

        The old heading fades out, and the new heading fades in. The video shows only one
        heading at a time, so that two headings do not overlap. text=None only removes
        the old heading.
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
        """Label for the top right of the frame. Parts 3 and 4 must show it when they use
        tiny-configuration data."""
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
    """Draw a list of data points as a polyline. Keep only the parts in the range of the axes.

    The line breaks where it goes out of the range. Use it for contour lines, parameter paths,
    and other curves that can go out of the range. Then no line goes into other areas of
    the frame.
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
    """Code block in a monospace font that keeps the indentation.

    Manim's Text removes the spaces at the start of a line. Thus each line renders only
    the text without the indentation. Then the line moves to the right by
    "number of indentation characters × width of one monospace character".
    Return a VGroup with one Text for each line, for highlights line by line:
    code_block(src)[2] is line 3.
    """
    lines = source.strip("\n").splitlines()
    char_w = Text("M" * 10, font=MONO_FONT, font_size=size).width / 10
    group = VGroup()
    indents = []
    for line in lines:
        stripped = line.lstrip(" ")
        indents.append(len(line) - len(stripped))
        group.add(Text(stripped if stripped else " ", font=MONO_FONT, font_size=size,
                       color=color))
    group.arrange(DOWN, aligned_edge=LEFT, buff=line_buff)
    left = group.get_left()[0]
    for m, ind in zip(group, indents):
        m.shift(RIGHT * (left + ind * char_w - m.get_left()[0]))
    return group
