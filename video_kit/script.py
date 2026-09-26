"""解析章节视频脚本 `video/script.md`，取出每一镜的旁白。

script.md 里每一镜的写法：

    ### S01 开场：一条直线
    - 画面：散点图淡入……
    - 屏幕文字：y = ax + b
    - 旁白：我们从最简单的模型开始：{y = ax + b|y 等于 a x 加 b}。

旁白里可以用 `{显示文字|读法}`：字幕显示前半部分，TTS 读后半部分。
旁白可以跨多行，直到下一个以 "- " 开头的字段或下一个标题为止。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SHOT_RE = re.compile(r"^###\s+(S\d+)\b\s*(.*)$")
_FIELD_RE = re.compile(r"^-\s*([^：:]+)[：:]\s*(.*)$")
_MARKUP_RE = re.compile(r"\{([^{}|]*)\|([^{}]*)\}")
# 断句：中文句末标点，以及后面跟空格的英文句号
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])|(?<=\.)\s")


@dataclass
class Sentence:
    display: str  # 字幕显示的文字
    spoken: str   # 送给 TTS 的文字


@dataclass
class Shot:
    shot_id: str
    title: str
    narration: str  # 原始旁白（含 {显示|读法} 标记）

    @property
    def sentences(self) -> list[Sentence]:
        return split_sentences(self.narration)


def to_display(text: str) -> str:
    return _MARKUP_RE.sub(lambda m: m.group(1), text).strip()


def to_spoken(text: str) -> str:
    return _MARKUP_RE.sub(lambda m: m.group(2), text).strip()


def split_sentences(narration: str) -> list[Sentence]:
    """按句切分，保证 `{显示|读法}` 标记不会被切断。"""
    # 先把标记替换成占位符，切完句再还原
    marks: list[str] = []

    def _hold(m: re.Match[str]) -> str:
        marks.append(m.group(0))
        return f"\x00{len(marks) - 1}\x00"

    held = _MARKUP_RE.sub(_hold, narration)
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(held) if p and p.strip()]
    out = []
    for p in parts:
        restored = re.sub(r"\x00(\d+)\x00", lambda m: marks[int(m.group(1))], p)
        out.append(Sentence(display=to_display(restored), spoken=to_spoken(restored)))
    return out


def parse_script(path: str | Path) -> list[Shot]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    shots: list[Shot] = []
    cur: Shot | None = None
    in_narration = False
    for line in lines:
        m = _SHOT_RE.match(line.strip())
        if m:
            cur = Shot(shot_id=m.group(1), title=m.group(2).strip(), narration="")
            shots.append(cur)
            in_narration = False
            continue
        if line.startswith("#"):
            cur = None
            in_narration = False
            continue
        if cur is None:
            continue
        f = _FIELD_RE.match(line.strip())
        if f:
            in_narration = f.group(1).strip() == "旁白"
            if in_narration:
                cur.narration = f.group(2).strip()
            continue
        if in_narration and line.strip():
            cur.narration += line.strip()
    ids = [s.shot_id for s in shots]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: 分镜编号重复：{ids}")
    missing = [s.shot_id for s in shots if not s.narration]
    if missing:
        raise ValueError(f"{path}: 这些分镜没有旁白：{missing}")
    return shots
