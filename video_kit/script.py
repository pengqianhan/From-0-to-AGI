"""Parse the chapter video script `video/script.md` and get the narration of each shot.

The format of one shot in script.md (the field names are Chinese, because the videos
are in Chinese; the parser reads only the narration field "旁白"):

    ### S01 开场：一条直线
    - 画面：散点图淡入……
    - 屏幕文字：y = ax + b
    - 旁白：我们从最简单的模型开始：{y = ax + b|y 等于 a x 加 b}。

The narration can contain `{display text|spoken form}`. The subtitle shows the first part,
and the TTS reads the second part.
The narration can continue on more lines, until the next field that starts with "- "
or the next heading.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SHOT_RE = re.compile(r"^###\s+(S\d+)\b\s*(.*)$")
_FIELD_RE = re.compile(r"^-\s*([^：:]+)[：:]\s*(.*)$")
_MARKUP_RE = re.compile(r"\{([^{}|]*)\|([^{}]*)\}")
# Split sentences at Chinese end-of-sentence punctuation, and at an English period that
# has a space after it.
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])|(?<=\.)\s")


@dataclass
class Sentence:
    display: str  # text that the subtitle shows
    spoken: str   # text that goes to the TTS


@dataclass
class Shot:
    shot_id: str
    title: str
    narration: str  # raw narration (with {display|spoken} markup)

    @property
    def sentences(self) -> list[Sentence]:
        return split_sentences(self.narration)


def to_display(text: str) -> str:
    return _MARKUP_RE.sub(lambda m: m.group(1), text).strip()


def to_spoken(text: str) -> str:
    return _MARKUP_RE.sub(lambda m: m.group(2), text).strip()


def split_sentences(narration: str) -> list[Sentence]:
    """Split into sentences. A `{display|spoken}` markup is never cut into two parts."""
    # Replace each markup with a placeholder first, and restore it after the split.
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
        raise ValueError(f"{path}: duplicate shot IDs: {ids}")
    missing = [s.shot_id for s in shots if not s.narration]
    if missing:
        raise ValueError(f"{path}: these shots have no narration: {missing}")
    return shots
