"""One visual language for the full course: colors, fonts, and layout.

A concept has the same color in the videos of all chapters:
    input = blue, parameter = orange, gradient = red, attention weight = purple,
    output/prediction = green.
"""

from __future__ import annotations

from functools import lru_cache

# ── Semantic colors (enough contrast on a dark background) ──────────────────
BG = "#101418"          # background
FG = "#E8EAED"          # body text
MUTED = "#9AA0A6"       # secondary text, axes
INPUT = "#4C9BE8"       # input / data
PARAM = "#F29E4C"       # parameter / weight
GRAD = "#E8615A"        # gradient / error
ATTN = "#B07CE8"        # attention weight
OUTPUT = "#5CC98A"      # output / prediction
HIGHLIGHT = "#F4D35E"   # emphasis

# ── Layout ──────────────────────────────────────────────────────────────────
# Keep a subtitle safe area at the bottom of the frame. ffmpeg burns the subtitles
# into this area, so do not put animation content there.
SUBTITLE_SAFE_BOTTOM = -2.75  # Manim coordinates (the frame is 8 units high, y from -4 to 4)
TITLE_Y = 3.3

_FONT_CANDIDATES = [
    "WenQuanYi Zen Hei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "Microsoft YaHei",
]


@lru_cache(maxsize=1)
def cjk_font() -> str:
    """Return the name of the first Chinese font that is available on this machine."""
    try:
        import manimpango

        available = set(manimpango.list_fonts())
    except Exception:  # noqa: BLE001 - if manimpango is missing, use the default
        available = set()
    for name in _FONT_CANDIDATES:
        if name in available:
            return name
    return _FONT_CANDIDATES[0]
