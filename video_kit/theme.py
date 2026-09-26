"""全课程统一的视觉语言：配色、字体、版式。

同一个概念在所有章节的视频里用同一种颜色：
    输入 = 蓝、参数 = 橙、梯度 = 红、注意力权重 = 紫、输出/预测 = 绿。
"""

from __future__ import annotations

from functools import lru_cache

# ── 语义配色（深色背景上对比度足够） ─────────────────────────────────────────
BG = "#101418"          # 背景
FG = "#E8EAED"          # 正文文字
MUTED = "#9AA0A6"       # 次要文字、坐标轴
INPUT = "#4C9BE8"       # 输入 / 数据
PARAM = "#F29E4C"       # 参数 / 权重
GRAD = "#E8615A"        # 梯度 / 误差
ATTN = "#B07CE8"        # 注意力权重
OUTPUT = "#5CC98A"      # 输出 / 预测
HIGHLIGHT = "#F4D35E"   # 强调

# ── 版式 ────────────────────────────────────────────────────────────────────
# 画面底部留出字幕安全区：字幕由 ffmpeg 烧录在这一带，动画内容不要放进去。
SUBTITLE_SAFE_BOTTOM = -2.75  # Manim 坐标（画面高 8 个单位，y 从 -4 到 4）
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
    """返回本机可用的第一个中文字体名。"""
    try:
        import manimpango

        available = set(manimpango.list_fonts())
    except Exception:  # noqa: BLE001 - manimpango 缺失时退回默认
        available = set()
    for name in _FONT_CANDIDATES:
        if name in available:
            return name
    return _FONT_CANDIDATES[0]
