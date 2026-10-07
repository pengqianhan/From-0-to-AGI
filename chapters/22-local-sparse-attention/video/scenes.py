"""Video for Chapter 22: local and sparse attention.

Look only at near tokens, but do not lose the far tokens.

The code in ../code/ calculates all values on screen (see the fact list in script.md).
The cache video/out/cache.json keeps the results. Delete it to calculate them again
(first run code/02_swa_model.py to train the 6 small models).
Render: bash chapters/22-local-sparse-attention/video/build.sh
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Axes,
    Create,
    DashedLine,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    LaggedStart,
    MathTex,
    Rectangle,
    RoundedRectangle,
    Square,
    Text,
    Transform,
    VGroup,
    Write,
)

from video_kit import theme
from video_kit.scene import NarratedScene, code_block, polyline_in_axes, zh

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
CACHE = HERE / "out" / "cache.json"
MONO = "Noto Sans Mono"
GiB = 2**30


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def compute() -> dict:
    """Calculate all numbers for the video with the real code in ../code."""
    import torch

    torch.set_num_threads(1)
    led = _load("masks_ledger", "01_masks_and_ledger.py")
    m = _load("swa_model", "02_swa_model.py")
    bc = _load("bounded_cache", "04_bounded_cache.py")
    tk = _load("topk_sparse", "05_topk_sparse.py")
    d: dict = {}
    d["pairs"] = [dict(T=T, full=T * (T + 1) // 2, swa=4 * T - 4 * 3 // 2) for T in (1024, 32768)]
    T = 64
    d["rf_swa"] = led.receptive_field([led.sliding_mask(T, 4)] * 6)
    d["rf_mix"] = led.receptive_field(
        [led.sliding_mask(T, 4) if (i + 1) % 3 else led.causal_mask(T) for i in range(6)]
    )
    d["ledger"] = []
    for name, n_layers, n_global, win, kvh, hd in led.MODELS:
        full_b = led.kv_bytes(n_layers, 0, win, 131072, kvh, hd)
        real_b = led.kv_bytes(n_global, n_layers - n_global, win, 131072, kvh, hd)
        d["ledger"].append(dict(name=name, full=full_b / GiB, real=real_b / GiB))
    d["lm"] = {v: m.lm_val_loss(m.load_or_train("lm", v, verbose=False)) for v in m.VARIANTS}
    d["needle"] = {
        v: m.needle_accuracy(m.load_or_train("needle", v, verbose=False)).tolist()
        for v in m.VARIANTS
    }
    data = m.CharData()
    report_at = (16, 64, 128, 256, 306)
    d["cache"] = {}
    for v in m.VARIANTS:
        model = m.load_or_train("lm", v, verbose=False)
        a, sizes = bc.generate(model, data.encode("ROMEO:\n"), 300, True, report_at)
        b, _ = bc.generate(model, data.encode("ROMEO:\n"), 300, False)
        d["cache"][v] = dict(sizes=[sizes[t] for t in report_at], same=a == b)
    d["report_at"] = list(report_at)
    needle = m.load_or_train("needle", "full", verbose=False)
    tk.set_recent(needle, 8)
    recent = m.needle_accuracy(needle)
    tk.set_recent(needle, None)
    needle.set_topk(8)
    top = m.needle_accuracy(needle)
    needle.set_topk(None)
    d["k8"] = dict(
        recent_far=recent[60:].mean().item(),
        top_far=top[60:].mean().item(),
        recent_all=recent.mean().item(),
        top_all=top.mean().item(),
    )
    return d


def get_data() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    d = compute()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


D = get_data()
COLORS = {"full": theme.ATTN, "sliding": theme.INPUT, "interleave": theme.OUTPUT}
NAMES = {"full": "全注意力", "sliding": "全部滑动窗口", "interleave": "3 局部 + 1 全局"}


def mono(text: str, size: float = 24, color: str = theme.FG) -> Text:
    return Text(text, font=MONO, font_size=size, color=color)


def cell(size: float, color: str, opacity: float = 0.9) -> Square:
    return Square(side_length=size * 0.9, fill_color=color, fill_opacity=opacity, stroke_width=0)


def matrix(n: int, size: float, center, visible) -> VGroup:
    """Draw an n×n attention matrix.

    Cells where visible(i, j) is True are purple. The other cells are dark.
    """
    g = VGroup()
    for i in range(n):
        for j in range(n):
            on = visible(i, j)
            sq = cell(size, theme.ATTN if on else theme.MUTED, 0.9 if on else 0.12)
            sq.move_to(
                [center[0] + (j - (n - 1) / 2) * size, center[1] - (i - (n - 1) / 2) * size, 0]
            )
            g.add(sq)
    return g


def hbar(width: float, color: str, height: float = 0.34, opacity: float = 0.85) -> Rectangle:
    return Rectangle(
        width=max(width, 0.02),
        height=height,
        fill_color=color,
        fill_opacity=opacity,
        stroke_width=0,
    )


class ChapterScene(NarratedScene):
    chapter_label = "第 22 章"
    chapter_title = "局部与稀疏注意力"

    def construct(self) -> None:
        for i in range(1, 14):
            getattr(self, f"s{i:02d}")()

    # ── S01 Opening ──────────────────────────────────────────────────────
    def s01(self) -> None:
        with self.shot("S01"):
            card = self.chapter_card()
            sub = zh("只看附近，也不丢掉远处", 32, theme.HIGHLIGHT).next_to(card, DOWN, 0.6)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(card), FadeOut(sub), run_time=self.fit(0.8))

    # ── S02 Two costs of full attention ──────────────────────────────────
    def s02(self) -> None:
        with self.shot("S02"):
            self.play(
                *self.set_heading("全注意力：算 T² 级别的对数，存 T 个位置"), run_time=self.fit(0.8)
            )
            n, s = 16, 0.27
            grid = matrix(n, s, (-3.3, 0.1), lambda i, j: False)
            self.play(FadeIn(grid), run_time=self.fit(0.8))
            qlab = zh("query 位置 i", 20, theme.MUTED).rotate(1.5708).next_to(grid, LEFT, 0.2)
            klab = zh("key 位置 j", 20, theme.MUTED).next_to(grid, UP, 0.15)
            self.play(FadeIn(qlab), FadeIn(klab), run_time=self.fit(0.6))
            formula = MathTex(r"\frac{T(T+1)}{2}", font_size=48).move_to([2.6, 1.6, 0])
            cap = zh("要算的对数", 22, theme.MUTED).next_to(formula, DOWN, 0.2)
            kv_base = [1.2, -1.9, 0]
            kv_lab = zh("KV cache", 22, theme.ATTN).move_to([4.8, 0.9, 0])
            self.play(Write(formula), FadeIn(cap), FadeIn(kv_lab), run_time=self.fit(1))
            bar = hbar(0.02, theme.ATTN, height=0.5).move_to(kv_base, aligned_edge=LEFT)
            self.add(bar)
            per_row = max(0.12, (self.remaining() - 2) / n)
            for i in range(n):
                row = [grid[i * n + j] for j in range(i + 1)]
                new_bar = hbar(0.3 * (i + 1), theme.ATTN, height=0.5).move_to(
                    kv_base, aligned_edge=LEFT
                )
                self.play(
                    *[c.animate.set_fill(theme.ATTN, 0.9) for c in row],
                    Transform(bar, new_bar),
                    run_time=per_row,
                )
            self.wait(self.remaining() - 0.6)
            self.play(
                FadeOut(VGroup(formula, cap, kv_lab, bar, qlab, klab)), run_time=self.fit(0.6)
            )
            self.grid = grid

    # ── S03 Sliding window ───────────────────────────────────────────────
    def s03(self) -> None:
        with self.shot("S03"):
            self.play(
                *self.set_heading("滑动窗口：每个 token 只看最近 W 个"), run_time=self.fit(0.8)
            )
            n, W = 16, 4
            outside = [
                self.grid[i * n + j] for i in range(n) for j in range(n) if j <= i and i - j >= W
            ]
            self.play(
                *[c.animate.set_fill(theme.MUTED, 0.12) for c in outside], run_time=self.fit(2)
            )
            f = MathTex(r"j \le i", r"\quad\text{and}\quad", r"i - j < W", font_size=46)
            f[2].set_color(theme.HIGHLIGHT)
            f.move_to([2.9, 1.2, 0])
            note = zh("W = 4：带子宽度固定", 24, theme.MUTED).next_to(f, DOWN, 0.4)
            code = code_block("mask = (j <= i) & (i - j < W)", 22).next_to(note, DOWN, 0.5)
            self.play(Write(f), run_time=self.fit(1.5))
            self.play(FadeIn(note), run_time=self.fit(0.8))
            self.play(FadeIn(code), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(f, note, code, self.grid)), run_time=self.fit(0.6))

    # ── S04 How much we save ─────────────────────────────────────────────
    def s04(self) -> None:
        with self.shot("S04"):
            self.play(*self.set_heading("数一数：窗口固定，越长省得越多"), run_time=self.fit(0.8))
            rows = VGroup()
            for p in D["pairs"]:
                r = VGroup(
                    zh(f"T = {p['T']:,}", 28, theme.MUTED),
                    zh(f"全因果 {p['full']:,} 对", 28, theme.ATTN),
                    zh(f"滑动窗口 {p['swa']:,} 对", 28, theme.INPUT),
                    zh(f"{p['full'] / p['swa']:,.0f} 倍", 34, theme.HIGHLIGHT),
                ).arrange(RIGHT, buff=0.5)
                rows.add(r)
            rows.arrange(DOWN, buff=0.8, aligned_edge=LEFT).move_to([0, 0.6, 0])
            for r in rows:
                self.play(FadeIn(r, shift=UP * 0.2), run_time=self.fit(1.2))
                self.wait(min(2.5, self.remaining() * 0.25))
            foot = zh("W = 4（真实模型的窗口是 128 到 4096）", 22, theme.MUTED).move_to(
                [0, -1.6, 0]
            )
            self.play(FadeIn(foot), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(rows, foot)), run_time=self.fit(0.6))

    # ── S05 Receptive field ──────────────────────────────────────────────
    def s05(self) -> None:
        with self.shot("S05"):
            self.play(*self.set_heading("感受野：信息一层层接力"), run_time=self.fit(0.8))
            T, s, h = 64, 0.1, 0.26

            def panel(rf: list[int], y0: float, title: str, color: str):
                cells = VGroup()
                rows = []
                for l in range(6):
                    row = VGroup()
                    for p in range(T):
                        sq = Rectangle(
                            width=s * 0.85,
                            height=h * 0.8,
                            stroke_width=0,
                            fill_color=theme.MUTED,
                            fill_opacity=0.12,
                        )
                        sq.move_to([-2.2 + p * s, y0 - l * h, 0])
                        row.add(sq)
                    rows.append(row)
                    cells.add(row)
                lab = zh(title, 22, color).next_to(cells, UP, 0.12).align_to(cells, LEFT)
                ll = VGroup(
                    *[
                        zh(f"第 {l + 1} 层", 16, theme.MUTED).next_to(rows[l], LEFT, 0.15)
                        for l in range(6)
                    ]
                )
                nums = VGroup(
                    *[mono(str(rf[l]), 18, color).next_to(rows[l], RIGHT, 0.2) for l in range(6)]
                )
                return cells, rows, lab, ll, nums

            a = panel(D["rf_swa"], 2.0, "全部滑动窗口（W = 4）：l × 3", theme.INPUT)
            b = panel(D["rf_mix"], -0.35, "第 3、6 层换成全局", theme.OUTPUT)
            for cells, _rows, lab, ll, _ in (a, b):
                self.play(FadeIn(cells), FadeIn(lab), FadeIn(ll), run_time=self.fit(0.8))
            for l in range(6):
                anims = []
                for (_c, rows, _l, _ll, nums), rf, col in (
                    (a, D["rf_swa"], theme.INPUT),
                    (b, D["rf_mix"], theme.OUTPUT),
                ):
                    for p in range(T - 1 - rf[l], T):
                        anims.append(rows[l][p].animate.set_fill(col, 0.85))
                    anims.append(FadeIn(nums[l]))
                self.play(*anims, run_time=self.fit(1.0, reserve=2))
                self.wait(min(1.5, max(0.1, (self.remaining() - 2) / (6 - l) - 1.0)))
            self.wait(self.remaining() - 0.6)
            self.play(
                FadeOut(VGroup(*a[0], a[2], a[3], a[4], *b[0], b[2], b[3], b[4])),
                run_time=self.fit(0.6),
            )

    # ── S06 The KV cache has a limit ─────────────────────────────────────
    def s06(self) -> None:
        with self.shot("S06"):
            self.play(*self.set_heading("滑动窗口层：KV cache 有上限"), run_time=self.fit(0.8))
            # Left: schematic of the ring buffer (8 slots)
            ring = VGroup()
            for k in range(8):
                ang = 1.5708 - k * 6.2832 / 8
                sq = Square(0.55, stroke_color=theme.INPUT, stroke_width=2)
                sq.move_to([-4.6 + 1.3 * math.cos(ang), 0.4 + 1.3 * math.sin(ang), 0])
                ring.add(sq)
            ring_lab = zh("环形缓冲区：新的覆盖最旧的", 20, theme.MUTED).move_to([-4.6, -1.5, 0])
            self.play(Create(ring), FadeIn(ring_lab), run_time=self.fit(1))
            fills = VGroup()
            for t in range(12):
                k = t % 8
                num = mono(str(t), 20, theme.FG).move_to(ring[k])
                old = [f for f in fills if f.slot == k]
                anims = [FadeIn(num)] + [FadeOut(o) for o in old]
                num.slot = k
                for o in old:
                    fills.remove(o)
                fills.add(num)
                self.play(*anims, run_time=self.fit(0.25, reserve=6))
            # Right: cache size of the 3 configurations
            axes_x, base_y, scale = [0.3, 2.4, 4.5], -1.6, 3.3 / max(D["cache"]["full"]["sizes"])
            bars, labs = VGroup(), VGroup()
            for x, v in zip(axes_x, ("full", "sliding", "interleave")):
                bars.add(
                    Rectangle(
                        width=1.0,
                        height=0.02,
                        fill_color=COLORS[v],
                        fill_opacity=0.85,
                        stroke_width=0,
                    ).move_to([x, base_y, 0], aligned_edge=DOWN)
                )
                short = {
                    "full": "全注意力",
                    "sliding": "全部滑动",
                    "interleave": "3 局部 + 1 全局",
                }[v]
                labs.add(zh(short, 20, COLORS[v]).move_to([x, base_y - 0.35, 0]))
            self.play(FadeIn(bars), FadeIn(labs), run_time=self.fit(0.6))
            t_lab, vals = None, None
            for k, t in enumerate(D["report_at"]):
                new_bars, new_vals = VGroup(), VGroup()
                for x, v in zip(axes_x, ("full", "sliding", "interleave")):
                    size = D["cache"][v]["sizes"][k]
                    nb = Rectangle(
                        width=1.0,
                        height=max(0.02, size * scale),
                        fill_color=COLORS[v],
                        fill_opacity=0.85,
                        stroke_width=0,
                    )
                    nb.move_to([x, base_y, 0], aligned_edge=DOWN)
                    new_bars.add(nb)
                    new_vals.add(mono(f"{size:,}", 16, COLORS[v]).next_to(nb, UP, 0.08))
                new_t = zh(f"已缓存 {t} 个位置（字节）", 22, theme.MUTED).move_to([2.4, 2.55, 0])
                if t_lab is None:
                    t_lab, vals = new_t, new_vals
                    self.play(
                        Transform(bars, new_bars),
                        FadeIn(vals),
                        FadeIn(t_lab),
                        run_time=self.fit(0.8, reserve=2),
                    )
                else:
                    self.play(
                        Transform(bars, new_bars),
                        Transform(vals, new_vals),
                        Transform(t_lab, new_t),
                        run_time=self.fit(0.8, reserve=2),
                    )
                self.wait(min(1.2, self.remaining() * 0.15))
            same = all(D["cache"][v]["same"] for v in D["cache"])
            ok = zh(f"截断缓存生成 = 不用缓存生成：{same}", 22, theme.OUTPUT).move_to(
                [2.4, -2.45, 0]
            )
            self.play(FadeIn(ok), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.play(
                FadeOut(VGroup(ring, ring_lab, fills, bars, labs, vals, t_lab, ok)),
                run_time=self.fit(0.6),
            )

    # ── S07 How real models configure it ─────────────────────────────────
    def s07(self) -> None:
        with self.shot("S07"):
            self.play(*self.set_heading("局部-全局交替：真实模型的配置"), run_time=self.fit(0.8))
            specs = [
                ("Mistral 7B v0.1", 32, None, "W = 4096"),
                ("gpt-oss-120b", 36, 2, "W = 128"),
                ("OLMo 3 7B", 32, 4, "W = 4096"),
                ("Gemma 3 27B", 62, 6, "W = 1024"),
            ]
            s = 0.1
            rows = VGroup()
            for k, (name, n, every, w) in enumerate(specs):
                y = 1.7 - k * 0.95
                strip = VGroup()
                for i in range(n):
                    is_global = every is not None and (i + 1) % every == 0
                    r = Rectangle(
                        width=s * 0.8,
                        height=0.42,
                        stroke_width=0,
                        fill_color=theme.ATTN if is_global else theme.INPUT,
                        fill_opacity=0.9,
                    )
                    r.move_to([-2.4 + i * s, y, 0])
                    strip.add(r)
                lab = zh(name, 22).move_to([-4.9, y, 0])
                wl = zh(w, 20, theme.MUTED).move_to([4.6, y, 0])
                rows.add(VGroup(lab, strip, wl))
            legend = (
                VGroup(
                    VGroup(hbar(0.3, theme.INPUT, 0.3), zh("滑动窗口层", 20)).arrange(
                        RIGHT, buff=0.12
                    ),
                    VGroup(hbar(0.3, theme.ATTN, 0.3), zh("全注意力层", 20)).arrange(
                        RIGHT, buff=0.12
                    ),
                    zh("来源：各模型 config.json 的 layer_types / sliding_window", 18, theme.MUTED),
                )
                .arrange(RIGHT, buff=0.5)
                .move_to([0, -2.2, 0])
            )
            self.play(FadeIn(legend), run_time=self.fit(0.6))
            for r in rows:
                self.play(
                    FadeIn(r[0]),
                    LaggedStart(*[FadeIn(c) for c in r[1]], lag_ratio=0.02),
                    FadeIn(r[2]),
                    run_time=self.fit(1.4, reserve=1),
                )
                self.wait(min(2.5, max(0.1, self.remaining() * 0.18)))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(rows, legend)), run_time=self.fit(0.6))

    # ── S08 Ledger ───────────────────────────────────────────────────────
    def s08(self) -> None:
        with self.shot("S08"):
            self.play(
                *self.set_heading("128K 上下文、BF16：KV cache 省多少"), run_time=self.fit(0.8)
            )
            scale = 6.0 / max(r["full"] for r in D["ledger"])
            items = VGroup()
            for k, r in enumerate(D["ledger"]):
                y = 1.8 - k * 1.0
                name = r["name"].split("（")[0]
                lab = zh(name, 22).move_to([-5.3, y, 0])
                full = hbar(r["full"] * scale, theme.MUTED, 0.3, 0.35).move_to(
                    [-4.1, y + 0.17, 0], aligned_edge=LEFT
                )
                real = hbar(r["real"] * scale, theme.OUTPUT, 0.3).move_to(
                    [-4.1, y - 0.17, 0], aligned_edge=LEFT
                )
                fv = zh(f"{r['full']:.2f}", 18, theme.MUTED).next_to(full, RIGHT, 0.1)
                rv = zh(
                    f"{r['real']:.2f} GiB，省 {1 - r['real'] / r['full']:.1%}", 18, theme.OUTPUT
                ).next_to(real, RIGHT, 0.1)
                items.add(VGroup(lab, full, real, fv, rv))
            legend = (
                VGroup(
                    VGroup(
                        hbar(0.3, theme.MUTED, 0.25, 0.35), zh("假如每层都是全注意力", 18)
                    ).arrange(RIGHT, buff=0.1),
                    VGroup(hbar(0.3, theme.OUTPUT, 0.25), zh("实际配置", 18)).arrange(
                        RIGHT, buff=0.1
                    ),
                )
                .arrange(RIGHT, buff=0.5)
                .move_to([2.5, -2.3, 0])
            )
            self.play(FadeIn(legend), run_time=self.fit(0.5))
            for it in items:
                self.play(
                    FadeIn(it[0]),
                    GrowFromEdge(it[1], LEFT),
                    GrowFromEdge(it[2], LEFT),
                    FadeIn(it[3]),
                    FadeIn(it[4]),
                    run_time=self.fit(1.2, reserve=1),
                )
                self.wait(min(2.5, max(0.1, self.remaining() * 0.2)))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(items, legend)), run_time=self.fit(0.6))

    # ── S09 Experiment 1: loss ───────────────────────────────────────────
    def s09(self) -> None:
        with self.shot("S09"):
            self.play(*self.set_heading("实验一：字符级语言建模"), run_time=self.fit(0.8))
            setup = zh("4 层小模型，窗口 W = 16，Shakespeare 字符级，600 步", 24, theme.MUTED)
            setup.move_to([0, 2.45, 0])
            self.play(FadeIn(setup), run_time=self.fit(0.8))
            rows = VGroup()
            for v in ("full", "sliding", "interleave"):
                rows.add(
                    VGroup(zh(NAMES[v], 30, COLORS[v]), mono(f"{D['lm'][v]:.3f}", 34)).arrange(
                        RIGHT, buff=1.2
                    )
                )
            rows.arrange(DOWN, buff=0.55, aligned_edge=LEFT).move_to([0, 0.0, 0])
            head = zh("验证集 loss（nats / 字符）", 22, theme.MUTED).next_to(rows, UP, 0.4)
            self.play(FadeIn(head), run_time=self.fit(0.5))
            for r in rows:
                self.play(FadeIn(r, shift=UP * 0.2), run_time=self.fit(0.8))
                self.wait(min(1.5, self.remaining() * 0.12))
            note = zh("几乎一样：预测下一个字符，附近的上下文就够了", 24, theme.HIGHLIGHT)
            note.move_to([0, -1.8, 0])
            self.play(FadeIn(note), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(setup, rows, head, note)), run_time=self.fit(0.6))

    # ── S10 Experiment 2: needle in a haystack ───────────────────────────
    def s10(self) -> None:
        with self.shot("S10"):
            self.play(*self.set_heading("实验二：大海捞针"), run_time=self.fit(0.8))
            ax = Axes(
                x_range=[0, 96, 16],
                y_range=[0, 1.05, 0.25],
                x_length=8.2,
                y_length=3.8,
                axis_config={"color": theme.MUTED, "include_numbers": True, "font_size": 20},
                y_axis_config={"decimal_number_config": {"num_decimal_places": 2}},
            ).move_to([-0.9, 0.05, 0])
            xl = zh("针与提问的距离 d", 20, theme.MUTED).next_to(ax.x_axis, DOWN, 0.35)
            yl = zh("准确率", 20, theme.MUTED).next_to(ax.y_axis, UP, 0.15)
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=self.fit(1.2))
            chance = DashedLine(ax.c2p(0, 0.125), ax.c2p(96, 0.125), color=theme.MUTED)
            cl = (
                zh("瞎猜 12.5%", 18, theme.MUTED)
                .next_to(ax.c2p(96, 0.125), UP, 0.08)
                .shift(LEFT * 0.6)
            )
            v16 = DashedLine(ax.c2p(16, 0), ax.c2p(16, 1.05), color=theme.HIGHLIGHT)
            v60 = DashedLine(ax.c2p(60, 0), ax.c2p(60, 1.05), color=theme.GRAD)
            l16 = zh("W = 16", 18, theme.HIGHLIGHT).next_to(v16, UP, 0.05)
            l60 = zh("4 × 15 = 60", 18, theme.GRAD).next_to(v60, UP, 0.05)
            self.play(
                Create(chance),
                FadeIn(cl),
                Create(v16),
                Create(v60),
                FadeIn(l16),
                FadeIn(l60),
                run_time=self.fit(1.2),
            )
            legend = VGroup()
            lines = VGroup()
            for v in ("full", "interleave", "sliding"):
                acc = D["needle"][v]
                # Average each 4 distances to make the curve smoother. Move the full-attention line and the
                # interleaved line a little apart, so that they do not overlap.
                pts = []
                for e in range(1, 96, 4):
                    seg = acc[e - 1 : e + 3]
                    y = sum(seg) / len(seg) - (0.03 if v == "interleave" else 0)
                    pts.append((e + 1.5, y))
                lines.add(polyline_in_axes(ax, pts, color=COLORS[v], stroke_width=4))
                legend.add(
                    VGroup(hbar(0.35, COLORS[v], 0.12), zh(NAMES[v], 20, COLORS[v])).arrange(
                        RIGHT, buff=0.12
                    )
                )
            legend.arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([5.3, 1.0, 0])
            for ln, lg in zip(lines, legend):
                self.play(Create(ln), FadeIn(lg), run_time=self.fit(1.6, reserve=2))
                self.wait(min(3.0, self.remaining() * 0.2))
            self.wait(self.remaining() - 0.6)
            self.play(
                FadeOut(VGroup(ax, xl, yl, chance, cl, v16, v60, l16, l60, lines, legend)),
                run_time=self.fit(0.6),
            )

    # ── S11 Sparse attention: select by content ──────────────────────────
    def s11(self) -> None:
        with self.shot("S11"):
            self.play(
                *self.set_heading("同样只看 k 个键：按位置挑 vs 按内容挑"), run_time=self.fit(0.8)
            )
            n, s = 24, 0.3
            needle_pos = 3
            recent = set(range(n - 8, n))
            top = {needle_pos, 9, 14, 18, 20, 21, 22, 23}

            def row(y: float, chosen: set, title: str, color: str):
                g = VGroup()
                for j in range(n):
                    on = j in chosen
                    sq = cell(s, color if on else theme.MUTED, 0.9 if on else 0.15)
                    sq.move_to([-3.5 + j * s, y, 0])
                    g.add(sq)
                star = zh("针", 18, theme.HIGHLIGHT).next_to(g[needle_pos], DOWN, 0.08)
                lab = zh(title, 22, color).next_to(g, UP, 0.15).align_to(g, LEFT)
                return VGroup(g, star, lab)

            r1 = row(1.6, recent, "最近 8 个（滑动窗口）", theme.INPUT)
            r2 = row(-0.2, top, "分数最高的 8 个（稀疏注意力）", theme.ATTN)
            demo = zh("示意图", 18, theme.MUTED).move_to([4.3, 2.4, 0])
            self.play(FadeIn(r1), FadeIn(demo), run_time=self.fit(1.2))
            self.wait(min(2.5, self.remaining() * 0.15))
            self.play(FadeIn(r2), run_time=self.fit(1.2))
            k8 = D["k8"]
            res = (
                VGroup(
                    zh(
                        "全注意力模型、推理时只留 8 个键，远处（d > 60）捞针准确率：",
                        20,
                        theme.MUTED,
                    ),
                    VGroup(
                        zh(f"最近 8 个：{k8['recent_far']:.1%}", 26, theme.INPUT),
                        zh(f"分数最高 8 个：{k8['top_far']:.1%}", 26, theme.ATTN),
                    ).arrange(RIGHT, buff=0.8),
                )
                .arrange(DOWN, buff=0.25)
                .move_to([0.3, -1.85, 0])
            )
            self.wait(min(3, self.remaining() * 0.2))
            self.play(FadeIn(res), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(r1, r2, res, demo)), run_time=self.fit(0.6))

    # ── S12 Sparse attention: consensus and costs ────────────────────────
    def s12(self) -> None:
        with self.shot("S12"):
            self.play(*self.set_heading("稀疏注意力：已经是大模型的共识"), run_time=self.fit(0.8))
            steps = VGroup()
            for t, c in (
                ("便宜的索引器\n给历史打分", theme.PARAM),
                ("选出 top-k\n（几千个）", theme.HIGHLIGHT),
                ("只对选中的\n做精确注意力", theme.ATTN),
            ):
                box = RoundedRectangle(width=2.8, height=1.2, corner_radius=0.15, stroke_color=c)
                steps.add(VGroup(box, zh(t, 22, c).move_to(box)))
            steps.arrange(RIGHT, buff=0.9).move_to([0, 1.5, 0])
            arrows = VGroup(
                *[
                    Arrow(
                        steps[i].get_right(), steps[i + 1].get_left(), buff=0.1, color=theme.MUTED
                    )
                    for i in range(2)
                ]
            )
            self.play(
                LaggedStart(*[FadeIn(s_) for s_ in steps], lag_ratio=0.3),
                Create(arrows),
                run_time=self.fit(2),
            )
            fams = VGroup(
                *[
                    zh(t, 22)
                    for t in (
                        "DeepSeek V3.2 / V4",
                        "GLM-5（DSA）",
                        "MiniMax-M3（MSA）",
                        "LongCat-2.0（LSA）",
                    )
                ]
            )
            fams.arrange(RIGHT, buff=0.55).move_to([0, -0.2, 0])
            self.wait(min(3, self.remaining() * 0.15))
            self.play(
                LaggedStart(*[FadeIn(f, shift=UP * 0.2) for f in fams], lag_ratio=0.3),
                run_time=self.fit(2),
            )
            tag = zh("都是 2000 亿参数以上的 MoE 旗舰", 22, theme.MUTED).next_to(fams, DOWN, 0.35)
            self.play(FadeIn(tag), run_time=self.fit(0.6))
            self.wait(min(4, self.remaining() * 0.35))
            warn = zh("省算力，不省 KV cache：将来哪个 token 会被选中，事先不知道", 24, theme.GRAD)
            warn.move_to([0, -1.9, 0])
            self.play(FadeIn(warn), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(steps, arrows, fams, tag, warn)), run_time=self.fit(0.6))

    # ── S13 From minimal code to production code ─────────────────────────
    def s13(self) -> None:
        with self.shot("S13"):
            self.play(*self.set_heading("从极简到生产级"), run_time=self.fit(0.8))
            left = code_block("mask = (k <= q)\nmask &= (q - k) < window", 22).move_to(
                [-3.6, 1.2, 0]
            )
            ll = zh("极简版：02_swa_model.py", 20, theme.MUTED).next_to(left, UP, 0.3)
            right = (
                VGroup(
                    *[
                        mono(t, 22, c)
                        for t, c in (
                            ("make_layer_types()", theme.INPUT),
                            ("SlidingWindowAttention", theme.ATTN),
                            ("SlidingWindowKVCache", theme.OUTPUT),
                            ("flash_attn window_size", theme.PARAM),
                        )
                    ]
                )
                .arrange(DOWN, aligned_edge=LEFT, buff=0.3)
                .move_to([2.9, 1.0, 0])
            )
            rl = zh("zero/arch/sliding_window.py", 20, theme.MUTED).next_to(right, UP, 0.3)
            self.play(FadeIn(left), FadeIn(ll), run_time=self.fit(1))
            self.play(
                LaggedStart(*[FadeIn(r) for r in right], lag_ratio=0.3),
                FadeIn(rl),
                run_time=self.fit(2),
            )
            gpu = (
                VGroup(
                    zh("RTX 3090：CUDA 上的正确性已验证", 20, theme.MUTED),
                    zh("FlexAttention 实测：64K 时比稠密因果快 39 倍", 20, theme.MUTED),
                )
                .arrange(DOWN, buff=0.15)
                .next_to(right, DOWN, 0.35)
            )
            self.play(FadeIn(gpu), run_time=self.fit(0.6))
            self.wait(max(0.1, self.remaining() - 4))
            nxt = zh("下一章：线性注意力与混合架构", 30, theme.HIGHLIGHT).move_to([0, -1.6, 0])
            self.play(FadeIn(nxt, shift=UP * 0.2), run_time=self.fit(1))
            self.wait(self.remaining() - 0.8)
            self.play(
                FadeOut(VGroup(left, ll, right, rl, gpu, nxt)),
                *self.set_heading(None),
                run_time=self.fit(0.8),
            )
