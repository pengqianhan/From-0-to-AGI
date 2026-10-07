"""Chapter 12 video: scaling laws and experiment design.

Calculate with small models first, then spend the money.

The code in ../code/ calculates all numbers in the frames (see the fact list in script.md):
  - Chinchilla curve, overtraining ledger: 02_chinchilla.py (coefficients of the Epoch AI
    replication; data from other people, only to show the shape)
  - FLOPs per token, budget: 05_plan_budget.py
  - mini ladder (tiny-configuration demo): the cache out/ch12/*.json of 03_lr_sweep.py,
    04_mini_ladder.py, and 06_muon.py
Render: bash chapters/12-scaling-laws/video/build.sh
"""

from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Arrow,
    Axes,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    GrowArrow,
    GrowFromEdge,
    LaggedStart,
    Line,
    MathTex,
    Rectangle,
    RoundedRectangle,
    VGroup,
    Write,
)

from video_kit import theme
from video_kit.scene import NarratedScene, polyline_in_axes, zh

CODE = Path(__file__).resolve().parent.parent / "code"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


chin = _load("chinchilla", "02_chinchilla.py")
plan = _load("plan_budget", "05_plan_budget.py")
sw = _load("lr_sweep", "03_lr_sweep.py")
lad = _load("mini_ladder", "04_mini_ladder.py")

FIT = chin.FITS["Epoch 2024 复现"]

# FLOPs per token of the main-line model (the same formulas as 05_plan_budget.py)
N_TOTAL, N_MATMUL, Q_DIM = plan.shape(1280, 28)
SIX_N = 6 * N_MATMUL
ATTN = 12 * 28 * Q_DIM * 4096
FPT = SIX_N + ATTN

# Mini ladder (tiny-configuration demo): read the cache
SWEEP = sw.run_sweep()
BEST = sw.best_lrs(SWEEP)
LADDER = lad.run_ladder()
PTS = LADDER["points"]
HELD = LADDER["held_out"]
LFIT = lad.fit_lnd([p["N"] for p in PTS], [p["D"] for p in PTS], [p["loss"] for p in PTS])
SIZE_COLORS = {"s1": theme.INPUT, "s2": theme.OUTPUT, "s3": theme.PARAM, "s4": theme.ATTN, "s5": theme.HIGHLIGHT}
MUON = json.loads((sw.OUT / "muon.json").read_text()) if (sw.OUT / "muon.json").exists() else []


# Fit and held-out results on the course build machine (the run in README section 5).
# This machine calculates all numbers in the frames. Only these two items are fixed values for
# comparison: on a different machine, the same tiny ladder can give an α that is almost 2× different
# (README section 5).
BUILD_MACHINE = {"alpha": 0.52, "beta": 0.58, "held_inside": 3}


@lru_cache(maxsize=1)
def boot_fits():
    return lad.bootstrap(PTS)


@lru_cache(maxsize=1)
def held_out_ci():
    fits = boot_fits()
    out = []
    for h in HELD:
        boots = [lad.predict(g, h["N"], h["D"]) for g in fits]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append((float(lad.predict(LFIT, h["N"], h["D"])), float(lo), float(hi), h["loss"], h["D"]))
    return out


def lr_law():
    ns = [BEST[s]["N"] for s, _, _ in sw.LADDER]
    lrs = [BEST[s]["lr"] for s, _, _ in sw.LADDER]
    return sw.fit_power_law(ns, lrs)


def tex(s: str, size: float = 30, color: str = theme.FG) -> MathTex:
    return MathTex(s, font_size=size, color=color)


def log_axes(x_range, y_range, x_len, y_len, center, x_ticks=(), y_ticks=(), x_fmt=None, y_fmt=None):
    """Axes + tick labels by hand (a log axis has labels of the form 10^k)."""
    ax = Axes(x_range=x_range, y_range=y_range, x_length=x_len, y_length=y_len,
              axis_config={"color": theme.MUTED, "stroke_width": 2}, tips=False).move_to(center)
    labels = VGroup()
    for x in x_ticks:
        labels.add(tex(x_fmt(x) if x_fmt else f"{x:g}", 20, theme.MUTED).next_to(ax.c2p(x, y_range[0]), DOWN, 0.12))
    for y in y_ticks:
        labels.add(tex(y_fmt(y) if y_fmt else f"{y:g}", 20, theme.MUTED).next_to(ax.c2p(x_range[0], y), LEFT, 0.12))
    return ax, labels


def box(text: str, w: float, h: float, color: str, size: float = 22) -> VGroup:
    r = RoundedRectangle(width=w, height=h, corner_radius=0.12, stroke_color=color, stroke_width=2.5,
                         fill_color=theme.BG, fill_opacity=1)
    t = zh(text, size, theme.FG)
    if t.width > w - 0.2:
        t.scale_to_fit_width(w - 0.2)
    return VGroup(r, t.move_to(r))


class ChapterScene(NarratedScene):
    chapter_label = "第 12 章"
    chapter_title = "Scaling Law 与实验设计"

    def construct(self) -> None:
        # ── S01 Opening ──────────────────────────────────────────────────
        with self.shot("S01"):
            card = self.chapter_card()
            sub = zh("先用小模型算清楚，再花大钱", 32, theme.HIGHLIGHT).next_to(card, DOWN, 0.6)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(card), FadeOut(sub), run_time=self.fit(0.8))

        # ── S02 One chance ───────────────────────────────────────────────
        with self.shot("S02"):
            self.play(*self.set_heading("大模型没法“试试看”"), run_time=self.fit(0.8))
            big = box("主线预训练  $5,000 · 8 卡 10 天", 5.2, 1.6, theme.GRAD, 26).move_to([3.3, 0.6, 0])
            smalls = VGroup(*[box(f"{n}", 1.0, 0.7, theme.INPUT, 20) for n in ["20M", "60M", "150M", "300M"]])
            smalls.arrange(RIGHT, buff=0.25).move_to([-3.6, 0.6, 0])
            cap = zh("阶梯实验：每个几美元到一百多美元", 22, theme.MUTED).next_to(smalls, DOWN, 0.3)
            self.play(FadeIn(big), run_time=self.fit(1))
            self.wait(self.remaining() * 0.25)
            self.play(LaggedStart(*[FadeIn(s, shift=UP * 0.2) for s in smalls], lag_ratio=0.3),
                      FadeIn(cap), run_time=self.fit(2))
            arrow = Arrow(smalls.get_right() + RIGHT * 0.1, big.get_left() + LEFT * 0.1, color=theme.HIGHLIGHT, buff=0)
            lbl = zh("外推", 26, theme.HIGHLIGHT).next_to(arrow, UP, 0.1)
            self.play(GrowArrow(arrow), FadeIn(lbl), run_time=self.fit(1))
            law = zh("scaling law：loss 随算力平滑变化", 28, theme.FG).move_to([0, -1.6, 0])
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(law), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [big, smalls, cap, arrow, lbl, law]], run_time=self.fit(0.6))

        # ── S03 The calculation ─────────────────────────────────────────
        with self.shot("S03"):
            self.play(*self.set_heading("算账：C ≈ 6ND"), run_time=self.fit(0.8))
            eq = tex(r"y = W x", 44).move_to([-3.5, 1.9, 0])
            fwd = VGroup(zh("前向：每个参数一次乘加", 24), tex(r"2N", 34, theme.PARAM)).arrange(RIGHT, buff=0.3)
            bwd1 = VGroup(zh("反向：对输入求梯度", 24), tex(r"2N", 34, theme.GRAD)).arrange(RIGHT, buff=0.3)
            bwd2 = VGroup(zh("反向：对权重求梯度", 24), tex(r"2N", 34, theme.GRAD)).arrange(RIGHT, buff=0.3)
            rows = VGroup(fwd, bwd1, bwd2).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([-3.3, 0.35, 0])
            total = tex(r"6N \text{ FLOPs / token}\quad\Rightarrow\quad C \approx 6ND", 36, theme.HIGHLIGHT)
            total.move_to([-2.2, -1.15, 0])
            self.play(Write(eq), run_time=self.fit(0.8))
            for r in rows:
                self.play(FadeIn(r, shift=RIGHT * 0.2), run_time=self.fit(0.9, reserve=6))
            self.play(Write(total), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.25)
            # FLOPs per token of the main-line model: 6N and the attention term
            L = 6.4
            bar_y = -2.1
            w1 = L * SIX_N / FPT
            b1 = Rectangle(width=w1, height=0.45, fill_color=theme.PARAM, fill_opacity=0.9, stroke_width=0)
            b2 = Rectangle(width=L - w1, height=0.45, fill_color=theme.ATTN, fill_opacity=0.9, stroke_width=0)
            b1.move_to([-3.2 + w1 / 2, bar_y, 0])
            b2.next_to(b1, RIGHT, buff=0)
            t1 = zh(f"6N：{SIX_N / 1e9:.2f} GFLOPs", 20, theme.BG).move_to(b1)
            t2 = zh(f"注意力 {ATTN / FPT:.0%}", 20, theme.BG).move_to(b2)
            cap = zh(f"主线模型每 token（序列 4096）：共 {FPT / 1e9:.2f} GFLOPs", 22, theme.MUTED)
            cap.next_to(VGroup(b1, b2), UP, 0.12)
            self.play(GrowFromEdge(b1, LEFT), FadeIn(t1), FadeIn(cap), run_time=self.fit(1))
            self.play(GrowFromEdge(b2, LEFT), FadeIn(t2), run_time=self.fit(1))
            note = zh("实测（FlopCounterMode）与公式逐位一致", 22, theme.OUTPUT).move_to([3.4, 1.0, 0])
            self.play(FadeIn(note), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [eq, rows, total, b1, b2, t1, t2, cap, note]], run_time=self.fit(0.6))

        # ── S04 The loss is a power law of compute ───────────────────────
        with self.shot("S04"):
            self.play(*self.set_heading("loss 随算力平滑下降"), run_time=self.fit(0.8))
            ax, lab = log_axes([18, 25, 1], [1.9, 3.3, 0.2], 7.2, 4.4, [-2.4, 0.15, 0],
                               x_ticks=range(18, 26), y_ticks=[2.0, 2.4, 2.8, 3.2],
                               x_fmt=lambda x: rf"10^{{{x}}}")
            xl = zh("训练算力 C（FLOPs）", 22, theme.MUTED).next_to(ax, DOWN, 0.5)
            yl = zh("loss", 22, theme.MUTED).next_to(ax, UP, 0.1).align_to(ax, LEFT)
            self.play(Create(ax), FadeIn(lab), FadeIn(xl), FadeIn(yl), run_time=self.fit(1.2))
            curves = VGroup()
            labels = VGroup()
            for N, c in zip([1e8, 3e8, 1e9, 3e9, 1e10], [theme.INPUT, theme.OUTPUT, theme.PARAM, theme.ATTN, theme.GRAD]):
                xs = np.linspace(18, 25, 300)
                pts = [(x, chin.loss(N, 10**x / (6 * N), **FIT)) for x in xs]
                curves.add(polyline_in_axes(ax, pts, color=c, stroke_width=3))
                labels.add(zh(f"{N / 1e9:g}B", 20, c).next_to(ax.c2p(25, chin.loss(N, 1e25 / (6 * N), **FIT)), RIGHT, 0.1))
            self.play(LaggedStart(*[Create(cv) for cv in curves], lag_ratio=0.3), FadeIn(labels), run_time=self.fit(3))
            env = [(x, chin.loss(*chin.compute_optimal(10**x, **FIT), **FIT)) for x in np.linspace(18, 25, 200)]
            env_line = polyline_in_axes(ax, env, color=theme.HIGHLIGHT, stroke_width=6)
            self.wait(self.remaining() * 0.2)
            self.play(Create(env_line), run_time=self.fit(1.5))
            front = zh("最优前沿", 24, theme.HIGHLIGHT).next_to(ax.c2p(21.2, 2.1), DOWN, 0.1)
            formula = tex(r"L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}", 34).move_to([4.3, 1.3, 0])
            parts = VGroup(zh("E：数据本身的不确定性", 20, theme.MUTED),
                           zh("A/N^α：模型太小的代价", 20, theme.PARAM),
                           zh("B/D^β：数据太少的代价", 20, theme.INPUT)).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
            parts.next_to(formula, DOWN, 0.35)
            src = zh("系数：Epoch AI 复现 Chinchilla", 18, theme.MUTED).next_to(parts, DOWN, 0.3)
            self.play(FadeIn(front), Write(formula), run_time=self.fit(1.5))
            self.play(FadeIn(parts), FadeIn(src), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [ax, lab, xl, yl, curves, labels, env_line, front, formula, parts, src]],
                      run_time=self.fit(0.6))

        # ── S05 IsoFLOP parabolas ───────────────────────────────────────
        with self.shot("S05"):
            self.play(*self.set_heading("Chinchilla：每个参数约 20 个 token"), run_time=self.fit(0.8))
            ax, lab = log_axes([7.5, 11, 0.5], [2.0, 3.4, 0.2], 6.6, 4.3, [-2.6, 0.15, 0],
                               x_ticks=[8, 9, 10, 11], y_ticks=[2.2, 2.6, 3.0, 3.4],
                               x_fmt=lambda x: rf"10^{{{x}}}")
            xl = zh("模型参数 N（固定算力，沿曲线换 N 与 D）", 22, theme.MUTED).next_to(ax, DOWN, 0.5)
            self.play(Create(ax), FadeIn(lab), FadeIn(xl), run_time=self.fit(1))
            parabolas, mins, clabels = VGroup(), VGroup(), VGroup()
            colors = [theme.INPUT, theme.OUTPUT, theme.PARAM, theme.ATTN]
            opt_pts = []
            for C, c in zip([1e19, 1e20, 1e21, 1e22], colors):
                xs = np.linspace(7.5, 11, 300)
                pts = [(x, chin.loss(10**x, C / (6 * 10**x), **FIT)) for x in xs]
                parabolas.add(polyline_in_axes(ax, pts, color=c, stroke_width=3))
                n_opt, _ = chin.compute_optimal(C, **FIT)
                y = chin.loss(n_opt, C / (6 * n_opt), **FIT)
                opt_pts.append(ax.c2p(np.log10(n_opt), y))
                mins.add(Dot(opt_pts[-1], color=theme.HIGHLIGHT, radius=0.07))
                exp = int(np.log10(C))
                clabels.add(tex(rf"C=10^{{{exp}}}", 20, c).next_to(opt_pts[-1], LEFT, 0.15))
            self.play(LaggedStart(*[Create(p) for p in parabolas], lag_ratio=0.3), run_time=self.fit(2.5))
            self.play(FadeIn(mins), FadeIn(clabels), run_time=self.fit(1))
            link = Line(opt_pts[0], opt_pts[-1], color=theme.HIGHLIGHT, stroke_width=3)
            self.play(Create(link), run_time=self.fit(0.8))
            n1, d1 = chin.compute_optimal(1.65e21, **FIT)
            info = VGroup(tex(r"D_{\mathrm{opt}} \approx 20\,N", 40, theme.HIGHLIGHT),
                          zh(f"例：C = 1.65e21 → {n1 / 1e9:.1f}B 参数、{d1 / 1e9:.0f}B token", 22),
                          zh("Kaplan 2020：N_opt ∝ C^0.73（模型长得更快）", 20, theme.MUTED),
                          zh("后来发现：差别来自小模型超参没调好、算力口径", 20, theme.MUTED)
                          ).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
            info.scale_to_fit_width(min(info.width, 5.8)).move_to([3.6, 0.4, 0])
            self.wait(self.remaining() * 0.15)
            self.play(FadeIn(info[0]), FadeIn(info[1]), run_time=self.fit(1))
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(info[2]), FadeIn(info[3]), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [ax, lab, xl, parabolas, mins, clabels, link, info]], run_time=self.fit(0.6))

        # ── S06 Overtraining: tokens per parameter ──────────────────────
        with self.shot("S06"):
            self.play(*self.set_heading("为什么小模型要“过训练”"), run_time=self.fit(0.8))
            models = [("Chinchilla 70B", 20, theme.MUTED), ("本课主线（计划）", 580, theme.HIGHLIGHT),
                      ("Puro-2B", 700, theme.INPUT), ("Llama 3 8B", 1875, theme.INPUT),
                      ("MobileLLM-R1-950M", 4426, theme.INPUT), ("Qwen3-0.6B", 60000, theme.INPUT)]
            x0, scale = -2.2, 1.55  # 1.55 units for each factor of 10
            bars, names, vals = VGroup(), VGroup(), VGroup()
            for i, (name, r, c) in enumerate(models):
                y = 2.2 - i * 0.6
                w = scale * np.log10(r)
                bar = Rectangle(width=w, height=0.46, fill_color=c, fill_opacity=0.85, stroke_width=0)
                bar.move_to([x0 + w / 2, y, 0])
                bars.add(bar)
                names.add(zh(name, 22).next_to([x0, y, 0], LEFT, 0.2))
                vals.add(zh(f"{r:,}", 22, c).next_to(bar, RIGHT, 0.15))
            ref = DashedLine([x0 + scale * np.log10(20), 2.55, 0], [x0 + scale * np.log10(20), -1.0, 0],
                             color=theme.GRAD, stroke_width=2)
            ref_l = zh("20：Chinchilla 最优", 20, theme.GRAD).next_to(ref, DOWN, 0.1)
            cap = zh("训练 token ÷ 参数（对数长度）", 22, theme.MUTED).move_to([3.3, -1.25, 0])
            self.play(LaggedStart(*[GrowFromEdge(b, LEFT) for b in bars], lag_ratio=0.2), FadeIn(names),
                      run_time=self.fit(2.5))
            self.play(FadeIn(vals), Create(ref), FadeIn(ref_l), FadeIn(cap), run_time=self.fit(1))
            self.wait(self.remaining() * 0.35)
            why = VGroup(zh("训练：6N × D（一次）", 22, theme.PARAM), zh("推理：2N × 每个生成的 token（上亿次）", 22, theme.OUTPUT))
            why.arrange(RIGHT, buff=0.8).move_to([0, -2.1, 0])
            self.play(FadeIn(why), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [bars, names, vals, ref, ref_l, cap, why]], run_time=self.fit(0.6))

        # ── S07 The price of overtraining ───────────────────────────────
        with self.shot("S07"):
            self.play(*self.set_heading("过训练的价格"), run_time=self.fit(0.8))
            N_ours, D_ours = 689.5e6, 400e9
            C = 6 * N_ours * D_ours
            L_ours = chin.loss(N_ours, D_ours, **FIT)
            n_opt, d_opt = chin.compute_optimal(C, **FIT)
            L_opt = chin.loss(n_opt, d_opt, **FIT)
            C_eq = chin.compute_to_reach(L_ours, FIT)
            # Inference ledger (the same search as ③ in 02_chinchilla.py)
            rows = [VGroup(zh("一生要生成的 token", 22, theme.MUTED), zh("最省总算力的模型", 22, theme.MUTED))]
            Ns = np.logspace(8, 11, 3001)
            for D_inf, lbl in [(0, "0"), (1e12, "1 万亿"), (1e13, "10 万亿"), (1e14, "100 万亿")]:
                best = None
                for N in Ns:
                    gap = L_ours - FIT["E"] - FIT["A"] / N ** FIT["alpha"]
                    if gap <= 0:
                        continue
                    D = (FIT["B"] / gap) ** (1 / FIT["beta"])
                    tot = 6 * N * D + 2 * N * D_inf
                    if best is None or tot < best[0]:
                        best = (tot, N, D)
                rows.append(VGroup(zh(lbl, 24), zh(f"{best[1] / 1e9:.2f}B · {best[2] / best[1]:,.0f} token/参数", 24, theme.OUTPUT)))
            table = VGroup(*[VGroup(a, b) for a, b in rows])
            for i, (a, b) in enumerate(table):
                a.move_to([-5.0, 1.8 - 0.7 * i, 0])
                b.move_to([-1.5, 1.8 - 0.7 * i, 0])
            note = zh(f"同一个目标 loss（= 主线的 {L_ours:.3f}，Epoch 系数）", 20, theme.MUTED).move_to([-3.3, -1.8, 0])
            self.play(FadeIn(table[0]), run_time=self.fit(0.6))
            for r in table[1:]:
                self.play(FadeIn(r, shift=RIGHT * 0.2), run_time=self.fit(0.8, reserve=5))
            self.play(FadeIn(note), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.25)
            price = VGroup(zh("主线 0.69B × 400B token", 24, theme.HIGHLIGHT),
                           zh(f"loss 比同算力最优高 {(L_ours - L_opt) / L_opt:.1%}", 26, theme.GRAD),
                           zh(f"同样的 loss，最优分配只要 {C_eq / C:.0%} 的算力", 24),
                           zh("换来：端侧可用、与 0.8B 标杆同级", 24, theme.OUTPUT)).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
            price.move_to([4.0, 0.4, 0])
            self.play(LaggedStart(*[FadeIn(p) for p in price], lag_ratio=0.5), run_time=self.fit(2.5))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(table), FadeOut(note), FadeOut(price), run_time=self.fit(0.6))

        # ── S08 Tune first (tiny-configuration demo) ─────────────────────
        with self.shot("S08"):
            self.play(*self.set_heading("先调参，再拟合"), run_time=self.fit(0.8))
            badge = self.show_badge()
            ax, lab = log_axes([-3.1, -1.2, 0.5], [3.1, 3.55, 0.1], 5.4, 3.9, [-3.3, 0.2, 0],
                               x_ticks=[-3, -2.5, -2, -1.5], y_ticks=[3.1, 3.3, 3.5],
                               x_fmt=lambda x: rf"10^{{{x:g}}}")
            xl = zh("学习率 η", 22, theme.MUTED).next_to(ax, DOWN, 0.5)
            yl = zh("验证 loss（bit/字节）", 20, theme.MUTED).next_to(ax, UP, 0.1).align_to(ax, LEFT)
            self.play(Create(ax), FadeIn(lab), FadeIn(xl), FadeIn(yl), run_time=self.fit(1))
            ucurves, stars = VGroup(), VGroup()
            for name, _, _ in sw.LADDER:
                rows = sorted((r["lr"], r["val_bpb"]) for r in SWEEP["rows"] if r["size"] == name)
                pts = [(np.log10(lr), v) for lr, v in rows]
                c = SIZE_COLORS[name]
                g = VGroup(polyline_in_axes(ax, pts, color=c, stroke_width=3),
                           *[Dot(ax.c2p(*p), radius=0.05, color=c) for p in pts if 3.1 <= p[1] <= 3.55])
                ucurves.add(g)
                b = BEST[name]
                stars.add(Dot(ax.c2p(np.log10(b["grid_lr"]), b["val_bpb"]), radius=0.1, color=theme.HIGHLIGHT))
            legend = VGroup(*[zh(f"{n}  N={BEST[n]['N']:,}", 18, SIZE_COLORS[n]) for n, _, _ in sw.LADDER])
            legend.arrange(DOWN, aligned_edge=LEFT, buff=0.08).next_to(ax, RIGHT, 0.15).align_to(ax, UP)
            self.play(LaggedStart(*[Create(u) for u in ucurves], lag_ratio=0.35), FadeIn(legend), run_time=self.fit(3))
            self.play(FadeIn(stars), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.2)
            c_, k_ = lr_law()
            law = VGroup(tex(rf"\eta^*(N) = {c_:.1f}\,N^{{-{k_:.2f}}}", 30, theme.HIGHLIGHT),
                         zh("最优学习率随模型变大而变小", 20, theme.MUTED)).arrange(DOWN, buff=0.15)
            law.move_to([4.1, -0.6, 0])
            fixed = BEST[sw.LADDER[0][0]]["grid_lr"]
            worse = [next(r["val_bpb"] for r in SWEEP["rows"] if r["size"] == n and r["lr"] == fixed) - BEST[n]["val_bpb"]
                     for n, _, _ in sw.LADDER]
            trap = zh(f"都用 η={fixed:g}：大模型差 +{worse[2]:.2f}、+{worse[3]:.2f}", 20, theme.GRAD).next_to(law, DOWN, 0.35)
            self.play(FadeIn(law), run_time=self.fit(1))
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(trap), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [ax, lab, xl, yl, ucurves, stars, legend, law, trap]], run_time=self.fit(0.6))

        # ── S09 Mini ladder (tiny-configuration demo) ────────────────────
        with self.shot("S09"):
            self.play(*self.set_heading("迷你阶梯：拟合 L(N, D)"), run_time=self.fit(0.8))
            ax, lab = log_axes([4.7, 6.1, 0.2], [2.7, 3.6, 0.1], 7.0, 4.3, [-2.3, 0.2, 0],
                               x_ticks=[5, 5.5, 6], y_ticks=[2.8, 3.0, 3.2, 3.4, 3.6],
                               x_fmt=lambda x: rf"10^{{{x:g}}}")
            xl = zh("训练 token 数 D（字节）", 22, theme.MUTED).next_to(ax, DOWN, 0.5)
            yl = zh("验证 loss（bit/字节）", 20, theme.MUTED).next_to(ax, UP, 0.1).align_to(ax, LEFT)
            self.play(Create(ax), FadeIn(lab), FadeIn(xl), FadeIn(yl), run_time=self.fit(1))
            dots, fits = VGroup(), VGroup()
            for name, _, _ in sw.LADDER:
                mine = [p for p in PTS if p["size"] == name]
                c = SIZE_COLORS[name]
                N = mine[0]["N"]
                xs = np.linspace(4.75, 6.05, 120)
                fits.add(polyline_in_axes(ax, [(x, lad.predict(LFIT, N, 10**x)) for x in xs], color=c, stroke_width=2.5))
                dots.add(*[Dot(ax.c2p(np.log10(p["D"]), p["loss"]), radius=0.07, color=c) for p in mine])
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.08), run_time=self.fit(2))
            wsd = zh("WSD 分叉：一条主干，多个预算", 22, theme.FG).move_to([4.3, 1.9, 0])
            self.play(FadeIn(wsd), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.25)
            self.play(LaggedStart(*[Create(f) for f in fits], lag_ratio=0.2), run_time=self.fit(2))
            errs = [abs(lad.predict(LFIT, p["N"], p["D"]) - p["loss"]) / p["loss"] for p in PTS]
            a_lo, a_hi = np.percentile([f["alpha"] for f in boot_fits()], [2.5, 97.5])
            info = VGroup(tex(rf"\alpha\approx{LFIT['alpha']:.2f},\ \ \beta\approx{LFIT['beta']:.2f}", 30),
                          zh(f"{len(PTS)} 个点，最大误差 {max(errs):.1%}", 22),
                          zh("两个小模型补了“过训练”预算", 20, theme.MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
            fragile = VGroup(zh(f"课程构建机上：α≈{BUILD_MACHINE['alpha']:.2f}、β≈{BUILD_MACHINE['beta']:.2f}", 20, theme.MUTED),
                             zh(f"bootstrap：α 的 95% 区间 {a_lo:.2f}–{a_hi:.2f}", 20, theme.GRAD),
                             zh("极小阶梯定不住指数，要看留出检验", 20, theme.GRAD)).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
            col = VGroup(info, fragile).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
            if col.width > 5.4:
                col.scale_to_fit_width(5.4)
            col.move_to([4.25, -0.3, 0])
            self.play(FadeIn(info), run_time=self.fit(1))
            self.wait(self.remaining() * 0.25)
            self.play(FadeIn(fragile), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [ax, lab, xl, yl, dots, fits, wsd, info, fragile]], run_time=self.fit(0.6))

        # ── S10 Extrapolation test (tiny-configuration demo) ─────────────
        with self.shot("S10"):
            self.play(*self.set_heading("留出检验：大一号的模型真的训出来"), run_time=self.fit(0.8))
            ci = held_out_ci()
            ax, lab = log_axes([0, 4, 1], [2.85, 3.5, 0.1], 6.2, 4.2, [-2.6, 0.15, 0],
                               y_ticks=[2.9, 3.1, 3.3, 3.5])
            ticks = VGroup(*[zh(f"{D // 1024}K", 20, theme.MUTED).next_to(ax.c2p(i + 1, 2.85), DOWN, 0.12)
                             for i, (_, _, _, _, D) in enumerate(ci)])
            xl = zh(f"留出尺寸 s5（N = {HELD[0]['N']:,}，阶梯最大的 {HELD[0]['N'] / max(p['N'] for p in PTS):.1f} 倍）的 token 预算",
                    20, theme.MUTED).next_to(ax, DOWN, 0.5)
            self.play(Create(ax), FadeIn(lab), FadeIn(ticks), FadeIn(xl), run_time=self.fit(1))
            bars, preds, acts, errs = VGroup(), VGroup(), VGroup(), VGroup()
            for i, (q, lo, hi, act, _) in enumerate(ci):
                x = i + 1
                bars.add(Line(ax.c2p(x, lo), ax.c2p(x, hi), color=theme.MUTED, stroke_width=10, stroke_opacity=0.6))
                preds.add(Dot(ax.c2p(x, q), radius=0.09, color=theme.HIGHLIGHT))
                acts.add(Dot(ax.c2p(x + 0.18, act), radius=0.09, color=theme.OUTPUT))
                errs.add(zh(f"{(q - act) / act:+.2%}", 20, theme.FG).next_to(ax.c2p(x, hi), UP, 0.12))
            self.play(FadeIn(bars), FadeIn(preds), run_time=self.fit(1.2))
            key = VGroup(VGroup(Dot(radius=0.08, color=theme.HIGHLIGHT), zh("拟合外推", 20)).arrange(RIGHT, buff=0.15),
                         VGroup(Line(ORIGIN, UP * 0.4, color=theme.MUTED, stroke_width=10), zh("95% 区间（bootstrap）", 20)).arrange(RIGHT, buff=0.15),
                         VGroup(Dot(radius=0.08, color=theme.OUTPUT), zh("实际训练", 20)).arrange(RIGHT, buff=0.15)
                         ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([4.2, 1.2, 0])
            self.play(FadeIn(key), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.2)
            self.play(LaggedStart(*[FadeIn(a, scale=1.5) for a in acts], lag_ratio=0.4), run_time=self.fit(2))
            self.play(FadeIn(errs), run_time=self.fit(0.8))
            n_out = sum(not (lo <= act <= hi) for _, lo, hi, act, _ in ci)
            verdict = zh("全部落在区间内" if n_out == 0 else f"{n_out} 个点落在区间外", 26,
                         theme.OUTPUT if n_out == 0 else theme.GRAD)
            verdict.move_to([4.2, -0.4, 0])
            lr_note = zh(f"学习率也是外推的：η = {HELD[0]['lr']:.4f}", 20, theme.MUTED).next_to(verdict, DOWN, 0.3)
            self.play(FadeIn(verdict), FadeIn(lr_note), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.3)
            n_in = BUILD_MACHINE["held_inside"]
            other = zh(f"课程构建机上：{n_in} 个都在区间内" if n_in == len(ci) else f"课程构建机上：{n_in} 个在区间内",
                       20, theme.MUTED).next_to(lr_note, DOWN, 0.3)
            self.play(FadeIn(other), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [ax, lab, ticks, xl, bars, preds, acts, errs, key, verdict, lr_note, other]],
                      FadeOut(badge), run_time=self.fit(0.6))

        # ── S11 Delphi ─────────────────────────────────────────────────
        with self.shot("S11"):
            self.play(*self.set_heading("真实世界：Delphi 外推 300 倍"), run_time=self.fit(0.8))
            axis = Line([-6.0, 0.3, 0], [6.0, 0.3, 0], color=theme.MUTED, stroke_width=2)

            def xc(e):  # compute 10^e → x in the frame
                return -6.0 + (e - 18) * 12 / 5.5

            ticks = VGroup(*[tex(rf"10^{{{e}}}", 22, theme.MUTED).next_to([xc(e), 0.3, 0], DOWN, 0.15) for e in range(18, 24)])
            fit_zone = Rectangle(width=xc(20.48) - xc(18.48), height=0.5, fill_color=theme.INPUT, fill_opacity=0.35,
                                 stroke_width=0).move_to([(xc(18.48) + xc(20.48)) / 2, 0.3, 0])
            fit_lbl = zh("拟合：7 个预算 3e18–3e20", 20, theme.INPUT).next_to(fit_zone, UP, 0.15)
            self.play(Create(axis), FadeIn(ticks), run_time=self.fit(1))
            self.play(FadeIn(fit_zone), FadeIn(fit_lbl), run_time=self.fit(1))
            held = VGroup(*[Dot([xc(e), 0.3, 0], radius=0.1, color=theme.HIGHLIGHT) for e in (21, 22, 23)])
            self.play(FadeIn(held), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.2)
            first = VGroup(zh("第一次配方", 22, theme.GRAD).move_to([-4.6, -0.9, 0]),
                           zh("?", 22, theme.MUTED).move_to([xc(21), -0.9, 0]),
                           zh("偏 2.5%", 22, theme.GRAD).move_to([xc(22), -0.9, 0]),
                           zh("发散", 22, theme.GRAD).move_to([xc(23), -0.9, 0]))
            self.play(FadeIn(first), run_time=self.fit(1))
            self.wait(self.remaining() * 0.25)
            fix = zh("修配方：学习率随训练长度下调（×(T₀/T)^0.3）+ AdamH", 20, theme.FG).move_to([0, -1.5, 0])
            second = VGroup(zh("修正后", 22, theme.OUTPUT).move_to([-4.6, -2.1, 0]),
                            *[zh(t, 22, theme.OUTPUT).move_to([xc(e), -2.1, 0]) for t, e in [("+0.5%", 21), ("+0.2%", 22), ("+0.2%", 23)]])
            self.play(FadeIn(fix), run_time=self.fit(0.8))
            self.play(FadeIn(second), run_time=self.fit(1))
            src = zh("来源：openathena.ai/blog/delphi", 18, theme.MUTED).move_to([4.2, 2.0, 0])
            self.play(FadeIn(src), run_time=self.fit(0.5))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [axis, ticks, fit_zone, fit_lbl, held, first, fix, second, src]], run_time=self.fit(0.6))

        # ── S12 Gate 1 ────────────────────────────────────────────────
        with self.shot("S12"):
            self.play(*self.set_heading("闸门 1：外推预测 + 配方验证"), run_time=self.fit(0.8))
            a = box("阶梯实验", 2.2, 0.8, theme.INPUT).move_to([-5.3, 1.5, 0])
            b = box("loss 外推 + 区间", 2.6, 0.8, theme.INPUT).move_to([-2.4, 1.5, 0])
            c = box("软指标 → S 形映射", 2.8, 0.8, theme.INPUT).move_to([0.8, 1.5, 0])
            d = box("基准分数预测", 2.3, 0.8, theme.INPUT).move_to([3.9, 1.5, 0])
            e = box("后训练配方", 2.2, 0.8, theme.PARAM).move_to([-5.3, -0.4, 0])
            f1 = box("(a) 套在阶梯 Base 上", 3.0, 0.7, theme.PARAM, 20).move_to([-2.0, 0.1, 0])
            f2 = box("(b) 套在同尺寸开源 Base 上", 3.4, 0.7, theme.PARAM, 20).move_to([-1.8, -0.9, 0])
            g = box("工具调用得分预测", 2.6, 0.8, theme.PARAM).move_to([1.8, -0.4, 0])
            h = box("能达到硬目标？", 2.4, 0.9, theme.HIGHLIGHT).move_to([5.3, 0.55, 0])
            yes = zh("是：开训", 22, theme.OUTPUT).move_to([5.3, -0.6, 0])
            no = zh("否：改配方或目标", 22, theme.GRAD).move_to([5.3, -1.2, 0])
            arrows = VGroup(Arrow(a.get_right(), b.get_left(), buff=0.05, color=theme.MUTED),
                            Arrow(b.get_right(), c.get_left(), buff=0.05, color=theme.MUTED),
                            Arrow(c.get_right(), d.get_left(), buff=0.05, color=theme.MUTED),
                            Arrow(e.get_right(), f1.get_left(), buff=0.05, color=theme.MUTED),
                            Arrow(e.get_right(), f2.get_left(), buff=0.05, color=theme.MUTED),
                            Arrow(f1.get_right(), g.get_left(), buff=0.05, color=theme.MUTED),
                            Arrow(f2.get_right(), g.get_left(), buff=0.05, color=theme.MUTED),
                            Arrow(d.get_bottom(), h.get_top() + LEFT * 0.4, buff=0.05, color=theme.MUTED),
                            Arrow(g.get_right(), h.get_left(), buff=0.05, color=theme.MUTED))
            self.play(FadeIn(a), FadeIn(b), GrowArrow(arrows[0]), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.12)
            self.play(FadeIn(c), FadeIn(d), GrowArrow(arrows[1]), GrowArrow(arrows[2]), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(e), FadeIn(f1), FadeIn(f2), *[GrowArrow(x) for x in arrows[3:5]], run_time=self.fit(1.2))
            self.play(FadeIn(g), *[GrowArrow(x) for x in arrows[5:7]], run_time=self.fit(1))
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(h), GrowArrow(arrows[7]), GrowArrow(arrows[8]), run_time=self.fit(1))
            self.play(FadeIn(yes), FadeIn(no), run_time=self.fit(0.8))
            tmpl = zh("模板：runs/gate1_report_template.md", 20, theme.MUTED).move_to([0, -2.1, 0])
            self.play(FadeIn(tmpl), run_time=self.fit(0.5))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [a, b, c, d, e, f1, f2, g, h, yes, no, arrows, tmpl]], run_time=self.fit(0.6))

        # ── S13 Decisions for the main-line model ─────────────────────
        with self.shot("S13"):
            self.play(*self.set_heading("主线决策：0.69B × 约 400B token"), run_time=self.fit(0.8))
            n_total, n_matmul, q = plan.shape(1280, 28)
            fpt = plan.flops_per_token(n_matmul, 28, q, 4096)
            rows = [VGroup(zh("MFU", 22, theme.MUTED), zh("$5,000 买到的 token", 22, theme.MUTED))]
            for mfu in (0.4, 0.45, 0.5):
                D = plan.tokens_for(plan.BUDGET, fpt, mfu)
                rows.append(VGroup(zh(f"{mfu:.2f}", 24), zh(f"{D / 1e9:.0f}B（{D / n_total:.0f} token/参数）", 24, theme.OUTPUT)))
            table = VGroup(*rows)
            for i, (a_, b_) in enumerate(table):
                a_.move_to([-5.4, 1.9 - 0.62 * i, 0])
                b_.move_to([-2.6, 1.9 - 0.62 * i, 0])
            cost500 = 500e9 * fpt / (plan.PEAK * 0.4) / 3600 * plan.PRICE
            over = zh(f"原计划 500B（MFU 0.4）要 ${cost500:,.0f}：超线", 22, theme.GRAD)
            wsd = VGroup(zh("WSD 稳定段随时可停：", 20, theme.FG), zh("实测吞吐后再定 400–500B", 20, theme.FG)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            for m_ in (over, wsd):
                m_.scale_to_fit_width(min(m_.width, 5.4))
            over.move_to([-4.0, -0.7, 0]).align_to(table, LEFT)
            wsd.next_to(over, DOWN, 0.3).align_to(table, LEFT)
            self.play(FadeIn(table), run_time=self.fit(1.2))
            self.play(FadeIn(over), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(wsd), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.25)
            muon_line = VGroup(zh("Muon：Kimi K2、GLM-4.5、DeepSeek-V4 采用 → 共识", 20, theme.FG))
            if MUON:
                best_m = {n: min((r for r in MUON if r["size"] == n), key=lambda r: r["val_bpb"]) for n in ("s2", "s3")}
                gain = [best_m[n]["val_bpb"] - BEST[n]["val_bpb"] for n in ("s2", "s3")]
                muon_line.add(zh(f"迷你阶梯：比调好的 AdamW 低 {-gain[0]:.2f}、{-gain[1]:.2f} bit/字节", 20, theme.OUTPUT))
            muon_line.add(zh("→ 第二步阶梯里与 AdamW 正面对比后再定", 20, theme.MUTED))
            muon_line.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
            muon_line.scale_to_fit_width(min(muon_line.width, 6.0)).move_to([3.7, 1.0, 0])
            fp8 = VGroup(zh("FP8：DeepSeek-V3、Llama 4、Nemotron-H 采用 → 共识", 20, theme.FG),
                         zh("zero 尚未实现、未在 GPU 上验证 → 不计入预算", 20, theme.MUTED)
                         ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
            fp8.scale_to_fit_width(min(fp8.width, 6.0)).move_to([3.7, -1.1, 0])
            badge = self.demo_badge()
            self.play(FadeIn(muon_line), FadeIn(badge), run_time=self.fit(1))
            self.play(FadeIn(fp8), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [table, over, wsd, muon_line, fp8, badge]], run_time=self.fit(0.6))

        # ── S14 Summary ───────────────────────────────────────────────
        with self.shot("S14"):
            self.play(*self.set_heading("小结"), run_time=self.fit(0.8))
            steps = VGroup(box("算账：C ≈ 6ND", 2.7, 1.0, theme.PARAM), box("每个尺寸先调参", 2.7, 1.0, theme.INPUT),
                           box("拟合 + 留出检验", 2.7, 1.0, theme.OUTPUT), box("外推，交人决定", 2.7, 1.0, theme.HIGHLIGHT))
            steps.arrange(RIGHT, buff=0.55).move_to([0, 0.6, 0])
            arrs = VGroup(*[Arrow(steps[i].get_right(), steps[i + 1].get_left(), buff=0.05, color=theme.MUTED) for i in range(3)])
            self.play(LaggedStart(*[FadeIn(s) for s in steps], lag_ratio=0.4), *[GrowArrow(a) for a in arrs], run_time=self.fit(3))
            self.wait(self.remaining() * 0.5)
            nxt = zh("下一章：数据与分词器", 30, theme.HIGHLIGHT).move_to([0, -1.4, 0])
            self.play(FadeIn(nxt), run_time=self.fit(1))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(steps), FadeOut(arrs), FadeOut(nxt), *self.set_heading(None), run_time=self.fit(0.8))
