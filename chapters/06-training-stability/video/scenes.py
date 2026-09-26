"""第 6 章视频：让训练稳定 —— 初始化、归一化、残差、AdamW 与学习率调度

画面里的所有数值都由 ../code/ 中的代码真实计算（见 script.md 事实清单）。
训练类实验（06、07、08）较慢，结果缓存到 video/out/cache.json（按代码文件内容的哈希失效）。
渲染：bash chapters/06-training-stability/video/build.sh
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
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
    GrowArrow,
    GrowFromEdge,
    LaggedStart,
    Line,
    MathTex,
    Rectangle,
    RoundedRectangle,
    SurroundingRectangle,
    Transform,
    VGroup,
    Write,
)

from video_kit import theme
from video_kit.scene import NarratedScene, code_block, polyline_in_axes, zh

VIDEO = Path(__file__).resolve().parent
CODE = VIDEO.parent / "code"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sig = _load("signal", "01_signal_propagation.py")
nrm = _load("norm", "02_normalization.py")
res = _load("residual", "03_residual.py")
opt = _load("optimizers", "04_optimizers.py")
sch = _load("sched", "05_lr_schedule.py")

# ── 快速的计算：直接跑 ──────────────────────────────────────────────────────
ACT = {name: sig.layer_stats(std) for name, std in sig.INITS.items()}          # (act, grad)
NORMED = {name: nrm.normed_layer_stats(sig.INITS[name])
          for name in ["std = 1.0", "std = 0.01", "std = 0.02", "Kaiming"]}
RES = {name: res.run(**kw) for name, kw in res.VARIANTS.items()}               # (cos, std, grad, dir)
OPT_RUNS = {"SGD": opt.sgd(200, 0.019), "动量": opt.momentum(200, 0.019), "Adam": opt.adam(200, 0.05)}
WD = {"Adam + L2": opt.weight_decay_demo(False), "AdamW": opt.weight_decay_demo(True)}


def steps_to(hist, thr=1e-3):
    return next((i + 1 for i, (t, _) in enumerate(hist) if opt.loss(t) < thr), None)


# ── 慢的计算（训练）：缓存 ──────────────────────────────────────────────────
CACHE = VIDEO / "out" / "cache.json"


def _code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(CODE.glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _heavy() -> dict:
    ab = _load("ablation", "06_ablation.py")
    ex = _load("experiments", "07_schedule_experiments.py")
    pv = _load("pytorch_version", "08_pytorch_version.py")
    out: dict = {"ladder": [], "loo": {}}
    for name, cfg, lrs in ab.LADDER:
        lr, r, _ = ab.best_of(cfg, lrs)
        out["ladder"].append(dict(name=name, lr=lr, val=r["val_loss"], diverged=r["diverged"]))
    r = ab.run(dict(ab.FULL, residual=False))
    out["loo"]["no_res"] = r["val_loss"]
    out["stress"] = {}
    for key, cfg in [("full", ab.FULL), ("no_warm", dict(ab.FULL, warmup=0)),
                     ("no_norm", dict(ab.FULL, norm=False))]:
        peak, _ = ex.stress(cfg, 0.3)
        out["stress"][key] = peak
    branches, trunk = ex.wsd_branching()
    out["wsd"] = {"trunk": trunk, "branches": {}}
    for b, (before, after, tail) in branches.items():
        cos, _ = ex.cosine_run(b + 100)
        out["wsd"]["branches"][str(b)] = dict(before=before, after=after, tail=tail, cosine=cos)
    bad = tuple(range(300, 305))
    out["clip"] = {}
    for key, clip in [("clip", 1.0), ("noclip", None)]:
        r = ab.run(dict(ab.FULL, norm=False, clip=clip), bad_steps=bad)
        out["clip"][key] = dict(losses=r["losses"][250:420], after=max(r["losses"][305:345]))
    import torch
    out["crosscheck64"] = pv.cross_check(torch.float64)[0]
    return out


def heavy() -> dict:
    key = _code_hash()
    if CACHE.exists():
        data = json.loads(CACHE.read_text(encoding="utf-8"))
        if data.get("key") == key:
            return data["data"]
    data = _heavy()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({"key": key, "data": data}), encoding="utf-8")
    return data


H = heavy()


# ── 画图小工具 ──────────────────────────────────────────────────────────────
def sci(v: float, digits: int = 2) -> str:
    """把 2.27e+31 写成 LaTeX 的 2.27\\times10^{31}。"""
    if v == 0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    m = v / 10**e
    return rf"{m:.{digits}f}\times10^{{{e}}}"


def log_axes(y_min: int, y_max: int, step: int, x_len=7.0, y_len=4.9, center=(-2.7, 0.2),
             n_layers: int = 30) -> tuple[Axes, VGroup]:
    """横轴层号 1..n，纵轴 log10 刻度（标签写成 10^k）。"""
    ax = Axes(x_range=[0, n_layers + 1, 5], y_range=[y_min, y_max, step], x_length=x_len,
              y_length=y_len, tips=False,
              axis_config={"color": theme.MUTED, "stroke_width": 2},
              x_axis_config={"include_numbers": True, "font_size": 18,
                             "numbers_to_include": list(range(5, n_layers + 1, 5))})
    ax.move_to([*center, 0])
    labels = VGroup(*[MathTex(rf"10^{{{k}}}", font_size=20, color=theme.MUTED)
                      .next_to(ax.c2p(0, k), LEFT, 0.12)
                      for k in range(y_min, y_max + 1, step)])
    return ax, labels


def log_bars(ax: Axes, values, color, width=0.16, offset=0.0) -> VGroup:
    """每层一根柱子，从 10^0 画到 log10(value)；超出坐标范围的截断。"""
    y0, y1 = ax.y_range[0], ax.y_range[1]
    bars = VGroup()
    for i, v in enumerate(values, start=1):
        lv = max(y0, min(y1, math.log10(v)))
        bottom, top = ax.c2p(i + offset, 0), ax.c2p(i + offset, lv)
        h = abs(top[1] - bottom[1])
        bar = Rectangle(width=width, height=max(h, 0.005), stroke_width=0, fill_color=color,
                        fill_opacity=0.9)
        bar.move_to([(top[0] + bottom[0]) / 2, (top[1] + bottom[1]) / 2, 0])
        bars.add(bar)
    return bars


def hbar(label: str, value: float, vmax: float, length: float, color, size=22,
         text: str | None = None) -> VGroup:
    lab = zh(label, size, theme.FG)
    bar = Rectangle(width=max(0.02, length * value / vmax), height=0.3, stroke_width=0,
                    fill_color=color, fill_opacity=0.9)
    val = zh(text if text is not None else f"{value:.3f}", size, color)
    return VGroup(lab, bar, val)


class ChapterScene(NarratedScene):
    chapter_label = "第 6 章"
    chapter_title = "让训练稳定"

    def construct(self) -> None:
        # ── S01 片头 ─────────────────────────────────────────────────────
        with self.shot("S01"):
            card = self.chapter_card()
            sub = zh("初始化 · 归一化 · 残差 · AdamW · 学习率调度", 30, theme.HIGHLIGHT)
            sub.next_to(card, DOWN, 0.6)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(card), FadeOut(sub), run_time=self.fit(0.8))

        # ── S02 30 层之后，信号去哪了 ────────────────────────────────────
        with self.shot("S02"):
            ax, ylab = log_axes(-30, 30, 10)
            xl = zh("层号", 20, theme.MUTED).next_to(ax.x_axis, DOWN, 0.35).shift(RIGHT * 3.0)
            yt = zh("激活值的标准差（对数刻度）", 20, theme.MUTED).next_to(ax, UP, 0.12)
            self.play(*self.set_heading("30 层之后，信号去哪了"), Create(ax), FadeIn(ylab),
                      FadeIn(xl), FadeIn(yt), run_time=self.fit(1.5))
            formula = MathTex(r"h_l=\mathrm{ReLU}(h_{l-1}W_l)", font_size=34).move_to([4.3, 2.3, 0])
            spec = zh("30 层，宽 256，不训练", 22, theme.MUTED).next_to(formula, DOWN, 0.25)
            self.play(Write(formula), FadeIn(spec), run_time=self.fit(1.2))
            act1 = ACT["std = 1.0"][0]
            bars1 = log_bars(ax, act1, theme.GRAD)
            self.wait(self.remaining() * 0.12)
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars1], lag_ratio=0.08),
                      run_time=self.fit(4, reserve=6))
            t1 = VGroup(zh("权重 std = 1", 24, theme.GRAD),
                        MathTex(r"\text{L30}:\ " + sci(act1[-1]), font_size=32, color=theme.GRAD)
                        ).arrange(DOWN, buff=0.15).move_to([4.3, 0.6, 0])
            self.play(FadeIn(t1), run_time=self.fit(0.8, reserve=5))
            self.wait(self.remaining() * 0.25)
            act2 = ACT["std = 0.01"][0]
            bars2 = VGroup()
            for b, v in zip(bars1, act2):
                nb = log_bars(ax, [v], theme.INPUT)[0]
                nb.move_to([b.get_center()[0], nb.get_center()[1], 0])
                bars2.add(nb)
            t2 = VGroup(zh("权重 std = 0.01", 24, theme.INPUT),
                        MathTex(r"\text{L30}:\ " + sci(act2[-1]), font_size=32, color=theme.INPUT)
                        ).arrange(DOWN, buff=0.15).move_to([4.3, -1.0, 0])
            self.play(Transform(bars1, bars2), FadeIn(t2), run_time=self.fit(2.5, reserve=0.7))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [ax, ylab, xl, yt, formula, spec, bars1, t1, t2]],
                      run_time=self.fit(0.6))

        # ── S03 连乘的诅咒 ───────────────────────────────────────────────
        with self.shot("S03"):
            self.play(*self.set_heading("深度 = 连乘"), run_time=self.fit(0.6))
            l1 = MathTex(r"1.1^{30}\approx", f"{1.1 ** 30:.0f}", font_size=48).move_to([-3.2, 1.6, 0])
            l2 = MathTex(r"0.9^{30}\approx", f"{0.9 ** 30:.2f}", font_size=48).move_to([3.2, 1.6, 0])
            l1[1].set_color(theme.GRAD)
            l2[1].set_color(theme.INPUT)
            self.play(Write(l1), run_time=self.fit(1.2))
            self.play(Write(l2), run_time=self.fit(1.2))
            factor = MathTex(r"\text{per-layer gain}\approx \sigma_W\sqrt{\text{fan\_in}/2}",
                             font_size=36).move_to([0, 0.3, 0])
            fz = zh("每层放大倍数", 24, theme.MUTED).next_to(factor, UP, 0.2)
            g1 = math.sqrt(sig.WIDTH / 2)
            ex = MathTex(rf"\sigma_W=1:\ {g1:.1f}^{{30}}\approx " + sci(g1 ** 30),
                         font_size=32, color=theme.GRAD).next_to(factor, DOWN, 0.35)
            self.wait(self.remaining() * 0.15)
            self.play(FadeIn(fz), Write(factor), run_time=self.fit(1.5))
            self.play(FadeIn(ex), run_time=self.fit(1))
            boom = VGroup(RoundedRectangle(width=4.6, height=1.0, corner_radius=0.12, color=theme.GRAD),
                          zh("梯度爆炸：一更新就溢出", 24, theme.GRAD)).move_to([-3.2, -1.9, 0])
            boom[1].move_to(boom[0])
            fade = VGroup(RoundedRectangle(width=4.6, height=1.0, corner_radius=0.12, color=theme.INPUT),
                          zh("梯度消失：前面的层学不到", 24, theme.INPUT)).move_to([3.2, -1.9, 0])
            fade[1].move_to(fade[0])
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(boom), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(fade), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [l1, l2, factor, fz, ex, boom, fade]],
                      run_time=self.fit(0.6))

        # ── S04 Kaiming 初始化 ──────────────────────────────────────────
        with self.shot("S04"):
            ax, ylab = log_axes(-6, 2, 2)
            yt = zh("激活值的标准差（对数刻度）", 20, theme.MUTED).next_to(ax, UP, 0.12)
            self.play(*self.set_heading("第一招：按 fan_in 缩放的初始化"), Create(ax), FadeIn(ylab),
                      FadeIn(yt), run_time=self.fit(1.2))
            rule = MathTex(r"\mathrm{Var}(W)=\frac{2}{\text{fan\_in}}", font_size=44)
            rule.move_to([4.3, 2.0, 0])
            name = zh("Kaiming 初始化", 24, theme.OUTPUT).next_to(rule, DOWN, 0.2)
            self.play(Write(rule), FadeIn(name), run_time=self.fit(1.5))
            self.wait(self.remaining() * 0.2)
            kai = log_bars(ax, ACT["Kaiming"][0], theme.OUTPUT, width=0.11, offset=0.1)
            xav = log_bars(ax, ACT["Xavier"][0], theme.PARAM, width=0.11, offset=-0.1)
            self.play(LaggedStart(*[GrowFromEdge(b, UP if math.log10(v) < 0 else DOWN)
                                    for b, v in zip(kai, ACT["Kaiming"][0])], lag_ratio=0.05),
                      run_time=self.fit(2.5, reserve=5))
            k_act, k_grad = ACT["Kaiming"]
            kt = VGroup(zh(f"30 层后激活 {k_act[-1]:.2f}", 22, theme.OUTPUT),
                        zh(f"误差信号传回第 1 层：{k_grad[0] / k_grad[-1]:.0%}", 22, theme.OUTPUT)
                        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([4.3, 0.2, 0])
            self.play(FadeIn(kt), run_time=self.fit(0.8, reserve=4))
            self.wait(self.remaining() * 0.35)
            self.play(LaggedStart(*[GrowFromEdge(b, UP) for b in xav], lag_ratio=0.05),
                      run_time=self.fit(2.5, reserve=2))
            xt = VGroup(zh("Xavier：Var(W) = 1/fan_in", 22, theme.PARAM),
                        zh(f"每层 ×{math.sqrt(0.5):.2f}，30 层后 {ACT['Xavier'][0][-1]:.1e}", 22,
                           theme.PARAM)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            xt.move_to([4.3, -1.4, 0])
            self.play(FadeIn(xt), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [ax, ylab, yt, rule, name, kai, xav, kt, xt]],
                      run_time=self.fit(0.6))

        # ── S05 大模型为什么用 0.02 ─────────────────────────────────────
        with self.shot("S05"):
            ax, ylab = log_axes(-22, 2, 4)
            yt = zh("std = 0.02 的普通网络", 20, theme.MUTED).next_to(ax, UP, 0.12)
            self.play(*self.set_heading("大模型为什么都用 0.02？"), Create(ax), FadeIn(ylab),
                      FadeIn(yt), run_time=self.fit(1))
            b02 = log_bars(ax, ACT["std = 0.02"][0], theme.INPUT)
            self.play(LaggedStart(*[GrowFromEdge(b, UP) for b in b02], lag_ratio=0.05),
                      run_time=self.fit(2.5, reserve=6))
            cfg = VGroup(*[zh(s, 22, theme.FG) for s in [
                "Qwen3-0.6B", "Gemma 3 1B", "SmolLM3-3B", "OLMo 2"]]).arrange(DOWN, aligned_edge=LEFT,
                                                                              buff=0.18)
            val = MathTex(r"\texttt{initializer\_range}=0.02", font_size=30, color=theme.PARAM)
            box = VGroup(val, cfg).arrange(DOWN, buff=0.3).move_to([4.3, 1.2, 0])
            self.play(FadeIn(box), run_time=self.fit(1))
            end = MathTex(r"\text{L30}:\ " + sci(ACT["std = 0.02"][0][-1]), font_size=30,
                          color=theme.INPUT).move_to([4.3, -0.9, 0])
            self.wait(self.remaining() * 0.25)
            self.play(FadeIn(end), run_time=self.fit(0.8))
            key = zh("关键：归一化 + 残差连接", 28, theme.HIGHLIGHT).move_to([4.3, -1.9, 0])
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(key), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [ax, ylab, yt, b02, box, end, key]],
                      run_time=self.fit(0.6))

        # ── S06 LayerNorm 与 RMSNorm ────────────────────────────────────
        with self.shot("S06"):
            ln_f = MathTex(r"\mathrm{LayerNorm}(x)=", r"\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}",
                           r"\gamma", r"+\beta", font_size=38).move_to([0, 2.2, 0])
            rms_f = MathTex(r"\mathrm{RMSNorm}(x)=", r"\frac{x}{\sqrt{\mathrm{mean}(x^2)+\epsilon}}",
                            r"\gamma", font_size=38).move_to([0, 0.95, 0])
            self.play(*self.set_heading("第二招：归一化"), Write(ln_f), run_time=self.fit(2))
            self.wait(self.remaining() * 0.12)
            self.play(Write(rms_f), run_time=self.fit(1.8))
            import torch
            x = torch.tensor([2.0, 4.0, 6.0, 8.0])
            one, zero = torch.ones(4), torch.zeros(4)
            ln = nrm.layer_norm(x, one, zero)
            rn = nrm.rms_norm(x, one)

            def vec(v, color):
                return MathTex(r"[" + ",\ ".join(f"{a:.2f}" for a in v.tolist()) + "]",
                               font_size=30, color=color)

            xin = VGroup(zh("输入 x", 22, theme.INPUT), vec(x, theme.INPUT)).arrange(DOWN, buff=0.15)
            xin.move_to([-4.6, -1.1, 0])
            lo = VGroup(zh(f"LayerNorm：均值 {ln.mean():.2f}，RMS {ln.pow(2).mean().sqrt():.2f}", 22,
                           theme.OUTPUT), vec(ln, theme.OUTPUT)).arrange(DOWN, buff=0.15)
            ro = VGroup(zh(f"RMSNorm：均值 {rn.mean():.2f}，RMS {rn.pow(2).mean().sqrt():.2f}", 22,
                           theme.PARAM), vec(rn, theme.PARAM)).arrange(DOWN, buff=0.15)
            lo.move_to([1.6, -0.75, 0])
            ro.move_to([1.6, -2.0, 0])
            a1 = Arrow(xin.get_right(), lo.get_left(), buff=0.15, color=theme.MUTED, stroke_width=3)
            a2 = Arrow(xin.get_right(), ro.get_left(), buff=0.15, color=theme.MUTED, stroke_width=3)
            self.wait(self.remaining() * 0.12)
            self.play(FadeIn(xin), run_time=self.fit(0.8))
            self.play(GrowArrow(a1), FadeIn(lo), run_time=self.fit(1))
            self.play(GrowArrow(a2), FadeIn(ro), run_time=self.fit(1))
            self.wait(self.remaining() * 0.3)
            drop = zh("去掉：减均值、β", 22, theme.GRAD).next_to(ln_f, RIGHT, 0.3)
            fast = zh("快 7%–64%（原论文）", 22, theme.HIGHLIGHT).next_to(rms_f, RIGHT, 0.3)
            self.play(FadeIn(drop), ln_f[3].animate.set_color(theme.MUTED),
                      run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(fast), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [ln_f, rms_f, xin, lo, ro, a1, a2, drop, fast]],
                      run_time=self.fit(0.6))

        # ── S07 归一化之后 ───────────────────────────────────────────────
        with self.shot("S07"):
            ax, ylab = log_axes(-2, 2, 1)
            yt = zh("每层前加 RMSNorm：激活值的标准差", 20, theme.MUTED).next_to(ax, UP, 0.12)
            self.play(*self.set_heading("归一化之后：任何初始化都稳"), Create(ax), FadeIn(ylab),
                      FadeIn(yt), run_time=self.fit(1.2))
            colors = {"std = 1.0": theme.GRAD, "std = 0.01": theme.INPUT, "std = 0.02": theme.ATTN,
                      "Kaiming": theme.OUTPUT}
            lines, tags = VGroup(), VGroup()
            for name, (act, _) in NORMED.items():
                pts = [(i + 1, math.log10(v)) for i, v in enumerate(act)]
                ln_ = polyline_in_axes(ax, pts, color=colors[name], stroke_width=4)
                lines.add(ln_)
                tags.add(zh(name, 20, colors[name]).next_to(ax.c2p(30, pts[-1][1]), RIGHT, 0.12))
            self.wait(self.remaining() * 0.1)
            for ln_, tg in zip(lines, tags):
                self.play(Create(ln_), FadeIn(tg), run_time=self.fit(1.2, reserve=3))
            ratio = NORMED["std = 1.0"][1][0] / NORMED["std = 1.0"][1][-1]
            note = zh(f"误差信号 第 1 层 / 第 30 层：四种初始化都是 {ratio:.2f}", 22, theme.HIGHLIGHT)
            note.move_to([0.6, -2.35, 0])
            self.play(FadeIn(note), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [ax, ylab, yt, lines, tags, note]],
                      run_time=self.fit(0.6))

        # ── S08 Pre-Norm ─────────────────────────────────────────────────
        with self.shot("S08"):
            def box(text, color, w=1.7):
                r = RoundedRectangle(width=w, height=0.7, corner_radius=0.12, color=color)
                return VGroup(r, zh(text, 22, color).move_to(r))

            def plus():
                return VGroup(RoundedRectangle(width=0.5, height=0.5, corner_radius=0.25,
                                               color=theme.FG), MathTex("+", font_size=32))

            # Post-Norm（左）：x → f → (+) → Norm
            px = -3.6
            p_in = MathTex("h", font_size=34).move_to([px, -1.9, 0])
            p_f = box("f（子层）", theme.PARAM).move_to([px + 0.9, -0.5, 0])
            p_add = plus().move_to([px, 0.6, 0])
            p_norm = box("Norm", theme.ATTN).move_to([px, 1.6, 0])
            p_arrows = VGroup(
                Arrow(p_in.get_top(), p_add.get_bottom(), buff=0.1, color=theme.MUTED),
                Arrow([px, -1.6, 0], p_f.get_bottom(), buff=0.05, color=theme.MUTED),
                Arrow(p_f.get_top(), p_add.get_right(), buff=0.05, color=theme.MUTED),
                Arrow(p_add.get_top(), p_norm.get_bottom(), buff=0.05, color=theme.MUTED),
            )
            p_t = zh("Post-Norm：相加之后再 Norm", 22, theme.MUTED).move_to([px, 2.45, 0])
            # Pre-Norm（右）：x → Norm → f → (+)
            qx = 2.6
            q_in = MathTex("h", font_size=34).move_to([qx, -1.9, 0])
            q_norm = box("Norm", theme.ATTN).move_to([qx + 1.3, -0.9, 0])
            q_f = box("f（子层）", theme.PARAM).move_to([qx + 1.3, 0.15, 0])
            q_add = plus().move_to([qx, 1.2, 0])
            q_arrows = VGroup(
                Arrow(q_in.get_top(), q_add.get_bottom(), buff=0.1, color=theme.OUTPUT,
                      stroke_width=6),
                Arrow([qx, -1.6, 0], q_norm.get_left(), buff=0.05, color=theme.MUTED),
                Arrow(q_norm.get_top(), q_f.get_bottom(), buff=0.05, color=theme.MUTED),
                Arrow(q_f.get_top(), q_add.get_right(), buff=0.05, color=theme.MUTED),
                Arrow(q_add.get_top(), [qx, 2.0, 0], buff=0.05, color=theme.OUTPUT, stroke_width=6),
            )
            q_t = zh("Pre-Norm：Norm 在分支入口", 22, theme.OUTPUT).move_to([qx, 2.45, 0])
            self.play(*self.set_heading("归一化放在哪：Pre-Norm"), FadeIn(p_t),
                      FadeIn(VGroup(p_in, p_f, p_add, p_norm)), run_time=self.fit(1.2))
            self.play(LaggedStart(*[GrowArrow(a) for a in p_arrows], lag_ratio=0.3),
                      run_time=self.fit(1.5))
            self.wait(self.remaining() * 0.25)
            self.play(FadeIn(q_t), FadeIn(VGroup(q_in, q_norm, q_f, q_add)), run_time=self.fit(1))
            self.play(LaggedStart(*[GrowArrow(a) for a in q_arrows], lag_ratio=0.3),
                      run_time=self.fit(1.5))
            eq = MathTex(r"h\leftarrow h+f(\mathrm{Norm}(h))", font_size=34, color=theme.OUTPUT)
            eq.move_to([qx, -2.4, 0])
            main = zh("主干上没有 Norm", 20, theme.OUTPUT).next_to(q_in, LEFT, 0.3).shift(UP * 1.4)
            self.play(Write(eq), FadeIn(main), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [p_in, p_f, p_add, p_norm, p_arrows, p_t, q_in, q_norm,
                                             q_f, q_add, q_arrows, q_t, eq, main]],
                      run_time=self.fit(0.6))

        # ── S09 残差连接 ─────────────────────────────────────────────────
        with self.shot("S09"):
            blocks = VGroup(*[RoundedRectangle(width=0.55, height=0.55, corner_radius=0.08,
                                               color=theme.PARAM) for _ in range(10)])
            blocks.arrange(RIGHT, buff=0.25).move_to([-1.2, 1.9, 0])
            dots = zh("… 30 块", 22, theme.MUTED).next_to(blocks, RIGHT, 0.25)
            links = VGroup(*[Arrow(blocks[i].get_right(), blocks[i + 1].get_left(), buff=0.02,
                                   color=theme.MUTED, stroke_width=2, max_tip_length_to_length_ratio=0.4)
                             for i in range(9)])
            eq = MathTex(r"h\leftarrow h+f(h)", font_size=40).move_to([4.8, 1.9, 0])
            self.play(*self.set_heading("第三招：残差连接"), FadeIn(blocks), FadeIn(links),
                      FadeIn(dots), Write(eq), run_time=self.fit(1.5))
            hw = Arrow(blocks[0].get_left() + DOWN * 0.55, blocks[-1].get_right() + DOWN * 0.55 + RIGHT * 1.3,
                       buff=0, color=theme.OUTPUT, stroke_width=7)
            hw_t = zh("梯度的直通高速路", 22, theme.OUTPUT).next_to(hw, DOWN, 0.1)
            jac = MathTex(r"\frac{\partial h_{l+1}}{\partial h_l}=", r"I", r"+\frac{\partial f}{\partial h_l}",
                          font_size=36).move_to([4.8, 0.75, 0])
            jac[1].set_color(theme.OUTPUT)
            self.wait(self.remaining() * 0.08)
            self.play(Write(jac), run_time=self.fit(1.2))
            self.play(GrowArrow(hw), FadeIn(hw_t), run_time=self.fit(1.2))
            plain = RES["普通堆叠"]
            good = RES["残差（W2 × 1/√(2L)）"]
            bad = RES["残差（W2 不缩放）"]
            rows = [("", "普通堆叠", "残差"),
                    ("30 块后输入相似度", f"{plain[0][-1]:.3f}", f"{good[0][-1]:.3f}"),
                    ("误差信号方向相似度", f"{plain[3]:.3f}", f"{good[3]:.3f}")]
            table = VGroup()
            for r_i, row in enumerate(rows):
                for c_i, cell in enumerate(row):
                    color = theme.MUTED if r_i == 0 or c_i == 0 else (
                        theme.GRAD if c_i == 1 else theme.OUTPUT)
                    table.add(zh(cell if cell else " ", 22, color))
            table.arrange_in_grid(rows=3, cols=3, buff=(0.6, 0.25), col_alignments="lcc")
            table.move_to([-2.2, -1.35, 0])
            self.wait(self.remaining() * 0.12)
            self.play(FadeIn(table[:3]), run_time=self.fit(0.6))
            self.play(FadeIn(table[3:6]), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(table[6:]), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.35)
            scale = VGroup(zh("残差流 std（30 块后）", 20, theme.MUTED),
                           zh(f"W2 不缩放：{bad[1][-1]:.2f}", 22, theme.GRAD),
                           MathTex(rf"W_2\times\frac{{1}}{{\sqrt{{2L}}}}:\ {good[1][-1]:.2f}",
                                   font_size=30, color=theme.OUTPUT)
                           ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([4.6, -1.35, 0])
            self.play(FadeIn(scale), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [blocks, links, dots, eq, hw, hw_t, jac, table, scale]],
                      run_time=self.fit(0.6))

        # ── S10 Adam ─────────────────────────────────────────────────────
        with self.shot("S10"):
            ax = Axes(x_range=[0, 5, 1], y_range=[-3, 1, 1], x_length=5.8, y_length=3.8, tips=False,
                      axis_config={"color": theme.MUTED}).move_to([-3.4, 0.0, 0])
            ylab = VGroup(*[MathTex(rf"10^{{{k}}}", font_size=20, color=theme.MUTED)
                            .next_to(ax.c2p(0, k), LEFT, 0.12) for k in range(-3, 2)])
            xlab = VGroup(*[MathTex(rf"h={h:g}", font_size=22, color=theme.MUTED)
                            .next_to(ax.c2p(i + 1, -3), DOWN, 0.15) for i, h in enumerate(opt.H)])
            yt = zh("第 1 步，每个参数走了多远", 22, theme.MUTED).next_to(ax, UP, 0.15)
            self.play(*self.set_heading("第四招：Adam —— 每个参数自己定步长"), Create(ax),
                      FadeIn(ylab), FadeIn(xlab), FadeIn(yt), run_time=self.fit(1.5))

            def step_bars(steps, color, off):
                bars = VGroup()
                for i, s in enumerate(np.abs(steps)):
                    lv = math.log10(s)
                    top, bot = ax.c2p(i + 1 + off, lv), ax.c2p(i + 1 + off, -3)
                    bars.add(Rectangle(width=0.3, height=top[1] - bot[1], stroke_width=0,
                                       fill_color=color, fill_opacity=0.9)
                             .move_to([top[0], (top[1] + bot[1]) / 2, 0]))
                return bars

            sgd_b = step_bars(OPT_RUNS["SGD"][0][1], theme.GRAD, -0.18)
            adam_b = step_bars(OPT_RUNS["Adam"][0][1], theme.OUTPUT, 0.18)
            legend = VGroup(zh("SGD", 22, theme.GRAD), zh("Adam", 22, theme.OUTPUT)).arrange(
                RIGHT, buff=0.5).next_to(ax, DOWN, 0.55)
            self.wait(self.remaining() * 0.1)
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in sgd_b], lag_ratio=0.2),
                      FadeIn(legend[0]), run_time=self.fit(1.5, reserve=8))
            sgd_v = zh(" / ".join(f"{abs(s):g}" for s in OPT_RUNS["SGD"][0][1]), 20, theme.GRAD)
            sgd_v.move_to([3.6, 2.3, 0])
            self.play(FadeIn(sgd_v), run_time=self.fit(0.6, reserve=7))
            self.wait(self.remaining() * 0.3)
            upd = MathTex(r"\theta\leftarrow\theta-\eta\,\frac{\hat m}{\sqrt{\hat v}+\epsilon}",
                          font_size=40).move_to([3.6, 1.3, 0])
            self.play(Write(upd), run_time=self.fit(1.2, reserve=5))
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in adam_b], lag_ratio=0.2),
                      FadeIn(legend[1]), run_time=self.fit(1.5, reserve=4))
            res_rows = VGroup(zh("损失 < 0.001 用了", 22, theme.MUTED))
            for name, color in [("SGD", theme.GRAD), ("动量", theme.PARAM), ("Adam", theme.OUTPUT)]:
                n = steps_to(OPT_RUNS[name])
                res_rows.add(zh(f"{name}：{n} 步" if n else f"{name}：超过 200 步", 24, color))
            res_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.6, -0.6, 0])
            self.wait(self.remaining() * 0.3)
            self.play(LaggedStart(*[FadeIn(r) for r in res_rows], lag_ratio=0.4),
                      run_time=self.fit(2))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [ax, ylab, xlab, yt, sgd_b, adam_b, legend, sgd_v, upd,
                                             res_rows]], run_time=self.fit(0.6))

        # ── S11 AdamW ────────────────────────────────────────────────────
        with self.shot("S11"):
            l2 = MathTex(r"\text{Adam+L2}:\ g\leftarrow g+\lambda\theta", r"\ \Rightarrow\ \div\sqrt{v}",
                         font_size=34).move_to([0, 2.3, 0])
            aw = MathTex(r"\text{AdamW}:\ \theta\leftarrow\theta-\eta\lambda\theta", r"\ \ (\text{no}\ \sqrt{v})",
                         font_size=34, color=theme.OUTPUT).move_to([0, 1.55, 0])
            self.play(*self.set_heading("AdamW：把权重衰减解耦出来"), Write(l2), run_time=self.fit(1.5))
            self.wait(self.remaining() * 0.3)
            self.play(Write(aw), run_time=self.fit(1.2))
            ax = Axes(x_range=[0, 5, 1], y_range=[0, 1, 0.25], x_length=6.5, y_length=2.6, tips=False,
                      axis_config={"color": theme.MUTED, "include_numbers": False},
                      y_axis_config={"include_numbers": True, "font_size": 18,
                                     "decimal_number_config": {"num_decimal_places": 2}})
            ax.move_to([-1.0, -0.95, 0])
            vals = [(WD["Adam + L2"][0], theme.GRAD, "L2：A 组"), (WD["Adam + L2"][1], theme.GRAD, "L2：B 组"),
                    (WD["AdamW"][0], theme.OUTPUT, "AdamW：A"), (WD["AdamW"][1], theme.OUTPUT, "AdamW：B")]
            bars, labels = VGroup(), VGroup()
            for i, (v, color, lab) in enumerate(vals):
                top, bot = ax.c2p(i + 1, max(v, 0.004)), ax.c2p(i + 1, 0)
                bars.add(Rectangle(width=0.7, height=top[1] - bot[1], stroke_width=0, fill_color=color,
                                   fill_opacity=0.9).move_to([top[0], (top[1] + bot[1]) / 2, 0]))
                labels.add(VGroup(zh(lab, 18, color), zh(f"{v:.3f}", 20, color))
                           .arrange(DOWN, buff=0.05).next_to(ax.c2p(i + 1, 0), DOWN, 0.1))
            theory = (1 - 0.01 * 0.1) ** 3000
            dl = DashedLine(ax.c2p(0.3, theory), ax.c2p(4.7, theory), color=theme.HIGHLIGHT)
            dt = zh(f"理论值 {theory:.3f}", 20, theme.HIGHLIGHT).next_to(dl, RIGHT, 0.1)
            note = VGroup(zh("A 组梯度噪声 0.01", 20, theme.MUTED), zh("B 组梯度噪声 10", 20, theme.MUTED),
                          zh("同一个 λ = 0.1", 20, theme.MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            note.move_to([4.9, -0.2, 0])
            self.wait(self.remaining() * 0.15)
            self.play(Create(ax), FadeIn(note), run_time=self.fit(1))
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars[:2]], lag_ratio=0.3),
                      FadeIn(labels[:2]), run_time=self.fit(1.5, reserve=4))
            self.wait(self.remaining() * 0.25)
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars[2:]], lag_ratio=0.3),
                      FadeIn(labels[2:]), Create(dl), FadeIn(dt), run_time=self.fit(1.5, reserve=2))
            nodecay = zh("RMSNorm 的 γ、偏置：通常不做衰减", 22, theme.PARAM).move_to([4.2, -1.9, 0])
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(nodecay), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [l2, aw, ax, bars, labels, dl, dt, note, nodecay]],
                      run_time=self.fit(0.6))

        # ── S12 学习率调度 ───────────────────────────────────────────────
        total, peak, warm = 1000, 3e-3, 100
        with self.shot("S12"):
            ax = Axes(x_range=[0, 1000, 200], y_range=[0, 3.5, 1], x_length=7.2, y_length=4.2,
                      tips=False, axis_config={"color": theme.MUTED, "include_numbers": True,
                                               "font_size": 18}).move_to([-2.6, 0.1, 0])
            yl = MathTex(r"\eta\ (\times10^{-3})", font_size=26, color=theme.MUTED).next_to(ax, UP, 0.1)
            yl.align_to(ax, LEFT)
            xl = zh("步数", 20, theme.MUTED).next_to(ax.x_axis, RIGHT, 0.1).shift(DOWN * 0.3)
            self.play(*self.set_heading("第五招：学习率调度"), Create(ax), FadeIn(yl), FadeIn(xl),
                      run_time=self.fit(1.2))
            s = np.arange(total)
            cos_pts = [(i, sch.warmup_cosine(i, total, peak, warm) * 1e3) for i in s]
            wsd_pts = [(i, sch.wsd(i, total, peak, warm) * 1e3) for i in s]
            warm_zone = Rectangle(width=ax.c2p(warm, 0)[0] - ax.c2p(0, 0)[0],
                                  height=ax.c2p(0, 3.5)[1] - ax.c2p(0, 0)[1],
                                  stroke_width=0, fill_color=theme.HIGHLIGHT, fill_opacity=0.18)
            warm_zone.move_to(ax.c2p(warm / 2, 1.75))
            wz_t = zh("warmup", 20, theme.HIGHLIGHT).next_to(warm_zone, UP, 0.05).shift(RIGHT * 0.3)
            cos_l = polyline_in_axes(ax, cos_pts, color=theme.PARAM, stroke_width=4)
            wsd_l = polyline_in_axes(ax, wsd_pts, color=theme.OUTPUT, stroke_width=4)
            self.wait(self.remaining() * 0.18)
            self.play(FadeIn(warm_zone), FadeIn(wz_t), run_time=self.fit(1))
            self.wait(self.remaining() * 0.25)
            cos_t = zh("warmup + 余弦", 22, theme.PARAM).move_to(ax.c2p(560, 2.6))
            self.play(Create(cos_l), FadeIn(cos_t), run_time=self.fit(2))
            self.wait(self.remaining() * 0.3)
            wsd_t = zh("WSD：恒定，最后 20% 降", 22, theme.OUTPUT).move_to(ax.c2p(470, 3.3))
            self.play(Create(wsd_l), FadeIn(wsd_t), run_time=self.fit(2))
            who = VGroup(zh("余弦：Llama 2、Llama 3、OLMo 2", 20, theme.PARAM),
                         zh("WSD：MiniCPM、SmolLM3、Kimi K2", 20, theme.OUTPUT),
                         zh("DeepSeek-V3（同形状）", 20, theme.OUTPUT)
                         ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([4.6, 0.2, 0])
            self.play(FadeIn(who), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [ax, yl, xl, warm_zone, wz_t, cos_l, wsd_l, cos_t, wsd_t,
                                             who]], run_time=self.fit(0.6))

        # ── S13 WSD 随时收尾 ─────────────────────────────────────────────
        with self.shot("S13"):
            def smooth(xs, k=25):
                out = []
                for i in range(len(xs)):
                    w = xs[max(0, i - k + 1): i + 1]
                    out.append(sum(w) / len(w))
                return out

            ax = Axes(x_range=[0, 900, 100], y_range=[0.6, 1.4, 0.2], x_length=7.4, y_length=4.2,
                      tips=False, axis_config={"color": theme.MUTED, "include_numbers": True,
                                               "font_size": 18}).move_to([-2.5, 0.1, 0])
            yl = zh("训练损失（滑动平均）", 20, theme.MUTED).next_to(ax, UP, 0.1).align_to(ax, LEFT)
            self.play(*self.set_heading("WSD：随时都能收尾"), Create(ax), FadeIn(yl),
                      run_time=self.fit(1))
            trunk = smooth(H["wsd"]["trunk"])
            tr = polyline_in_axes(ax, list(enumerate(trunk)), color=theme.MUTED, stroke_width=3)
            tr_t = zh("恒定学习率主干", 20, theme.MUTED).move_to(ax.c2p(250, 1.3))
            self.play(Create(tr), FadeIn(tr_t), run_time=self.fit(2.5, reserve=8))
            info = VGroup()
            for b_s, color, y in [("400", theme.OUTPUT, 1.8), ("800", theme.PARAM, 0.2)]:
                b = int(b_s)
                br = H["wsd"]["branches"][b_s]
                full = H["wsd"]["trunk"][:b] + br["tail"]
                sm = smooth(full)[b:]
                seg = polyline_in_axes(ax, [(b + i, v) for i, v in enumerate(sm)], color=color,
                                       stroke_width=5)
                t = VGroup(zh(f"第 {b} 步分叉，衰减 100 步", 20, color),
                           zh(f"验证损失 {br['before']:.3f} → {br['after']:.3f}", 22, color),
                           zh(f"专门跑一次余弦：{br['cosine']:.3f}", 20, theme.MUTED)
                           ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([4.7, y, 0])
                self.wait(self.remaining() * 0.12)
                self.play(Create(seg), FadeIn(t), run_time=self.fit(1.8, reserve=3))
                info.add(seg, t)
            cost = zh("WSD 共 1000 步；两次余弦共 1400 步", 24, theme.HIGHLIGHT).move_to([0, -2.4, 0])
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(cost), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [ax, yl, tr, tr_t, info, cost]], run_time=self.fit(0.6))

        # ── S14 梯度裁剪 ─────────────────────────────────────────────────
        with self.shot("S14"):
            import numpy as _np
            g = [_np.array([3.0, 4.0]), _np.array([12.0])]
            before = sch.clip_by_global_norm(g, 1.0)
            after = math.sqrt(sum(float((x * x).sum()) for x in g))
            origin = np.array([-5.6, -0.4, 0])
            d = np.array([5.0, 12.0, 0]) / 13.0
            long_a = Arrow(origin, origin + d * 3.4, buff=0, color=theme.GRAD, stroke_width=6)
            short_a = Arrow(origin, origin + d * 3.4 * after / before, buff=0, color=theme.OUTPUT,
                            stroke_width=8, max_tip_length_to_length_ratio=0.5)
            la = MathTex(rf"\|g\|={before:.0f}", font_size=32, color=theme.GRAD).next_to(
                long_a.get_end(), RIGHT, 0.15)
            lb = MathTex(rf"\|g\|={after:.0f}", font_size=32, color=theme.OUTPUT).next_to(
                short_a.get_end(), RIGHT, 0.15)
            rule = MathTex(r"g\leftarrow g\cdot\min\!\left(1,\frac{1}{\|g\|}\right)", font_size=34)
            rule.move_to([-4.3, -1.9, 0])
            self.play(*self.set_heading("第六招：梯度裁剪"), GrowArrow(long_a), FadeIn(la),
                      run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.2)
            self.play(Transform(long_a, short_a), Transform(la, lb), Write(rule), run_time=self.fit(1.5))
            ax = Axes(x_range=[250, 420, 50], y_range=[0.6, 2.2, 0.4], x_length=5.6, y_length=3.6,
                      tips=False, axis_config={"color": theme.MUTED, "include_numbers": True,
                                               "font_size": 18}).move_to([3.4, 0.2, 0])
            yl = zh("训练损失（无 RMSNorm）", 20, theme.MUTED).next_to(ax, UP, 0.1)
            bad_zone = Rectangle(width=ax.c2p(305, 0.6)[0] - ax.c2p(300, 0.6)[0],
                                 height=ax.c2p(0, 2.2)[1] - ax.c2p(0, 0.6)[1], stroke_width=0,
                                 fill_color=theme.GRAD, fill_opacity=0.3).move_to(ax.c2p(302.5, 1.4))
            bz = zh("坏数据", 18, theme.GRAD).next_to(bad_zone, UP, 0.05)
            self.wait(self.remaining() * 0.12)
            self.play(Create(ax), FadeIn(yl), FadeIn(bad_zone), FadeIn(bz), run_time=self.fit(1))
            curves = VGroup()
            for key, color in [("noclip", theme.GRAD), ("clip", theme.OUTPUT)]:
                ls = H["clip"][key]["losses"]
                curves.add(polyline_in_axes(ax, [(250 + i, v) for i, v in enumerate(ls)], color=color,
                                            stroke_width=3))
            leg = VGroup(zh(f"不裁剪：之后最高 {H['clip']['noclip']['after']:.3f}", 20, theme.GRAD),
                         zh(f"裁剪：之后最高 {H['clip']['clip']['after']:.3f}", 20, theme.OUTPUT)
                         ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(ax, DOWN, 0.35)
            self.play(Create(curves[0]), FadeIn(leg[0]), run_time=self.fit(1.5, reserve=2))
            self.play(Create(curves[1]), FadeIn(leg[1]), run_time=self.fit(1.5))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [long_a, la, rule, ax, yl, bad_zone, bz, curves, leg]],
                      run_time=self.fit(0.6))

        # ── S15 对比实验 ─────────────────────────────────────────────────
        with self.shot("S15"):
            names = ["A 普通深网络", "B + Kaiming 初始化", "C + 残差连接", "D + RMSNorm",
                     "E SGD → AdamW", "F + warmup + 衰减", "G + 梯度裁剪"]
            vmax, length = 2.4, 4.0
            rows = VGroup()
            for n, r in zip(names, H["ladder"]):
                if r["diverged"]:
                    row = VGroup(zh(n, 20), Rectangle(width=0.02, height=0.3, stroke_width=0),
                                 zh("第 2 步就溢出（NaN）", 20, theme.GRAD))
                else:
                    color = theme.OUTPUT if r["val"] < 1.0 else theme.PARAM
                    row = hbar(n, r["val"], vmax, length, color, size=20)
                rows.add(row)
            rows.add(hbar("全套去掉残差", H["loo"]["no_res"], vmax, length, theme.GRAD, size=20))
            x_lab, x_bar = -5.0, -2.6
            for i, row in enumerate(rows):
                y = 2.3 - i * 0.55 - (0.2 if i == 7 else 0)
                row[0].move_to([x_lab, y, 0], aligned_edge=LEFT)
                row[1].move_to([x_bar, y, 0], aligned_edge=LEFT)
                row[2].next_to(row[1], RIGHT, 0.15)
            head = zh("验证损失（24 层网络，800 步；随机猜 ≈ 2.30）", 20, theme.MUTED).move_to([-2.2, 2.85, 0])
            ref = DashedLine([x_bar + length * math.log(10) / vmax, 2.6, 0],
                             [x_bar + length * math.log(10) / vmax, -1.95, 0], color=theme.MUTED)
            self.play(*self.set_heading(None), FadeIn(head), Create(ref), run_time=self.fit(0.8))
            for i in range(7):
                self.play(FadeIn(rows[i][0]), GrowFromEdge(rows[i][1], LEFT), FadeIn(rows[i][2]),
                          run_time=self.fit(0.9, reserve=9 - i))
                if i in (1, 2):
                    self.wait(self.remaining() * 0.08)
            self.play(FadeIn(rows[7][0]), GrowFromEdge(rows[7][1], LEFT), FadeIn(rows[7][2]),
                      run_time=self.fit(0.9, reserve=5))
            st = H["stress"]
            stress = VGroup(zh("η 放大 100 倍（0.3）", 20, theme.HIGHLIGHT),
                            zh("前 150 步最大损失", 18, theme.MUTED),
                            zh(f"全套：{st['full']:.2f}", 20, theme.OUTPUT),
                            zh(f"去掉 warmup：{st['no_warm']:.2f}", 20, theme.PARAM),
                            MathTex(r"\text{no RMSNorm}:\ " + sci(st["no_norm"], 1), font_size=28,
                                    color=theme.GRAD)).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
            frame = SurroundingRectangle(stress, color=theme.HIGHLIGHT, buff=0.2, corner_radius=0.1)
            VGroup(stress, frame).move_to([5.0, -0.3, 0])
            nz = zh("去掉 RMSNorm", 20, theme.GRAD).move_to(stress[4]).align_to(stress[4], LEFT)
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(frame), FadeIn(stress[:4]), FadeIn(nz), run_time=self.fit(1))
            self.play(Transform(nz, stress[4]), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [head, ref, rows, stress[:4], nz, frame]],
                      run_time=self.fit(0.6))

        # ── S16 从极简到生产级 ───────────────────────────────────────────
        with self.shot("S16"):
            left = code_block("""
rms_norm(h, gamma)
if p.dim() >= 2: p *= 1 - lr*wd
p -= lr * m_hat / (sqrt(v_hat) + eps)
lr = warmup_cosine(step, ...)
clip_by_global_norm(grads, 1.0)
""", 19)
            right = code_block("""
nn.RMSNorm(width, eps=1e-6)
AdamW([{decay}, {no_decay}])
    betas=(0.9, 0.95)
LambdaLR(opt, lr_lambda)
clip_grad_norm_(params, 1.0)
""", 19)
            left.move_to([-3.5, 0.4, 0])
            right.move_to([3.5, 0.4, 0])
            lt = zh("手写（本章 code/）", 24, theme.MUTED).next_to(left, UP, 0.4)
            rt = zh("PyTorch 标准写法", 24, theme.HIGHLIGHT).next_to(right, UP, 0.4)
            self.play(*self.set_heading("从极简到生产级"), FadeIn(lt), FadeIn(left),
                      run_time=self.fit(1.2))
            self.play(FadeIn(rt), FadeIn(right), run_time=self.fit(1.2))
            pairs = [(0, 0), (1, 1), (2, 1), (3, 3), (4, 4)]
            links = VGroup(*[Line(left[a].get_right(), right[b].get_left(), color=theme.MUTED,
                                  stroke_width=1.5, buff=0.15) for a, b in pairs])
            self.play(LaggedStart(*[Create(ln_) for ln_ in links], lag_ratio=0.3),
                      run_time=self.fit(2.5))
            ok = MathTex(r"\text{float64}:\ \max|\Delta\,\text{loss}|=" + sci(H["crosscheck64"], 1),
                         font_size=34, color=theme.OUTPUT).move_to([0, -1.9, 0])
            zt = zh("逐步对拍 200 步", 22, theme.OUTPUT).next_to(ok, UP, 0.15)
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(zt), Write(ok), run_time=self.fit(1.2))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [left, right, lt, rt, links, ok, zt]],
                      run_time=self.fit(0.6))

        # ── S17 第一部分结束 ─────────────────────────────────────────────
        with self.shot("S17"):
            items = [("初始化", "开局不爆不缩", theme.PARAM), ("RMSNorm", "每层自动对齐", theme.ATTN),
                     ("残差连接", "梯度高速路", theme.OUTPUT), ("AdamW", "按参数定步长", theme.PARAM),
                     ("warmup + 衰减", "管住学习率", theme.INPUT), ("梯度裁剪", "兜住意外", theme.GRAD)]
            cards = VGroup()
            for title, desc, color in items:
                r = RoundedRectangle(width=3.4, height=1.15, corner_radius=0.15, color=color)
                t = VGroup(zh(title, 26, color), zh(desc, 20, theme.FG)).arrange(DOWN, buff=0.12)
                cards.add(VGroup(r, t.move_to(r)))
            cards.arrange_in_grid(rows=2, cols=3, buff=(0.45, 0.35)).move_to([0, 1.3, 0])
            self.play(*self.set_heading("六招回顾"),
                      LaggedStart(*[FadeIn(c) for c in cards], lag_ratio=0.35),
                      run_time=self.fit(5, reserve=6))
            part = zh("第一部分完成 → 可以开始 Stanford CS336", 30, theme.HIGHLIGHT).move_to([0, -0.9, 0])
            self.wait(self.remaining() * 0.25)
            self.play(FadeIn(part, shift=UP * 0.2), run_time=self.fit(1))
            nxt = zh("下一章：语言建模与分词", 32, theme.FG).move_to([0, -2.0, 0])
            nbox = RoundedRectangle(width=nxt.width + 0.6, height=nxt.height + 0.35, corner_radius=0.1,
                                    color=theme.HIGHLIGHT).move_to(nxt)
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(nxt), Create(nbox), run_time=self.fit(1))
            self.wait(self.remaining() - 1.0)
            self.play(*[FadeOut(m) for m in [cards, part, nxt, nbox]], *self.set_heading(None),
                      run_time=self.fit(1.0))
