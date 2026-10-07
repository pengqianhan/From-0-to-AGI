"""Chapter 3 video: nonlinearity and neural networks — build a curve from line segments.

The code in ../code/ calculates all numbers in the frames (see the fact list in script.md).
Render: bash chapters/03-neural-network/video/build.sh
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Axes,
    ChangeDecimalToValue,
    Circle,
    Create,
    DashedLine,
    DecimalNumber,
    Dot,
    FadeIn,
    FadeOut,
    GrowArrow,
    LaggedStart,
    Line,
    MathTex,
    Rectangle,
    RoundedRectangle,
    SurroundingRectangle,
    Text,
    Transform,
    ValueTracker,
    VGroup,
    VMobject,
    Write,
    always_redraw,
)

from video_kit import theme
from video_kit.scene import NarratedScene, polyline_in_axes, zh

CODE = Path(__file__).resolve().parent.parent / "code"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lin = _load("linear_is_not_enough", "01_linear_is_not_enough.py")
actm = _load("activations", "02_activations.py")
mlp = _load("mlp_numpy", "03_mlp_numpy.py")
ptv = _load("pytorch_version", "04_pytorch_version.py")

# ── Real calculations ───────────────────────────────────────────────────────
X, Y = mlp.make_data()
LINE_A, LINE_B, LINE_MSE = lin.best_line(X, Y)
VAR_Y = float(np.var(Y))
STACKS = lin.stack_demo(X)                              # [(name, params, W, b, max diff), ...]
# The network names in the code are now English. The video still shows the Chinese names.
STACK_ZH = {"2 layers 1→8→1": "两层 1→8→1", "3 layers 1→8→8→1": "三层 1→8→8→1"}
_, LIN_LOSSES, _ = mlp.train(X, Y, 8, act="linear")     # loss of 2 linear layers, no activation
SNAP_STEPS = (0, 1000, 5000, mlp.STEPS)
P64, L64, SNAP64 = mlp.train(X, Y, 64, snapshot_steps=SNAP_STEPS)
P8, L8, _ = mlp.train(X, Y, 8)
P2, L2, _ = mlp.train(X, Y, 2)
PIECES8, KINKS8 = mlp.pieces(P8, X)
_p0 = mlp.init_params(8, seed=0)
_ga, _gn = mlp.gradients(_p0, X, Y)[1], mlp.numerical_gradients(_p0, X, Y)
GRAD_ERR = max(float(np.max(np.abs(_ga[k] - _gn[k]))) for k in _p0)
SEED_MED = {h: float(np.median([mlp.train(X, Y, h, seed=s)[1][-1] for s in range(5)]))
            for h in (2, 8, 64)}
PT_TORCH, PT_NUMPY = ptv.run_match(verbose=False)
PT_DEFAULT, _ = ptv.run_default(verbose=False)

XS = np.linspace(-3, 3, 601).reshape(-1, 1)             # dense grid for the curves
MONO = "Noto Sans Mono"


def mono(text: str, size: float = 24, color: str = theme.FG) -> Text:
    return Text(text, font=MONO, font_size=size, color=color)


def sci(v: float) -> str:
    """1.1e-10 → 1.1 × 10⁻¹⁰ (easier to read on the screen)."""
    m, e = f"{v:.1e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{m} × 10{str(int(e)).translate(sup)}"


def data_axes(center, x_len=6.2, y_len=4.2, y_range=(-1.5, 2.5, 1), numbers=True) -> Axes:
    return Axes(
        x_range=[-3.2, 3.2, 1], y_range=list(y_range), x_length=x_len, y_length=y_len,
        axis_config={"color": theme.MUTED, "include_numbers": numbers, "font_size": 18},
        tips=False,
    ).move_to(center)


def data_dots(ax: Axes, radius=0.04, opacity=1.0) -> VGroup:
    return VGroup(*[Dot(ax.c2p(x, y), radius=radius, color=theme.INPUT, fill_opacity=opacity)
                    for x, y in zip(X[:, 0], Y[:, 0], strict=False)])


def curve(ax: Axes, xs, ys, color=theme.OUTPUT, width=4) -> VGroup:
    return polyline_in_axes(ax, list(zip(np.ravel(xs), np.ravel(ys), strict=False)), color=color,
                            stroke_width=width)


def net_curve(ax: Axes, params: dict, color=theme.OUTPUT, width=4) -> VMobject:
    """Network output on a dense grid as one polyline (fixed point count, for Transform)."""
    y_hat, _ = mlp.forward(params, XS)
    lo, hi = ax.y_range[0], ax.y_range[1]
    pts = [ax.c2p(x, float(np.clip(y, lo, hi))) for x, y in zip(XS[:, 0], y_hat[:, 0], strict=False)]
    m = VMobject(color=color, stroke_width=width)
    m.set_points_as_corners(pts)
    return m


class ChapterScene(NarratedScene):
    chapter_label = "第 3 章"
    chapter_title = "非线性与神经网络"

    def construct(self) -> None:
        # ── S01 Opening ──────────────────────────────────────────────────
        with self.shot("S01"):
            card = self.chapter_card()
            sub = zh("用折线拼出曲线", 32, theme.HIGHLIGHT).next_to(card, DOWN, 0.6)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(card), FadeOut(sub), run_time=self.fit(0.8))

        # ── S02 A straight line cannot fit a curve ───────────────────────
        with self.shot("S02"):
            ax = data_axes([-3.4, 0.2, 0])
            xl = MathTex("x", color=theme.MUTED, font_size=28).next_to(ax.x_axis, RIGHT, 0.1)
            dots = data_dots(ax)
            self.play(*self.set_heading("直线拟合不了曲线"), Create(ax), FadeIn(xl),
                      run_time=self.fit(1.2))
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.03),
                      run_time=self.fit(3))
            f = MathTex(r"y=\sin(2x)", color=theme.INPUT, font_size=40).move_to([3.5, 2.0, 0])
            self.play(Write(f), run_time=self.fit(1))
            self.wait(self.remaining() * 0.2)
            line = curve(ax, [-3, 3], [LINE_A * -3 + LINE_B, LINE_A * 3 + LINE_B])
            resid = VGroup(*[Line(ax.c2p(x, y), ax.c2p(x, LINE_A * x + LINE_B), color=theme.GRAD,
                                  stroke_width=1.5) for x, y in zip(X[::2, 0], Y[::2, 0], strict=False)])
            self.play(Create(line), run_time=self.fit(1))
            self.play(LaggedStart(*[Create(r) for r in resid], lag_ratio=0.02),
                      run_time=self.fit(1.5))
            best = VGroup(
                zh("最好的直线", 26, theme.OUTPUT),
                MathTex(rf"y={LINE_A:.3f}\,x{LINE_B:+.3f}", font_size=34),
                zh(f"MSE = {LINE_MSE:.4f}", 30, theme.GRAD),
                zh(f"（直接猜平均值：{VAR_Y:.4f}）", 22, theme.MUTED),
            ).arrange(DOWN, buff=0.25).move_to([3.5, 0.1, 0])
            self.play(FadeIn(best), run_time=self.fit(1))
            self.wait(self.remaining() * 0.45)
            verdict = zh("不是没训练好，是直线弯不过来", 26, theme.HIGHLIGHT).move_to([3.5, -1.9, 0])
            self.play(FadeIn(verdict), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [ax, xl, dots, f, line, resid, best, verdict]],
                      run_time=self.fit(0.6))

        # ── S03 Stack more linear layers? ────────────────────────────────
        with self.shot("S03"):
            self.play(*self.set_heading("多叠几层线性层？"), run_time=self.fit(0.6))
            e1 = MathTex(r"(XW_1+b_1)\,W_2+b_2", font_size=48).move_to([0, 2.0, 0])
            e2 = MathTex(r"=\;X\,", r"(W_1W_2)", r"\;+\;", r"(b_1W_2+b_2)",
                         font_size=48).move_to([0, 0.7, 0])
            e3 = MathTex(r"=\;X\,", r"W", r"\;+\;", r"b", font_size=48).move_to([0, -1.2, 0])
            for e in (e2, e3):
                e[1].set_color(theme.PARAM)
                e[3].set_color(theme.PARAM)
            self.play(Write(e1), run_time=self.fit(1.5))
            self.wait(self.remaining() * 0.25)
            self.play(Write(e2), run_time=self.fit(1.5))
            b1 = SurroundingRectangle(e2[1], color=theme.PARAM, buff=0.08)
            b2 = SurroundingRectangle(e2[3], color=theme.PARAM, buff=0.08)
            t1 = zh("还是一个矩阵", 22, theme.PARAM).next_to(b1, DOWN, 0.15)
            t2 = zh("还是一个向量", 22, theme.PARAM).next_to(b2, DOWN, 0.15)
            self.play(Create(b1), Create(b2), FadeIn(t1), FadeIn(t2), run_time=self.fit(1))
            self.wait(self.remaining() * 0.35)
            self.play(Write(e3), run_time=self.fit(1))
            concl = zh("两层线性 = 一层线性；叠多少层都一样", 30, theme.HIGHLIGHT).move_to([0, -2.2, 0])
            self.play(FadeIn(concl), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [e1, e2, e3, b1, b2, t1, t2, concl]],
                      run_time=self.fit(0.6))

        # ── S04 Check with numbers ───────────────────────────────────────
        with self.shot("S04"):
            lax = Axes(x_range=[0, 50, 10], y_range=[0, 2.2, 0.5], x_length=5.8, y_length=3.8,
                       axis_config={"color": theme.MUTED, "include_numbers": True, "font_size": 18},
                       tips=False).move_to([-3.5, 0.1, 0])
            lxl = zh("步数", 20, theme.MUTED).next_to(lax.x_axis, DOWN, 0.4)
            lyl = zh("损失", 20, theme.MUTED).next_to(lax.y_axis, UP, 0.1)
            self.play(*self.set_heading("用数字验证：线性叠起来还是线性"), run_time=self.fit(0.6))
            rows = VGroup()
            for name, n_p, _, _, diff in STACKS:
                rows.add(VGroup(zh(f"{STACK_ZH[name]}：{n_p} 个参数", 26, theme.FG),
                                zh(f"逐层算 vs 合并算：差 {sci(diff)}", 22, theme.MUTED)
                                ).arrange(DOWN, aligned_edge=LEFT, buff=0.12))
            rows.arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to([3.4, 1.3, 0])
            self.play(FadeIn(rows[0]), run_time=self.fit(1))
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(rows[1]), run_time=self.fit(1))
            self.wait(self.remaining() * 0.1)
            self.play(Create(lax), FadeIn(lxl), FadeIn(lyl), run_time=self.fit(1))
            ref = DashedLine(lax.c2p(0, LINE_MSE), lax.c2p(50, LINE_MSE), color=theme.HIGHLIGHT)
            ref_l = zh(f"最好的直线 {LINE_MSE:.4f}", 20, theme.HIGHLIGHT).next_to(
                lax.c2p(50, LINE_MSE), UP, 0.12).shift(LEFT * 1.0)
            loss_c = curve(lax, np.arange(51), LIN_LOSSES[:51], color=theme.GRAD, width=3)
            self.play(Create(ref), FadeIn(ref_l), run_time=self.fit(0.8))
            self.play(Create(loss_c), run_time=self.fit(2.5))
            trained = VGroup(zh("两层线性（宽 8）梯度下降", 22, theme.GRAD),
                             zh(f"{mlp.STEPS} 步后损失：{LIN_LOSSES[-1]:.4f}", 24, theme.GRAD)
                             ).arrange(DOWN, buff=0.12).move_to([3.4, -0.8, 0])
            stuck = zh("参数再多，也逃不出直线", 28, theme.HIGHLIGHT).move_to([3.4, -1.8, 0])
            self.play(FadeIn(trained), run_time=self.fit(0.8))
            self.play(FadeIn(stuck), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [lax, lxl, lyl, rows, ref, ref_l, loss_c, trained,
                                             stuck]], run_time=self.fit(0.6))

        # ── S05 ReLU ─────────────────────────────────────────────────────
        with self.shot("S05"):
            rax = Axes(x_range=[-3, 3, 1], y_range=[-1, 3, 1], x_length=5.6, y_length=3.8,
                       axis_config={"color": theme.MUTED, "include_numbers": True, "font_size": 18},
                       tips=False).move_to([-3.4, 0.2, 0])
            zl = MathTex("z", color=theme.MUTED, font_size=28).next_to(rax.x_axis, RIGHT, 0.1)
            self.play(*self.set_heading("激活函数：在线性层之间加一道弯"), Create(rax), FadeIn(zl),
                      run_time=self.fit(1))
            neg = curve(rax, [-3, 0], [0, 0], color=theme.GRAD, width=6)
            pos = curve(rax, [0, 3], [0, 3], color=theme.OUTPUT, width=6)
            formula = MathTex(r"\mathrm{ReLU}(z)=\max(0,\,z)", font_size=46).move_to([3.5, 1.8, 0])
            self.wait(self.remaining() * 0.25)
            self.play(Write(formula), run_time=self.fit(1.2))
            self.play(Create(neg), run_time=self.fit(0.8))
            self.play(Create(pos), run_time=self.fit(0.8))
            notes = VGroup(zh("z < 0：变成 0", 26, theme.GRAD), zh("z ≥ 0：原样保留", 26, theme.OUTPUT),
                           zh("逐个元素作用，各管各的", 24, theme.MUTED)
                           ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.5, 0.2, 0])
            self.play(FadeIn(notes[0]), FadeIn(notes[1]), run_time=self.fit(0.8))
            kink = Dot(rax.c2p(0, 0), color=theme.HIGHLIGHT, radius=0.1)
            kl = zh("折角", 24, theme.HIGHLIGHT).next_to(kink, DOWN + RIGHT, 0.1)
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(kink, scale=2), FadeIn(kl), run_time=self.fit(0.8))
            no_merge = zh("有了折角，线性层就合并不了", 26, theme.HIGHLIGHT).move_to([3.5, -1.3, 0])
            self.play(FadeIn(no_merge), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(notes[2]), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [rax, zl, neg, pos, formula, notes, kink, kl,
                                             no_merge]], run_time=self.fit(0.6))

        # ── S06 Other activation functions ───────────────────────────────
        with self.shot("S06"):
            self.play(*self.set_heading("激活函数家族"), run_time=self.fit(0.6))
            zs = np.linspace(-3, 3, 301)
            specs = [("sigmoid", actm.sigmoid, "早期", theme.MUTED),
                     ("tanh", actm.tanh, "早期", theme.MUTED),
                     ("SiLU", actm.silu, "现代", theme.OUTPUT),
                     ("GELU", actm.gelu, "现代", theme.OUTPUT)]
            panels = VGroup()
            for i, (name, fn, era, color) in enumerate(specs):
                pax = Axes(x_range=[-3, 3, 1], y_range=[-1.2, 3, 1], x_length=2.9, y_length=2.6,
                           axis_config={"color": theme.MUTED, "stroke_width": 1}, tips=False)
                pax.move_to([-5.1 + 3.4 * i, 0.5, 0])
                title = mono(name, 26, theme.FG).next_to(pax, UP, 0.15)
                tag = zh(era, 22, color).next_to(pax, DOWN, 0.15)
                ref = curve(pax, zs, actm.relu(zs), color=theme.MUTED, width=1.5)
                ref.set_stroke(opacity=0.5)
                cur = curve(pax, zs, fn(zs), color=color if era == "现代" else theme.INPUT, width=4)
                panels.add(VGroup(pax, title, tag, ref, cur))
            self.play(FadeIn(panels[0][:3]), FadeIn(panels[1][:3]), run_time=self.fit(0.8))
            self.play(Create(panels[0][4]), Create(panels[1][4]), run_time=self.fit(1.2))
            s_note = zh("S 形：两头太平，梯度接近 0", 22, theme.MUTED).move_to([-3.4, -1.6, 0])
            self.play(FadeIn(s_note), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(panels[2][:4]), FadeIn(panels[3][:4]), run_time=self.fit(0.8))
            self.play(Create(panels[2][4]), Create(panels[3][4]), run_time=self.fit(1.2))
            m_note = zh("ReLU 的平滑版（灰线是 ReLU）", 22, theme.OUTPUT).move_to([3.4, -1.6, 0])
            self.play(FadeIn(m_note), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.35)
            swiglu = zh("大模型前馈层的 SwiGLU = 门控 + SiLU → 第 9 章", 26, theme.HIGHLIGHT)
            swiglu.move_to([0, -2.3, 0])
            self.play(FadeIn(swiglu), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(panels), FadeOut(s_note), FadeOut(m_note), FadeOut(swiglu),
                      run_time=self.fit(0.6))

        # ── S07 The structure of a two-layer MLP ─────────────────────────
        with self.shot("S07"):
            self.play(*self.set_heading("两层 MLP：线性 → ReLU → 线性"), run_time=self.fit(0.6))
            inp = VGroup(Circle(0.32, color=theme.INPUT, fill_opacity=0.2),
                         MathTex("x", color=theme.INPUT, font_size=34)).move_to([-5.8, 0.5, 0])
            out = VGroup(Circle(0.32, color=theme.OUTPUT, fill_opacity=0.2),
                         MathTex(r"\hat{y}", color=theme.OUTPUT, font_size=34)).move_to([-1.0, 0.5, 0])
            hidden = VGroup()
            ys_h = [2.2, 1.35, 0.5, -0.35, -1.2]
            for yy in ys_h:
                c = Circle(0.3, color=theme.FG)
                glyph = VMobject(color=theme.HIGHLIGHT, stroke_width=2.5)
                glyph.set_points_as_corners([[-0.15, -0.08, 0], [0, -0.08, 0], [0.14, 0.1, 0]])
                hidden.add(VGroup(c, glyph).move_to([-3.4, yy, 0]))
            dots_v = MathTex(r"\vdots", color=theme.MUTED, font_size=30).move_to([-3.4, -1.8, 0])
            edges1 = VGroup(*[Line(inp.get_right(), h.get_left(), color=theme.PARAM,
                                   stroke_width=1.5, stroke_opacity=0.7) for h in hidden])
            edges2 = VGroup(*[Line(h.get_right(), out.get_left(), color=theme.PARAM,
                                   stroke_width=1.5, stroke_opacity=0.7) for h in hidden])
            w1l = MathTex(r"W_1,b_1", color=theme.PARAM, font_size=28).move_to([-4.8, 2.6, 0])
            w2l = MathTex(r"W_2,b_2", color=theme.PARAM, font_size=28).move_to([-2.0, 2.6, 0])
            hl = zh("H 个隐藏单元（宽度 H）", 22, theme.FG).move_to([-3.4, -2.35, 0])
            self.play(FadeIn(inp), run_time=self.fit(0.6))
            self.play(Create(edges1), FadeIn(hidden), FadeIn(dots_v), FadeIn(w1l),
                      run_time=self.fit(1.2))
            self.play(Create(edges2), FadeIn(out), FadeIn(w2l), run_time=self.fit(1.0))
            eqs = VGroup(
                MathTex(r"Z=XW_1+b_1", font_size=38),
                MathTex(r"A=\mathrm{ReLU}(Z)", font_size=38),
                MathTex(r"\hat{Y}=AW_2+b_2", font_size=38),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([2.3, 1.1, 0])
            shapes = VGroup(
                MathTex(r"(N,1)\to(N,H)", color=theme.MUTED, font_size=30),
                MathTex(r"(N,H)", color=theme.MUTED, font_size=30),
                MathTex(r"(N,H)\to(N,1)", color=theme.MUTED, font_size=30),
            )
            for s, e in zip(shapes, eqs, strict=False):
                s.next_to(e, RIGHT, 0.4)
                s.align_to([4.6, 0, 0], LEFT)
            self.wait(self.remaining() * 0.15)
            for e, s in zip(eqs, shapes, strict=False):
                self.play(Write(e), FadeIn(s), run_time=self.fit(1.0, reserve=3))
                self.wait(self.remaining() * 0.12)
            self.play(FadeIn(hl), run_time=self.fit(0.6))
            npar = zh("参数量：3H + 1", 26, theme.PARAM).move_to([3.4, -1.3, 0])
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(npar), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [inp, out, hidden, dots_v, edges1, edges2, w1l, w2l, hl,
                                             eqs, shapes, npar]], run_time=self.fit(0.6))

        # ── S08 One hidden unit = one kink ────────────────────────────────
        with self.shot("S08"):
            hax = Axes(x_range=[-3, 3, 1], y_range=[-2.5, 2.5, 1], x_length=5.8, y_length=4.2,
                       axis_config={"color": theme.MUTED, "include_numbers": True, "font_size": 18},
                       tips=False).move_to([-3.4, 0.1, 0])
            w_t, b_t, v_t = ValueTracker(1.0), ValueTracker(0.5), ValueTracker(0.8)
            xs = np.linspace(-3, 3, 241)

            def hinge_mob():
                ys = actm.hinge(xs, w_t.get_value(), b_t.get_value(), v_t.get_value())
                return curve(hax, xs, ys, color=theme.OUTPUT, width=5)

            def kink_mob():
                k = -b_t.get_value() / w_t.get_value()
                return Dot(hax.c2p(k, 0), color=theme.HIGHLIGHT, radius=0.09)

            h = always_redraw(hinge_mob)
            kd = always_redraw(kink_mob)
            self.play(*self.set_heading("一个隐藏单元 = 一个折点"), Create(hax), run_time=self.fit(1))
            form = MathTex(r"v\cdot\mathrm{ReLU}(w\,x+b)", font_size=44).move_to([3.5, 2.0, 0])
            kf = VGroup(zh("折点：", 26, theme.HIGHLIGHT),
                        MathTex(r"x=-b/w", color=theme.HIGHLIGHT, font_size=38)).arrange(RIGHT, buff=0.15)
            kf.move_to([3.5, 1.0, 0])
            self.play(Write(form), run_time=self.fit(1.2))
            self.play(Create(h), FadeIn(kd), run_time=self.fit(1))
            self.play(FadeIn(kf), run_time=self.fit(0.8))
            steps = [
                ("改 b：折点左右滑动", [(b_t, 2.0), (b_t, -1.5), (b_t, 0.5)]),
                ("改 w：斜的一边变陡或变缓", [(w_t, 2.5), (w_t, 0.6)]),
                ("v < 0：整条折线翻下去", [(v_t, -0.8)]),
            ]
            labels = VGroup(*[zh(t, 24, theme.FG) for t, _ in steps]).arrange(
                DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.5, -0.6, 0])
            per = (self.remaining() - 2.5) / 3
            for lab, (_, moves) in zip(labels, steps, strict=False):
                self.play(FadeIn(lab), run_time=self.fit(0.5))
                for tr, val in moves:
                    self.play(tr.animate.set_value(val), run_time=self.fit(max(0.5, (per - 0.5) / len(moves))))
            each = zh("每个隐藏单元，贡献一个折", 26, theme.HIGHLIGHT).move_to([3.5, -2.2, 0])
            self.play(FadeIn(each), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.remove(h, kd)
            self.play(*[FadeOut(m) for m in [hax, form, kf, labels, each]], run_time=self.fit(0.6))

        # ── S09 Build shapes from line segments ──────────────────────────
        with self.shot("S09"):
            self.play(*self.set_heading("折线可以相加"), run_time=self.fit(0.6))
            xs = np.linspace(-2, 2, 201)

            def small_axes(cx):
                return Axes(x_range=[-2, 2, 1], y_range=[-2, 2.2, 1], x_length=5.0, y_length=3.6,
                            axis_config={"color": theme.MUTED, "include_numbers": True,
                                         "font_size": 16}, tips=False).move_to([cx, 0.2, 0])

            a1, a2 = small_axes(-3.5), small_axes(3.5)
            self.play(Create(a1), run_time=self.fit(0.8))
            r1 = curve(a1, xs, actm.hinge(xs, 1, 0, 1), color=theme.INPUT, width=3)
            r2 = curve(a1, xs, actm.hinge(xs, -1, 0, 1), color=theme.PARAM, width=3)
            absx = curve(a1, xs, actm.hinge(xs, 1, 0, 1) + actm.hinge(xs, -1, 0, 1), width=6)
            t1 = MathTex(r"\mathrm{ReLU}(x)+\mathrm{ReLU}(-x)=|x|", font_size=30).next_to(a1, UP, 0.15)
            self.play(Create(r1), Create(r2), run_time=self.fit(1.2))
            self.play(Create(absx), FadeIn(t1), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.12)
            self.play(Create(a2), run_time=self.fit(0.8))
            parts = [(1, 1, 1, theme.INPUT), (1, 0, -2, theme.GRAD), (1, -1, 1, theme.PARAM)]
            pcs = VGroup(*[curve(a2, xs, actm.hinge(xs, w, b, v), color=c, width=3)
                           for w, b, v, c in parts])
            tent_y = sum(actm.hinge(xs, w, b, v) for w, b, v, _ in parts)
            tent = curve(a2, xs, tent_y, width=6)
            t2 = MathTex(r"\mathrm{ReLU}(x{+}1)-2\,\mathrm{ReLU}(x)+\mathrm{ReLU}(x{-}1)",
                         font_size=28).next_to(a2, UP, 0.15)
            self.play(LaggedStart(*[Create(p) for p in pcs], lag_ratio=0.4), FadeIn(t2),
                      run_time=self.fit(2))
            self.play(Create(tent), run_time=self.fit(1.2))
            tl = zh("帐篷：只在中间鼓一个包", 24, theme.OUTPUT).next_to(a2, DOWN, 0.2)
            self.play(FadeIn(tl), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [a1, a2, r1, r2, absx, t1, pcs, tent, t2, tl]],
                      run_time=self.fit(0.6))

        # ── S10 Eight kinks build a sine ─────────────────────────────────
        with self.shot("S10"):
            ax = data_axes([-3.2, 0.0, 0], x_len=6.6, y_len=4.4, y_range=(-1.5, 1.5, 0.5))
            dots = data_dots(ax, radius=0.035, opacity=0.5)
            self.play(*self.set_heading("宽 8 的网络：8 条折线之和"), Create(ax), FadeIn(dots),
                      run_time=self.fit(1))
            # Small plot at the top right: the 8 piecewise-linear pieces (large amplitudes).
            lo = float(np.floor(PIECES8.min() / 2) * 2)
            hi = float(np.ceil(PIECES8.max() / 2) * 2)
            pax = Axes(x_range=[-3, 3, 1], y_range=[lo, hi, 4], x_length=4.6, y_length=2.6,
                       axis_config={"color": theme.MUTED, "include_numbers": True, "font_size": 14},
                       tips=False).move_to([3.9, 1.1, 0])
            palette = [theme.INPUT, theme.PARAM, theme.GRAD, theme.ATTN, theme.OUTPUT,
                       theme.HIGHLIGHT, theme.FG, theme.MUTED]
            order = np.argsort(KINKS8)
            pieces = VGroup(*[curve(pax, X[:, 0], PIECES8[:, j], color=palette[i], width=2.5)
                              for i, j in enumerate(order)])
            ptitle = zh("8 条折线（每个隐藏单元一条）", 20, theme.MUTED).next_to(pax, UP, 0.1)
            self.play(Create(pax), FadeIn(ptitle), run_time=self.fit(0.8))
            self.play(LaggedStart(*[Create(p) for p in pieces], lag_ratio=0.2), run_time=self.fit(2.5))
            big = zh(f"单条最大幅度 {np.max(np.abs(PIECES8)):.2f}，相加后互相抵消", 20,
                     theme.MUTED).next_to(pax, DOWN, 0.3)
            self.play(FadeIn(big), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.1)
            # Left plot: draw the output segment by segment from left to right.
            # The line bends at each kink.
            inside = sorted(k for k in KINKS8 if -3 < k < 3)
            bounds = [-3.0] + inside + [3.0]
            kink_dots = VGroup(*[Dot(ax.c2p(k, float(mlp.forward(P8, np.array([[k]]))[0][0, 0])),
                                     color=theme.HIGHLIGHT, radius=0.07) for k in inside])
            per = (self.remaining() - 4) / (len(bounds) - 1)
            for i in range(len(bounds) - 1):
                seg_x = np.linspace(bounds[i], bounds[i + 1], 30).reshape(-1, 1)
                seg_y, _ = mlp.forward(P8, seg_x)
                seg = curve(ax, seg_x, seg_y, width=5)
                anims = [Create(seg)]
                if i < len(kink_dots):
                    anims.append(FadeIn(kink_dots[i], scale=1.5))
                self.play(*anims, run_time=self.fit(max(0.3, per)))
                self.add(seg)
                if i == 0:
                    segs = VGroup(seg)
                else:
                    segs.add(seg)
            y_hat8, _ = mlp.forward(P8, X)
            diff = float(np.max(np.abs(PIECES8.sum(axis=1, keepdims=True) + P8["b2"] - y_hat8)))
            eq = VGroup(zh("输出 = 8 条折线之和 + b₂", 24, theme.OUTPUT),
                        zh(f"与网络输出最大差 {sci(diff)}", 20, theme.MUTED)).arrange(DOWN, buff=0.15)
            eq.move_to([3.9, -1.7, 0])
            self.play(FadeIn(eq), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [ax, dots, pax, pieces, ptitle, big, kink_dots, segs,
                                             eq]], run_time=self.fit(0.6))

        # ── S11 Training: gradients by hand ──────────────────────────────
        with self.shot("S11"):
            self.play(*self.set_heading("训练：从损失往回推梯度"), run_time=self.fit(0.6))
            names = [("X", theme.INPUT), ("Z", theme.FG), ("A", theme.FG), (r"\hat{Y}", theme.OUTPUT),
                     ("L", theme.GRAD)]
            chain = VGroup()
            for n, c in names:
                chain.add(VGroup(RoundedRectangle(width=1.0, height=0.62, corner_radius=0.1, color=c),
                                 MathTex(n, color=c, font_size=32)))
            chain.arrange(DOWN, buff=0.28).move_to([-5.3, 0.15, 0])
            for box in chain:
                box[1].move_to(box[0])
            fwd = VGroup(*[Arrow(chain[i].get_bottom(), chain[i + 1].get_top(), buff=0.03,
                                 color=theme.MUTED, stroke_width=3, max_tip_length_to_length_ratio=0.4)
                           for i in range(4)])
            back = Arrow(chain[4].get_right() + RIGHT * 0.3, chain[0].get_right() + RIGHT * 0.3,
                         color=theme.GRAD, buff=0, stroke_width=4)
            back_l = zh("反向", 20, theme.GRAD).next_to(back, RIGHT, 0.08)
            self.play(LaggedStart(*[FadeIn(b) for b in chain], lag_ratio=0.2),
                      *[GrowArrow(a) for a in fwd], run_time=self.fit(1.2))
            grads = VGroup(
                MathTex(r"\frac{\partial L}{\partial \hat{Y}}=\frac{2}{N}(\hat{Y}-Y)", font_size=32),
                MathTex(r"\frac{\partial L}{\partial W_2}=A^{\top}\frac{\partial L}{\partial \hat{Y}}",
                        font_size=32),
                MathTex(r"\frac{\partial L}{\partial A}=\frac{\partial L}{\partial \hat{Y}}W_2^{\top}",
                        font_size=32),
                MathTex(r"\frac{\partial L}{\partial Z}=\frac{\partial L}{\partial A}\odot",
                        r"\mathrm{ReLU}'(Z)", font_size=32),
                MathTex(r"\frac{\partial L}{\partial W_1}=X^{\top}\frac{\partial L}{\partial Z}",
                        font_size=32),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([-0.6, 0.15, 0])
            self.play(GrowArrow(back), FadeIn(back_l), run_time=self.fit(0.8))
            per = (self.remaining() - 5) / 5
            for g in grads:
                self.play(Write(g), run_time=self.fit(min(1.0, max(0.4, per * 0.6))))
                self.wait(max(0.05, per * 0.4))
            hl = SurroundingRectangle(grads[3][1], color=theme.HIGHLIGHT, buff=0.06)
            hl_t = zh("没激活的单元：梯度为 0", 22, theme.HIGHLIGHT).move_to([4.6, 0.3, 0])
            self.play(Create(hl), FadeIn(hl_t), run_time=self.fit(0.8))
            ch4 = VGroup(zh("链式法则 = 反向传播", 22, theme.MUTED),
                         zh("第 4 章系统讲", 22, theme.MUTED)).arrange(DOWN, buff=0.1)
            ch4.move_to([4.6, 1.5, 0])
            self.play(FadeIn(ch4), run_time=self.fit(0.6))
            gc = VGroup(zh("梯度检验：手推 vs 数值梯度", 22, theme.OUTPUT),
                        zh(f"最大差 {sci(GRAD_ERR)}", 24, theme.OUTPUT)).arrange(DOWN, buff=0.12)
            gc.move_to([4.6, -1.0, 0])
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(gc), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [chain, fwd, back, back_l, grads, hl, hl_t, ch4, gc]],
                      run_time=self.fit(0.6))

        # ── S12 Watch the curve fit the data ─────────────────────────────
        with self.shot("S12"):
            ax = data_axes([-3.4, 0.2, 0])
            dots = data_dots(ax)
            self.play(*self.set_heading(f"宽 64，学习率 {mlp.LR}：梯度下降 {mlp.STEPS} 步"),
                      Create(ax), FadeIn(dots), run_time=self.fit(1))
            fit_c = net_curve(ax, SNAP64[0])
            table = VGroup(
                zh("步数", 24, theme.MUTED), DecimalNumber(0, num_decimal_places=0, font_size=34,
                                                          group_with_commas=False),
                zh("损失", 24, theme.GRAD), DecimalNumber(L64[0], num_decimal_places=4, font_size=34),
            ).arrange_in_grid(rows=2, cols=2, buff=(0.5, 0.3), col_alignments="lr")
            table.move_to([3.6, 0.8, 0])
            self.play(Create(fit_c), FadeIn(table), run_time=self.fit(1.2))
            per = (self.remaining() - 3) / 3
            for s in SNAP_STEPS[1:]:
                self.wait(per * 0.35)
                self.play(Transform(fit_c, net_curve(ax, SNAP64[s])),
                          ChangeDecimalToValue(table[1], s), ChangeDecimalToValue(table[3], L64[s]),
                          run_time=self.fit(per * 0.65))
            same = VGroup(zh("梯度下降没变", 26, theme.HIGHLIGHT),
                          zh("变的是模型：它能弯了", 26, theme.HIGHLIGHT)).arrange(DOWN, buff=0.15)
            same.move_to([3.6, -1.1, 0])
            self.play(FadeIn(same), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [ax, dots, fit_c, table, same]], run_time=self.fit(0.6))

        # ── S13 What the width does ──────────────────────────────────────
        with self.shot("S13"):
            self.play(*self.set_heading("宽度 = 能拼出多少个折"), run_time=self.fit(0.6))
            configs = [(2, P2, L2), (8, P8, L8), (64, P64, L64)]
            panels = VGroup()
            for i, (h, p, ls) in enumerate(configs):
                pax = data_axes([-4.5 + 4.5 * i, 0.75, 0], x_len=3.9, y_len=2.8,
                                y_range=(-1.5, 1.5, 1), numbers=False)
                d = data_dots(pax, radius=0.03, opacity=0.8)
                c = net_curve(pax, p, width=3.5)
                title = zh(f"宽 {h} · {mlp.n_params(h)} 个参数", 24, theme.FG).next_to(pax, UP, 0.15)
                cap = zh(f"MSE = {ls[-1]:.4f}", 26, theme.GRAD).next_to(pax, DOWN, 0.2)
                panels.add(VGroup(pax, d, title, c, cap))
            gray = curve(panels[0][0], [-3, 3], [LINE_A * -3 + LINE_B, LINE_A * 3 + LINE_B],
                         color=theme.MUTED, width=2)
            for p in panels:
                self.play(FadeIn(VGroup(p[0], p[1], p[2])), run_time=self.fit(0.6, reserve=5))
                self.play(Create(p[3]), FadeIn(p[4]), run_time=self.fit(1.5, reserve=4))
                if p is panels[0]:
                    self.play(Create(gray), run_time=self.fit(0.6, reserve=4))
                self.wait(self.remaining() * 0.18)
            med = zh("5 个随机种子的中位数：" + " / ".join(f"{SEED_MED[h]:.4f}" for h in (2, 8, 64)),
                     24, theme.MUTED).move_to([0, -1.85, 0])
            self.play(FadeIn(med), run_time=self.fit(0.8))
            uat = zh("隐藏单元足够多 → 一维曲线可以拟合得任意好", 26, theme.HIGHLIGHT).move_to([0, -2.4, 0])
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(uat), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(panels), FadeOut(gray), FadeOut(med), FadeOut(uat),
                      run_time=self.fit(0.6))

        # ── S14 From minimal code to production code ─────────────────────
        with self.shot("S14"):
            left_code = VGroup(*[mono(s, 20) for s in [
                "z = x @ W1 + b1",
                "a = np.maximum(0, z)",
                "y_hat = a @ W2 + b2",
                "d_W2 = a.T @ d_yhat",
                "d_z = (d_yhat @ W2.T) * (z > 0)",
                "d_W1 = x.T @ d_z   # ……",
            ]]).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([-3.6, 0.5, 0])
            right_code = VGroup(*[mono(s, 20) for s in [
                "model = nn.Sequential(",
                "    nn.Linear(1, 64),",
                "    nn.ReLU(),",
                "    nn.Linear(64, 1))",
                "loss.backward()",
                "optimizer.step()",
            ]]).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.4, 0.5, 0])
            for ln in right_code[1:4]:
                ln.shift(RIGHT * 0.5)                  # Text removes leading spaces
            lt = zh("NumPy：手推梯度", 26, theme.MUTED).next_to(left_code, UP, 0.4)
            rt = zh("PyTorch 标准写法", 26, theme.HIGHLIGHT).next_to(right_code, UP, 0.4)
            self.play(*self.set_heading("从极简到生产级"), FadeIn(lt), FadeIn(left_code),
                      run_time=self.fit(1))
            self.wait(self.remaining() * 0.1)
            self.play(FadeIn(rt), FadeIn(right_code), run_time=self.fit(1))
            r1 = SurroundingRectangle(right_code[1:4], color=theme.HIGHLIGHT, buff=0.08)
            self.play(Create(r1), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.2)
            r2 = SurroundingRectangle(right_code[4], color=theme.GRAD, buff=0.08)
            auto = zh("自动求梯度", 22, theme.GRAD).next_to(r2, RIGHT, 0.2)
            self.play(Transform(r1, r2), FadeIn(auto), run_time=self.fit(0.8))
            res = VGroup(
                zh(f"同样的初始参数：PyTorch 与手推版损失差 {sci(abs(PT_TORCH - PT_NUMPY))}", 24,
                   theme.OUTPUT),
                zh(f"PyTorch 默认初始化：{mlp.STEPS} 步后损失 {PT_DEFAULT:.4f}", 24, theme.OUTPUT),
            ).arrange(DOWN, buff=0.2).move_to([0, -1.9, 0])
            self.wait(self.remaining() * 0.15)
            self.play(FadeIn(res[0]), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(res[1]), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [left_code, right_code, lt, rt, r1, auto, res]],
                      run_time=self.fit(0.6))

        # ── S15 Summary and the next chapter ─────────────────────────────
        with self.shot("S15"):
            pts = VGroup(
                zh("1. 线性层叠多少层，还是线性", 30, theme.FG),
                zh("2. 激活函数：每个隐藏单元贡献一个折", 30, theme.FG),
                zh("3. 很多个折相加 → 任意形状的曲线", 30, theme.FG),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([0, 1.3, 0])
            self.play(*self.set_heading("小结"), FadeIn(pts[0]), run_time=self.fit(1))
            self.wait(self.remaining() * 0.08)
            self.play(FadeIn(pts[1]), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.08)
            self.play(FadeIn(pts[2]), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.12)
            pain = zh("两层就要手推 6 行梯度；几十层、上百种运算呢？", 26, theme.GRAD).move_to([0, -0.5, 0])
            self.play(FadeIn(pain), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.35)
            nxt = zh("下一章：反向传播与自动微分", 34, theme.HIGHLIGHT).move_to([0, -1.8, 0])
            nbox = Rectangle(width=nxt.width + 0.6, height=nxt.height + 0.3, color=theme.HIGHLIGHT)
            nbox.move_to(nxt)
            self.play(FadeIn(nxt), Create(nbox), run_time=self.fit(1))
            self.wait(self.remaining() - 1.0)
            self.play(*[FadeOut(m) for m in [pts, pain, nxt, nbox]], *self.set_heading(None),
                      run_time=self.fit(1.0))
