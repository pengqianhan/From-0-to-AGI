"""Chapter 1 video: y = ax + b — learn "training" with a straight line.

The code in ../code/ calculates all numbers in the frames (see the fact list F4–F8 in script.md).
Render: bash chapters/01-linear-regression/video/build.sh
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
    NumberLine,
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


fit = _load("fit_line", "01_fit_line.py")
lrmod = _load("learning_rate", "02_learning_rate.py")

X, Y = fit.make_data()
N = len(X)
HIST = fit.gradient_descent(X, Y, lr=0.05, steps=200)
A_LS, B_LS = np.polyfit(X, Y, deg=1)
LR_C = lrmod.critical_lr(X)
# The loss is a quadratic function: L(θ) = L* + ½ (θ−θ*)ᵀ H (θ−θ*).
# Thus the contour lines are exact ellipses.
H = 2 / N * np.array([[np.sum(X**2), np.sum(X)], [np.sum(X), N]])
THETA_STAR = np.array([A_LS, B_LS])
L_STAR = fit.mse_loss(A_LS, B_LS, X, Y)

MONO = "Noto Sans Mono"


def mono(text: str, size: float = 24, color: str = theme.FG) -> Text:
    return Text(text, font=MONO, font_size=size, color=color)


def loss_ellipse(axes: Axes, level: float, color=theme.MUTED, width=1.5) -> VGroup:
    """Contour line (an ellipse) where the loss = L* + level.

    Draw only the part in the range of the axes.
    """
    evals, evecs = np.linalg.eigh(H)
    radii = np.sqrt(2 * level / evals)
    pts = []
    for t in np.linspace(0, 2 * np.pi, 400):
        a, b = THETA_STAR + evecs @ (radii * np.array([np.cos(t), np.sin(t)]))
        pts.append((a, b))
    return polyline_in_axes(axes, pts, color=color, stroke_width=width)


def clipped_path(axes: Axes, points: list[tuple[float, float]], color: str) -> VMobject:
    """Polyline of the parameter path.

    Cut the parts outside the axes (a diverging path flies out of the frame).
    """
    xr, yr = axes.x_range, axes.y_range
    kept = []
    for a, b in points:
        if not (xr[0] <= a <= xr[1] and yr[0] <= b <= yr[1]):
            break
        kept.append(axes.c2p(a, b))
    path = VMobject(color=color, stroke_width=3)
    if len(kept) >= 2:
        path.set_points_as_corners(kept)
    return path


class ChapterScene(NarratedScene):
    chapter_label = "第 1 章"
    chapter_title = "y = ax + b"

    def construct(self) -> None:
        # ── S01 Opening ──────────────────────────────────────────────────
        with self.shot("S01"):
            card = self.chapter_card()
            sub = zh("从一条直线学会“训练”", 32, theme.HIGHLIGHT).next_to(card, DOWN, 0.6)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(card), FadeOut(sub), run_time=self.fit(0.8))

        # Data axes on the left. They stay from S02 to S07.
        axes = Axes(
            x_range=[0, 5.5, 1], y_range=[-2, 14, 2], x_length=6.2, y_length=4.2,
            axis_config={"color": theme.MUTED, "include_numbers": True, "font_size": 20},
            tips=False,
        ).move_to([-3.4, 0.45, 0])
        xl = MathTex("x", color=theme.MUTED, font_size=30).next_to(axes.x_axis, RIGHT, 0.15)
        yl = MathTex("y", color=theme.MUTED, font_size=30).next_to(axes.y_axis, UP, 0.1)
        dots = VGroup(*[Dot(axes.c2p(x, y), radius=0.05, color=theme.INPUT) for x, y in zip(X, Y)])

        # ── S02 A set of points ──────────────────────────────────────────
        with self.shot("S02"):
            self.play(*self.set_heading("问题：给定数据，找一条最合适的直线"), Create(axes), FadeIn(xl), FadeIn(yl), run_time=self.fit(1.5))
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.08),
                      run_time=self.fit(4))
            note = zh("每个点：一套房子的（面积 x，价格 y）", 26, theme.FG).move_to([3.4, 1.2, 0])
            q = zh("新的 x 来了，y 该猜多少？", 30, theme.HIGHLIGHT).move_to([3.4, 0.2, 0])
            self.play(FadeIn(note), run_time=self.fit(1))
            self.wait(self.remaining() * 0.45)
            self.play(Write(q), run_time=self.fit(1.5))

        # ── S03 The model ────────────────────────────────────────────────
        a_t, b_t = ValueTracker(0.8), ValueTracker(5.0)
        line = always_redraw(lambda: axes.plot(
            lambda x: a_t.get_value() * x + b_t.get_value(), x_range=[0, 5.3],
            color=theme.OUTPUT, stroke_width=4, use_smoothing=False))
        with self.shot("S03"):
            self.play(FadeOut(note), FadeOut(q), run_time=self.fit(0.5))
            model = MathTex(r"\hat{y}", "=", "a", "x", "+", "b", font_size=60).move_to([3.4, 1.9, 0])
            model[2].set_color(theme.PARAM)
            model[5].set_color(theme.PARAM)
            model[3].set_color(theme.INPUT)
            self.play(Write(model), run_time=self.fit(1.5))
            self.play(Create(line), run_time=self.fit(1))
            a_lbl = VGroup(zh("a：斜率", 28, theme.PARAM), zh("b：截距", 28, theme.PARAM)).arrange(
                DOWN, aligned_edge=LEFT, buff=0.25).move_to([3.4, 0.5, 0])
            self.play(FadeIn(a_lbl), run_time=self.fit(1))
            self.wait(self.remaining() * 0.25)
            self.play(a_t.animate.set_value(2.8), run_time=self.fit(2))
            self.play(a_t.animate.set_value(0.3), run_time=self.fit(2))
            self.play(b_t.animate.set_value(1.0), run_time=self.fit(1.5))
            self.play(b_t.animate.set_value(8.0), run_time=self.fit(1.5))
            params = zh("参数 = 模型知道的一切", 26, theme.FG).move_to([3.4, -0.8, 0])
            self.play(FadeIn(params), run_time=self.fit(1))

        # ── S04 The loss ─────────────────────────────────────────────────
        with self.shot("S04"):
            self.play(FadeOut(a_lbl), FadeOut(params),
                      a_t.animate.set_value(HIST[0][0]), b_t.animate.set_value(HIST[0][1]),
                      run_time=self.fit(1.5))
            a0, b0 = HIST[0][0], HIST[0][1]
            resid = VGroup(*[
                Line(axes.c2p(x, y), axes.c2p(x, a0 * x + b0), color=theme.GRAD, stroke_width=2)
                for x, y in zip(X, Y)
            ])
            self.play(LaggedStart(*[Create(r) for r in resid], lag_ratio=0.04),
                      run_time=self.fit(3))
            rl = zh("红线：残差 ŷᵢ − yᵢ", 26, theme.GRAD).move_to([3.4, 0.9, 0])
            self.play(FadeIn(rl), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.3)
            mse = MathTex(r"L(a,b)=\frac{1}{N}\sum_{i=1}^{N}\left(a x_i+b-y_i\right)^2",
                          font_size=36).move_to([3.7, -0.3, 0])
            self.play(Write(mse), run_time=self.fit(2.5))
            val = zh(f"现在：L = {HIST[0][2]:.2f}", 28, theme.HIGHLIGHT).move_to([3.4, -1.5, 0])
            self.play(FadeIn(val), run_time=self.fit(1))

        # ── S05 The loss is a bowl ───────────────────────────────────────
        cax = Axes(
            x_range=[-1.5, 4, 1], y_range=[-2, 6, 2], x_length=4.8, y_length=3.5,
            axis_config={"color": theme.MUTED, "include_numbers": True, "font_size": 18},
            tips=False,
        ).move_to([3.5, -0.35, 0])
        cal = MathTex("a", color=theme.PARAM, font_size=28).next_to(cax.x_axis, RIGHT, 0.1)
        cbl = MathTex("b", color=theme.PARAM, font_size=28).next_to(cax.y_axis, UP, 0.05)
        levels = [0.3, 1.5, 5, 12, 25, 45]
        ellipses = VGroup(*[loss_ellipse(cax, lv) for lv in levels])
        star = Dot(cax.c2p(*THETA_STAR), color=theme.HIGHLIGHT, radius=0.07)
        with self.shot("S05"):
            self.play(*self.set_heading("损失是 (a, b) 的函数：一个碗"),
                      FadeOut(rl), FadeOut(mse), FadeOut(val), FadeOut(resid),
                      model.animate.scale(0.7).move_to([3.5, 2.55, 0]), run_time=self.fit(1))
            self.play(Create(cax), FadeIn(cal), FadeIn(cbl), run_time=self.fit(1))
            self.play(LaggedStart(*[Create(e) for e in reversed(ellipses)], lag_ratio=0.25),
                      run_time=self.fit(3))
            bowl = zh("等高线：损失相同的 (a, b)", 22, theme.MUTED).next_to(cax, UP, 0.12)
            self.play(FadeIn(bowl), run_time=self.fit(0.8))
            self.play(FadeIn(star, scale=2), run_time=self.fit(0.8))
            bottom = zh("碗底", 24, theme.HIGHLIGHT).next_to(star, RIGHT, 0.1)
            self.play(FadeIn(bottom), run_time=self.fit(0.6))

        # ── S06 The gradient ─────────────────────────────────────────────
        start = Dot(cax.c2p(HIST[0][0], HIST[0][1]), color=theme.FG, radius=0.07)
        with self.shot("S06"):
            self.play(*self.set_heading("梯度：脚下的坡度"), FadeOut(bowl),
                      FadeIn(start, scale=2), run_time=self.fit(1))
            ga, gb = fit.gradients(HIST[0][0], HIST[0][1], X, Y)
            g = np.array([ga, gb]) / np.linalg.norm([ga, gb])
            p0 = np.array([HIST[0][0], HIST[0][1]])
            up_arrow = Arrow(cax.c2p(*p0), cax.c2p(*(p0 + 1.1 * g)), color=theme.GRAD, buff=0,
                             stroke_width=5, max_tip_length_to_length_ratio=0.25)
            up_lbl = zh("梯度：上坡最陡", 22, theme.GRAD).next_to(up_arrow.get_end(), UP, 0.15)
            self.wait(self.remaining() * 0.25)
            self.play(GrowArrow(up_arrow), FadeIn(up_lbl), run_time=self.fit(1.2))
            down_arrow = Arrow(cax.c2p(*p0), cax.c2p(*(p0 - 1.1 * g)), color=theme.OUTPUT,
                               buff=0, stroke_width=5, max_tip_length_to_length_ratio=0.25)
            self.play(GrowArrow(down_arrow), run_time=self.fit(1))
            grads = MathTex(
                r"\frac{\partial L}{\partial a}=\frac{2}{N}\sum(\hat{y}_i-y_i)\,x_i",
                r"\qquad",
                r"\frac{\partial L}{\partial b}=\frac{2}{N}\sum(\hat{y}_i-y_i)",
                font_size=34,
            ).move_to([0, -2.3, 0])
            box = RoundedRectangle(width=grads.width + 0.4, height=grads.height + 0.25,
                                   corner_radius=0.1, fill_color=theme.BG, fill_opacity=0.92,
                                   stroke_color=theme.GRAD).move_to(grads)
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(box), Write(grads), run_time=self.fit(2.5))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(box), FadeOut(grads), FadeOut(up_arrow), FadeOut(up_lbl),
                      FadeOut(down_arrow), run_time=self.fit(0.6))

        # ── S07 Gradient descent ─────────────────────────────────────────
        with self.shot("S07"):
            rule = MathTex(r"a \leftarrow a-\eta\,\frac{\partial L}{\partial a}", r"\quad",
                           r"b \leftarrow b-\eta\,\frac{\partial L}{\partial b}",
                           font_size=34).move_to([3.5, 2.55, 0])
            self.play(*self.set_heading("梯度下降：往梯度的反方向走一小步"), Transform(model, rule),
                      run_time=self.fit(1.2))
            step_t = ValueTracker(0)
            table = VGroup(
                zh("步数", 22, theme.MUTED), DecimalNumber(0, num_decimal_places=0, font_size=26),
                zh("a", 22, theme.PARAM), DecimalNumber(HIST[0][0], num_decimal_places=3, font_size=26),
                zh("b", 22, theme.PARAM), DecimalNumber(HIST[0][1], num_decimal_places=3, font_size=26),
                zh("损失", 22, theme.GRAD), DecimalNumber(HIST[0][2], num_decimal_places=2, font_size=26),
            ).arrange_in_grid(rows=2, cols=4, col_alignments="cccc", flow_order="dr",
                              buff=(0.55, 0.12))
            table.move_to([-3.4, -2.25, 0])
            self.play(FadeIn(table), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.2)
            trail_pts = [cax.c2p(HIST[0][0], HIST[0][1])]
            trail = VMobject(color=theme.HIGHLIGHT, stroke_width=3)
            trail.set_points_as_corners([trail_pts[0], trail_pts[0]])
            self.add(trail)
            shown = [1, 2, 3, 5, 10, 20, 35, 50, 75, 100, 140, 200]
            per = self.remaining() / (len(shown) + 2)
            for s in shown:
                a, b, loss = HIST[s]
                trail_pts.append(cax.c2p(a, b))
                new_trail = VMobject(color=theme.HIGHLIGHT, stroke_width=3)
                new_trail.set_points_as_corners(trail_pts)
                self.play(
                    a_t.animate.set_value(a), b_t.animate.set_value(b),
                    start.animate.move_to(cax.c2p(a, b)),
                    Transform(trail, new_trail),
                    ChangeDecimalToValue(table[1], s), ChangeDecimalToValue(table[3], a),
                    ChangeDecimalToValue(table[5], b), ChangeDecimalToValue(table[7], loss),
                    run_time=self.fit(per),
                )
            step_t.set_value(200)

        # ── S08 Why not calculate it with the formula ────────────────────
        with self.shot("S08"):
            everything = VGroup(axes, xl, yl, dots, cax, cal, cbl, ellipses, star, bottom, start,
                                trail, table, model)
            self.remove(line)
            self.play(FadeOut(everything), run_time=self.fit(1))
            gd = VGroup(zh("梯度下降 200 步", 28, theme.OUTPUT),
                        MathTex(rf"a={HIST[200][0]:.3f},\ b={HIST[200][1]:.3f}", font_size=40)
                        ).arrange(DOWN, buff=0.3).move_to([-3.2, 1.0, 0])
            ls = VGroup(zh("解析解（最小二乘）", 28, theme.HIGHLIGHT),
                        MathTex(rf"a={A_LS:.3f},\ b={B_LS:.3f}", font_size=40)
                        ).arrange(DOWN, buff=0.3).move_to([3.2, 1.0, 0])
            self.play(*self.set_heading("线性模型有解析解，神经网络没有"), FadeIn(gd),
                      run_time=self.fit(1))
            self.wait(self.remaining() * 0.25)
            self.play(FadeIn(ls), run_time=self.fit(1))
            self.wait(self.remaining() * 0.3)
            curve_ax = Axes(x_range=[0, 6, 1], y_range=[-2, 2, 1], x_length=6, y_length=2,
                            axis_config={"color": theme.MUTED}, tips=False).move_to([0, -1.4, 0])
            curve = curve_ax.plot(lambda x: np.sin(1.7 * x) * np.exp(-0.15 * x) + 0.3 * np.cos(4 * x),
                                  color=theme.INPUT, stroke_width=4)
            qmark = zh("？没有公式", 34, theme.GRAD).next_to(curve_ax, RIGHT, 0.2)
            self.play(Create(curve_ax), Create(curve), run_time=self.fit(1.5))
            self.play(FadeIn(qmark), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [gd, ls, curve_ax, curve, qmark]],
                      run_time=self.fit(0.6))

        # ── S09 Three cases of the learning rate ─────────────────────────
        with self.shot("S09"):
            self.play(*self.set_heading("学习率 η：太小、合适、太大"), run_time=self.fit(0.6))
            configs = [(0.005, "太小：慢", theme.INPUT), (0.05, "合适：收敛", theme.OUTPUT),
                       (1.05 * LR_C, "太大：发散", theme.GRAD)]
            panels = VGroup()
            for i, (lr, label, color) in enumerate(configs):
                pax = Axes(x_range=[-1.5, 4, 1], y_range=[-2, 6, 2], x_length=3.6, y_length=2.8,
                           axis_config={"color": theme.MUTED, "stroke_width": 1}, tips=False)
                pax.move_to([-4.4 + 4.4 * i, 0.35, 0])
                ell = VGroup(*[loss_ellipse(pax, lv, width=1) for lv in levels])
                steps = 100 if lr < 0.08 else 12
                hist = fit.gradient_descent(X, Y, lr=lr, steps=steps)
                path = clipped_path(pax, [(h[0], h[1]) for h in hist], color)
                title = zh(f"η = {lr:.4g}", 24, theme.FG).next_to(pax, UP, 0.15)
                cap = zh(label, 26, color).next_to(pax, DOWN, 0.15)
                panels.add(VGroup(pax, ell, title, cap, path))
            for p in panels:
                self.play(FadeIn(VGroup(p[0], p[1], p[2])), run_time=self.fit(0.6, reserve=6))
                self.play(Create(p[4]), FadeIn(p[3]), run_time=self.fit(2.2, reserve=3))
            final = zh(f"η = {1.05 * LR_C:.4g}：100 步后损失 ≈ 7.7 × 10⁹", 24, theme.GRAD)
            final.move_to([0, -2.35, 0])
            self.play(FadeIn(final), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(panels), FadeOut(final), run_time=self.fit(0.6))

        # ── S10 The critical learning rate ───────────────────────────────
        with self.shot("S10"):
            nl = NumberLine(x_range=[0, 0.15, 0.025], length=10, color=theme.MUTED,
                            include_numbers=True, decimal_number_config={"num_decimal_places": 3},
                            font_size=20).move_to([0, -0.2, 0])
            self.play(Create(nl), *self.set_heading("收敛条件：η < 2 / λmax（最陡方向的弯曲程度）"),
                      run_time=self.fit(1))
            c_pt = nl.n2p(LR_C)
            good = Line(nl.n2p(0), c_pt, color=theme.OUTPUT, stroke_width=10)
            bad = Line(c_pt, nl.n2p(0.15), color=theme.GRAD, stroke_width=10)
            crit = DashedLine(c_pt + UP * 1.3, c_pt + DOWN * 0.3, color=theme.HIGHLIGHT)
            crit_lbl = MathTex(rf"2/\lambda_{{max}}={LR_C:.4f}", color=theme.HIGHLIGHT,
                               font_size=34).next_to(crit, UP, 0.1)
            self.wait(self.remaining() * 0.35)
            self.play(Create(crit), Write(crit_lbl), run_time=self.fit(1.2))
            self.play(Create(good), Create(bad), run_time=self.fit(1))
            ok = VGroup(Circle(0.12, color=theme.OUTPUT, fill_opacity=1).move_to(nl.n2p(0.9 * LR_C)),
                        zh("0.0914 收敛", 22, theme.OUTPUT).next_to(nl.n2p(0.9 * LR_C), DOWN, 0.55))
            ng = VGroup(Circle(0.12, color=theme.GRAD, fill_opacity=1).move_to(nl.n2p(1.05 * LR_C)),
                        zh("0.1066 发散", 22, theme.GRAD).next_to(nl.n2p(1.05 * LR_C), DOWN, 1.05))
            self.play(FadeIn(ok), run_time=self.fit(0.8))
            self.play(FadeIn(ng), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [nl, good, bad, crit, crit_lbl, ok, ng]],
                      run_time=self.fit(0.6))

        # ── S11 From minimal code to production code ─────────────────────
        with self.shot("S11"):
            left_code = VGroup(*[mono(s, 22) for s in [
                "err = (a*x + b) - y",
                "grad_a = 2*np.mean(err*x)",
                "grad_b = 2*np.mean(err)",
                "a = a - lr*grad_a",
                "b = b - lr*grad_b",
            ]]).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
            right_code = VGroup(*[mono(s, 22) for s in [
                "y_hat = model(x)",
                "loss = loss_fn(y_hat, y)",
                "optimizer.zero_grad()",
                "loss.backward()",
                "optimizer.step()",
            ]]).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
            left_code.move_to([-3.6, 0.2, 0])
            right_code.move_to([3.4, 0.2, 0])
            lt = zh("手算梯度（NumPy）", 26, theme.MUTED).next_to(left_code, UP, 0.4)
            rt = zh("PyTorch 标准写法", 26, theme.HIGHLIGHT).next_to(right_code, UP, 0.4)
            self.play(*self.set_heading("从极简到生产级：PyTorch 五行训练循环"), FadeIn(lt),
                      FadeIn(left_code), run_time=self.fit(1))
            self.play(FadeIn(rt), FadeIn(right_code), run_time=self.fit(1))
            notes = ["前向", "损失", "清梯度", "反向", "更新"]
            per = (self.remaining() - 3) / 5
            prev = None
            for line_m, note in zip(right_code, notes):
                rect = SurroundingRectangle(line_m, color=theme.HIGHLIGHT, buff=0.08)
                tag = zh(note, 22, theme.HIGHLIGHT).next_to(rect, RIGHT, 0.2)
                grp = VGroup(rect, tag)
                if prev is None:
                    self.play(FadeIn(grp), run_time=self.fit(min(per, 1)))
                else:
                    self.play(Transform(prev, grp), run_time=self.fit(min(per, 1)))
                    grp = prev
                prev = grp
                self.wait(max(0.1, per - 1))
            same = zh("201 步后：a = 2.039，b = 0.925 —— 两种写法逐位一致", 24, theme.OUTPUT)
            same.move_to([0, -1.9, 0])
            self.play(FadeIn(same), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [lt, rt, left_code, right_code, prev, same]],
                      run_time=self.fit(0.6))

        # ── S12 The link to large models ─────────────────────────────────
        with self.shot("S12"):
            names = [("模型", theme.OUTPUT), ("损失", theme.GRAD), ("梯度", theme.GRAD),
                     ("更新", theme.PARAM)]
            boxes = VGroup()
            for name, color in names:
                r = RoundedRectangle(width=2.2, height=1.0, corner_radius=0.15, color=color)
                boxes.add(VGroup(r, zh(name, 32, color)))
            boxes.arrange(RIGHT, buff=1.0).move_to([0, 1.6, 0])
            arrows = VGroup(*[Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.1,
                                    color=theme.MUTED) for i in range(3)])
            back = Arrow(boxes[3].get_bottom(), boxes[0].get_bottom(), buff=0.1, color=theme.MUTED,
                         path_arc=-0.9)
            self.play(*self.set_heading("四步循环"),
                      LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.3), run_time=self.fit(2))
            self.play(*[GrowArrow(a) for a in arrows], Create(back), run_time=self.fit(1.2))
            scale = VGroup(zh("本章：2 个参数，均方误差", 26, theme.FG),
                           zh("大语言模型：几十亿个参数，预测下一个词的交叉熵", 26, theme.FG)
                           ).arrange(DOWN, buff=0.25).move_to([0, -0.7, 0])
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(scale[0]), run_time=self.fit(0.8))
            self.play(FadeIn(scale[1]), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.45)
            nxt = zh("下一章：从标量到矩阵", 34, theme.HIGHLIGHT).move_to([0, -2.3, 0])
            nbox = Rectangle(width=nxt.width + 0.6, height=nxt.height + 0.3, color=theme.HIGHLIGHT)
            nbox.move_to(nxt)
            self.play(FadeIn(nxt), Create(nbox), run_time=self.fit(1))
            self.wait(self.remaining() - 1.0)
            self.play(*[FadeOut(m) for m in [boxes, arrows, back, scale, nxt, nbox]],
                      *self.set_heading(None),
                      run_time=self.fit(1.0))
