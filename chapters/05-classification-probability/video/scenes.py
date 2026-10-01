"""第 5 章视频：分类与概率 —— 从"猜一个数"到"猜哪一类"

画面里的所有数值都由 ../code/ 中的代码真实计算（见 script.md 事实清单 F2–F13）。
渲染：bash chapters/05-classification-probability/video/build.sh
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
    UR,
    Arrow,
    Axes,
    ChangeDecimalToValue,
    Create,
    DashedLine,
    DecimalNumber,
    Dot,
    FadeIn,
    FadeOut,
    GrowArrow,
    ImageMobject,
    LaggedStart,
    Line,
    MathTex,
    NumberLine,
    Rectangle,
    RoundedRectangle,
    SurroundingRectangle,
    Text,
    Transform,
    VGroup,
    Write,
)
from manim.utils.color import ManimColor

from video_kit import theme
from video_kit.scene import NarratedScene, code_block, polyline_in_axes, zh

CODE = Path(__file__).resolve().parent.parent / "code"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


smx = _load("softmax_mod", "01_softmax.py")
cem = _load("ce_mod", "02_cross_entropy.py")
clf = _load("clf_mod", "03_train_classifier.py")
lm = _load("lm_mod", "04_next_token.py")

# ── 真实数据（全部由 code/ 算出） ──────────────────────────────────────────
LOGITS = smx.LOGITS
EXPS = np.exp(LOGITS)
PROBS = smx.softmax(LOGITS)
BIG = LOGITS * 500
with np.errstate(over="ignore", invalid="ignore"):
    NAIVE_BIG = smx.softmax_naive(BIG)
STABLE_BIG = smx.softmax(BIG)
CE_EACH = [cem.cross_entropy(LOGITS, np.array([k])) for k in range(3)]
GRAD_CAT = cem.ce_grad(LOGITS[None, :].copy(), np.array([0]))[0]
GRAD_NUM = cem.numerical_grad(lambda t: cem.cross_entropy(t, np.array([0])), LOGITS[None, :].copy())[0]
GRAD_DIFF = float(np.abs(GRAD_CAT - GRAD_NUM).max())
S_GRID = np.linspace(0, 10, 101)
WRONG_ROWS = cem.confident_wrong_table(gaps=S_GRID)
WRONG_END = cem.confident_wrong_table(gaps=(0, 10))

X, Y = clf.make_spirals()
SNAP_STEPS = [0, 20, 50, 100, 200, 500, 3000]
_, LOG_CE, SNAPS = clf.train(X, Y, loss="ce", snapshot_steps=SNAP_STEPS)
LOG_BY_STEP = {s: (loss, acc) for s, loss, acc in LOG_CE}
XT, YT = clf.make_spirals(seed=1)
TEST_ACC = clf.accuracy(SNAPS[3000], XT, YT)
W_LIN, B_LIN = clf.train_linear(X, Y)
LIN_ACC = float(((X @ W_LIN + B_LIN).argmax(1) == Y).mean())
_, LOG_BAD_CE, _ = clf.train(X, Y, loss="ce", out_std=10.0)
_, LOG_BAD_MSE, _ = clf.train(X, Y, loss="mse", out_std=10.0)
# MSE 卡多久对舍入极其敏感（课程构建机上卡七八百步，另一台服务器上两三百步），
# 所以画面只强调两台机器都成立的部分：第 100 步时的差距，以及 MSE 在这台机器上第几步过 90%。
STUCK_STEP = 100
STUCK = {}
for _loss in ["ce", "mse"]:
    _p, _, _ = clf.train(X, Y, loss=_loss, steps=STUCK_STEP, out_std=10.0)
    _pt = cem.softmax(clf.forward(_p, X)[0])[np.arange(len(Y)), Y]
    STUCK[_loss] = int(np.sum(_pt < 0.01))
MSE_CATCH = next(s for s, _, a in LOG_BAD_MSE if a >= 0.9)

VOCAB, STOI, LM_X, LM_Y = lm.build_dataset(lm.TEXT)
V = len(VOCAB)
W_LM, LOG_LM = lm.train_bigram(LM_X, LM_Y, V)
LM_START, LM_END = LOG_LM[0][1], LOG_LM[-1][1]
P_FEN = cem.softmax(W_LM[STOI["分"]])
TOP_FEN = np.argsort(-P_FEN)[:3]

MONO = "Noto Sans Mono"
CLASS_COLORS = [theme.INPUT, theme.PARAM, theme.OUTPUT]  # 三类：蓝、橙、绿
NAMES = smx.CLASSES  # 猫 狗 鸟


def mono(text: str, size: float = 24, color: str = theme.FG) -> Text:
    return Text(text, font=MONO, font_size=size, color=color)


def bars(values, labels, colors, center_x: float, base_y: float, unit: float,
         width: float = 0.7, gap: float = 1.2, fmt: str = "{:.2f}", label_y: float | None = None,
         font: float = 24, value_color=None) -> VGroup:
    """简易柱状图：返回 VGroup(柱子, 数值, 类别名, 基线)。负值向下画。"""
    n = len(values)
    xs = [center_x + (i - (n - 1) / 2) * gap for i in range(n)]
    rects, vals, names = VGroup(), VGroup(), VGroup()
    for x, v, lab, c in zip(xs, values, labels, colors, strict=True):
        h = max(abs(v) * unit, 0.02)
        r = Rectangle(width=width, height=h, stroke_width=0, fill_color=c, fill_opacity=0.9)
        if v >= 0:
            r.move_to([x, base_y + h / 2, 0])
            t = mono(fmt.format(v), font, value_color or theme.FG).next_to(r, UP, 0.1)
        else:
            r.move_to([x, base_y - h / 2, 0])
            t = mono(fmt.format(v), font, value_color or theme.FG).next_to(r, DOWN, 0.1)
        rects.add(r)
        vals.add(t)
        ly = label_y if label_y is not None else base_y - 0.35
        names.add(zh(lab, font, c).move_to([x, ly, 0]))
    base = Line([xs[0] - gap / 2, base_y, 0], [xs[-1] + gap / 2, base_y, 0], color=theme.MUTED,
                stroke_width=2)
    return VGroup(rects, vals, names, base)


def _rgb(hex_color: str) -> np.ndarray:
    return np.array(ManimColor(hex_color).to_rgb())


def region_image(prob_fn, axes: Axes, res: int = 160) -> ImageMobject:
    """决策区域：每个像素的颜色 = 各类颜色按预测概率加权，再和背景混合。"""
    lo, hi = axes.x_range[0], axes.x_range[1]
    g = np.linspace(lo, hi, res)
    gx, gy = np.meshgrid(g, g[::-1])  # 图像第 0 行在最上面 = y 最大
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    p = prob_fn(pts)  # (res², 3)
    cols = np.stack([_rgb(c) for c in CLASS_COLORS])
    mix = p @ cols
    img = 0.42 * mix + 0.58 * _rgb(theme.BG)
    arr = (np.clip(img, 0, 1) * 255).astype(np.uint8).reshape(res, res, 3)
    im = ImageMobject(arr)
    im.stretch_to_fit_width(axes.x_length)
    im.stretch_to_fit_height(axes.y_length)
    im.move_to(axes.c2p((lo + hi) / 2, (lo + hi) / 2))
    return im


def mlp_probs(params):
    return lambda pts: cem.softmax(clf.forward(params, pts)[0])


class ChapterScene(NarratedScene):
    chapter_label = "第 5 章"
    chapter_title = "分类与概率"

    def construct(self) -> None:
        # ── S01 片头 ─────────────────────────────────────────────────────
        with self.shot("S01"):
            card = self.chapter_card()
            sub = zh("从“猜一个数”到“猜哪一类”", 32, theme.HIGHLIGHT).next_to(card, DOWN, 0.6)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(card), FadeOut(sub), run_time=self.fit(0.8))

        # ── S02 为什么不预测编号 ─────────────────────────────────────────
        with self.shot("S02"):
            self.play(*self.set_heading("答案不是一个数，而是“哪一类”"), run_time=self.fit(0.8))
            nl = NumberLine(x_range=[0, 2, 1], length=5, color=theme.MUTED, include_numbers=True,
                            font_size=26).move_to([-3.4, 0.6, 0])
            tags = VGroup(*[zh(n, 30, c).next_to(nl.n2p(i), UP, 0.35)
                            for i, (n, c) in enumerate(zip(NAMES, CLASS_COLORS, strict=True))])
            self.play(Create(nl), FadeIn(tags), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.2)
            pred = Dot(nl.n2p(1), radius=0.12, color=theme.GRAD)
            pred_lbl = zh("拿不准猫还是鸟 → 平均 = 1 = 狗？", 24, theme.GRAD).next_to(nl, DOWN, 0.7)
            self.play(FadeIn(pred, scale=2), FadeIn(pred_lbl), run_time=self.fit(1))
            self.wait(self.remaining() * 0.35)
            boxes = VGroup()
            for n, c in zip(NAMES, CLASS_COLORS, strict=True):
                r = RoundedRectangle(width=2.6, height=0.75, corner_radius=0.1, color=c)
                boxes.add(VGroup(r, zh(f"{n} 的分数", 26, c).move_to(r)))
            boxes.arrange(DOWN, buff=0.3).move_to([3.6, 0.5, 0])
            head = zh("有几类，就输出几个分数", 28, theme.HIGHLIGHT).next_to(boxes, UP, 0.4)
            self.play(FadeIn(head), LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.3),
                      run_time=self.fit(1.5))
            lg = zh("这些分数叫 logits：任意实数", 24, theme.FG).next_to(boxes, DOWN, 0.4)
            self.play(FadeIn(lg), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [nl, tags, pred, pred_lbl, boxes, head, lg]],
                      run_time=self.fit(0.6))

        # ── S03 softmax ──────────────────────────────────────────────────
        with self.shot("S03"):
            self.play(*self.set_heading("softmax：把分数变成概率"), run_time=self.fit(0.8))
            cx, by = -3.4, -0.4
            b_logit = bars(LOGITS, NAMES, CLASS_COLORS, cx, by, 0.9, fmt="{:.1f}", label_y=-1.75)
            stage = zh("logits  z", 28, theme.MUTED).move_to([cx, 2.55, 0])
            self.play(FadeIn(b_logit), FadeIn(stage), run_time=self.fit(1.2))
            formula = MathTex(r"p_k=\frac{e^{z_k}}{\sum_j e^{z_j}}", font_size=56).move_to([3.5, 1.2, 0])
            self.play(Write(formula), run_time=self.fit(1.5))
            notes = VGroup(zh("① 取指数：一定为正", 26, theme.FG),
                           zh("② 除以总和：加起来 = 1", 26, theme.FG)
                           ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.5, -0.4, 0])
            self.play(FadeIn(notes[0]), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.12)
            b_exp = bars(EXPS, NAMES, CLASS_COLORS, cx, by, 0.3, label_y=-1.75)
            stage2 = zh("exp(z)", 28, theme.MUTED).move_to(stage)
            self.play(Transform(b_logit, b_exp), Transform(stage, stage2), run_time=self.fit(1.5))
            self.play(FadeIn(notes[1]), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.15)
            b_p = bars(PROBS, NAMES, CLASS_COLORS, cx, by, 3.0, label_y=-1.75,
                       value_color=theme.OUTPUT)
            stage3 = zh("概率 p（和 = 1）", 28, theme.OUTPUT).move_to(stage)
            self.play(Transform(b_logit, b_p), Transform(stage, stage3), run_time=self.fit(1.5))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [b_logit, stage, formula, notes]], run_time=self.fit(0.6))

        # ── S04 数值稳定 ─────────────────────────────────────────────────
        with self.shot("S04"):
            self.play(*self.set_heading("数值稳定：先减去最大值"), run_time=self.fit(0.8))
            of = zh("float32：exp(x) 在 x > 88.7 时溢出成 ∞", 26, theme.MUTED).move_to([0, 2.2, 0])
            self.play(FadeIn(of), run_time=self.fit(0.8))
            z_big = MathTex(r"z=[1000,\ 500,\ -500]", font_size=40).move_to([0, 1.3, 0])
            self.play(Write(z_big), run_time=self.fit(1))
            naive = VGroup(zh("直接算：", 28, theme.GRAD),
                           MathTex(r"e^{1000}=\infty\ \Rightarrow\ p=[\,\mathrm{nan},\ 0,\ 0\,]",
                                   font_size=38, color=theme.GRAD)).arrange(RIGHT, buff=0.3)
            naive.move_to([0, 0.3, 0])
            self.wait(self.remaining() * 0.1)
            self.play(FadeIn(naive), run_time=self.fit(1))
            self.wait(self.remaining() * 0.3)
            prop = MathTex(r"\mathrm{softmax}(z)=\mathrm{softmax}(z-c)", font_size=40,
                           color=theme.HIGHLIGHT).move_to([0, -0.7, 0])
            self.play(Write(prop), run_time=self.fit(1.2))
            stable = VGroup(zh("先减最大值：", 28, theme.OUTPUT),
                            MathTex(r"z-1000=[0,\ -500,\ -1500]\ \Rightarrow\ p=[\,"
                                    + ",\\ ".join(f"{v:.0f}" for v in STABLE_BIG) + r"\,]",
                                    font_size=38, color=theme.OUTPUT)).arrange(RIGHT, buff=0.3)
            stable.move_to([0, -1.7, 0])
            self.play(FadeIn(stable), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [of, z_big, naive, prop, stable]], run_time=self.fit(0.6))

        # ── S05 最大似然 ─────────────────────────────────────────────────
        with self.shot("S05"):
            self.play(*self.set_heading("最大似然：让正确答案的概率尽量大"), run_time=self.fit(0.8))
            lik = MathTex(r"\text{likelihood}=\prod_{i=1}^{N} p_{i,\,y_i}", font_size=44).move_to([-3.4, 1.9, 0])
            self.play(Write(lik), run_time=self.fit(1.2))
            rows = VGroup()
            for n in [10, 100, 1000, 10000]:
                with np.errstate(under="ignore"):
                    v = float(np.prod(np.full(n, 0.9)))
                mant, ex = f"{v:.2e}".split("e")
                rows.add(MathTex(rf"0.9^{{{n}}}={mant}\times10^{{{int(ex)}}}", font_size=34))
            rows.arrange(DOWN, aligned_edge=LEFT, buff=0.28).move_to([-3.4, -0.3, 0])
            self.play(LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.5), run_time=self.fit(3, reserve=4))
            under = zh("连乘会下溢", 26, theme.GRAD).next_to(rows, DOWN, 0.3)
            self.play(FadeIn(under), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.25)
            arrow = Arrow([-0.6, 0.3, 0], [0.6, 0.3, 0], color=theme.MUTED, buff=0)
            logt = MathTex(r"\log\prod_i p_i=\sum_i \log p_i", font_size=40).move_to([3.7, 1.2, 0])
            nll = MathTex(r"L=-\frac{1}{N}\sum_{i=1}^{N}\log p_{i,\,y_i}", font_size=46,
                          color=theme.GRAD).move_to([3.7, -0.3, 0])
            nll_lbl = zh("负对数似然：越小越好", 26, theme.GRAD).next_to(nll, DOWN, 0.35)
            self.play(GrowArrow(arrow), Write(logt), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.25)
            self.play(Write(nll), FadeIn(nll_lbl), run_time=self.fit(1.5))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [lik, rows, under, arrow, logt, nll, nll_lbl]],
                      run_time=self.fit(0.6))

        # ── S06 交叉熵 ───────────────────────────────────────────────────
        with self.shot("S06"):
            self.play(*self.set_heading("交叉熵：−log（正确答案的概率）"), run_time=self.fit(0.8))
            ax = Axes(x_range=[0, 1, 0.2], y_range=[0, 4, 1], x_length=5.6, y_length=4.0,
                      axis_config={"color": theme.MUTED, "include_numbers": True, "font_size": 20},
                      tips=False).move_to([-3.3, 0.1, 0])
            xl = MathTex(r"p_y", color=theme.MUTED, font_size=30).next_to(ax.x_axis, RIGHT, 0.15)
            yl = MathTex(r"-\ln p_y", color=theme.MUTED, font_size=28).next_to(ax.y_axis, UP, 0.1)
            curve = ax.plot(lambda p: -np.log(p), x_range=[np.exp(-4), 1], color=theme.GRAD,
                            stroke_width=4)
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=self.fit(1))
            ce_def = MathTex(r"H(q,p)=-\sum_k q_k\log p_k", font_size=40).move_to([3.6, 1.7, 0])
            ce_one = MathTex(r"=-\log p_y", font_size=46, color=theme.GRAD).next_to(ce_def, DOWN, 0.3)
            oh = zh("q 是 onehot：只剩正确那一项", 24, theme.MUTED).next_to(ce_one, DOWN, 0.3)
            self.play(Write(ce_def), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.12)
            self.play(Write(ce_one), FadeIn(oh), run_time=self.fit(1.2))
            self.play(Create(curve), run_time=self.fit(1.5))
            pts = VGroup()
            for k in range(3):
                d = Dot(ax.c2p(PROBS[k], CE_EACH[k]), color=CLASS_COLORS[k], radius=0.09)
                lab = zh(f"{NAMES[k]}：p={PROBS[k]:.2f} → {CE_EACH[k]:.2f}", 22, CLASS_COLORS[k])
                pts.add(VGroup(d, lab))
            pts[0][1].next_to(pts[0][0], UR, 0.1)
            pts[1][1].next_to(pts[1][0], UR, 0.12)
            pts[2][1].next_to(pts[2][0], RIGHT, 0.2)
            self.play(LaggedStart(*[FadeIn(p) for p in pts], lag_ratio=0.5), run_time=self.fit(2.5, reserve=3))
            ln3 = DashedLine(ax.c2p(0, np.log(3)), ax.c2p(1, np.log(3)), color=theme.HIGHLIGHT)
            ln3_lbl = zh(f"均匀乱猜：ln 3 = {np.log(3):.4f}", 24, theme.HIGHLIGHT).move_to([3.6, -1.2, 0])
            self.wait(self.remaining() * 0.4)
            self.play(Create(ln3), FadeIn(ln3_lbl), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [ax, xl, yl, curve, ce_def, ce_one, oh, pts, ln3, ln3_lbl]],
                      run_time=self.fit(0.6))

        # ── S07 梯度 p − onehot ──────────────────────────────────────────
        with self.shot("S07"):
            self.play(*self.set_heading("梯度：p − onehot"), run_time=self.fit(0.8))
            d1 = MathTex(r"L=-z_y+\log\sum_j e^{z_j}", font_size=40).move_to([3.4, 2.0, 0])
            d2 = MathTex(r"\frac{\partial L}{\partial z_k}=p_k-[k=y]", font_size=44,
                         color=theme.GRAD).move_to([3.4, 0.8, 0])
            self.play(Write(d1), run_time=self.fit(1.2))
            self.play(Write(d2), run_time=self.fit(1.2))
            onehot = np.array([1.0, 0.0, 0.0])
            short = ["猫", "狗", "鸟"]
            g_p = bars(PROBS, short, CLASS_COLORS, -5.3, -0.2, 1.8, width=0.45, gap=0.65,
                       fmt="{:.2f}", font=20, label_y=-1.9)
            g_o = bars(onehot, short, [theme.HIGHLIGHT] * 3, -3.2, -0.2, 1.8, width=0.45, gap=0.65,
                       fmt="{:.0f}", font=20, label_y=-1.9)
            g_g = bars(GRAD_CAT, short, [theme.GRAD] * 3, -1.1, -0.2, 1.8, width=0.45, gap=0.65,
                       fmt="{:.2f}", font=20, label_y=-1.9)
            ts = VGroup(zh("p", 24, theme.OUTPUT).move_to([-5.3, 2.3, 0]),
                        zh("onehot", 22, theme.HIGHLIGHT).move_to([-3.2, 2.3, 0]),
                        zh("p − onehot", 22, theme.GRAD).move_to([-1.1, 2.3, 0]))
            ts[1].add(zh("（正确 = 猫）", 18, theme.HIGHLIGHT).next_to(ts[1], DOWN, 0.08))
            ops = VGroup(MathTex("-", font_size=48).move_to([-4.25, 0.5, 0]),
                         MathTex("=", font_size=48).move_to([-2.15, 0.5, 0]))
            self.wait(self.remaining() * 0.1)
            self.play(FadeIn(g_p), FadeIn(ts[0]), run_time=self.fit(0.8))
            self.play(FadeIn(ops[0]), FadeIn(g_o), FadeIn(ts[1]), run_time=self.fit(0.8))
            self.play(FadeIn(ops[1]), FadeIn(g_g), FadeIn(ts[2]), run_time=self.fit(1))
            meaning = VGroup(zh("正确类：往上推，力度 = 1 − p", 24, theme.FG),
                             zh("错误类：往下压，力度 = p", 24, theme.FG)
                             ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.4, -0.5, 0])
            self.wait(self.remaining() * 0.1)
            self.play(FadeIn(meaning), run_time=self.fit(1))
            check = zh(f"数值梯度对拍：最大差 {GRAD_DIFF:.1e}", 24, theme.OUTPUT).move_to([3.4, -1.8, 0])
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(check), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [d1, d2, g_p, g_o, g_g, ts, ops, meaning, check]],
                      run_time=self.fit(0.6))

        # ── S08 MSE vs CE：梯度 ──────────────────────────────────────────
        with self.shot("S08"):
            self.play(*self.set_heading("自信地错了：MSE 的梯度会消失"), run_time=self.fit(0.8))
            # 纵轴画 log10(梯度) + 4，让横轴落在最底下（10⁻⁴ 处）
            ax = Axes(x_range=[0, 10, 2], y_range=[0, 4.5, 1], x_length=6.8, y_length=4.2,
                      axis_config={"color": theme.MUTED, "font_size": 20}, tips=False,
                      x_axis_config={"include_numbers": True}).move_to([-2.4, 0.05, 0])
            ylabels = VGroup(*[MathTex(rf"10^{{{k}}}", font_size=24, color=theme.MUTED)
                               .next_to(ax.c2p(0, k + 4), LEFT, 0.12) for k in range(-4, 1)])
            xl = zh("错误类别的 logit s（越大越自信地错）", 20, theme.MUTED).next_to(ax.x_axis, DOWN, 0.45)
            yl = zh("梯度大小", 20, theme.MUTED).next_to(ax.y_axis, UP, 0.1)
            self.play(Create(ax), FadeIn(ylabels), FadeIn(xl), FadeIn(yl), run_time=self.fit(1.2))
            ce_pts = [(r[0], np.log10(r[3]) + 4) for r in WRONG_ROWS]
            mse_pts = [(r[0], np.log10(r[5]) + 4) for r in WRONG_ROWS]
            ce_line = polyline_in_axes(ax, ce_pts, color=theme.GRAD, stroke_width=5)
            mse_line = polyline_in_axes(ax, mse_pts, color=theme.FG, stroke_width=5)
            leg = VGroup(zh("交叉熵", 26, theme.GRAD), zh("MSE", 26, theme.FG)
                         ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([3.9, 1.8, 0])
            self.wait(self.remaining() * 0.18)
            self.play(Create(ce_line), FadeIn(leg[0]), run_time=self.fit(1.5))
            self.play(Create(mse_line), FadeIn(leg[1]), run_time=self.fit(2))
            s0, s10 = WRONG_END
            nums = VGroup(
                zh(f"s = 10：CE 梯度 {s10[3]:.3f}", 24, theme.GRAD),
                zh(f"MSE 梯度 {s10[5]:.1e}", 24, theme.FG),
                zh(f"相差 {s10[3] / s10[5]:.0f} 倍", 24, theme.HIGHLIGHT),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([4.2, 0.1, 0])
            self.play(FadeIn(nums), run_time=self.fit(1))
            why = MathTex(r"\frac{\partial p}{\partial z}=\mathrm{diag}(p)-pp^{\top}", font_size=34
                          ).move_to([4.2, -1.55, 0])
            self.wait(self.remaining() * 0.3)
            self.play(Write(why), run_time=self.fit(1.2))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [ax, ylabels, xl, yl, ce_line, mse_line, leg, nums, why]],
                      run_time=self.fit(0.6))

        # ── S09 MSE vs CE：训练 ──────────────────────────────────────────
        with self.shot("S09"):
            self.play(*self.set_heading("一开始就自信地乱猜：MSE 起步慢得多"), run_time=self.fit(0.8))
            ax = Axes(x_range=[0, 3000, 500], y_range=[0.2, 1.0, 0.2], x_length=6.8, y_length=4.0,
                      axis_config={"color": theme.MUTED, "include_numbers": True, "font_size": 20},
                      tips=False).move_to([-2.4, 0.15, 0])
            xl = zh("步数", 20, theme.MUTED).next_to(ax.x_axis, DOWN, 0.4)
            yl = zh("训练准确率", 20, theme.MUTED).next_to(ax.y_axis, UP, 0.1)
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=self.fit(1))
            ce_curve = polyline_in_axes(ax, [(s, a) for s, _, a in LOG_BAD_CE], color=theme.GRAD,
                                        stroke_width=5)
            mse_curve = polyline_in_axes(ax, [(s, a) for s, _, a in LOG_BAD_MSE], color=theme.FG,
                                         stroke_width=5)
            leg = VGroup(zh("交叉熵", 26, theme.GRAD), zh("MSE", 26, theme.FG)
                         ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([3.9, 2.1, 0])
            self.wait(self.remaining() * 0.1)
            self.play(Create(ce_curve), FadeIn(leg[0]), run_time=self.fit(2))
            self.play(Create(mse_curve), FadeIn(leg[1]), run_time=self.fit(2.5))
            acc_ce = {s: a for s, _, a in LOG_BAD_CE}
            acc_mse = {s: a for s, _, a in LOG_BAD_MSE}
            vline = DashedLine(ax.c2p(STUCK_STEP, 0.2), ax.c2p(STUCK_STEP, 1.0), color=theme.HIGHLIGHT,
                               stroke_width=2)
            d_ce = Dot(ax.c2p(STUCK_STEP, acc_ce[STUCK_STEP]), radius=0.08, color=theme.GRAD)
            d_mse = Dot(ax.c2p(STUCK_STEP, acc_mse[STUCK_STEP]), radius=0.08, color=theme.FG)
            self.play(Create(vline), FadeIn(d_ce), FadeIn(d_mse), run_time=self.fit(1))
            stuck = VGroup(
                zh(f"第 {STUCK_STEP} 步", 22, theme.HIGHLIGHT), zh("交叉熵", 22, theme.GRAD),
                zh("MSE", 22, theme.FG),
                zh("训练准确率", 22, theme.MUTED), zh(f"{acc_ce[STUCK_STEP]:.0%}", 26, theme.GRAD),
                zh(f"{acc_mse[STUCK_STEP]:.0%}", 26, theme.FG),
                zh("p(正确) < 1% 的样本", 22, theme.MUTED), zh(f"{STUCK['ce']} 个", 26, theme.GRAD),
                zh(f"{STUCK['mse']} 个", 26, theme.FG),
            ).arrange_in_grid(rows=3, cols=3, buff=(0.45, 0.32), col_alignments="lcc"
                              ).move_to([4.1, 0.35, 0])
            self.play(FadeIn(stuck), run_time=self.fit(1))
            note = VGroup(zh("MSE 卡多久，对舍入误差极其敏感：", 18, theme.HIGHLIGHT),
                          zh("课程构建机：在 65% 附近卡了七八百步", 18, theme.MUTED),
                          zh(f"渲染本视频的机器：第 {MSE_CATCH} 步就过了 90%", 18, theme.MUTED)
                          ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([4.2, -1.75, 0])
            note.shift(RIGHT * (1.6 - note.get_left()[0]))  # 左边避开横轴的 3,000，右边留出安全边距
            self.wait(self.remaining() * 0.25)
            self.play(FadeIn(note), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [ax, xl, yl, ce_curve, mse_curve, leg, vline, d_ce, d_mse,
                                             stuck, note]],
                      run_time=self.fit(0.6))

        # ── S10 螺旋数据 + 线性分类器 ────────────────────────────────────
        sax = Axes(x_range=[-1.2, 1.2, 0.5], y_range=[-1.2, 1.2, 0.5], x_length=4.7, y_length=4.7,
                   axis_config={"color": theme.MUTED, "stroke_width": 1}, tips=False
                   ).move_to([-3.4, 0.1, 0])
        frame = Rectangle(width=4.7, height=4.7, color=theme.MUTED, stroke_width=1.5).move_to(sax)
        dots = VGroup(*[Dot(sax.c2p(*p), radius=0.035, color=CLASS_COLORS[c])
                        for p, c in zip(X, Y, strict=True)])
        with self.shot("S10"):
            self.play(*self.set_heading("三类螺旋：直线切不开"), Create(frame), run_time=self.fit(0.8))
            self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.005), run_time=self.fit(2))
            info = VGroup(zh("300 个点，3 类，代码里生成", 24, theme.FG),
                          zh("对照：线性 softmax 分类器", 26, theme.MUTED),
                          MathTex(r"z = XW + b", font_size=36)
                          ).arrange(DOWN, buff=0.3).move_to([3.5, 1.3, 0])
            self.play(FadeIn(info[0]), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.2)
            lin_img = region_image(lambda pts: cem.softmax(pts @ W_LIN + B_LIN), sax)
            lin_img.set_z_index(-1)
            self.play(FadeIn(info[1]), FadeIn(info[2]), FadeIn(lin_img), run_time=self.fit(1.5))
            lin_acc = zh(f"准确率 {LIN_ACC:.1%}", 34, theme.GRAD).move_to([3.5, -0.4, 0])
            lin_why = zh("边界都是直线", 24, theme.MUTED).next_to(lin_acc, DOWN, 0.3)
            self.play(FadeIn(lin_acc), FadeIn(lin_why), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(lin_img), FadeOut(info), FadeOut(lin_acc), FadeOut(lin_why),
                      run_time=self.fit(0.6))

        # ── S11 MLP 训练 ─────────────────────────────────────────────────
        with self.shot("S11"):
            arch = MathTex(r"2\to64\ (\mathrm{ReLU})\to3\ \to\ \mathrm{softmax}", font_size=34
                           ).move_to([3.5, 2.2, 0])
            self.play(*self.set_heading("两层 MLP：决策区域长成螺旋"), Write(arch), run_time=self.fit(1))
            loss0, acc0 = LOG_BY_STEP[0]
            table = VGroup(
                zh("步数", 24, theme.MUTED), DecimalNumber(0, num_decimal_places=0, font_size=32),
                zh("交叉熵", 24, theme.GRAD), DecimalNumber(loss0, num_decimal_places=4, font_size=32),
                zh("准确率 %", 24, theme.OUTPUT), DecimalNumber(acc0 * 100, num_decimal_places=1, font_size=32),
            ).arrange_in_grid(rows=3, cols=2, col_alignments="lr", buff=(0.6, 0.3)).move_to([3.5, 0.6, 0])
            img = region_image(mlp_probs(SNAPS[0]), sax)
            img.set_z_index(-1)
            self.play(FadeIn(table), FadeIn(img), run_time=self.fit(1))
            ln3 = zh(f"初始 ≈ ln 3 = {np.log(3):.4f}", 22, theme.HIGHLIGHT).move_to([3.5, -0.8, 0])
            self.play(FadeIn(ln3), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.3)
            steps = SNAP_STEPS[1:]
            per = (self.remaining() - 3.5) / len(steps)
            for s in steps:
                new = region_image(mlp_probs(SNAPS[s]), sax)
                new.set_z_index(-1)
                loss, acc = LOG_BY_STEP[s]
                self.play(FadeOut(img), FadeIn(new), ChangeDecimalToValue(table[1], s),
                          ChangeDecimalToValue(table[3], loss), ChangeDecimalToValue(table[5], acc * 100),
                          run_time=self.fit(min(per, 1.5)))
                img = new
                self.wait(max(0.05, per - 1.5))
            test = zh(f"另一份测试数据：{TEST_ACC:.1%}", 26, theme.OUTPUT).move_to([3.5, -1.6, 0])
            self.play(FadeIn(test), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [img, dots, frame, arch, table, ln3, test]],
                      run_time=self.fit(0.6))

        # ── S12 温度 ─────────────────────────────────────────────────────
        with self.shot("S12"):
            self.play(*self.set_heading("温度 T：logits 先除以 T"), run_time=self.fit(0.8))
            cx, by = -3.0, -1.2
            b = bars(smx.softmax(LOGITS, 1.0), NAMES, CLASS_COLORS, cx, by, 3.0, label_y=-1.6)
            t_lbl = MathTex(r"T=", font_size=44)
            t_val = DecimalNumber(1.0, num_decimal_places=1, font_size=44)
            t_grp = VGroup(t_lbl, t_val).arrange(RIGHT, buff=0.15).move_to([cx, 2.2, 0])
            formula = MathTex(r"p=\mathrm{softmax}(z/T)", font_size=44).move_to([3.5, 1.5, 0])
            self.play(FadeIn(b), FadeIn(t_grp), Write(formula), run_time=self.fit(1.2))
            notes = VGroup(zh("T 小：差距放大，更尖", 26, theme.FG),
                           zh("T 大：差距压小，更平", 26, theme.FG),
                           zh("第 10 章采样会用到", 24, theme.HIGHLIGHT)
                           ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.5, -0.2, 0])
            self.wait(self.remaining() * 0.12)
            temps = [0.5, 10.0, 1.0]
            per = (self.remaining() - 2) / len(temps)
            for i, t in enumerate(temps):
                nb = bars(smx.softmax(LOGITS, t), NAMES, CLASS_COLORS, cx, by, 3.0, label_y=-1.6)
                anims = [Transform(b, nb), ChangeDecimalToValue(t_val, t)]
                if i < 2:
                    anims.append(FadeIn(notes[i]))
                self.play(*anims, run_time=self.fit(1.5))
                self.wait(max(0.05, per - 1.5))
            self.play(FadeIn(notes[2]), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [b, t_grp, formula, notes]], run_time=self.fit(0.6))

        # ── S13 语言模型 = 词表上的分类 ──────────────────────────────────
        with self.shot("S13"):
            self.play(*self.set_heading("语言模型 = 在词表上做分类"), run_time=self.fit(0.8))
            chars = VGroup()
            for ch in "看前面的字，猜下一个字":
                r = RoundedRectangle(width=0.55, height=0.6, corner_radius=0.06, color=theme.INPUT)
                chars.add(VGroup(r, zh(ch, 26, theme.FG).move_to(r)))
            chars.arrange(RIGHT, buff=0.08).move_to([-3.2, 2.2, 0])
            self.play(LaggedStart(*[FadeIn(c) for c in chars], lag_ratio=0.08), run_time=self.fit(1.5))
            vs = zh(f"类别数 = 词表大小 V（本例 V = {V}）", 24, theme.MUTED).next_to(chars, DOWN, 0.3)
            self.play(FadeIn(vs), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.12)
            fen = VGroup(RoundedRectangle(width=0.7, height=0.7, corner_radius=0.08, color=theme.INPUT),
                         zh("分", 32, theme.FG)).move_to([-5.9, -0.5, 0])
            fen[1].move_to(fen[0])
            arr = Arrow(fen.get_right(), fen.get_right() + RIGHT * 0.9, color=theme.MUTED, buff=0.1)
            top_vals = [float(P_FEN[i]) for i in TOP_FEN]
            top_names = [VOCAB[i] for i in TOP_FEN]
            nb = bars(top_vals, top_names, [theme.OUTPUT] * 3, -2.9, -1.6, 2.2, width=0.6, gap=1.1,
                      label_y=-2.0)
            cap = zh("下一个字的概率", 22, theme.OUTPUT).move_to([-2.9, 0.55, 0])
            self.play(FadeIn(fen), GrowArrow(arr), FadeIn(nb), FadeIn(cap), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.15)
            b0, p0 = lm.to_bits_and_ppl(LM_START)
            b1, p1 = lm.to_bits_and_ppl(LM_END)
            hdr = [zh(t, 22, theme.MUTED) for t in ["", "nats", "bits", "困惑度"]]
            r0 = [zh("第 0 步", 22, theme.FG)] + [mono(f"{v:.2f}", 24) for v in (LM_START, b0, p0)]
            r1 = [zh(f"第 {LOG_LM[-1][0]} 步", 22, theme.FG)] + [mono(f"{v:.2f}", 24, theme.OUTPUT)
                                                                for v in (LM_END, b1, p1)]
            tbl = VGroup(*hdr, *r0, *r1).arrange_in_grid(rows=3, cols=4, buff=(0.45, 0.28)
                                                         ).move_to([3.6, 0.2, 0])
            self.play(FadeIn(tbl), run_time=self.fit(1))
            conv = zh("bits = nats / ln 2　　困惑度 = e^nats", 22, theme.MUTED).next_to(tbl, DOWN, 0.35)
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(conv), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.45)
            gpt = zh(f"GPT-2（V = 50257）：初始 ≈ ln V = {np.log(50257):.2f}", 22, theme.HIGHLIGHT
                     ).next_to(conv, DOWN, 0.4)
            self.play(FadeIn(gpt), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [chars, vs, fen, arr, nb, cap, tbl, conv, gpt]],
                      run_time=self.fit(0.6))

        # ── S14 从极简到生产级 ───────────────────────────────────────────
        with self.shot("S14"):
            left_code = code_block("""
z = z - z.max(-1, keepdims=True)
logp = z - log(exp(z).sum(-1))
loss = -logp[range(N), y].mean()
""", 20, line_buff=0.22).move_to([-3.4, 1.0, 0])
            right_code = code_block("""
logits = model(x)
loss = F.cross_entropy(logits, y)
loss.backward()
""", 20, line_buff=0.22).move_to([3.5, 1.0, 0])
            lt = zh("手写（NumPy）", 24, theme.MUTED).next_to(left_code, UP, 0.35)
            rt = zh("PyTorch 标准写法", 24, theme.HIGHLIGHT).next_to(right_code, UP, 0.35)
            self.play(*self.set_heading("从极简到生产级：F.cross_entropy"), FadeIn(lt), FadeIn(left_code),
                      FadeIn(rt), FadeIn(right_code), run_time=self.fit(1.2))
            hl = SurroundingRectangle(right_code[1], color=theme.HIGHLIGHT, buff=0.08)
            tag = zh("吃 logits，内部融合 log-softmax", 22, theme.HIGHLIGHT).next_to(right_code, DOWN, 0.3)
            self.wait(self.remaining() * 0.12)
            self.play(Create(hl), FadeIn(tag), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.25)
            chk = VGroup(zh("[1000, 500, −500]：朴素写法 inf，F.cross_entropy 1500.0", 22, theme.FG),
                         zh("随机 8 样本 × 5 类：3.1764553511 = 3.1764553511", 22, theme.OUTPUT)
                         ).arrange(DOWN, buff=0.22).move_to([0, -1.0, 0])
            self.play(FadeIn(chk[0]), run_time=self.fit(0.8))
            self.play(FadeIn(chk[1]), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.3)
            shape = mono("logits (B, T, V) → view(-1, V)", 22, theme.ATTN).move_to([0, -2.1, 0])
            self.play(FadeIn(shape), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [left_code, right_code, lt, rt, hl, tag, chk, shape]],
                      run_time=self.fit(0.6))

        # ── S15 小结 ─────────────────────────────────────────────────────
        with self.shot("S15"):
            steps = [("logits", theme.PARAM), ("softmax", theme.FG), ("概率 p", theme.OUTPUT),
                     ("−log p_y", theme.GRAD), ("p − onehot", theme.GRAD)]
            boxes = VGroup()
            for name, color in steps:
                r = RoundedRectangle(width=2.25, height=0.9, corner_radius=0.12, color=color)
                boxes.add(VGroup(r, zh(name, 24, color).move_to(r)))
            boxes.arrange(RIGHT, buff=0.42).move_to([0, 1.5, 0])
            arrows = VGroup(*[Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.06,
                                    color=theme.MUTED, max_tip_length_to_length_ratio=0.35)
                              for i in range(4)])
            subs = VGroup(zh("分数", 20, theme.MUTED), zh("指数 + 归一", 20, theme.MUTED),
                          zh("和为 1", 20, theme.MUTED), zh("交叉熵", 20, theme.MUTED),
                          zh("梯度", 20, theme.MUTED))
            for s, bx in zip(subs, boxes, strict=True):
                s.next_to(bx, DOWN, 0.2)
            self.play(*self.set_heading("本章小结"),
                      LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.3), run_time=self.fit(2))
            self.play(*[GrowArrow(a) for a in arrows], FadeIn(subs), run_time=self.fit(1))
            self.wait(self.remaining() * 0.35)
            lmline = zh("语言模型 = 在词表上做这件事", 30, theme.HIGHLIGHT).move_to([0, -0.4, 0])
            self.play(FadeIn(lmline), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.5)
            nxt = zh("下一章：让训练稳定", 34, theme.HIGHLIGHT).move_to([0, -1.9, 0])
            nbox = Rectangle(width=nxt.width + 0.6, height=nxt.height + 0.3, color=theme.HIGHLIGHT)
            nbox.move_to(nxt)
            self.play(FadeIn(nxt), Create(nbox), run_time=self.fit(1))
            self.wait(self.remaining() - 1.0)
            self.play(*[FadeOut(m) for m in [boxes, arrows, subs, lmline, nxt, nbox]],
                      *self.set_heading(None), run_time=self.fit(1.0))
