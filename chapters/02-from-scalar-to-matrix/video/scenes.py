"""Chapter 2 video: from scalars to matrices — y = XW + b.

The code in ../code/ calculates all numbers in the frames (see the fact list F1–F17 in script.md).
The timings in S11 come from a run of 05_loop_vs_vectorized.py during the render.
Render: bash chapters/02-from-scalar-to-matrix/video/build.sh
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
    Brace,
    ChangeDecimalToValue,
    Create,
    DecimalNumber,
    Dot,
    FadeIn,
    FadeOut,
    GrowArrow,
    GrowFromEdge,
    LaggedStart,
    MathTex,
    Matrix,
    Rectangle,
    RoundedRectangle,
    SurroundingRectangle,
    Text,
    Transform,
    TransformFromCopy,
    VGroup,
    VMobject,
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


mm = _load("matrix_multiply", "02_matrix_multiply.py")
lin = _load("linear_layer", "03_linear_layer.py")
reg = _load("multivariate_regression", "04_multivariate_regression.py")
bench = _load("loop_vs_vectorized", "05_loop_vs_vectorized.py")
pt = _load("pytorch_version", "06_pytorch_version.py")

# ── F1/F2: house vectors and the dot product ─────────────────────────────
HOUSES = lin.HOUSES                         # (4, 3)
W_DEMO = reg.W_TRUE.ravel()                 # [0.8, 5, -3]
B_DEMO = reg.B_TRUE                         # 20
X0 = HOUSES[0]
PRODS = X0 * W_DEMO                         # [64, 10, -15]
DOT0 = float(W_DEMO @ X0)                   # 59
PREDS = HOUSES @ W_DEMO + B_DEMO            # [79, 125, 49, 104.5]

# ── F3: matrix multiplication by hand ─────────────────────────────────────
A_MM = [[1, 2, 3], [4, 5, 6]]
B_MM = [[7, 8], [9, 10], [11, 12]]
C_MM = mm.matmul(A_MM, B_MM)

# ── F6: broadcasting example ──────────────────────────────────────────────
W1, B1, W2, B2 = lin.init_params()
XW1 = HOUSES @ W1                           # (4, 2)
H1 = lin.linear(HOUSES, W1, B1)
W12 = W1 @ W2

# ── F8–F11: multiple linear regression ────────────────────────────────────
XR, YR = reg.make_data()
XS, MU, SIGMA = reg.standardize(XR)
LR_RAW, LR_STD = reg.critical_lr(XR), reg.critical_lr(XS)
HIST = reg.gradient_descent(XS, YR, lr=0.1, steps=200)
LOSSES = [h[2] for h in HIST]

# ── F12/F13: timings measured during the render ───────────────────────────
BENCH = bench.run_benchmarks()

# ── F16: parity check with PyTorch ────────────────────────────────────────
import torch  # noqa: E402

_W_N, _B_N, _ = HIST[-1]
_m64 = pt.train_torch(XS, YR, dtype=torch.float64)
_m32 = pt.train_torch(XS, YR, dtype=torch.float32)


def _diff(m) -> float:
    return max(np.abs(m.weight.detach().numpy().T - _W_N).max(),
               np.abs(m.bias.detach().numpy() - _B_N).max())


DIFF64, DIFF32 = _diff(_m64), _diff(_m32)

MONO = "Noto Sans Mono"


def mono(text: str, size: float = 24, color: str = theme.FG) -> Text:
    return Text(text, font=MONO, font_size=size, color=color)


def fmt(v: float) -> str:
    return f"{v:g}"


def mat(rows, color=theme.FG, fs=36, h_buff=1.1, v_buff=0.7) -> Matrix:
    m = Matrix([[fmt(v) for v in r] for r in rows], h_buff=h_buff, v_buff=v_buff,
               element_to_mobject_config={"font_size": fs})
    m.get_entries().set_color(color)
    return m


def block(w: float, h: float, color: str, label: str) -> VGroup:
    r = Rectangle(width=w, height=h, color=color, fill_color=color, fill_opacity=0.18)
    t = MathTex(label, font_size=30, color=color).move_to(r)
    return VGroup(r, t)


class ChapterScene(NarratedScene):
    chapter_label = "第 2 章"
    chapter_title = "从标量到矩阵"

    def construct(self) -> None:
        # ── S01 Opening ──────────────────────────────────────────────────
        with self.shot("S01"):
            card = self.chapter_card()
            sub = MathTex(r"\hat{y}=ax+b\ \longrightarrow\ Y=XW+b", font_size=44,
                          color=theme.HIGHLIGHT).next_to(card, DOWN, 0.6)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.wait(max(0.05, self.remaining() - 0.8))
            self.play(FadeOut(card), FadeOut(sub), run_time=self.fit(0.8))

        # ── S02 One input is not sufficient ──────────────────────────────
        with self.shot("S02"):
            old = MathTex(r"\hat{y}", "=", "a", "x", "+", "b", font_size=56).move_to([-3.6, 1.3, 0])
            old[2].set_color(theme.PARAM)
            old[5].set_color(theme.PARAM)
            old[3].set_color(theme.INPUT)
            old_lbl = zh("第 1 章：只看面积", 26, theme.MUTED).next_to(old, DOWN, 0.35)
            self.play(*self.set_heading("一个输入不够"), FadeIn(old), FadeIn(old_lbl),
                      run_time=self.fit(1.2))
            feats = VGroup(zh("面积　80 ㎡", 28, theme.INPUT), zh("卧室　2 间", 28, theme.INPUT),
                           zh("距市中心　5 km", 28, theme.INPUT)).arrange(DOWN, aligned_edge=LEFT,
                                                                         buff=0.3)
            house = VGroup(RoundedRectangle(width=feats.width + 0.8, height=feats.height + 0.7,
                                            corner_radius=0.15, color=theme.INPUT), feats)
            feats.move_to(house[0])
            house.move_to([3.4, 1.0, 0])
            house_t = zh("一套房子", 24, theme.MUTED).next_to(house, UP, 0.15)
            self.wait(self.remaining() * 0.12)
            self.play(FadeIn(house_t), Create(house[0]), run_time=self.fit(0.8))
            self.play(LaggedStart(*[FadeIn(f, shift=RIGHT * 0.2) for f in feats], lag_ratio=0.5),
                      run_time=self.fit(2))
            self.wait(self.remaining() * 0.3)
            new = MathTex(r"\hat{y}", "=", "w_1", "x_1", "+", "w_2", "x_2", "+", "w_3", "x_3",
                          "+", "b", font_size=54).move_to([0, -1.55, 0])
            for i in (2, 5, 8, 11):
                new[i].set_color(theme.PARAM)
            for i in (3, 6, 9):
                new[i].set_color(theme.INPUT)
            arrow = Arrow([0, -0.2, 0], [0, -1.05, 0], color=theme.MUTED, buff=0)
            self.play(GrowArrow(arrow), run_time=self.fit(0.5))
            self.play(Write(new), run_time=self.fit(2.5))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(*[FadeOut(m) for m in [old, old_lbl, house, house_t, arrow, new]],
                      run_time=self.fit(0.6))

        # ── S03 Vectors and the dot product ──────────────────────────────
        with self.shot("S03"):
            xv = mat([[v] for v in X0], theme.INPUT, fs=40, v_buff=0.9).move_to([-5.2, 0.4, 0])
            wv = mat([[v] for v in W_DEMO], theme.PARAM, fs=40, v_buff=0.9).move_to([-3.4, 0.4, 0])
            xl = MathTex("x", font_size=40, color=theme.INPUT).next_to(xv, UP, 0.2)
            wl = MathTex("w", font_size=40, color=theme.PARAM).next_to(wv, UP, 0.2)
            self.play(*self.set_heading("向量与点积"), FadeIn(xv), FadeIn(xl),
                      run_time=self.fit(1))
            self.play(FadeIn(wv), FadeIn(wl), run_time=self.fit(1))
            self.wait(self.remaining() * 0.18)
            prods = VGroup()
            for i in range(3):
                y = xv.get_entries()[i].get_center()[1]
                p = MathTex(fmt(X0[i]), r"\times", fmt(W_DEMO[i]), "=", fmt(PRODS[i]),
                            font_size=38).move_to([-0.4, y, 0])
                p[0].set_color(theme.INPUT)
                p[2].set_color(theme.PARAM)
                prods.add(p)
            for p in prods:
                self.play(FadeIn(p, shift=RIGHT * 0.3), run_time=self.fit(1.2, reserve=4))
                self.wait(min(0.8, max(0.1, self.remaining() - 6)))
            summ = VGroup(
                MathTex(r"\Sigma", "=", fmt(DOT0), font_size=42),
                MathTex("+b", "=", fmt(B_DEMO), font_size=42),
                MathTex(r"\hat{y}", "=", fmt(DOT0 + B_DEMO), font_size=46, color=theme.OUTPUT),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([3.9, 0.4, 0])
            summ[1][0].set_color(theme.PARAM)
            brace = Brace(prods, RIGHT, color=theme.MUTED)
            self.play(Create(brace), FadeIn(summ[0]), run_time=self.fit(1))
            self.play(FadeIn(summ[1]), run_time=self.fit(0.8))
            self.play(FadeIn(summ[2]), run_time=self.fit(0.8))
            formula = MathTex(r"w\cdot x", "=", r"w_1x_1+w_2x_2+w_3x_3", font_size=40,
                              color=theme.HIGHLIGHT).move_to([0, -2.1, 0])
            self.play(Write(formula), run_time=self.fit(1.5))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(*[FadeOut(m) for m in [xv, wv, xl, wl, prods, summ, brace, formula]],
                      run_time=self.fit(0.6))

        # ── S04 A batch of samples: a matrix ─────────────────────────────
        with self.shot("S04"):
            Xm = mat(HOUSES, theme.INPUT, fs=34, h_buff=1.15, v_buff=0.62)
            at = MathTex("@", font_size=44)
            Wm = mat([[v] for v in W_DEMO], theme.PARAM, fs=34, v_buff=0.62)
            plus = MathTex("+", str(int(B_DEMO)), font_size=44)
            plus[1].set_color(theme.PARAM)
            eq = MathTex("=", font_size=44)
            Ym = mat([[v] for v in PREDS], theme.OUTPUT, fs=34, h_buff=1.4, v_buff=0.62)
            row = VGroup(Xm, at, Wm, plus, eq, Ym).arrange(RIGHT, buff=0.3).move_to([0, 0.5, 0])
            labels = VGroup(
                MathTex(r"X\ (4,3)", font_size=30, color=theme.INPUT).next_to(Xm, DOWN, 0.25),
                MathTex(r"W\ (3,1)", font_size=30, color=theme.PARAM).next_to(Wm, DOWN, 0.25),
                MathTex(r"\hat{y}\ (4,1)", font_size=30, color=theme.OUTPUT).next_to(Ym, DOWN, 0.25),
            )
            self.play(*self.set_heading("一批样本摞起来：矩阵"), run_time=self.fit(0.6))
            rows = Xm.get_rows()
            self.play(FadeIn(Xm.get_brackets()),
                      LaggedStart(*[FadeIn(r, shift=DOWN * 0.3) for r in rows], lag_ratio=0.3),
                      run_time=self.fit(2.5))
            row_note = zh("每一行 = 一套房子（一个样本）", 24, theme.MUTED).next_to(Xm, UP, 0.3)
            self.play(FadeIn(row_note), FadeIn(labels[0]), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.25)
            self.play(FadeIn(at), FadeIn(Wm), FadeIn(labels[1]), FadeIn(plus),
                      run_time=self.fit(1))
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(eq), FadeIn(Ym.get_brackets()), FadeIn(labels[2]),
                      run_time=self.fit(0.6))
            per = max(0.3, (self.remaining() - 2.5) / 4)
            ents = Ym.get_entries()
            for i in range(4):
                rect = SurroundingRectangle(rows[i], color=theme.HIGHLIGHT, buff=0.08)
                self.play(Create(rect), FadeIn(ents[i]), run_time=self.fit(min(per * 0.6, 0.8)))
                self.play(FadeOut(rect), run_time=self.fit(min(per * 0.4, 0.4)))
            note = zh("一次矩阵乘法 = 所有样本的点积", 28, theme.HIGHLIGHT).move_to([0, -2.2, 0])
            self.play(FadeIn(note), run_time=self.fit(0.6))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(*[FadeOut(m) for m in [row, labels, row_note, note]], run_time=self.fit(0.6))

        # ── S05 How to calculate a matrix multiplication ─────────────────
        with self.shot("S05"):
            Am = mat(A_MM, theme.INPUT, fs=38, h_buff=0.9)
            Bm = mat(B_MM, theme.PARAM, fs=38, h_buff=0.9)
            Cm = mat(C_MM, theme.OUTPUT, fs=38, h_buff=1.2)
            grp = VGroup(Am, MathTex("@", font_size=44), Bm, MathTex("=", font_size=44), Cm)
            grp.arrange(RIGHT, buff=0.35).move_to([0, 0.6, 0])
            for e in Cm.get_entries():
                e.set_opacity(0)
            names = VGroup(MathTex(r"A\ (2,3)", font_size=28, color=theme.INPUT).next_to(Am, UP, 0.2),
                           MathTex(r"B\ (3,2)", font_size=28, color=theme.PARAM).next_to(Bm, UP, 0.2),
                           MathTex(r"C\ (2,2)", font_size=28, color=theme.OUTPUT).next_to(Cm, UP, 0.2))
            self.play(*self.set_heading("矩阵乘法：行 · 列"), FadeIn(grp), FadeIn(names),
                      run_time=self.fit(1.2))
            rule = zh("C[i][j] = A 的第 i 行 · B 的第 j 列", 28, theme.HIGHLIGHT).move_to([0, -1.2, 0])
            self.wait(self.remaining() * 0.12)
            self.play(FadeIn(rule), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.3)
            calc = None
            order = [(0, 0), (0, 1), (1, 0), (1, 1)]
            for idx, (i, j) in enumerate(order):
                ra = SurroundingRectangle(Am.get_rows()[i], color=theme.HIGHLIGHT, buff=0.1)
                cb = SurroundingRectangle(Bm.get_columns()[j], color=theme.HIGHLIGHT, buff=0.1)
                terms = "+".join(rf"{A_MM[i][p]}\times{B_MM[p][j]}" for p in range(3))
                new_calc = MathTex(terms + "=" + fmt(C_MM[i][j]), font_size=38).move_to([0, -2.05, 0])
                slow = idx == 0
                anims = [Create(ra), Create(cb)]
                anims.append(FadeIn(new_calc) if calc is None else Transform(calc, new_calc))
                self.play(*anims, run_time=self.fit(1.0 if slow else 0.6, reserve=1))
                if calc is None:
                    calc = new_calc
                if slow:
                    self.wait(max(0.1, (self.remaining() - 6) * 0.5))
                self.play(Cm.get_entries()[i * 2 + j].animate.set_opacity(1),
                          run_time=self.fit(0.6, reserve=0.5))
                self.wait(min(0.6, max(0.05, self.remaining() - 3)))
                self.play(FadeOut(ra), FadeOut(cb), run_time=self.fit(0.3, reserve=0.3))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(*[FadeOut(m) for m in [grp, names, rule, calc]], run_time=self.fit(0.6))

        # ── S06 The shape rule ───────────────────────────────────────────
        with self.shot("S06"):
            rule = MathTex("(m,", "k", r")\ @\ (", "k", r",n)\ \to\ (m,n)", font_size=54)
            rule.move_to([0, 2.15, 0])
            rule[1].set_color(theme.HIGHLIGHT)
            rule[3].set_color(theme.HIGHLIGHT)
            self.play(*self.set_heading("形状规则"), Write(rule), run_time=self.fit(1.5))
            u = 0.45
            bA = block(3 * u, 4 * u, theme.INPUT, r"m\times k")
            bB = block(2 * u, 3 * u, theme.PARAM, r"k\times n")
            bC = block(2 * u, 4 * u, theme.OUTPUT, r"m\times n")
            ops = [MathTex("@", font_size=40), MathTex(r"\to", font_size=40)]
            blocks = VGroup(bA, ops[0], bB, ops[1], bC).arrange(RIGHT, buff=0.6)
            blocks.move_to([-3.2, 0.3, 0])
            VGroup(bA, ops[0]).shift(LEFT * 0.5)   # make space for the brace and k on the left of B
            kA = Brace(bA[0], DOWN, color=theme.HIGHLIGHT)
            kA_t = MathTex("k", font_size=32, color=theme.HIGHLIGHT).next_to(kA, DOWN, 0.1)
            kB = Brace(bB[0], LEFT, color=theme.HIGHLIGHT)
            kB_t = MathTex("k", font_size=32, color=theme.HIGHLIGHT).next_to(kB, LEFT, 0.1)
            self.play(FadeIn(bA), FadeIn(ops[0]), FadeIn(bB), run_time=self.fit(1))
            self.play(Create(kA), FadeIn(kA_t), Create(kB), FadeIn(kB_t), run_time=self.fit(1))
            same = zh("两个 k 必须相等：点积的两个向量一样长", 24, theme.HIGHLIGHT).move_to([-3.6, -2.1, 0])
            self.play(FadeIn(same), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(ops[1]), FadeIn(bC), run_time=self.fit(1))
            gone = zh("k 被消掉", 24, theme.OUTPUT).next_to(bC, UP, 0.2)
            self.play(FadeIn(gone), run_time=self.fit(0.6))
            ex = VGroup(
                VGroup(MathTex(r"(4,3)\,@\,(3,1)\to(4,1)", font_size=34), zh("可以", 24, theme.OUTPUT)),
                VGroup(MathTex(r"(32,128)\,@\,(128,64)\to(32,64)", font_size=34),
                       zh("可以", 24, theme.OUTPUT)),
                VGroup(MathTex(r"(3,2)\,@\,(3,5)", font_size=34), zh("报错：2 ≠ 3", 24, theme.GRAD)),
            )
            for e in ex:
                e.arrange(RIGHT, buff=0.3)
            ex.arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([3.6, 0.0, 0])
            self.wait(self.remaining() * 0.1)
            for e in ex:
                self.play(FadeIn(e, shift=LEFT * 0.2), run_time=self.fit(0.8, reserve=2))
                self.wait(min(1.6, max(0.1, self.remaining() - 5)))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(*[FadeOut(m) for m in [rule, blocks, kA, kA_t, kB, kB_t, same, gone, ex]],
                      run_time=self.fit(0.6))

        # ── S07 Each letter in y = XW + b ────────────────────────────────
        with self.shot("S07"):
            f = MathTex("Y", "=", "X", "W", "+", "b", font_size=80).move_to([0, 1.5, 0])
            f[0].set_color(theme.OUTPUT)
            f[2].set_color(theme.INPUT)
            f[3].set_color(theme.PARAM)
            f[5].set_color(theme.PARAM)
            self.play(*self.set_heading("Y = XW + b 的形状"), Write(f), run_time=self.fit(1.2))
            shapes = VGroup(
                MathTex("(N,n)", font_size=32, color=theme.OUTPUT).next_to(f[0], DOWN, 0.3),
                MathTex("(N,k)", font_size=32, color=theme.INPUT).next_to(f[2], DOWN, 0.3).shift(LEFT * 0.35),
                MathTex("(k,n)", font_size=32, color=theme.PARAM).next_to(f[3], DOWN, 0.3).shift(RIGHT * 0.35),
                MathTex("(n,)", font_size=32, color=theme.PARAM).next_to(f[5], DOWN, 0.3),
            )
            notes = VGroup(
                zh("N：一批有多少个样本 —— 批量维（batch）", 26, theme.INPUT),
                zh("k：输入特征数　　n：输出个数（W 的每一列算一个输出）", 26, theme.PARAM),
                zh("W 的形状和 N 无关：4 个样本还是 1 万个，都用同一个 W", 26, theme.HIGHLIGHT),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([0, -1.1, 0])
            self.wait(self.remaining() * 0.1)
            self.play(FadeIn(shapes[1]), run_time=self.fit(0.8))
            self.play(FadeIn(notes[0]), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.25)
            self.play(FadeIn(shapes[2]), FadeIn(notes[1]), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(shapes[0]), FadeIn(shapes[3]), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.25)
            self.play(FadeIn(notes[2]), run_time=self.fit(0.8))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(FadeOut(f), FadeOut(shapes), FadeOut(notes), run_time=self.fit(0.6))

        # ── S08 Broadcasting ─────────────────────────────────────────────
        with self.shot("S08"):
            xw = mat(np.round(XW1, 2), theme.INPUT, fs=30, h_buff=1.5, v_buff=0.6)
            bb = mat([B1] * 4, theme.PARAM, fs=30, h_buff=1.0, v_buff=0.6)
            hh = mat(np.round(H1, 2), theme.OUTPUT, fs=30, h_buff=1.5, v_buff=0.6)
            grp = VGroup(xw, MathTex("+", font_size=40), bb, MathTex("=", font_size=40), hh)
            grp.arrange(RIGHT, buff=0.3).move_to([0, 0.35, 0])
            lbls = VGroup(
                MathTex(r"XW\ (4,2)", font_size=28, color=theme.INPUT).next_to(xw, UP, 0.2),
                MathTex(r"b\ (2,)", font_size=28, color=theme.PARAM).next_to(bb, UP, 0.2),
                MathTex(r"(4,2)", font_size=28, color=theme.OUTPUT).next_to(hh, UP, 0.2),
            )
            self.play(*self.set_heading("广播：b 加到每一行"), FadeIn(xw), FadeIn(lbls[0]),
                      run_time=self.fit(1))
            b_rows = bb.get_rows()
            self.play(FadeIn(grp[1]), FadeIn(b_rows[0]), FadeIn(lbls[1]), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.2)
            copies = [TransformFromCopy(b_rows[0], b_rows[i]) for i in range(1, 4)]
            self.play(FadeIn(bb.get_brackets()), LaggedStart(*copies, lag_ratio=0.3),
                      run_time=self.fit(2))
            virt = zh("像复制了 N 份，但内存里没有真的复制", 22, theme.MUTED).next_to(bb, DOWN, 0.25)
            self.play(FadeIn(virt), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.15)
            self.play(FadeIn(grp[3]), FadeIn(hh), FadeIn(lbls[2]), run_time=self.fit(1))
            rule = zh("规则：从最后一维对齐，长度相等、或其中一个是 1", 26, theme.HIGHLIGHT)
            rule.move_to([0, -1.85, 0])
            self.wait(self.remaining() * 0.15)
            self.play(FadeIn(rule), run_time=self.fit(0.8))
            err = MathTex(r"(4,2)+(3,)\ \to", font_size=32).move_to([-0.9, -2.45, 0])
            err_t = zh("报错", 26, theme.GRAD).next_to(err, RIGHT, 0.2)
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(err), FadeIn(err_t), run_time=self.fit(0.8))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(*[FadeOut(m) for m in [grp, lbls, virt, rule, err, err_t]],
                      run_time=self.fit(0.6))

        # ── S09 The gradient is also a matrix ────────────────────────────
        with self.shot("S09"):
            g1 = MathTex(r"\frac{\partial L}{\partial a}=\frac{2}{N}\sum_i(\hat{y}_i-y_i)\,x_i",
                         font_size=40, color=theme.MUTED).move_to([0.8, 1.9, 0])
            g1_l = zh("第 1 章", 24, theme.MUTED).next_to(g1, LEFT, 0.4)
            self.play(*self.set_heading("梯度也写成矩阵"), FadeIn(g1), FadeIn(g1_l),
                      run_time=self.fit(1))
            self.wait(self.remaining() * 0.3)
            g2 = MathTex(r"\frac{\partial L}{\partial W}", "=", r"\frac{2}{N}", "X^{T}",
                         r"(\hat{y}-y)", font_size=60).move_to([0.8, 0.3, 0])
            g2[0].set_color(theme.GRAD)
            g2[3].set_color(theme.INPUT)
            g2[4].set_color(theme.GRAD)
            g2_l = zh("本章", 24, theme.HIGHLIGHT).next_to(g2, LEFT, 0.4)
            self.play(Write(g2), FadeIn(g2_l), run_time=self.fit(2))
            box = SurroundingRectangle(g2, color=theme.HIGHLIGHT, buff=0.15)
            self.play(Create(box), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.2)
            shp = MathTex(r"\underbrace{(k,N)}_{X^{T}}", r"\ @\ ", r"\underbrace{(N,1)}_{\hat{y}-y}",
                          r"\ \to\ ", r"\underbrace{(k,1)}_{\phantom{W}}", font_size=42)
            shp.move_to([0, -1.35, 0])
            shp[0].set_color(theme.INPUT)
            shp[2].set_color(theme.GRAD)
            shp[4].set_color(theme.PARAM)
            self.play(FadeIn(shp[0]), FadeIn(shp[1]), FadeIn(shp[2]), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.2)
            same_w = zh("与 W 同形状", 22, theme.PARAM).next_to(shp[4], DOWN, 0.0)
            self.play(FadeIn(shp[3]), FadeIn(shp[4]), FadeIn(same_w), run_time=self.fit(1))
            same = zh("梯度和参数永远同形状", 28, theme.HIGHLIGHT).move_to([0, -2.4, 0])
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(same), run_time=self.fit(0.8))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(*[FadeOut(m) for m in [g1, g1_l, g2, g2_l, box, shp, same_w, same]],
                      run_time=self.fit(0.6))

        # ── S10 Train a multiple linear regression ───────────────────────
        with self.shot("S10"):
            setup = zh("200 套房子 · 学习率 0.1 · 从全 0 出发", 24, theme.MUTED).move_to([0, 2.5, 0])
            self.play(*self.set_heading("训练：多元线性回归"), FadeIn(setup), run_time=self.fit(1))
            std = VGroup(
                zh("先标准化：每个特征 (x − 均值) / 标准差", 26, theme.FG),
                zh(f"临界学习率：原始特征 {LR_RAW:.2e}　→　标准化后 {LR_STD:.3f}", 26,
                   theme.HIGHLIGHT),
            ).arrange(DOWN, buff=0.35).move_to([0, 0.6, 0])
            self.wait(self.remaining() * 0.12)
            self.play(FadeIn(std[0]), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.12)
            self.play(FadeIn(std[1]), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.12)
            self.play(FadeOut(std), run_time=self.fit(0.5))

            ax = Axes(x_range=[0, 50, 10], y_range=[1, 4, 1], x_length=5.4, y_length=3.6,
                      axis_config={"color": theme.MUTED, "font_size": 20}, tips=False,
                      x_axis_config={"include_numbers": True}).move_to([-3.3, 0.3, 0])
            ylabels = VGroup(*[MathTex(f"10^{k}", font_size=22, color=theme.MUTED).next_to(
                ax.c2p(0, k), LEFT, 0.12) for k in (1, 2, 3, 4)])
            xlab = zh("步数", 20, theme.MUTED).next_to(ax.x_axis, RIGHT, 0.2)
            ylab = zh("损失", 20, theme.GRAD).next_to(ax, UP, 0.12).align_to(ax, LEFT)

            names = ["面积", "卧室", "距离", "b"]
            true_vals = list(reg.W_TRUE.ravel()) + [reg.B_TRUE]
            head = VGroup(zh("", 22), zh("学到", 22, theme.PARAM), zh("真实", 22, theme.MUTED))
            cells = [head[0], head[1], head[2]]
            decs = []
            for nm, tv in zip(names, true_vals, strict=True):
                d = DecimalNumber(0, num_decimal_places=3, font_size=30, color=theme.PARAM)
                decs.append(d)
                cells += [zh(nm, 24, theme.FG), d, MathTex(fmt(tv), font_size=30, color=theme.MUTED)]
            table = VGroup(*cells).arrange_in_grid(rows=5, cols=3, buff=(0.6, 0.3),
                                                   col_alignments="lrr")
            table.move_to([3.4, 0.55, 0])
            step_lbl = VGroup(zh("步数", 22, theme.MUTED),
                              DecimalNumber(0, num_decimal_places=0, font_size=28),
                              zh("损失", 22, theme.GRAD),
                              DecimalNumber(LOSSES[0], num_decimal_places=1, font_size=28))
            step_lbl[0].next_to(table, DOWN, 0.45).align_to(table, LEFT)
            step_lbl[1].next_to(step_lbl[0], RIGHT, 0.3)
            step_lbl[2].next_to(step_lbl[0], DOWN, 0.3).align_to(table, LEFT)
            step_lbl[3].next_to(step_lbl[2], RIGHT, 0.3)
            self.play(Create(ax), FadeIn(ylabels), FadeIn(xlab), FadeIn(ylab), FadeIn(table),
                      FadeIn(step_lbl), run_time=self.fit(1.2))
            shown = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 200]
            per = max(0.3, (self.remaining() - 3.5) / len(shown))
            curve = VMobject(color=theme.GRAD, stroke_width=4)
            pts = [ax.c2p(0, np.log10(LOSSES[0]))]
            curve.set_points_as_corners([pts[0], pts[0]])
            self.add(curve)
            prev = 0
            for s in shown:
                for t in range(prev + 1, min(s, 50) + 1):
                    pts.append(ax.c2p(t, np.log10(LOSSES[t])))
                prev = min(s, 50)
                new_curve = VMobject(color=theme.GRAD, stroke_width=4)
                new_curve.set_points_as_corners(pts)
                W, b, loss = HIST[s]
                W_o, b_o = reg.to_original_units(W, b, MU, SIGMA)
                vals = list(W_o.ravel()) + [b_o[0]]
                self.play(Transform(curve, new_curve),
                          *[ChangeDecimalToValue(d, v) for d, v in zip(decs, vals, strict=True)],
                          ChangeDecimalToValue(step_lbl[1], s),
                          ChangeDecimalToValue(step_lbl[3], loss),
                          run_time=self.fit(per, reserve=2.5))
            final = zh(f"最终损失 {LOSSES[-1]:.1f}（≈ 噪声方差 25），与最小二乘解一致", 22,
                       theme.OUTPUT).move_to([0, -2.4, 0])
            self.play(FadeIn(final), run_time=self.fit(0.8))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(*[FadeOut(m) for m in [setup, ax, ylabels, xlab, ylab, table, step_lbl,
                                             curve, final]], run_time=self.fit(0.6))

        # ── S11 Loop vs vectorization ────────────────────────────────────
        with self.shot("S11"):
            fw, tr = BENCH["forward"], BENCH["train"]
            sub = MathTex(r"(1000,100)\ @\ (100,10)", font_size=32, color=theme.MUTED).move_to([0, 2.4, 0])
            self.play(*self.set_heading("同一个计算，三种写法"), FadeIn(sub), run_time=self.fit(1))
            items = [("Python 三重循环", fw["loop"], theme.GRAD),
                     ("每行一次 np.dot", fw["rows"], theme.PARAM),
                     ("一次 X @ W", fw["vec"], theme.OUTPUT)]
            x0, maxlen = -2.2, 5.6
            bars = VGroup()
            for i, (name, t, color) in enumerate(items):
                y = 1.45 - 0.95 * i
                lbl = zh(name, 26, theme.FG).move_to([x0 - 0.3, y, 0], aligned_edge=RIGHT)
                length = max(0.04, maxlen * t / fw["loop"])
                bar = Rectangle(width=length, height=0.5, color=color, fill_color=color,
                                fill_opacity=0.85, stroke_width=0)
                bar.move_to([x0, y, 0], aligned_edge=LEFT)
                txt = f"{t * 1e3:.1f} ms" if t > 1e-3 else f"{t * 1e3:.3f} ms"
                if i > 0:
                    txt += f"   快 {fw['loop'] / t:.0f} 倍"
                val = zh(txt, 24, color).next_to(bar, RIGHT, 0.2)
                if i == 0:
                    val.move_to([x0 + maxlen - 0.1, y, 0], aligned_edge=RIGHT).set_color(theme.BG)
                bars.add(VGroup(lbl, bar, val))
            for grp in bars:
                self.play(FadeIn(grp[0]), GrowFromEdge(grp[1], LEFT), run_time=self.fit(1.0, reserve=4))
                self.play(FadeIn(grp[2]), run_time=self.fit(0.5, reserve=4))
                self.wait(min(2.2, max(0.1, self.remaining() - 8)))
            train = zh(f"训练 200 步（5000 套房子）：循环 {tr['loop'] * 1e3:.0f} ms　"
                       f"矩阵 {tr['vec'] * 1e3:.1f} ms（快 {tr['loop'] / tr['vec']:.0f} 倍）", 26,
                       theme.FG).move_to([0, -1.5, 0])
            self.play(FadeIn(train), run_time=self.fit(0.8))
            same = zh("本机渲染时实测，每次运行略有不同；三种写法结果完全一致", 20,
                      theme.MUTED).move_to([0, -2.2, 0])
            self.play(FadeIn(same), run_time=self.fit(0.6))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(FadeOut(sub), FadeOut(bars), FadeOut(train), FadeOut(same),
                      run_time=self.fit(0.6))

        # ── S12 Why it is fast ───────────────────────────────────────────
        with self.shot("S12"):
            def panel(title, lines, color, x):
                t = zh(title, 28, color)
                body = VGroup(*[zh(s, 24, theme.FG) for s in lines]).arrange(
                    DOWN, aligned_edge=LEFT, buff=0.28)
                content = VGroup(t, body).arrange(DOWN, buff=0.4)
                box = RoundedRectangle(width=5.8, height=content.height + 0.7,
                                       corner_radius=0.15, color=color)
                content.move_to(box)
                return VGroup(box, content).move_to([x, 0.75, 0])

            left = panel("Python 循环", ["每做一次乘加，都要：", "· 解释一行代码", "· 检查类型",
                                        "· 创建新对象"], theme.GRAD, -3.3)
            right = panel("X @ W", ["整块交给 BLAS（C / 汇编）：", "· 数据连续存放",
                                    "· 一条指令算好几个数（SIMD）", "· 多个核心一起算"],
                          theme.OUTPUT, 3.3)
            self.play(*self.set_heading("为什么快：把循环交给底层库"), FadeIn(left),
                      run_time=self.fit(1))
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(right), run_time=self.fit(1))
            self.wait(self.remaining() * 0.35)
            gpu = zh("GPU：成千上万个核心同时做乘加 —— 最擅长的就是矩阵乘法", 26, theme.FG)
            gpu.move_to([0, -1.6, 0])
            self.play(FadeIn(gpu), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.3)
            big = zh("大模型里绝大部分计算 = 矩阵乘法", 30, theme.HIGHLIGHT).move_to([0, -2.3, 0])
            self.play(FadeIn(big), run_time=self.fit(0.8))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(*[FadeOut(m) for m in [left, right, gpu, big]], run_time=self.fit(0.6))

        # ── S13 From minimal code to production code ─────────────────────
        with self.shot("S13"):
            left_code = VGroup(*[mono(s, 22) for s in [
                "err = X @ W + b - y",
                "grad_W = 2/N * X.T @ err",
                "grad_b = 2/N * err.sum(0)",
                "W = W - lr * grad_W",
                "b = b - lr * grad_b",
            ]]).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([-3.6, 0.55, 0])
            right_code = VGroup(*[mono(s, 22) for s in [
                "model = nn.Linear(3, 1)",
                "y_hat = model(x)",
                "loss = loss_fn(y_hat, y)",
                "optimizer.zero_grad()",
                "loss.backward()",
                "optimizer.step()",
            ]]).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([3.3, 0.55, 0])
            lt = zh("手写矩阵梯度（NumPy）", 26, theme.MUTED).next_to(left_code, UP, 0.4)
            rt = zh("PyTorch 标准写法", 26, theme.HIGHLIGHT).next_to(right_code, UP, 0.4)
            lt.align_to(rt, UP)
            self.play(*self.set_heading("从极简到生产级：nn.Linear"), FadeIn(lt), FadeIn(left_code),
                      run_time=self.fit(1))
            self.play(FadeIn(rt), FadeIn(right_code[0]), run_time=self.fit(1))
            r0 = SurroundingRectangle(right_code[0], color=theme.HIGHLIGHT, buff=0.08)
            shape_t = zh("输入 (200, 3) → 输出 (200, 1)", 22, theme.HIGHLIGHT).next_to(
                right_code, DOWN, 0.3)
            self.play(Create(r0), FadeIn(shape_t), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.12)
            wnote = zh("weight 形状 (1, 3) = (输出, 输入)，前向 = x @ weight.T + bias", 22,
                       theme.PARAM).move_to([0, -1.95, 0])
            self.play(FadeIn(wnote), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.2)
            loop_box = SurroundingRectangle(right_code[1:], color=theme.OUTPUT, buff=0.1)
            loop_t = zh("第 1 章的五行训练循环，一字未改", 22, theme.OUTPUT).move_to(shape_t)
            self.play(Transform(r0, loop_box), FadeOut(shape_t), FadeIn(loop_t),
                      LaggedStart(*[FadeIn(m) for m in right_code[1:]], lag_ratio=0.2),
                      run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.25)
            res = zh(f"200 步后对拍：float64 差 {DIFF64:.1e}　float32 差 {DIFF32:.1e}", 24,
                     theme.OUTPUT).move_to([0, -2.45, 0])
            self.play(FadeIn(res), run_time=self.fit(0.8))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(*[FadeOut(m) for m in [lt, rt, left_code, right_code, r0, loop_t, wnote, res]],
                      run_time=self.fit(0.6))

        # ── S14 Summary and the next chapter ─────────────────────────────
        with self.shot("S14"):
            summ = VGroup(
                zh("一组输入 → 向量、点积", 26, theme.FG),
                zh("一批样本 → 矩阵，(N,k) @ (k,n) → (N,n)", 26, theme.FG),
                zh("偏置 b → 广播到每一行", 26, theme.FG),
                zh("四个步骤：一个没变", 26, theme.HIGHLIGHT),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.26).move_to([-3.2, 1.65, 0])
            self.play(*self.set_heading("小结"), run_time=self.fit(0.5))
            for s in summ:
                self.play(FadeIn(s, shift=RIGHT * 0.2), run_time=self.fit(0.8, reserve=10))
                self.wait(min(1.2, max(0.1, self.remaining() - 16)))
            u = 0.42
            b1 = block(2 * u, 3 * u, theme.PARAM, "W_1")
            b2 = block(1 * u, 2 * u, theme.PARAM, "W_2")
            bw = block(1 * u, 3 * u, theme.OUTPUT, "W")
            for bl in (b1, b2, bw):
                bl[1].scale(0.8)
            coll = VGroup(b1, MathTex("@", font_size=34), b2, MathTex("=", font_size=34), bw)
            coll.arrange(RIGHT, buff=0.25).move_to([3.6, 1.55, 0])
            shapes = VGroup(MathTex("(3,2)", font_size=24, color=theme.MUTED).next_to(b1, DOWN, 0.12),
                            MathTex("(2,1)", font_size=24, color=theme.MUTED).next_to(b2, DOWN, 0.12),
                            MathTex("(3,1)", font_size=24, color=theme.MUTED).next_to(bw, DOWN, 0.12))
            q = zh("两层线性 = 一层线性", 28, theme.HIGHLIGHT).next_to(shapes, DOWN, 0.3)
            q.set_x(3.6)
            self.play(FadeIn(coll), FadeIn(shapes), run_time=self.fit(1.2))
            self.play(FadeIn(q), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.2)
            cax = Axes(x_range=[0, 6, 1], y_range=[-1.5, 1.5, 1], x_length=5.6, y_length=1.9,
                       axis_config={"color": theme.MUTED}, tips=False).move_to([-3.3, -1.3, 0])
            xs = np.linspace(0.2, 5.8, 24)
            ys = np.sin(1.3 * xs)
            dots = VGroup(*[Dot(cax.c2p(x, y), radius=0.05, color=theme.INPUT) for x, y in zip(xs, ys, strict=True)])
            sl, ic = np.polyfit(xs, ys, 1)
            fit_line = polyline_in_axes(cax, [(x, sl * x + ic) for x in np.linspace(0, 6, 50)],
                                        color=theme.OUTPUT, stroke_width=4)
            self.play(Create(cax), LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.05),
                      run_time=self.fit(1.5))
            self.play(Create(fit_line), run_time=self.fit(1))
            miss = zh("直线拟合不了曲线", 24, theme.GRAD).next_to(cax, UP, 0.1).align_to(cax, RIGHT)
            self.play(FadeIn(miss), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.35)
            nxt = zh("下一章：非线性与神经网络", 32, theme.HIGHLIGHT).move_to([3.6, -1.3, 0])
            nbox = Rectangle(width=nxt.width + 0.6, height=nxt.height + 0.35, color=theme.HIGHLIGHT)
            nbox.move_to(nxt)
            self.play(FadeIn(nxt), Create(nbox), run_time=self.fit(1))
            self.wait(max(0.05, self.remaining() - 1.0))
            self.play(*[FadeOut(m) for m in [summ, coll, shapes, q, cax, dots, fit_line, miss,
                                             nxt, nbox]],
                      *self.set_heading(None), run_time=self.fit(1.0))
