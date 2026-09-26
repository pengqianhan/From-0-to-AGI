"""第 8 章视频：注意力 —— 让每个位置自己决定看哪里

画面里的所有数值都由 ../code/ 中的代码真实计算（见 script.md 事实清单）。
训练小模型（约 1 分钟）的结果缓存在 video/out/cache.json，删掉即可重算。
渲染：bash chapters/08-attention/video/build.sh
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Axes,
    Create,
    CurvedArrow,
    FadeIn,
    FadeOut,
    GrowArrow,
    GrowFromEdge,
    LaggedStart,
    Line,
    MathTex,
    Rectangle,
    RoundedRectangle,
    Square,
    SurroundingRectangle,
    Text,
    Transform,
    VGroup,
    Write,
)

from video_kit import theme
from video_kit.scene import MONO_FONT, NarratedScene, code_block, zh

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
CACHE = HERE / "out" / "cache.json"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m01 = _load("avg01", "01_average_to_attention.py")
m02 = _load("attn02", "02_attention_from_scratch.py")
m03 = _load("sqrt03", "03_why_sqrt_d.py")
m05 = _load("zero05", "05_zero_parity.py")
PARITY = m05.parity()

# ── 01：5 个 2 维 token 向量 ──────────────────────────────────────────────
X = m01.make_x()
T5 = len(X)
W_UNI = m01.uniform_weights(T5)
W_DOT = m01.dot_product_weights(X)

# ── 02：随机输入上的多头注意力（4 个头的权重图 + 对拍误差） ─────────────────
torch.manual_seed(0)
_B, _T, _C, _H = 2, 8, 32, 4
_x = torch.randn(_B, _T, _C)
_mha = m02.MultiHeadAttention(_C, _H)
with torch.no_grad():
    _out, _w = _mha(_x, return_weights=True)
    SDPA_DIFF = float((_out - m02.mha_with_sdpa(_mha, _x)).abs().max())
    _x2 = _x.clone()
    _x2[:, 5:] = torch.randn(_B, 3, _C)
    CAUSAL_DIFF = float((_mha(_x2)[:, :5] - _out[:, :5]).abs().max())
HEAD_MAPS = _w[0].numpy()  # (4, 8, 8)

# ── 03：为什么除以 √d（与脚本 main() 同样的随机数顺序，数字和终端输出一致） ──
_rng = np.random.default_rng(0)
SQRT_STATS = {s: {d: m03.stats(d, s, _rng) for d in m03.DIMS} for s in (False, True)}
_rng2 = np.random.default_rng(1)
_q, _k = _rng2.normal(size=1024), _rng2.normal(size=(16, 1024))
_sc = _k @ _q
BARS_RAW = m03.softmax(_sc[None])[0]
BARS_SCALED = m03.softmax(_sc[None] / math.sqrt(1024))[0]

# ── 04：训练单层注意力模型（缓存） ──────────────────────────────────────────
HEAT_TEXT = "further, hear me speak."
SMALL_TEXT = "speak."
SMALL_HEAD = 1


def _train_results() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    m04 = _load("train04", "04_train_attention.py")
    losses, models = [], {}
    for mode in ("bigram", "average", "attention"):
        model, hist = m04.train(mode)
        models[mode] = model
        loss = hist[-1][1]
        losses.append({"mode": mode, "params": sum(p.numel() for p in model.parameters()),
                       "loss": loss, "bpb": loss / math.log(2)})
    model = models["attention"]
    chars, stoi, _, val = m04.load_data()
    prof = m04.head_profile(model, val).tolist()
    w = m04.attention_on(model, m04.SAMPLE)
    s = len(m04.SAMPLE) - len(HEAT_TEXT)
    assert m04.SAMPLE[s:] == HEAT_TEXT
    heat = w[:, s:, s:].tolist()
    # 小例子：模型直接读 "speak."，取一个头的缩放分数和权重
    with torch.no_grad():
        idx = torch.tensor([[stoi[c] for c in SMALL_TEXT]])
        T = idx.shape[1]
        h = model.tok(idx) + model.pos(torch.arange(T))
        mix = model.mix
        q = mix.wq(h).view(1, T, mix.H, mix.d).transpose(1, 2)[0, SMALL_HEAD]
        k = mix.wk(h).view(1, T, mix.H, mix.d).transpose(1, 2)[0, SMALL_HEAD]
        scores = (q @ k.T / math.sqrt(mix.d)).tolist()
        _, ws = model(idx, return_weights=True)
        small_w = ws[0, SMALL_HEAD].tolist()
    res = {"losses": losses, "profile": prof, "heat": heat, "small_scores": scores,
           "small_w": small_w}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


RES = _train_results()


def mono(text: str, size: float = 22, color: str = theme.FG) -> Text:
    return Text(text, font=MONO_FONT, font_size=size, color=color)


def show_c(c: str) -> str:
    return {" ": "_", "\n": "↵"}.get(c, c)


def grid(values: np.ndarray, cell: float, color: str = theme.ATTN, numbers: bool = False,
         fs: float = 16, fmt: str = "{:.2f}", opacity_scale: float = 1.0) -> VGroup:
    """T×T 的格子：填充透明度 = 权重。返回 VGroup(cells, texts)，cells[i][j]。"""
    n, m = values.shape
    cells, texts = VGroup(), VGroup()
    for i in range(n):
        row = VGroup()
        for j in range(m):
            sq = Square(cell, stroke_color=theme.MUTED, stroke_width=1,
                        fill_color=color, fill_opacity=float(np.clip(values[i, j] * opacity_scale, 0, 1)))
            sq.move_to([j * cell, -i * cell, 0])
            row.add(sq)
            if numbers:
                texts.add(mono(fmt.format(values[i, j]), fs).move_to(sq))
        cells.add(row)
    return VGroup(cells, texts)


class ChapterScene(NarratedScene):
    chapter_label = "第 8 章"
    chapter_title = "注意力"

    def construct(self) -> None:
        # ── S01 片头 ─────────────────────────────────────────────────────
        with self.shot("S01"):
            card = self.chapter_card()
            sub = zh("让每个位置自己决定看哪里", 32, theme.ATTN).next_to(card, DOWN, 0.6)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(card), FadeOut(sub), run_time=self.fit(0.8))

        # ── S02 只看前一个字不够 ─────────────────────────────────────────
        with self.shot("S02"):
            text = "hear me speak"
            boxes = VGroup()
            for c in text:
                sq = Square(0.6, stroke_color=theme.INPUT, stroke_width=2)
                boxes.add(VGroup(sq, mono(show_c(c), 30, theme.FG).move_to(sq)))
            boxes.arrange(RIGHT, buff=0.08).move_to([0, 0.2, 0])
            self.play(*self.set_heading("bigram 只看前一个 token"),
                      LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.05), run_time=self.fit(1.5))
            last = boxes[-1]
            bi = CurvedArrow(last.get_top() + UP * 0.05, boxes[-2].get_top() + UP * 0.05,
                             angle=1.2, color=theme.HIGHLIGHT, stroke_width=4)
            bl = zh("bigram：只看前一个", 26, theme.HIGHLIGHT).move_to([3.3, 1.7, 0])
            self.play(Create(bi), FadeIn(bl), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.3)
            far = VGroup(*[CurvedArrow(last.get_bottom() + DOWN * 0.05,
                                       boxes[i].get_bottom() + DOWN * 0.05,
                                       angle=-0.9 + 0.02 * i, color=theme.ATTN, stroke_width=3)
                           for i in (0, 1, 2, 3, 5, 6)])
            q = zh("更早的字也有用：怎么把整段前文用上？", 28, theme.ATTN).move_to([0, -2.1, 0])
            self.play(LaggedStart(*[Create(a) for a in far], lag_ratio=0.15), run_time=self.fit(2))
            self.play(FadeIn(q), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(boxes, bi, bl, far, q)), run_time=self.fit(0.6))

        # ── S03 RNN ─────────────────────────────────────────────────────
        with self.shot("S03"):
            chars = "hear m"
            xs = [-5.0 + 1.6 * i for i in range(len(chars))]
            ins = VGroup(*[VGroup(Square(0.55, stroke_color=theme.INPUT),
                                  mono(show_c(c), 26)).move_to([x, -1.2, 0]) for c, x in zip(chars, xs)])
            for g in ins:
                g[1].move_to(g[0])
            states = VGroup(*[RoundedRectangle(width=0.9, height=0.6, corner_radius=0.12,
                                               color=theme.PARAM).move_to([x, 0.5, 0]) for x in xs])
            hl = VGroup(*[MathTex(f"h_{i + 1}", font_size=28, color=theme.PARAM).move_to(s)
                          for i, s in enumerate(states)])
            ups = VGroup(*[Arrow(i.get_top(), s.get_bottom(), buff=0.08, color=theme.MUTED,
                                 stroke_width=3) for i, s in zip(ins, states)])
            rights = VGroup(*[Arrow(states[i].get_right(), states[i + 1].get_left(), buff=0.06,
                                    color=theme.PARAM, stroke_width=3) for i in range(len(xs) - 1)])
            self.play(*self.set_heading("RNN：把前文压进一个固定大小的状态"),
                      FadeIn(ins), run_time=self.fit(1))
            per = min(0.9, (self.remaining() * 0.55) / len(xs))
            for i in range(len(xs)):
                anims = [GrowArrow(ups[i]), FadeIn(states[i]), FadeIn(hl[i])]
                if i > 0:
                    anims.append(GrowArrow(rights[i - 1]))
                self.play(*anims, run_time=self.fit(per))
            n1 = zh("① 状态大小固定：前文越长越挤", 26, theme.FG).move_to([0, 2.0, 0])
            n2 = zh("② 必须一步一步算，无法并行", 26, theme.FG).move_to([0, -2.2, 0])
            self.play(FadeIn(n1), run_time=self.fit(0.6))
            self.play(FadeIn(n2), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(ins, states, hl, ups, rights, n1, n2)), run_time=self.fit(0.6))

        # ── S04 取平均 ───────────────────────────────────────────────────
        ax = Axes(x_range=[-1.5, 1.5, 0.5], y_range=[-1.5, 1.5, 0.5], x_length=4.4, y_length=4.4,
                  axis_config={"color": theme.MUTED, "stroke_width": 1.5}, tips=False).move_to([-4.2, 0.0, 0])
        tok_cols = [theme.INPUT, theme.OUTPUT, theme.PARAM, theme.HIGHLIGHT, theme.GRAD]
        vecs = VGroup()
        for i, (v, col) in enumerate(zip(X, tok_cols)):
            a = Arrow(ax.c2p(0, 0), ax.c2p(*v), buff=0, color=col, stroke_width=4,
                      max_tip_length_to_length_ratio=0.15)
            lbl = zh(m01.TOKENS[i], 22, col).next_to(ax.c2p(*v), UP if v[1] >= 0 else DOWN, 0.08)
            vecs.add(VGroup(a, lbl))
        cell = 0.74
        g_uni = grid(W_UNI, cell, theme.ATTN, numbers=False)
        frac = VGroup()
        for i in range(T5):
            for j in range(T5):
                s = "0" if j > i else ("1" if i == 0 else f"1/{i + 1}")
                frac.add(mono(s, 18, theme.FG if j <= i else theme.MUTED).move_to(g_uni[0][i][j]))
        mat_uni = VGroup(g_uni, frac)
        row_lbl = VGroup(*[zh(t, 20, theme.MUTED) for t in m01.TOKENS])
        with self.shot("S04"):
            self.play(*self.set_heading("最朴素的办法：把前面的向量取平均"), Create(ax),
                      LaggedStart(*[GrowArrow(v[0]) for v in vecs], lag_ratio=0.15),
                      LaggedStart(*[FadeIn(v[1]) for v in vecs], lag_ratio=0.15), run_time=self.fit(2))
            mat_uni.move_to([2.3, 0.5, 0])
            for i, lab in enumerate(row_lbl):
                lab.next_to(g_uni[0][i][0], LEFT, 0.15)
            wl = MathTex("W", font_size=40, color=theme.ATTN).next_to(mat_uni, UP, 0.2)
            self.wait(self.remaining() * 0.15)
            self.play(FadeIn(row_lbl), FadeIn(wl), LaggedStart(*[FadeIn(VGroup(g_uni[0][i], *frac[i * T5:(i + 1) * T5]))
                                                                for i in range(T5)], lag_ratio=0.3),
                      run_time=self.fit(2.5))
            eq = MathTex(r"\text{out} = W\,X", font_size=40).next_to(mat_uni, DOWN, 0.35)
            self.play(Write(eq), run_time=self.fit(1))
            avg = W_UNI[-1] @ X
            avg_arrow = Arrow(ax.c2p(0, 0), ax.c2p(*avg), buff=0, color=theme.FG, stroke_width=6,
                              max_tip_length_to_length_ratio=0.3)
            box_last = SurroundingRectangle(g_uni[0][T5 - 1], color=theme.HIGHLIGHT, buff=0.03)
            self.wait(self.remaining() * 0.25)
            avg_lbl = zh("平均", 20, theme.FG).next_to(ax.c2p(*avg), DOWN + RIGHT, 0.05)
            self.play(Create(box_last), GrowArrow(avg_arrow), FadeIn(avg_lbl), run_time=self.fit(1.2))
            same = zh(f"和逐个循环求平均的最大差：{np.abs(W_UNI @ X - m01.prefix_mean_loop(X)).max():.0e}",
                      22, theme.OUTPUT).move_to([2.3, -2.45, 0])
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(same), run_time=self.fit(0.8))

        # ── S05 权重由数据决定 ───────────────────────────────────────────
        with self.shot("S05"):
            g_dot = grid(W_DOT, cell, theme.ATTN, numbers=True, fs=16).move_to(g_uni)
            wl2 = MathTex(r"W=\mathrm{softmax}(XX^{\top}+\text{mask})", font_size=34,
                          color=theme.ATTN).next_to(mat_uni, UP, 0.2)
            self.play(*self.set_heading("让权重由数据决定：点积相似度 → softmax"),
                      FadeOut(same), FadeOut(box_last), FadeOut(eq), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.2)
            self.play(Transform(g_uni, g_dot[0]), FadeOut(frac), FadeIn(g_dot[1]),
                      Transform(wl, wl2), run_time=self.fit(2))
            wavg = W_DOT[-1] @ X
            new_arrow = Arrow(ax.c2p(0, 0), ax.c2p(*wavg), buff=0, color=theme.FG, stroke_width=6,
                              max_tip_length_to_length_ratio=0.3)
            new_lbl = zh("加权平均", 20, theme.FG).next_to(ax.c2p(*wavg), RIGHT, 0.1)
            self.play(Transform(avg_arrow, new_arrow), Transform(avg_lbl, new_lbl), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.35)
            diag = VGroup(*[SurroundingRectangle(g_uni[i][i], color=theme.HIGHLIGHT, buff=0.02)
                            for i in range(T5)])
            self_top = sum(int(np.argmax(W_DOT[t]) == t) for t in range(T5))
            warn = zh(f"问题：{self_top}/{T5} 个位置最关注的都是自己", 24, theme.HIGHLIGHT).move_to([2.3, -2.45, 0])
            self.play(Create(diag), FadeIn(warn), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(ax, vecs, g_uni, g_dot[1], wl, row_lbl, avg_arrow, avg_lbl, diag, warn)),
                      run_time=self.fit(0.6))

        # ── S06 Q K V ────────────────────────────────────────────────────
        with self.shot("S06"):
            xb = VGroup(RoundedRectangle(width=1.2, height=0.8, corner_radius=0.1, color=theme.INPUT),
                        MathTex("X", font_size=40, color=theme.INPUT)).move_to([-5.2, 0.2, 0])
            rows = []
            names = [("Q", "W_q", "Q：我在找什么"), ("K", "W_k", "K：我有什么特征"), ("V", "W_v", "V：关注我，就给你这个")]
            for r, (n, wname, desc) in enumerate(names):
                y = 1.6 - 1.4 * r
                wbox = VGroup(RoundedRectangle(width=1.1, height=0.7, corner_radius=0.1, color=theme.PARAM),
                              MathTex(wname, font_size=34, color=theme.PARAM)).move_to([-2.6, y, 0])
                obox = VGroup(RoundedRectangle(width=1.0, height=0.7, corner_radius=0.1, color=theme.ATTN),
                              MathTex(n, font_size=36, color=theme.ATTN)).move_to([-0.4, y, 0])
                a1 = Arrow(xb.get_right(), wbox.get_left(), buff=0.1, color=theme.MUTED, stroke_width=3)
                a2 = Arrow(wbox.get_right(), obox.get_left(), buff=0.1, color=theme.MUTED, stroke_width=3)
                d = zh(desc, 26, theme.FG).next_to(obox, RIGHT, 0.5)
                rows.append(VGroup(wbox, obox, a1, a2, d))
            self.play(*self.set_heading("Q、K、V：同一个输入，三个线性投影"), FadeIn(xb), run_time=self.fit(1))
            for r in rows:
                self.play(GrowArrow(r[2]), FadeIn(r[0]), GrowArrow(r[3]), FadeIn(r[1]),
                          run_time=self.fit(1.0, reserve=4))
                self.play(FadeIn(r[4]), run_time=self.fit(0.6, reserve=3))
                self.wait(min(1.5, max(0.1, self.remaining() * 0.12)))
            sc = MathTex(r"\text{score}(t, s) = q_t \cdot k_s", font_size=40).move_to([0.5, -2.25, 0])
            self.play(Write(sc), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(xb, *rows, sc)), run_time=self.fit(0.6))

        # ── S07 分数 → mask → softmax → 乘 V ────────────────────────────────
        S = np.array(RES["small_scores"])
        Wsm = np.array(RES["small_w"])
        n = len(SMALL_TEXT)
        cell7 = 0.72
        with self.shot("S07"):
            gs = grid(np.zeros((n, n)), cell7, theme.ATTN).move_to([-2.4, -0.1, 0])
            cells = gs[0]
            col_l = VGroup(*[mono(show_c(c), 24, theme.INPUT).next_to(cells[0][j], UP, 0.12)
                             for j, c in enumerate(SMALL_TEXT)])
            row_l = VGroup(*[mono(show_c(c), 24, theme.INPUT).next_to(cells[i][0], LEFT, 0.15)
                             for i, c in enumerate(SMALL_TEXT)])
            kq = VGroup(zh("键 k（被看的）", 20, theme.MUTED).next_to(col_l, UP, 0.1),
                        zh("查询 q", 20, theme.MUTED).next_to(row_l, LEFT, 0.1).rotate(math.pi / 2))
            steps = VGroup(
                MathTex(r"S = QK^{\top}/\sqrt{d}", font_size=36),
                MathTex(r"S_{ts} = -\infty\ \ (s > t)", font_size=36),
                MathTex(r"W = \mathrm{softmax}(S)", font_size=36, color=theme.ATTN),
                MathTex(r"\text{out} = W\,V", font_size=36),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([4.2, 0.2, 0])
            tag = zh(f"训练好的模型 · 头 {SMALL_HEAD} · 输入 “{SMALL_TEXT}”", 20, theme.MUTED).move_to([4.2, 2.25, 0])
            self.play(*self.set_heading("一个头的完整计算（真实数值）"), FadeIn(cells), FadeIn(col_l),
                      FadeIn(row_l), FadeIn(kq), FadeIn(tag), run_time=self.fit(1))
            nums = VGroup(*[mono(f"{S[i, j]:.1f}", 18).move_to(cells[i][j]) for i in range(n) for j in range(n)])
            self.play(FadeIn(steps[0]), LaggedStart(*[FadeIn(t) for t in nums], lag_ratio=0.04),
                      run_time=self.fit(3))
            self.wait(self.remaining() * 0.18)
            upper = [(i, j) for i in range(n) for j in range(n) if j > i]
            infs = VGroup(*[mono("-∞", 18, theme.MUTED).move_to(cells[i][j]) for i, j in upper])
            grays = [cells[i][j].animate.set_fill(theme.MUTED, opacity=0.35) for i, j in upper]
            self.play(FadeIn(steps[1]), *grays, *[Transform(nums[i * n + j], infs[k])
                                                  for k, (i, j) in enumerate(upper)], run_time=self.fit(1.5))
            self.wait(self.remaining() * 0.2)
            wnums = VGroup(*[mono(f"{Wsm[i, j]:.2f}" if j <= i else "0", 18,
                                  theme.FG if j <= i else theme.MUTED).move_to(cells[i][j])
                             for i in range(n) for j in range(n)])
            fills = [cells[i][j].animate.set_fill(theme.ATTN if j <= i else theme.BG,
                                                  opacity=float(Wsm[i, j]) if j <= i else 0.0)
                     for i in range(n) for j in range(n)]
            self.play(FadeIn(steps[2]), *fills, Transform(nums, wnums), run_time=self.fit(2))
            rs = zh("每行和为 1", 22, theme.ATTN).next_to(gs, DOWN, 0.2)
            self.play(FadeIn(rs), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(steps[3]), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(cells, nums, col_l, row_l, kq, steps, tag, rs)), run_time=self.fit(0.6))

        # ── S08 为什么除以 √d ────────────────────────────────────────────
        with self.shot("S08"):
            hdr = ["d", "方差", "最大权重", "方差", "最大权重"]
            tbl = VGroup()
            for h in hdr:
                tbl.add(zh(h, 22, theme.MUTED))
            for d in m03.DIMS:
                r, s = SQRT_STATS[False][d], SQRT_STATS[True][d]
                tbl.add(mono(str(d), 22), mono(f"{r['var']:.1f}", 22, theme.GRAD),
                        mono(f"{r['max_w']:.3f}", 22, theme.GRAD), mono(f"{s['var']:.1f}", 22, theme.OUTPUT),
                        mono(f"{s['max_w']:.3f}", 22, theme.OUTPUT))
            tbl.arrange_in_grid(rows=5, cols=5, buff=(0.45, 0.28)).move_to([-3.0, 0.0, 0])
            g1 = zh("不缩放 q·k", 22, theme.GRAD).next_to(VGroup(tbl[1], tbl[2]), UP, 0.25)
            g2 = zh("缩放 q·k/√d", 22, theme.OUTPUT).next_to(VGroup(tbl[3], tbl[4]), UP, 0.25)
            self.play(*self.set_heading("为什么除以 √d：点积的方差 ≈ d"),
                      FadeIn(VGroup(*tbl[:5])), run_time=self.fit(1))
            self.wait(self.remaining() * 0.15)
            self.play(FadeIn(g1), LaggedStart(*[FadeIn(VGroup(tbl[5 + 5 * i], tbl[6 + 5 * i], tbl[7 + 5 * i]))
                                                for i in range(4)], lag_ratio=0.3), run_time=self.fit(2))

            def bar_chart(ws, color, y, title):
                base = Line([1.2, y, 0], [6.6, y, 0], color=theme.MUTED, stroke_width=1.5)
                bars = VGroup()
                for i, w in enumerate(ws):
                    h = max(0.01, 1.5 * float(w))
                    b = Rectangle(width=0.26, height=h, fill_color=color, fill_opacity=0.9, stroke_width=0)
                    b.move_to([1.4 + 0.335 * i, y + h / 2, 0])
                    bars.add(b)
                t = zh(title, 20, theme.FG).next_to(base, UP, 1.65).align_to(base, LEFT)
                return VGroup(base, bars, t)

            c1 = bar_chart(BARS_RAW, theme.ATTN, 0.45, "d = 1024，不缩放：16 个位置的权重")
            c2 = bar_chart(BARS_SCALED, theme.ATTN, -2.2, "除以 √d 之后")
            self.play(FadeIn(c1[0]), FadeIn(c1[2]), GrowFromEdge(c1[1], DOWN), run_time=self.fit(1.5))
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(g2), LaggedStart(*[FadeIn(VGroup(tbl[8 + 5 * i], tbl[9 + 5 * i]))
                                                for i in range(4)], lag_ratio=0.3), run_time=self.fit(1.5))
            self.play(FadeIn(c2[0]), FadeIn(c2[2]), GrowFromEdge(c2[1], DOWN), run_time=self.fit(1.5))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(tbl, g1, g2, c1, c2)), run_time=self.fit(0.6))

        # ── S09 对拍与因果性 ─────────────────────────────────────────────
        with self.shot("S09"):
            src = code_block("out = F.scaled_dot_product_attention(\n    q, k, v, is_causal=True)", 24)
            src.move_to([0, 1.3, 0])
            r1 = zh(f"从零实现 vs 官方 SDPA：最大差 {SDPA_DIFF:.1e}", 30, theme.OUTPUT).move_to([0, 0.0, 0])
            r2 = zh(f"改掉位置 5–7 的输入 → 位置 0–4 的输出变化 {CAUSAL_DIFF:.1f}", 30, theme.ATTN)
            r2.move_to([0, -1.1, 0])
            self.play(*self.set_heading("对拍：和 PyTorch 官方实现逐位比较"), FadeIn(src), run_time=self.fit(1))
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(r1), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(r2), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(src, r1, r2)), run_time=self.fit(0.6))

        # ── S10 多头 ─────────────────────────────────────────────────────
        with self.shot("S10"):
            head_cols = [theme.INPUT, theme.OUTPUT, theme.PARAM, theme.HIGHLIGHT]
            strip = VGroup(*[Rectangle(width=0.16, height=0.5, stroke_width=0.5, stroke_color=theme.BG,
                                       fill_color=head_cols[i // 8], fill_opacity=0.85) for i in range(32)])
            strip.arrange(RIGHT, buff=0).move_to([0, 2.2, 0])
            cl = MathTex("C = 32", font_size=30).move_to([-5.9, 2.2, 0])
            self.play(*self.set_heading("多头注意力：切成 H 份，各算各的"), FadeIn(strip), FadeIn(cl),
                      run_time=self.fit(1))
            groups = VGroup(*[VGroup(*strip[8 * h:8 * h + 8]) for h in range(4)])
            self.play(*[groups[h].animate.shift(RIGHT * (h - 1.5) * 0.5) for h in range(4)],
                      run_time=self.fit(1))
            maps = VGroup()
            for h in range(4):
                gm = grid(HEAD_MAPS[h], 0.19, theme.ATTN).move_to([-4.35 + 2.9 * h, 0.35, 0])
                lab = MathTex("d = 8", font_size=24, color=head_cols[h]).next_to(gm, UP, 0.12)
                hl = zh(f"头 {h}", 20, head_cols[h]).next_to(gm, DOWN, 0.12)
                maps.add(VGroup(gm, lab, hl))
            arrows = VGroup(*[Arrow(groups[h].get_bottom(), maps[h][1].get_top(), buff=0.08,
                                    color=head_cols[h], stroke_width=3) for h in range(4)])
            self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.2),
                      LaggedStart(*[FadeIn(mp) for mp in maps], lag_ratio=0.2), run_time=self.fit(2))
            shapes = [r"(B,T,C)", r"\to (B,H,T,d)", r"\to W:(B,H,T,T)", r"\to (B,T,C)", r"\to \cdot W_o"]
            sh = MathTex(*shapes, font_size=32).move_to([0, -1.55, 0])
            self.wait(self.remaining() * 0.12)
            for part in sh:
                self.play(FadeIn(part), run_time=self.fit(0.7, reserve=3))
                self.wait(min(1.2, self.remaining() * 0.1))
            pc = zh("参数量 4C²（Wq、Wk、Wv、Wo），和头数无关", 24, theme.PARAM).move_to([0, -2.35, 0])
            self.play(FadeIn(pc), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(strip, cl, maps, arrows, sh, pc)), run_time=self.fit(0.6))

        # ── S11 训练结果 ─────────────────────────────────────────────────
        with self.shot("S11"):
            names = {"bigram": "bigram（只看当前）", "average": "均匀平均", "attention": "注意力"}
            cols = {"bigram": theme.MUTED, "average": theme.INPUT, "attention": theme.ATTN}
            unit = 2.0
            rows = VGroup()
            for i, r in enumerate(RES["losses"]):
                y = 1.2 - 1.2 * i
                lab = zh(names[r["mode"]], 26, cols[r["mode"]]).move_to([-4.3, y, 0])
                bar = Rectangle(width=unit * r["loss"], height=0.55, fill_color=cols[r["mode"]],
                                fill_opacity=0.85, stroke_width=0)
                bar.move_to([-2.5 + unit * r["loss"] / 2, y, 0])
                val = mono(f"{r['loss']:.3f}", 24).next_to(bar, RIGHT, 0.15)
                bpb = mono(f"{r['bpb']:.3f} bpb", 20, theme.MUTED).next_to(val, RIGHT, 0.3)
                rows.add(VGroup(lab, bar, val, bpb))
            axis_note = zh("验证损失（nats/字符，越低越好）", 22, theme.MUTED).move_to([0, 2.2, 0])
            base = Line([-2.5, 1.9, 0], [-2.5, -1.9, 0], color=theme.MUTED, stroke_width=1.5)
            self.play(*self.set_heading("Tiny Shakespeare · 字符级 · 各训练 2000 步"),
                      FadeIn(axis_note), Create(base), run_time=self.fit(1))
            self.wait(self.remaining() * 0.2)
            for r in rows:
                self.play(FadeIn(r[0]), GrowFromEdge(r[1], LEFT), run_time=self.fit(1.0, reserve=3))
                self.play(FadeIn(r[2]), FadeIn(r[3]), run_time=self.fit(0.5, reserve=2))
                self.wait(min(2.0, self.remaining() * 0.18))
            last = zh("看得见前文还不够，关键是会挑", 28, theme.HIGHLIGHT).move_to([0, -2.35, 0])
            self.play(FadeIn(last), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(rows, axis_note, base, last)), run_time=self.fit(0.6))

        # ── S12 真实热力图 ───────────────────────────────────────────────
        with self.shot("S12"):
            heat = np.array(RES["heat"])
            prof = np.array(RES["profile"])
            c12 = 0.19
            panels = VGroup()
            for p, (h, k, desc) in enumerate([(1, 1, "前 1 个"), (0, 2, "前 2 个")]):
                gm = grid(heat[h], c12, theme.ATTN).move_to([-3.2 + 6.6 * p, -0.2, 0])
                cells12 = gm[0]
                cl_ = VGroup(*[mono(show_c(c), 13, theme.INPUT).next_to(cells12[0][j], UP, 0.06)
                               for j, c in enumerate(HEAT_TEXT)])
                rl_ = VGroup(*[mono(show_c(c), 13, theme.INPUT).next_to(cells12[i][0], LEFT, 0.08)
                               for i, c in enumerate(HEAT_TEXT)])
                title = zh(f"头 {h}", 26, theme.ATTN).next_to(cl_, UP, 0.15)
                cap = zh(f"平均 {prof[h][k]:.2f} 的权重给{desc}字符", 22, theme.FG).next_to(gm, DOWN, 0.2)
                panels.add(VGroup(gm, cl_, rl_, title, cap))
            self.play(*self.set_heading("训练好的模型：真实的注意力权重"),
                      FadeIn(VGroup(panels[0][0], panels[0][1], panels[0][2], panels[0][3])),
                      run_time=self.fit(1.5))
            axes_note = zh("行：当前字符　列：被看的字符（_ 是空格）", 20, theme.MUTED)
            axes_note.to_corner(UP + RIGHT, buff=0.45)
            self.play(FadeIn(axes_note), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.12)
            self.play(FadeIn(panels[0][4]), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.12)
            self.play(FadeIn(VGroup(*panels[1][:4])), run_time=self.fit(1.2))
            self.play(FadeIn(panels[1][4]), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(panels, axes_note)), run_time=self.fit(0.6))

        # ── S13 从极简到生产级 ───────────────────────────────────────────
        with self.shot("S13"):
            src = """q = q_norm(wq(x).view(B, T, H, d))
k = k_norm(wk(x).view(B, T, Hkv, d))
v = wv(x).view(B, T, Hkv, d)
q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
if kv_cache is not None:
    k, v = kv_cache.update(layer_idx, start_pos, k, v)
out = F.scaled_dot_product_attention(
    q, k, v, is_causal=..., enable_gqa=Hkv != H)
return wo(out.reshape(B, T, H * d))"""
            cb = code_block(src, 17).move_to([-2.4, 0.35, 0]).to_edge(LEFT, buff=0.4)
            fn = zh("zero/model.py · Attention.forward（省略 transpose）", 20, theme.MUTED).next_to(cb, UP, 0.25)
            fn.align_to(cb, LEFT)
            self.play(*self.set_heading("从极简到生产级：zero/model.py 的 Attention"),
                      FadeIn(fn), FadeIn(cb), run_time=self.fit(1.2))
            tags = [
                ([0, 1], "QK-Norm（第 9 章）", theme.HIGHLIGHT),
                ([3], "RoPE（第 9 章）", theme.HIGHLIGHT),
                ([1, 2], "Hkv < H：GQA（第 10 章）", theme.PARAM),
                ([4, 5], "KV cache（第 10 章）", theme.PARAM),
                ([6, 7], "GPU 上 → FlashAttention（未在 GPU 验证）", theme.ATTN),
                ([8], "输出投影 Wo", theme.OUTPUT),
            ]
            per = max(0.5, (self.remaining() - 6) / len(tags))
            prev = None
            for lines, text, col in tags:
                rect = SurroundingRectangle(VGroup(*[cb[i] for i in lines]), color=col, buff=0.05)
                t = zh(text, 20, col).move_to([4.6, 1.8, 0])
                grp = VGroup(rect, t)
                if prev is None:
                    self.play(FadeIn(grp), run_time=self.fit(0.6))
                    prev = grp
                else:
                    self.play(Transform(prev, grp), run_time=self.fit(0.6))
                self.wait(max(0.1, per - 0.6))
            par = VGroup(
                zh(f"关掉 QK-Norm、RoPE：与极简版最大差 {PARITY['mha_diff']:.1e}", 20, theme.OUTPUT),
                zh("整个模型 vs HF Qwen3：tests/test_model_hf_parity.py", 20, theme.OUTPUT),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([0, -2.3, 0])
            self.play(FadeIn(par), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(cb, fn, prev, par)), run_time=self.fit(0.6))

        # ── S14 小结 ─────────────────────────────────────────────────────
        with self.shot("S14"):
            chain = [(r"X", theme.INPUT), (r"Q,K,V", theme.ATTN), (r"QK^{\top}/\sqrt{d}", theme.FG),
                     (r"+\text{mask}", theme.MUTED), (r"\mathrm{softmax}", theme.ATTN),
                     (r"\cdot V", theme.FG), (r"\text{concat}\cdot W_o", theme.OUTPUT)]
            items = VGroup(*[MathTex(t, font_size=34, color=c) for t, c in chain]).arrange(RIGHT, buff=0.55)
            items.move_to([0, 1.4, 0])
            arrs = VGroup(*[Arrow(items[i].get_right(), items[i + 1].get_left(), buff=0.08,
                                  color=theme.MUTED, stroke_width=2.5, max_tip_length_to_length_ratio=0.35)
                            for i in range(len(items) - 1)])
            self.play(*self.set_heading("注意力 = 由数据决定权重的加权平均"),
                      LaggedStart(*[FadeIn(i) for i in items], lag_ratio=0.2),
                      LaggedStart(*[GrowArrow(a) for a in arrs], lag_ratio=0.2), run_time=self.fit(2.5))
            self.wait(self.remaining() * 0.35)
            lack = zh("但光有注意力还不是完整的模型：它只搬运、混合信息", 26, theme.FG).move_to([0, -0.2, 0])
            self.play(FadeIn(lack), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.3)
            nxt = zh("下一章：现代 Transformer", 34, theme.HIGHLIGHT).move_to([0, -1.7, 0])
            nbox = Rectangle(width=nxt.width + 0.6, height=nxt.height + 0.3, color=theme.HIGHLIGHT).move_to(nxt)
            self.play(FadeIn(nxt), Create(nbox), run_time=self.fit(1))
            self.wait(self.remaining() - 1.0)
            self.play(FadeOut(VGroup(items, arrs, lack, nxt, nbox)), *self.set_heading(None),
                      run_time=self.fit(1.0))
