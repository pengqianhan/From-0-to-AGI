"""Video for Chapter 25: multi-token prediction and speculative decoding.

A small model guesses first. A large model corrects all guesses in one pass.

The code in ../code/ calculates all numbers on screen (see the fact list in script.md).
The cache video/out/cache.json keeps the results.
Render: bash chapters/25-mtp-speculative-decoding/video/build.sh
"""

from __future__ import annotations

import importlib.util
import json
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
    FadeIn,
    FadeOut,
    LaggedStart,
    Line,
    MathTex,
    Rectangle,
    RoundedRectangle,
    Text,
    VGroup,
    Write,
)

from video_kit import theme
from video_kit.scene import NarratedScene, code_block, polyline_in_axes, zh

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
CACHE = HERE / "out" / "cache.json"
MONO = "Noto Sans Mono"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def compute() -> dict:
    """Calculate all numbers for the video with the real code in ../code.

    The first run is slow. Later runs read the cache.
    """
    import statistics
    import time

    m1 = _load("ch25_models", "01_models_and_cost.py")
    m2 = _load("ch25_greedy", "02_greedy_speculative.py")
    m3 = _load("ch25_sampling", "03_speculative_sampling.py")
    m4 = _load("ch25_formula", "04_speedup_formula.py")
    m5 = _load("ch25_mtp", "05_mtp.py")
    d: dict = {}
    target, draft = m1.load_target(), m1.load_draft()
    data = m1.ch10.CharData()

    # 01: time of one forward pass with T tokens (relative to T=1)
    cost = [(T, m1.forward_time(target, 200, T)) for T in (1, 2, 3, 5, 9, 17)]
    d["cost"] = [(T, t / cost[0][1]) for T, t in cost]
    d["c"] = m1.forward_time(draft, 200, 1) / m1.forward_time(target, 200, 1)
    d["params"] = (m1.n_params(target), m1.n_params(draft))

    # 02: one real "draft–verify" round
    # (the first round that has a rejection and accepted at least 2 tokens)
    P = m2.prompts()
    trace: list = []
    m2.speculative_greedy(target, draft, P[1], 120, 4, trace)
    ex = next(t for t in trace if 2 <= t[2] < 4)
    d["example"] = {
        "drafts": [data.decode([x]) for x in ex[0]],
        "choice": [data.decode([x]) for x in ex[1]],
        "m": ex[2],
    }
    N = 200
    base = [m2.greedy_generate(target, p, N) for p in P]
    d["identical"] = all(
        m2.speculative_greedy(target, draft, p, N, k)[0] == b
        for k in (1, 3, 5, 8)
        for p, b in zip(P, base)
    )
    stats = {}
    for k in (1, 2, 3, 4, 6, 8):
        agg = dict(rounds=0, accepted=0, examined=0)
        for p in P:
            _, s = m2.speculative_greedy(target, draft, p, N, k)
            for key in agg:
                agg[key] += s[key]
        stats[k] = (agg["accepted"] / agg["examined"], agg["rounds"])
    d["stats"] = {str(k): v for k, v in stats.items()}
    tb, ts = [], []
    for _ in range(3):
        t0 = time.process_time()
        for p in P:
            m2.greedy_generate(target, p, N)
        tb.append(time.process_time() - t0)
        t0 = time.process_time()
        for p in P:
            m2.speculative_greedy(target, draft, p, N, 3)
        ts.append(time.process_time() - t0)
    d["speed_k3"] = (statistics.median(tb), statistics.median(ts))

    # 03: toy example of one rejection-sampling step
    r = m3.toy_single_step()
    d["toy"] = {key: [float(x) for x in r[key]] for key in ("p", "q", "spec", "naive")}
    d["toy_alpha"] = (r["accept"], r["alpha"])
    d["toy_tv"] = (m3.tv(r["spec"], r["p"]), m3.tv(r["naive"], r["p"]))
    d["toy_chi_p"] = r["chi"][2]

    # 04: speedup curves (c = 0.05)
    alphas = [0.3 + 0.01 * i for i in range(67)]
    d["curves"] = {str(k): [(a, m4.speedup(a, k, 0.05)) for a in alphas] for k in (1, 3, 6)}

    # 05: MTP
    lm, head = m5.load_mtp(0)
    base0 = m1.ch10.load_or_train(n_kv_heads=4, steps=600, seed=0, verbose=False)
    d["mtp_eval"] = (m5.evaluate(base0), m5.evaluate(lm, head))
    outs = [m5.mtp_self_speculative(lm, head, p, N) for p in P]
    ref = [m5.greedy(lm, p, N) for p in P]
    d["mtp_identical"] = all(o[0] == r for o, r in zip(outs, ref))
    d["mtp_accept"] = sum(o[2] for o in outs) / sum(o[1] for o in outs)
    d["mtp_rounds"] = sum(o[1] for o in outs)
    return d


def get_data() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    d = compute()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


D = get_data()


def mono(text: str, size: float = 24, color: str = theme.FG) -> Text:
    return Text(text, font=MONO, font_size=size, color=color)


def tok_box(ch: str, color: str, w: float = 0.62, opacity: float = 0.25) -> VGroup:
    """Draw one token box. A space shows as ␣, a newline shows as ↵."""
    shown = {" ": "␣", "\n": "↵"}.get(ch, ch)
    box = RoundedRectangle(
        width=w,
        height=w,
        corner_radius=0.08,
        stroke_color=color,
        fill_color=color,
        fill_opacity=opacity,
        stroke_width=2,
    )
    return VGroup(box, mono(shown, 26, theme.FG).move_to(box))


def bars(
    values: list[float], color: str, width: float = 0.34, scale: float = 4.0, opacity: float = 0.75
) -> VGroup:
    """Draw bars with aligned bases (left to right), height = value × scale."""
    g = VGroup()
    for i, v in enumerate(values):
        r = Rectangle(
            width=width,
            height=max(v * scale, 0.001),
            stroke_width=0,
            fill_color=color,
            fill_opacity=opacity,
        )
        r.move_to([i * (width * 2.2), 0, 0], aligned_edge=DOWN)
        g.add(r)
    return g


def table(
    rows: list[tuple],
    col_w: list[float],
    size: float = 22,
    row_h: float = 0.46,
    colors: list[str] | None = None,
) -> VGroup:
    out = VGroup()
    for r, row in enumerate(rows):
        x = 0.0
        line = VGroup()
        for c, (cell, w) in enumerate(zip(row, col_w)):
            color = theme.MUTED if r == 0 else (colors[c] if colors else theme.FG)
            t = zh(str(cell), size, color).move_to([x + w / 2, -r * row_h, 0])
            line.add(t)
            x += w
        out.add(line)
    return out


class ChapterScene(NarratedScene):
    chapter_label = "第 25 章"
    chapter_title = "多 token 预测与推测解码"

    def construct(self) -> None:
        for i in range(1, 15):
            getattr(self, f"s{i:02d}")()

    # ── S01 Opening ──────────────────────────────────────────────────────
    def s01(self):
        with self.shot("S01"):
            card = self.chapter_card()
            sub = zh("让小模型先猜，大模型一次改完", 32, theme.HIGHLIGHT).next_to(card, DOWN, 0.6)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.wait(max(0.1, self.remaining() - 0.8))
            self.play(FadeOut(card), FadeOut(sub), run_time=self.fit(0.8))

    # ── S02 Decode makes one token at a time; the compute is not used fully ───
    def s02(self):
        with self.shot("S02"):
            self.play(*self.set_heading("decode：一次只出一个 token"), run_time=self.fit(0.8))
            steps = VGroup()
            for i in range(5):
                blk = Rectangle(
                    width=1.0, height=0.6, stroke_width=0, fill_color=theme.PARAM, fill_opacity=0.7
                )
                t = tok_box("tok"[i % 3], theme.OUTPUT, w=0.5).next_to(blk, DOWN, 0.15)
                steps.add(VGroup(blk, t))
            steps.arrange(RIGHT, buff=0.25).move_to([-3.4, 1.4, 0])
            lab = zh("每一步：读一遍全部权重 → 只算 1 个 token", 22, theme.MUTED)
            lab.next_to(steps, DOWN, 0.35)
            self.play(
                LaggedStart(*[FadeIn(s) for s in steps], lag_ratio=0.3), run_time=self.fit(2.5)
            )
            self.play(FadeIn(lab), run_time=self.fit(0.6))
            # Right side: time of one forward pass with T tokens (relative to T=1)
            ax = Axes(
                x_range=[0, 18, 4],
                y_range=[0, 2, 0.5],
                x_length=5.2,
                y_length=3.0,
                axis_config={"color": theme.MUTED, "include_numbers": True, "font_size": 20},
            ).move_to([3.5, 0.2, 0])
            xl = zh("一次喂几个 token", 20, theme.MUTED).next_to(ax, DOWN, 0.15)
            yl = zh("耗时（相对 1 个）", 20, theme.MUTED).next_to(ax, UP, 0.1).align_to(ax, LEFT)
            pts = VGroup(
                *[
                    RoundedRectangle(
                        width=0.14,
                        height=0.14,
                        corner_radius=0.07,
                        stroke_width=0,
                        fill_color=theme.PARAM,
                        fill_opacity=1,
                    ).move_to(ax.c2p(T, r))
                    for T, r in D["cost"]
                ]
            )
            ideal = ax.plot(lambda x: x, x_range=[1, 2], color=theme.GRAD, stroke_width=2)
            ideal_lab = zh("红线：如果按计算量涨", 18, theme.GRAD).move_to(ax.c2p(12, 0.55))
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=self.fit(1.2))
            self.play(LaggedStart(*[FadeIn(p) for p in pts], lag_ratio=0.2), run_time=self.fit(1.5))
            self.play(Create(ideal), FadeIn(ideal_lab), run_time=self.fit(1.0))
            r17 = dict(D["cost"]).get(17, D["cost"][-1][1])
            note = zh(f"喂 17 个只慢 {r17:.1f} 倍（本机 CPU 实测）", 22, theme.HIGHLIGHT)
            note.move_to([0, -2.2, 0])
            self.play(FadeIn(note), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.8)
            self.play(
                FadeOut(VGroup(steps, lab, ax, xl, yl, pts, ideal, ideal_lab, note)),
                run_time=self.fit(0.8),
            )

    # ── S03 The draft guesses k tokens; the target checks them in one pass ───
    def s03(self):
        with self.shot("S03"):
            self.play(*self.set_heading("推测解码：草稿先猜，目标一次检查"), run_time=self.fit(0.8))
            ex = D["example"]
            k = len(ex["drafts"])
            dl = zh("草稿模型（1 层）猜 4 个：", 24, theme.INPUT).move_to([-4.2, 1.8, 0])
            drow = VGroup(*[tok_box(c, theme.INPUT) for c in ex["drafts"]]).arrange(RIGHT, buff=0.2)
            drow.next_to(dl, RIGHT, 0.4)
            self.play(FadeIn(dl), run_time=self.fit(0.6))
            self.play(
                LaggedStart(*[FadeIn(b, shift=RIGHT * 0.2) for b in drow], lag_ratio=0.4),
                run_time=self.fit(2.0),
            )
            tl = zh("目标模型一次前向的答案：", 24, theme.PARAM).move_to([-4.2, 0.3, 0])
            trow = VGroup(*[tok_box(c, theme.PARAM) for c in ex["choice"]]).arrange(RIGHT, buff=0.2)
            trow.next_to(tl, RIGHT, 0.4).align_to(drow, LEFT)
            bracket = Rectangle(
                width=trow.width + 0.3,
                height=trow.height + 0.3,
                stroke_color=theme.PARAM,
                stroke_width=2,
            ).move_to(trow)
            self.play(FadeIn(tl), Create(bracket), FadeIn(trow), run_time=self.fit(1.5))
            marks = VGroup()
            for i in range(k):
                ok = i < ex["m"]
                sym = mono("✓" if ok else "✗", 30, theme.OUTPUT if ok else theme.GRAD)
                sym.next_to(drow[i], UP, 0.12)
                marks.add(sym)
                if not ok:
                    break
            self.play(
                LaggedStart(*[FadeIn(s) for s in marks], lag_ratio=0.5), run_time=self.fit(2.0)
            )
            m = ex["m"]
            out = VGroup(
                *[tok_box(c, theme.OUTPUT, opacity=0.4) for c in ex["drafts"][:m]],
                tok_box(ex["choice"][m], theme.OUTPUT, opacity=0.4),
            )
            out.arrange(RIGHT, buff=0.2).move_to([0.6, -1.3, 0]).align_to(drow, LEFT)
            ol = zh("本轮产出：", 24, theme.OUTPUT).next_to(out, LEFT, 0.4)
            note = zh(
                f"接受 {m} 个 + 目标自己的 1 个纠正 = {m + 1} 个 token，只跑了 1 次目标模型",
                22,
                theme.HIGHLIGHT,
            ).move_to([0, -2.25, 0])
            self.play(FadeIn(ol), FadeIn(out), run_time=self.fit(1.0))
            self.play(FadeIn(note), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.8)
            self.play(
                FadeOut(VGroup(dl, drow, tl, trow, bracket, marks, out, ol, note)),
                run_time=self.fit(0.8),
            )

    # ── S04 Greedy output is identical token for token + rollback ───────────
    def s04(self):
        with self.shot("S04"):
            self.play(
                *self.set_heading("贪心：输出逐字不变；拒绝后回滚缓存"), run_time=self.fit(0.8)
            )
            code = code_block(
                """m = 0
while m < k and drafts[m] == choice[m]:
    m += 1
seq += drafts[:m] + [choice[m]]
truncate(cache, len(seq) - 1)   # 回滚""",
                size=22,
            ).move_to([-3.0, 0.9, 0])
            self.play(FadeIn(code), run_time=self.fit(1.5))
            cache = (
                VGroup(
                    *[
                        Rectangle(
                            width=0.36,
                            height=0.5,
                            stroke_width=1,
                            stroke_color=theme.ATTN,
                            fill_color=theme.ATTN if i < 6 else theme.GRAD,
                            fill_opacity=0.45,
                        )
                        for i in range(9)
                    ]
                )
                .arrange(RIGHT, buff=0.04)
                .move_to([3.4, 1.0, 0])
            )
            cl = zh("KV cache", 22, theme.ATTN).next_to(cache, UP, 0.2)
            self.play(FadeIn(cache), FadeIn(cl), run_time=self.fit(1.0))
            self.play(*[FadeOut(c) for c in cache[6:]], run_time=self.fit(1.2))
            ok = zh(
                f"4 段提示词 × 200 个字符，k = 1、3、5、8：与目标模型贪心解码逐字相同 = "
                f"{'是' if D['identical'] else '否'}",
                22,
                theme.OUTPUT,
            ).move_to([0, -1.4, 0])
            self.play(FadeIn(ok), run_time=self.fit(1.0))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(VGroup(code, cache[:6], cl, ok)), run_time=self.fit(0.8))

    # ── S05 Sampling: p and q ─────────────────────────────────────────────
    def _pq_axes(self):
        base = Line([-6.2, -1.6, 0], [-0.4, -1.6, 0], color=theme.MUTED, stroke_width=2)
        return base

    def s05(self):
        with self.shot("S05"):
            self.play(*self.set_heading("采样时：草稿分布 q ≠ 目标分布 p"), run_time=self.fit(0.8))
            p, q = D["toy"]["p"], D["toy"]["q"]
            base = self._pq_axes()
            pb = bars(p, theme.PARAM, width=0.3).move_to([-3.6, -1.6, 0], aligned_edge=DOWN)
            qb = bars(q, theme.INPUT, width=0.3).move_to([-3.6, -1.6, 0], aligned_edge=DOWN)
            qb.shift(RIGHT * 0.32)
            legend = VGroup(zh("p：目标", 22, theme.PARAM), zh("q：草稿", 22, theme.INPUT))
            legend.arrange(RIGHT, buff=0.6).move_to([-3.3, 2.2, 0])
            self.play(Create(base), FadeIn(legend), run_time=self.fit(0.8))
            self.play(FadeIn(pb, shift=UP * 0.3), run_time=self.fit(1.0))
            self.play(FadeIn(qb, shift=UP * 0.3), run_time=self.fit(1.0))
            rule = (
                VGroup(
                    zh("草稿按 q 抽到 x：", 24),
                    MathTex(
                        r"\text{accept with prob. } \min\!\left(1, \frac{p(x)}{q(x)}\right)",
                        font_size=34,
                        color=theme.HIGHLIGHT,
                    ),
                    zh("p(x) ≥ q(x)：一定接受", 22, theme.OUTPUT),
                    zh("p(x) < q(x)：草稿多给了，按比例拒绝", 22, theme.GRAD),
                )
                .arrange(DOWN, aligned_edge=LEFT, buff=0.3)
                .move_to([3.5, 0.6, 0])
            )
            self.play(
                LaggedStart(*[FadeIn(r) for r in rule], lag_ratio=0.5), run_time=self.fit(3.0)
            )
            a_meas, a_th = D["toy_alpha"]
            alpha = (
                VGroup(
                    MathTex(r"\alpha = \sum_x \min(p, q)", font_size=34, color=theme.FG),
                    zh(f"= {a_th:.3f}（实测接受率 {a_meas:.3f}）", 22, theme.MUTED),
                )
                .arrange(DOWN, buff=0.2)
                .move_to([3.5, -1.6, 0])
            )
            self.play(FadeIn(alpha), run_time=self.fit(1.0))
            self.wait(self.remaining() - 0.8)
            self._s05 = VGroup(base, pb, qb, legend)
            self.play(FadeOut(VGroup(rule, alpha)), run_time=self.fit(0.8))

    # ── S06 Residual distribution: the result is exactly p ─────────────────
    def s06(self):
        with self.shot("S06"):
            self.play(
                *self.set_heading("被拒绝时：从残差 max(0, p − q) 里重抽"), run_time=self.fit(0.8)
            )
            p, q = D["toy"]["p"], D["toy"]["q"]
            res = [max(0.0, a - b) for a, b in zip(p, q)]
            rb = bars(res, theme.GRAD, width=0.3).move_to([-3.6, -1.6, 0], aligned_edge=DOWN)
            rb.align_to(self._s05[1], LEFT)
            rl = zh("残差 max(0, p − q)", 22, theme.GRAD).move_to([-3.3, 1.7, 0])
            self.play(FadeOut(self._s05[2]), FadeIn(rb), FadeIn(rl), run_time=self.fit(1.5))
            eq = MathTex(
                r"P(x) = \underbrace{\min(p,q)}_{\text{accepted}} + "
                r"\underbrace{(1-\alpha)\,\frac{\max(0,p-q)}{1-\alpha}}_{\text{resampled}} = p(x)",
                font_size=30,
                color=theme.FG,
            ).move_to([3.3, 1.0, 0])
            self.play(Write(eq), run_time=self.fit(2.5))
            tv_s, tv_n = D["toy_tv"]
            spec = D["toy"]["spec"]
            sb = bars(spec, theme.OUTPUT, width=0.3, opacity=0.9).move_to(
                [-3.6, -1.6, 0], aligned_edge=DOWN
            )
            sb.align_to(self._s05[1], LEFT).shift(RIGHT * 0.32)
            res_txt = (
                VGroup(
                    zh("抽 20 万次的结果（绿）与 p（橙）重合：", 22, theme.OUTPUT),
                    zh(f"TV 距离 {tv_s:.4f}，卡方检验 p 值 {D['toy_chi_p']:.2f}", 22),
                    zh(f"直接用草稿的样本：TV 距离 {tv_n:.2f}", 22, theme.GRAD),
                )
                .arrange(DOWN, aligned_edge=LEFT, buff=0.25)
                .move_to([3.3, -1.1, 0])
            )
            self.play(FadeOut(rb), FadeOut(rl), FadeIn(sb), run_time=self.fit(1.2))
            self.play(FadeIn(res_txt), run_time=self.fit(1.2))
            self.wait(self.remaining() - 0.8)
            self.play(
                FadeOut(VGroup(self._s05[0], self._s05[1], self._s05[3], sb, eq, res_txt)),
                run_time=self.fit(0.8),
            )

    # ── S07 One full round ──────────────────────────────────────────────
    def s07(self):
        with self.shot("S07"):
            self.play(*self.set_heading("一轮最多 k+1 个，至少 1 个"), run_time=self.fit(0.8))
            k = 4
            rows = VGroup()
            for m in range(k + 1):
                cells = VGroup()
                for i in range(k + 1):
                    if i < m:
                        col, op = theme.OUTPUT, 0.5
                    elif i == m:
                        col, op = (theme.HIGHLIGHT if m == k else theme.GRAD), 0.6
                    else:
                        col, op = theme.MUTED, 0.1
                    cells.add(
                        Rectangle(
                            width=0.5,
                            height=0.4,
                            stroke_width=1,
                            stroke_color=col,
                            fill_color=col,
                            fill_opacity=op,
                        )
                    )
                cells.arrange(RIGHT, buff=0.05)
                lab = zh(f"接受 {m} 个 → 产出 {m + 1} 个", 20, theme.MUTED).next_to(
                    cells, RIGHT, 0.3
                )
                rows.add(VGroup(cells, lab))
            rows.arrange(DOWN, buff=0.18, aligned_edge=LEFT).move_to([-3.2, 0.3, 0])
            leg = VGroup(
                zh("绿：接受的草稿", 20, theme.OUTPUT),
                zh("红：纠正", 20, theme.GRAD),
                zh("黄：全对时的奖励", 20, theme.HIGHLIGHT),
            )
            leg.arrange(RIGHT, buff=0.4).move_to([-3.0, -2.2, 0])
            self.play(
                LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.3), run_time=self.fit(2.5)
            )
            self.play(FadeIn(leg), run_time=self.fit(0.6))
            f = (
                VGroup(
                    MathTex(r"E[\text{tokens}] = 1 + \alpha + \cdots + \alpha^k", font_size=32),
                    MathTex(
                        r"= \frac{1-\alpha^{k+1}}{1-\alpha}", font_size=36, color=theme.HIGHLIGHT
                    ),
                    MathTex(
                        r"\text{speedup} = \frac{1-\alpha^{k+1}}{(1-\alpha)(1+kc)}", font_size=32
                    ),
                    zh("c = 草稿一步 ÷ 目标一步", 20, theme.MUTED),
                )
                .arrange(DOWN, buff=0.3)
                .move_to([3.6, 0.4, 0])
            )
            self.play(LaggedStart(*[FadeIn(x) for x in f], lag_ratio=0.6), run_time=self.fit(3.0))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(VGroup(rows, leg, f)), run_time=self.fit(0.8))

    # ── S08 Speedup curves ───────────────────────────────────────────────
    def s08(self):
        with self.shot("S08"):
            self.play(*self.set_heading("加速比 vs 接受率 α（c = 0.05）"), run_time=self.fit(0.8))
            ax = Axes(
                x_range=[0.3, 0.97, 0.1],
                y_range=[1, 4.5, 0.5],
                x_length=7.0,
                y_length=4.0,
                axis_config={"color": theme.MUTED, "include_numbers": True, "font_size": 20},
            ).move_to([-1.4, 0.0, 0])
            xl = MathTex(r"\alpha", font_size=30, color=theme.MUTED).next_to(
                ax.x_axis.get_end(), DOWN, 0.25
            )
            yl = zh("加速比", 20, theme.MUTED).next_to(ax, UP, 0.1)
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=self.fit(1.2))
            colors = {"1": theme.INPUT, "3": theme.OUTPUT, "6": theme.ATTN}
            labels = VGroup()
            for k, pts in D["curves"].items():
                line = polyline_in_axes(ax, pts, color=colors[k], stroke_width=4)
                lab = zh(f"k = {k}", 22, colors[k]).next_to(ax.c2p(*pts[-1]), RIGHT, 0.1)
                labels.add(lab)
                self.play(Create(line), FadeIn(lab), run_time=self.fit(1.2))
                labels.add(line)
            note = VGroup(
                zh("α 低：猜多了白费草稿的算力", 22, theme.GRAD),
                zh("α 高：值得多猜几个", 22, theme.OUTPUT),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
            note.move_to(ax.c2p(0.3, 4.1), aligned_edge=UP + LEFT).shift(RIGHT * 0.3)
            self.play(FadeIn(note), run_time=self.fit(1.0))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(VGroup(ax, xl, yl, labels, note)), run_time=self.fit(0.8))

    # ── S09 Measurements ────────────────────────────────────────────────
    def s09(self):
        with self.shot("S09"):
            self.play(*self.set_heading("小实验：0.86M 目标 + 0.06M 草稿"), run_time=self.fit(0.8))
            badge = self.demo_badge("极小规模实验")
            self.play(FadeIn(badge), run_time=0.3)
            rows = [("k", "接受率 α", "目标前向次数", "每次前向产出")]
            for k in ("1", "3", "6"):
                a, r = D["stats"][k]
                rows.append((k, f"{a:.2f}", f"{r}（原来 800）", f"{800 / r:.2f} 个"))
            t = table(rows, [1.0, 2.2, 3.2, 2.6], size=24).move_to([0, 0.9, 0])
            self.play(FadeIn(t), run_time=self.fit(1.5))
            tb, ts = D["speed_k3"]
            sp = (
                VGroup(
                    zh(
                        f"k = 3 的 CPU 时间：{tb:.1f} s → {ts:.1f} s，加速 {tb / ts:.2f} 倍",
                        24,
                        theme.HIGHLIGHT,
                    ),
                    zh(
                        f"公式按 α 与 c≈{D['c']:.2f} 预测更高：验证 k+1 个并不完全免费，Python 循环也有开销",
                        20,
                        theme.MUTED,
                    ),
                )
                .arrange(DOWN, buff=0.25)
                .move_to([0, -1.4, 0])
            )
            self.play(FadeIn(sp), run_time=self.fit(1.2))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(VGroup(t, sp, badge)), run_time=self.fit(0.8))

    # ── S10 Where the draft comes from ──────────────────────────────────
    def s10(self):
        with self.shot("S10"):
            self.play(*self.set_heading("草稿从哪来"), run_time=self.fit(0.8))
            items = [
                ("同家族的小模型", "同一个分词器\n最好再用大模型的输出蒸馏", theme.INPUT),
                ("提示词查找", "从上下文里复制\n代价几乎为零", theme.PARAM),
                ("自带草稿头：MTP", "读目标模型的隐藏状态\n猜得最准", theme.OUTPUT),
            ]
            cards = VGroup()
            for title, desc, col in items:
                box = RoundedRectangle(
                    width=3.9, height=2.2, corner_radius=0.15, stroke_color=col, stroke_width=2
                )
                tt = zh(title, 24, col).move_to(box.get_top() + DOWN * 0.5)
                dd = zh(desc, 20, theme.FG, line_spacing=1.2).next_to(tt, DOWN, 0.35)
                if dd.width > 3.5:
                    dd.scale_to_fit_width(3.5)
                cards.add(VGroup(box, tt, dd))
            cards.arrange(RIGHT, buff=0.35).move_to([0, 0.4, 0])
            self.play(
                LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in cards], lag_ratio=0.5),
                run_time=self.fit(3.0),
            )
            note = zh("无论草稿是谁，验证规则保证输出分布不变；草稿只影响快慢", 22, theme.HIGHLIGHT)
            note.move_to([0, -1.8, 0])
            self.play(FadeIn(note), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(VGroup(cards, note)), run_time=self.fit(0.8))

    # ── S11 Structure of the MTP module ─────────────────────────────────
    def s11(self):
        with self.shot("S11"):
            self.play(*self.set_heading("DeepSeek-V3 的 MTP 模块"), run_time=self.fit(0.8))

            def block(text, col, w=2.6, h=0.55):
                b = RoundedRectangle(
                    width=w,
                    height=h,
                    corner_radius=0.1,
                    stroke_color=col,
                    fill_color=col,
                    fill_opacity=0.2,
                    stroke_width=2,
                )
                return VGroup(b, zh(text, 20, theme.FG).move_to(b))

            main = block("主模型（L 层）", theme.PARAM, w=3.0).move_to([-4.3, -0.9, 0])
            hi = zh("表示 h[i]", 20, theme.PARAM).next_to(main, UP, 0.2).shift(RIGHT * 0.9)
            ti = zh("输入 t[i]", 20, theme.INPUT).next_to(main, DOWN, 0.2)
            emb = block("Emb(t[i+1])（共享）", theme.INPUT, w=2.8).move_to([-0.6, -0.9, 0])
            n1 = block("RMSNorm", theme.MUTED, w=1.6).move_to([-4.3, 0.55, 0])
            n2 = block("RMSNorm", theme.MUTED, w=1.6).move_to([-0.6, 0.55, 0])
            proj = block("拼接 → 线性投影 2d→d", theme.ATTN, w=3.6).move_to([-2.45, 1.5, 0])
            trm = block("1 个 Transformer block", theme.ATTN, w=3.6).move_to([2.6, 1.5, 0])
            head = block("输出头（共享）", theme.OUTPUT, w=2.6).move_to([2.6, 0.45, 0])
            out = zh("预测 t[i+2]", 22, theme.OUTPUT).next_to(head, DOWN, 0.3)
            arrows = VGroup(
                Arrow(main.get_top(), n1.get_bottom(), buff=0.1, color=theme.MUTED),
                Arrow(emb.get_top(), n2.get_bottom(), buff=0.1, color=theme.MUTED),
                Arrow(n1.get_top(), proj.get_bottom() + LEFT * 1.0, buff=0.1, color=theme.MUTED),
                Arrow(n2.get_top(), proj.get_bottom() + RIGHT * 1.0, buff=0.1, color=theme.MUTED),
                Arrow(proj.get_right(), trm.get_left(), buff=0.1, color=theme.MUTED),
                Arrow(trm.get_bottom(), head.get_top(), buff=0.1, color=theme.MUTED),
            )
            self.play(FadeIn(main), FadeIn(ti), FadeIn(hi), run_time=self.fit(1.0))
            self.play(
                FadeIn(emb),
                FadeIn(n1),
                FadeIn(n2),
                Create(arrows[0]),
                Create(arrows[1]),
                run_time=self.fit(1.5),
            )
            self.play(FadeIn(proj), Create(arrows[2]), Create(arrows[3]), run_time=self.fit(1.2))
            self.play(FadeIn(trm), Create(arrows[4]), run_time=self.fit(1.0))
            self.play(FadeIn(head), Create(arrows[5]), FadeIn(out), run_time=self.fit(1.0))
            loss = MathTex(
                r"\mathcal{L} = \mathcal{L}_{\text{main}} + \lambda\,\mathcal{L}_{\text{MTP}}",
                font_size=32,
                color=theme.HIGHLIGHT,
            ).move_to([4.6, -1.3, 0])
            lam = zh("V3：λ = 0.3，后期 0.1", 18, theme.MUTED).next_to(loss, DOWN, 0.15)
            self.play(FadeIn(loss), FadeIn(lam), run_time=self.fit(1.0))
            self.wait(self.remaining() - 0.8)
            self.play(
                FadeOut(VGroup(main, hi, ti, emb, n1, n2, proj, trm, head, out, arrows, loss, lam)),
                run_time=self.fit(0.8),
            )

    # ── S12 MTP small experiment ───────────────────────────────────────
    def s12(self):
        with self.shot("S12"):
            self.play(
                *self.set_heading("小实验：给第 10 章的模型加一个 MTP 模块"), run_time=self.fit(0.8)
            )
            badge = self.demo_badge("极小规模实验")
            self.play(FadeIn(badge), run_time=0.3)
            (lb, ab, _), (lm, am, a2) = D["mtp_eval"]
            rows = [
                ("", "验证 loss", "下一个字符准确率", "下下个字符准确率"),
                ("不加 MTP", f"{lb:.3f}", f"{ab:.3f}", "—"),
                ("加 MTP（λ=0.3）", f"{lm:.3f}", f"{am:.3f}", f"{a2:.3f}（MTP 模块）"),
            ]
            t = table(rows, [2.8, 1.9, 2.8, 3.2], size=22).move_to([0, 1.2, 0])
            self.play(FadeIn(t), run_time=self.fit(1.5))
            sp = (
                VGroup(
                    zh(
                        f"拿 MTP 当草稿做自推测：接受率 {D['mtp_accept']:.2f}，"
                        f"主模型前向 800 → {D['mtp_rounds']} 次",
                        24,
                        theme.OUTPUT,
                    ),
                    zh(
                        f"输出与主模型贪心解码逐字相同 = {'是' if D['mtp_identical'] else '否'}", 22
                    ),
                    zh(
                        "DeepSeek-V3：第二个 token 接受率 85%–90%，解码速度约 1.8 倍",
                        22,
                        theme.HIGHLIGHT,
                    ),
                )
                .arrange(DOWN, buff=0.3)
                .move_to([0, -1.2, 0])
            )
            self.play(LaggedStart(*[FadeIn(s) for s in sp], lag_ratio=0.5), run_time=self.fit(2.5))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(VGroup(t, sp, badge)), run_time=self.fit(0.8))

    # ── S13 Adopters ────────────────────────────────────────────────────
    def s13(self):
        with self.shot("S13"):
            self.play(*self.set_heading("谁在用"), run_time=self.fit(0.8))
            rows = [
                ("模型家族", "MTP（config / 技术报告）"),
                ("DeepSeek", "V3、V4：num_nextn_predict_layers = 1"),
                ("千问 Qwen", "Qwen3-Next、Qwen3.5（连 0.8B 都有）"),
                ("智谱 GLM", "GLM-4.5、GLM-5：1 层 MTP，用于推测解码"),
                ("MiniMax / 小米 MiMo", "M2：3 个 MTP 模块；MiMo-7B、V2-Flash"),
                ("NVIDIA / Google", "Nemotron 3 Super；Gemma 4 发布 MTP 草稿模型"),
            ]
            t = table(rows, [3.6, 6.8], size=22, row_h=0.52).move_to([0, 0.6, 0])
            self.play(LaggedStart(*[FadeIn(r) for r in t], lag_ratio=0.3), run_time=self.fit(3.0))
            note = zh(
                "推理引擎 vLLM、SGLang 都内置推测解码：草稿模型、n-gram、MTP、EAGLE",
                22,
                theme.HIGHLIGHT,
            ).move_to([0, -2.0, 0])
            self.play(FadeIn(note), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(VGroup(t, note)), run_time=self.fit(0.8))

    # ── S14 From minimal code to production code + next chapter ─────────
    def s14(self):
        with self.shot("S14"):
            self.play(*self.set_heading("从极简到生产级"), run_time=self.fit(0.8))
            left = VGroup(
                zh("zero/arch/speculative.py", 24, theme.INPUT),
                zh("预分配 KV cache 回滚、top-p、提示词查找", 20, theme.MUTED),
                zh("zero/arch/mtp.py", 24, theme.OUTPUT),
                zh("DeepSeek 式 MTP 模块 + 损失 + 自推测", 20, theme.MUTED),
                zh("测试：贪心逐字相同、分布不变、梯度流动", 20, theme.FG),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
            left.scale_to_fit_width(min(left.width, 6.0)).move_to([-3.6, 0.5, 0])
            right = VGroup(
                zh("主线模型不加 MTP（不冒架构风险）", 22, theme.PARAM),
                zh("第二步：另训一个同分词器的小草稿", 22, theme.FG),
                zh("vLLM / SGLang 上的实测加速尚未验证", 20, theme.MUTED),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
            right.scale_to_fit_width(min(right.width, 6.0)).move_to([3.6, 0.5, 0])
            self.play(FadeIn(left), run_time=self.fit(1.5))
            self.play(FadeIn(right), run_time=self.fit(1.5))
            nxt = zh("下一章：当前最先进开源模型全景", 28, theme.HIGHLIGHT).move_to([0, -1.9, 0])
            self.wait(max(0.1, self.remaining() - 2.5))
            self.play(FadeIn(nxt), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.8)
            self.play(
                FadeOut(VGroup(left, right, nxt)), *self.set_heading(None), run_time=self.fit(0.8)
            )
