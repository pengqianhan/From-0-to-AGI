"""第 15 章视频：中期训练与长上下文 —— 最后一段怎么训，读不长怎么办

画面里的数值都由 ../code/ 中的代码真实计算（见 script.md 事实清单）：
  01_rope_wavelengths.py、02_yarn_from_scratch.py 现算；
  03_context_extension.py、04_anneal_mixture.py 读它们的缓存 code/out/*.pt（先运行这两个脚本）。
tiny 配置的 zero 运行日志（S15）来自 README"主线进度"里的两条命令，数字写在 TINY_RUN 里。
结果缓存在 video/out/cache.json；删掉它会重新计算。
渲染：bash chapters/15-midtraining-long-context/video/build.sh
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    PI,
    RIGHT,
    UP,
    Arc,
    Axes,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    LaggedStart,
    Line,
    MathTex,
    Rectangle,
    SurroundingRectangle,
    Text,
    ValueTracker,
    VGroup,
    Write,
    always_redraw,
)

from video_kit import theme
from video_kit.scene import NarratedScene, code_block, polyline_in_axes, zh

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
ROOT = HERE.parents[2]
CACHE = HERE / "out" / "cache.json"
MONO = "Noto Sans Mono"

NIAH_LENGTHS, NIAH_DEPTHS = (64, 128, 240), (0.0, 0.5, 1.0)

# tiny 配置的真实运行日志（README"主线进度"的两条命令；out/tiny/ch15/）
TINY_RUN = {
    "pre_val": [(100, 6.0048), (200, 5.7116)],
    "mid_changes": ["rope_scaling: None → yarn ×2（原长 128）", "seq_len: 128 → 256",
                    "配比：英 0.45 / 中 0.45 / 代码 0.1 → 0.3 / 0.6 / 0.1"],
    "mid_val": [(30, 5.6488), (60, 5.6101)],
}


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def compute() -> dict:
    import torch

    w1 = _load("rope_wavelengths", "01_rope_wavelengths.py")
    y = _load("yarn_from_scratch", "02_yarn_from_scratch.py")
    sys.path.insert(0, str(ROOT))
    from zero.config import load_model_config
    from zero.model import compute_rope_inv_freq, estimate_flops_per_token

    d: dict = {}
    bases = (1e4, 5e5, 1e6)
    d["wl"] = {str(i): [float(w1.wavelengths(128, b)[i]) for b in bases] for i in (0, 16, 32, 48, 63)}
    d["not_full"] = {str(L): [int((w1.wavelengths(128, b) > L).sum()) for b in bases]
                     for L in (4096, 32768)}
    wt = w1.inv_freq(128, 1e4)
    d["unseen"] = {"none": w1.unseen_pairs(wt, wt, 4096, 32768),
                   "abf": w1.unseen_pairs(wt, w1.inv_freq(128, 1e6), 4096, 32768),
                   "pi": w1.unseen_pairs(wt, wt / 8, 4096, 32768)}
    d["slow"] = [float(wt[i] / w1.inv_freq(128, 1e6)[i]) for i in (0, 16, 32, 48, 63)]
    d["clock_w"] = {str(i): float(wt[i]) for i in (24, 48, 63)}
    w32 = y.rope_inv_freq(32, 1e4)
    wy, keep, m = y.yarn_inv_freq(32, 1e4, 4.0, 64)
    d["tiny_keep"] = keep.tolist()
    d["tiny_turns"] = (64 * w32 / (2 * math.pi)).tolist()
    d["mscale4"] = m
    diffs = []
    for dim, theta, s, L in ((32, 1e4, 4.0, 64), (128, 1e6, 4.0, 32768)):
        mine, _, _ = y.yarn_inv_freq(dim, theta, s, L)
        ref, _ = compute_rope_inv_freq(dim, theta, {"factor": s, "original_max_position_embeddings": L})
        diffs.append(float((mine - ref).abs().max()))
    d["parity"] = max(diffs)
    cfg = load_model_config(ROOT / "configs/main/pretrain.toml")
    f0 = estimate_flops_per_token(cfg, 0)
    d["flops"] = {str(T): [estimate_flops_per_token(cfg, T) / 1e9,
                           1 - f0 / estimate_flops_per_token(cfg, T)] for T in (4096, 32768)}
    r3 = torch.load(CODE / "out" / "context_extension.pt", weights_only=False)
    d["ctx"] = {k: {v: r3[k][v]["len"] for v in ("none", "pi", "yarn", "abf")}
                for k in ("zero_shot", "finetuned")}
    d["ctx"] = {k: {v: {str(L): x for L, x in lens.items()} for v, lens in vv.items()}
                for k, vv in d["ctx"].items()}
    d["ctx_tokens"] = [r3["tokens_pre"], r3["tokens_ft"]]
    r4 = torch.load(CODE / "out" / "anneal_mixture.pt", weights_only=False)
    d["anneal"] = {"trunk": r4["trunk"], "branches": r4["branches"],
                   "steps": [r4["trunk_steps"], r4["branch_steps"]]}
    # 大海捞针：tiny 中期训练的结果（README"主线进度"的命令产生 out/tiny/ch15/midtrain/ckpt）
    from zero.post.common import load_policy
    from zero.tools.needle import run_grid

    torch.set_num_threads(1)
    model, tok = load_policy(ROOT / "out/tiny/ch15/midtrain/ckpt", ROOT / "out/tiny/tokenizer.json")
    res = run_grid(model.eval(), tok, NIAH_LENGTHS, NIAH_DEPTHS, n=5)
    d["niah"] = {f"{r.length}_{r.depth}": [r.accuracy, r.nll_gain] for r in res}
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


def box(w: float, h: float, color: str, opacity: float = 0.85) -> Rectangle:
    return Rectangle(width=max(w, 0.02), height=max(h, 0.02), fill_color=color,
                     fill_opacity=opacity, stroke_width=0)


def fmt_len(x: float) -> str:
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    if x >= 1e4:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}" if x >= 100 else f"{x:.1f}"


VARIANT_COLOR = {"none": theme.GRAD, "pi": theme.PARAM, "yarn": theme.OUTPUT, "abf": theme.INPUT}
VARIANT_NAME = {"none": "不改", "pi": "PI", "yarn": "YaRN", "abf": "调大基频"}


class ChapterScene(NarratedScene):
    chapter_label = "第 15 章"
    chapter_title = "中期训练与长上下文"

    def construct(self) -> None:
        for i in range(1, 17):
            getattr(self, f"s{i:02d}")()

    # ── S01 片头 ─────────────────────────────────────────────────────────
    def s01(self) -> None:
        with self.shot("S01"):
            card = self.chapter_card()
            sub = zh("最后一段怎么训，读不长怎么办", 30, theme.HIGHLIGHT).next_to(card, DOWN, 0.6)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.wait(max(0.05, self.remaining() - 0.8))
            self.play(FadeOut(card), FadeOut(sub), run_time=self.fit(0.8))

    # ── S02 两个问题：时间线 ─────────────────────────────────────────────
    def s02(self) -> None:
        with self.shot("S02"):
            self.play(*self.set_heading("预训练收尾的两个问题"), run_time=self.fit(0.8))
            segs = [("预训练（稳定段）", 5.0, theme.MUTED, "4K 片段 · 学习率恒定"),
                    ("中期训练", 1.9, theme.PARAM, "学习率降到 0"),
                    ("长上下文", 1.6, theme.ATTN, "32K 片段")]
            x = -6.2
            bars = VGroup()
            for name, w, c, note in segs:
                b = box(w, 0.7, c, 0.75).move_to([x + w / 2, 1.2, 0])
                t = zh(name, 22, theme.BG if c != theme.MUTED else theme.BG).move_to(b)
                n = zh(note, 18, theme.MUTED).next_to(b, DOWN, 0.15)
                bars.add(VGroup(b, t, n))
                x += w + 0.08
            base = zh("Base 模型 → 闸门 2", 24, theme.HIGHLIGHT).move_to([4.85, 1.2, 0])
            self.play(LaggedStart(*[GrowFromEdge(g[0], LEFT) for g in bars], lag_ratio=0.3),
                      run_time=self.fit(1.5))
            self.play(*[FadeIn(g[1:]) for g in bars], FadeIn(base), run_time=self.fit(0.8))
            q1 = zh("问题一：学习率最低的那一段，喂什么数据？", 28, t2c={"喂什么数据": theme.PARAM}
                    ).move_to([0, -0.5, 0])
            q2 = zh("问题二：只见过 4K 的模型，怎么读 32K？", 28, t2c={"怎么读 32K": theme.ATTN}
                    ).move_to([0, -1.4, 0])
            arr1 = Line(q1.get_top() + UP * 0.1, bars[1][2].get_bottom() + DOWN * 0.05,
                        color=theme.PARAM, stroke_width=2)
            self.wait(self.remaining() * 0.15)
            self.play(FadeIn(q1), Create(arr1), run_time=self.fit(1))
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(q2), run_time=self.fit(1))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(bars, base, q1, q2, arr1)), run_time=self.fit(0.6))

    # ── S03 WSD 曲线 + 衰减段换数据 ─────────────────────────────────────
    def s03(self) -> None:
        with self.shot("S03"):
            self.play(*self.set_heading("中期训练：衰减段换上最好的数据"), run_time=self.fit(0.8))
            ax = Axes(x_range=[0, 100, 20], y_range=[0, 1.1, 0.5], x_length=7.2, y_length=3.0,
                      tips=False, axis_config={"color": theme.MUTED, "include_ticks": False}
                      ).move_to([-2.6, 1.0, 0])
            xl = zh("训练进度", 20, theme.MUTED).next_to(ax.x_axis, DOWN, 0.15)
            yl = zh("学习率", 20, theme.MUTED).next_to(ax.y_axis, UP, 0.1)

            def lr(p: float) -> float:
                if p < 3:
                    return p / 3
                return 1.0 if p < 90 else 1 - (p - 90) / 10
            pts = [(p, lr(p)) for p in np.linspace(0, 100, 201)]
            stable = polyline_in_axes(ax, [q for q in pts if q[0] <= 90], color=theme.MUTED,
                                      stroke_width=4)
            decay = polyline_in_axes(ax, [q for q in pts if q[0] >= 90], color=theme.PARAM,
                                     stroke_width=5)
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=self.fit(1))
            self.play(Create(stable), run_time=self.fit(1.5))
            self.play(Create(decay), run_time=self.fit(1))
            band = Rectangle(width=ax.c2p(100, 0)[0] - ax.c2p(90, 0)[0], height=3.0,
                             fill_color=theme.PARAM, fill_opacity=0.15, stroke_width=0)
            band.move_to([(ax.c2p(90, 0)[0] + ax.c2p(100, 0)[0]) / 2, ax.get_center()[1], 0])
            lab = zh("5–10% 的算力", 20, theme.PARAM).next_to(band, UP, 0.08)
            self.play(FadeIn(band), FadeIn(lab), run_time=self.fit(0.6))
            # 右侧：两个配比条
            def mix(parts, y):
                g, x0 = VGroup(), 3.3
                for name, frac, c in parts:
                    b = box(3.4 * frac, 0.5, c, 0.85).move_to([x0 + 3.4 * frac / 2, y, 0])
                    g.add(b)
                    if frac >= 0.15:
                        g.add(zh(name, 16, theme.BG).move_to(b))
                    x0 += 3.4 * frac
                return g
            m1 = mix([("网页", 0.8, theme.MUTED), ("代码", 0.12, theme.INPUT),
                      ("", 0.08, theme.OUTPUT)], 1.8)
            m2 = mix([("高质量网页", 0.5, theme.MUTED), ("数学", 0.18, theme.OUTPUT),
                      ("代码", 0.17, theme.INPUT), ("指令", 0.15, theme.ATTN)], 0.4)
            t1 = zh("稳定段的配比", 20, theme.MUTED).next_to(m1, UP, 0.12)
            t2 = zh("衰减段的配比", 20, theme.PARAM).next_to(m2, UP, 0.12)
            self.play(FadeIn(t1), FadeIn(m1), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.15)
            self.play(FadeIn(t2), FadeIn(m2), run_time=self.fit(0.8))
            who = zh("OLMo 2 · Llama 3 · SmolLM3 · MiniCPM · Qwen3 · MobileLLM-R1", 22,
                     theme.FG).move_to([0, -1.6, 0])
            self.play(FadeIn(who), run_time=self.fit(0.8))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(ax, xl, yl, stable, decay, band, lab, m1, m2, t1, t2, who)),
                      run_time=self.fit(0.6))

    # ── S04 为什么有效 + OLMo 2 数字 ────────────────────────────────────
    def s04(self) -> None:
        with self.shot("S04"):
            self.play(*self.set_heading("为什么放在衰减段"), run_time=self.fit(0.8))
            pts = VGroup(
                zh("① 学习率降下来，参数落进谷底：这一段决定模型停在哪", 24),
                zh("② 好数据少，集中用在最有分量的一段", 24),
                zh("③ 从同一点分叉衰减，就能便宜地比较数据", 24),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([-2.4, 1.3, 0])
            for p in pts:
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=self.fit(0.7))
                self.wait(min(2.5, self.remaining() * 0.18))
            # OLMo 2 7B：中期训练前后（报告表 9）
            rows = [("10 项平均", 53.0, 62.9), ("GSM8K", 24.1, 67.5)]
            g = VGroup()
            for k, (name, a, b) in enumerate(rows):
                y0 = -0.75 - 0.75 * k
                ba = box(a * 0.06, 0.28, theme.MUTED).move_to([-3.0, y0 + 0.16, 0], aligned_edge=LEFT)
                bb = box(b * 0.06, 0.28, theme.PARAM).move_to([-3.0, y0 - 0.16, 0], aligned_edge=LEFT)
                g.add(VGroup(zh(name, 20).move_to([-4.2, y0, 0]), ba, bb,
                             mono(f"{a}", 18, theme.MUTED).next_to(ba, RIGHT, 0.1),
                             mono(f"{b}", 18, theme.PARAM).next_to(bb, RIGHT, 0.1)))
            leg = VGroup(VGroup(box(0.3, 0.2, theme.MUTED), zh("预训练后", 18)).arrange(RIGHT, buff=0.1),
                         VGroup(box(0.3, 0.2, theme.PARAM), zh("+ 中期训练 50B token", 18)
                                ).arrange(RIGHT, buff=0.1)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            leg.move_to([3.9, -1.0, 0])
            src = zh("OLMo 2 7B（报告表 9）", 18, theme.MUTED).next_to(leg, DOWN, 0.25)
            self.play(LaggedStart(*[FadeIn(r) for r in g], lag_ratio=0.3), FadeIn(leg), FadeIn(src),
                      run_time=self.fit(1.5))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(pts, g, leg, src)), run_time=self.fit(0.6))

    # ── S05 小实验：分叉衰减 × 换数据 ───────────────────────────────────
    def s05(self) -> None:
        with self.shot("S05"):
            self.play(*self.set_heading("小实验：衰减 × 换数据"), run_time=self.fit(0.8))
            badge = self.show_badge()
            A = D["anneal"]
            trunk_steps, br_steps = A["steps"]
            # 左：分叉示意
            o = np.array([-5.6, 0.3, 0])
            fork = np.array([-2.9, 0.3, 0])
            trunk = Line(o, fork, color=theme.MUTED, stroke_width=5)
            tl = zh(f"主干 {trunk_steps} 步：恒定学习率", 18, theme.MUTED).next_to(trunk, UP, 0.1)
            names = list(A["branches"])
            ends = [fork + np.array([1.4, 1.2 - 0.8 * k, 0]) for k in range(4)]
            colors = [theme.MUTED, theme.PARAM, theme.INPUT, theme.OUTPUT]
            brs = VGroup(*[Line(fork, e, color=c, stroke_width=4) for e, c in zip(ends, colors)])
            bls = VGroup(*[zh(n, 17, c).next_to(e, RIGHT, 0.08) for n, e, c in zip(names, ends, colors)])
            self.play(Create(trunk), FadeIn(tl), run_time=self.fit(1))
            self.play(Create(brs), FadeIn(bls), run_time=self.fit(1.2))
            note = zh(f"每条支路 {br_steps} 步 · 新配比：代码 0.10 → 0.50", 18, theme.MUTED
                      ).move_to([-3.4, -2.0, 0])
            self.play(FadeIn(note), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.12)
            # 右：代码与平均的 bits-per-byte
            vals = [A["branches"][n]["代码"] for n in names]
            lo = min(vals) - 0.15
            chart = VGroup()
            for k, (n, v, c) in enumerate(zip(names, vals, colors)):
                y0 = 1.5 - 0.8 * k
                b = box((v - lo) * 3.2, 0.42, c).move_to([2.6, y0, 0], aligned_edge=LEFT)
                chart.add(VGroup(zh(n[0], 22, c).move_to([2.25, y0, 0]), b,
                                 mono(f"{v:.3f}", 20, c).next_to(b, RIGHT, 0.1)))
            ttl = zh("代码验证集 bits/字节（越低越好）", 20).move_to([4.1, 2.35, 0])
            base_v = A["trunk"]["代码"]
            tb = zh(f"分叉点 {base_v:.3f}", 18, theme.MUTED).move_to([4.1, -1.65, 0])
            self.play(FadeIn(ttl), LaggedStart(*[FadeIn(r) for r in chart], lag_ratio=0.25),
                      FadeIn(tb), run_time=self.fit(1.8))
            best = SurroundingRectangle(chart[3], color=theme.HIGHLIGHT, buff=0.08)
            self.wait(self.remaining() * 0.4)
            self.play(Create(best), run_time=self.fit(0.6))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(trunk, tl, brs, bls, note, chart, ttl, tb, best, badge)),
                      run_time=self.fit(0.6))

    # ── S06 RoPE 钟表：没见过的角度 ─────────────────────────────────────
    def s06(self) -> None:
        with self.shot("S06"):
            self.play(*self.set_heading("为什么读不长：RoPE 的慢指针"), run_time=self.fit(0.8))
            pos = ValueTracker(0.0)
            ids = ["24", "48", "63"]
            centers = [np.array([x, 0.9, 0]) for x in (-4.3, 0.0, 4.3)]
            R = 1.3
            clocks = VGroup()
            for i, c in zip(ids, centers):
                lam = 2 * math.pi / D["clock_w"][i]
                clocks.add(VGroup(Circle(radius=R, color=theme.MUTED, stroke_width=2).move_to(c),
                                  zh(f"第 {i} 对 · 波长 {fmt_len(lam)}", 20).next_to(
                                      Circle(radius=R).move_to(c), DOWN, 0.2)))
            self.play(FadeIn(clocks), run_time=self.fit(1))

            def hand(i: str, c: np.ndarray):
                def f():
                    ang = PI / 2 - pos.get_value() * D["clock_w"][i]
                    return Line(c, c + R * np.array([math.cos(ang), math.sin(ang), 0]),
                                color=theme.HIGHLIGHT, stroke_width=5)
                return always_redraw(f)

            def seen_arc(i: str, c: np.ndarray):
                def f():
                    a = min(min(pos.get_value(), 4096) * D["clock_w"][i], 2 * PI)
                    return Arc(radius=R, start_angle=PI / 2, angle=-max(a, 1e-3), arc_center=c,
                               color=theme.OUTPUT, stroke_width=9)
                return always_redraw(f)

            def unseen_arc(i: str, c: np.ndarray):
                def f():
                    seen = 4096 * D["clock_w"][i]
                    now = pos.get_value() * D["clock_w"][i]
                    if seen >= 2 * PI or now <= seen:
                        return VGroup()
                    return Arc(radius=R, start_angle=PI / 2 - seen, angle=-min(now - seen, 2 * PI - seen),
                               arc_center=c, color=theme.GRAD, stroke_width=9)
                return always_redraw(f)

            hands = VGroup(*[hand(i, c) for i, c in zip(ids, centers)])
            seen = VGroup(*[seen_arc(i, c) for i, c in zip(ids, centers)])
            unseen = VGroup(*[unseen_arc(i, c) for i, c in zip(ids, centers)])
            self.add(seen, unseen, hands)
            counter = always_redraw(lambda: zh(f"位置 {int(pos.get_value()):,}", 26).move_to([0, -1.5, 0]))
            self.add(counter)
            self.play(pos.animate.set_value(4096), run_time=self.fit(4, reserve=6), rate_func=lambda t: t)
            l1 = zh("绿色：训练时（4K 以内）见过的角度", 20, theme.OUTPUT).move_to([-3.2, -2.2, 0])
            self.play(FadeIn(l1), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.2)
            self.play(pos.animate.set_value(32768), run_time=self.fit(4, reserve=2), rate_func=lambda t: t)
            l2 = zh("红色：读到 32K 时才转到的角度", 20, theme.GRAD).move_to([3.2, -2.2, 0])
            self.play(FadeIn(l2), run_time=self.fit(0.6))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.remove(seen, unseen, hands, counter)
            self.play(FadeOut(VGroup(clocks, l1, l2)), run_time=self.fit(0.6))

    # ── S07 波长表 ───────────────────────────────────────────────────────
    def s07(self) -> None:
        with self.shot("S07"):
            self.play(*self.set_heading("主线模型的波长表（head_dim = 128）"), run_time=self.fit(0.8))
            f = MathTex(r"\lambda_i = 2\pi\cdot\theta^{2i/d}", font_size=40).move_to([-4.3, 2.3, 0])
            self.play(Write(f), run_time=self.fit(1))
            heads = ["i", "θ = 1 万", "θ = 50 万", "θ = 100 万"]
            xs = [-5.8, -4.2, -2.4, -0.6]
            table = VGroup(*[zh(h, 22, theme.MUTED).move_to([x, 1.45, 0]) for h, x in zip(heads, xs)])
            for r, i in enumerate(("0", "16", "32", "48", "63")):
                y0 = 0.95 - 0.5 * r
                row = VGroup(mono(i, 22).move_to([xs[0], y0, 0]))
                for k, v in enumerate(D["wl"][i]):
                    col = theme.GRAD if v > 4096 else theme.FG
                    row.add(mono(fmt_len(v), 22, col).move_to([xs[k + 1], y0, 0]))
                table.add(row)
            self.play(FadeIn(table), run_time=self.fit(1.5))
            nf = D["not_full"]["4096"]
            un = D["unseen"]
            facts = VGroup(
                zh("4K 内转不满一圈（红色）的维度对：", 20),
                zh(f"θ = 1 万：{nf[0]} 对    θ = 100 万：{nf[2]} 对", 20, theme.GRAD),
                zh("读到 32K 时会遇到“没见过的角度”的：", 20),
                zh(f"不改：{un['none']} 对    调大基频到 100 万：{un['abf']} 对", 20, theme.OUTPUT),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([3.7, 0.2, 0])
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(facts[:2]), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(facts[2:]), run_time=self.fit(0.8))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(f, table, facts)), run_time=self.fit(0.6))

    # ── S08 调大基频 ─────────────────────────────────────────────────────
    def s08(self) -> None:
        with self.shot("S08"):
            self.play(*self.set_heading("办法一：调大基频（ABF）"), run_time=self.fit(0.8))
            f = MathTex(r"\theta:\ 10^4 \rightarrow 10^6", font_size=44).move_to([0, 2.3, 0])
            self.play(Write(f), run_time=self.fit(0.8))
            ids = [0, 16, 32, 48, 63]
            g = VGroup()
            for k, (i, s) in enumerate(zip(ids, D["slow"])):
                x = -4.4 + 2.2 * k
                h = 0.35 + 0.95 * math.log10(s) if s > 1.01 else 0.08
                b = box(1.0, h, theme.INPUT).move_to([x, -1.6, 0], aligned_edge=DOWN)
                g.add(VGroup(b, mono(f"{s:.1f}×", 22, theme.INPUT).next_to(b, UP, 0.1),
                             zh(f"第 {i} 对", 20, theme.MUTED).next_to(b, DOWN, 0.12)))
            cap = zh("每个维度对变慢了多少倍", 22, theme.MUTED).move_to([0, 1.5, 0])
            self.play(FadeIn(cap), LaggedStart(*[GrowFromEdge(x[0], DOWN) for x in g], lag_ratio=0.2),
                      run_time=self.fit(1.8))
            self.play(*[FadeIn(x[1:]) for x in g], run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.25)
            who = zh("Qwen3：长上下文阶段 1 万 → 100 万 · SmolLM3：150 万 → 500 万 · Llama 3：50 万",
                     20, theme.FG).move_to([0, -2.45, 0])
            self.play(FadeIn(who), run_time=self.fit(0.8))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(f, g, cap, who)), run_time=self.fit(0.6))

    # ── S09 位置内插 PI ─────────────────────────────────────────────────
    def s09(self) -> None:
        with self.shot("S09"):
            self.play(*self.set_heading("铺垫：位置内插（PI）把所有指针一起压慢"), run_time=self.fit(0.8))
            f = MathTex(r"\omega_i \rightarrow \omega_i / s", font_size=44).move_to([0, 2.45, 0])
            self.play(Write(f), run_time=self.fit(0.8))
            R = 1.25
            groups = VGroup()
            for c, step, name, col in ((np.array([-3.4, -0.05, 0]), 1.0, "原来：相邻 token 差 1 弧度", theme.OUTPUT),
                                       (np.array([3.4, -0.05, 0]), 1 / 8, "PI ×8：只差 0.125 弧度", theme.GRAD)):
                circ = Circle(radius=R, color=theme.MUTED, stroke_width=2).move_to(c)
                dots = VGroup(*[Dot(c + R * np.array([math.cos(PI / 2 - k * step),
                                                       math.sin(PI / 2 - k * step), 0]),
                                    radius=0.08, color=col) for k in range(5)])
                lab = zh(name, 22, col).next_to(circ, DOWN, 0.25)
                groups.add(VGroup(circ, dots, lab))
            sub = zh("最快那一对（秒针）上，位置 0–4 的落点", 20, theme.MUTED).move_to([0, 1.8, 0])
            self.play(FadeIn(sub), FadeIn(groups[0][0]), LaggedStart(*[FadeIn(d) for d in groups[0][1]]),
                      FadeIn(groups[0][2]), run_time=self.fit(1.5))
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(groups[1][0]), LaggedStart(*[FadeIn(d) for d in groups[1][1]]),
                      FadeIn(groups[1][2]), run_time=self.fit(1.5))
            note = zh("不越界了，但近处的位置挤在一起、难以分辨", 24, theme.HIGHLIGHT).move_to([0, -2.35, 0])
            self.wait(self.remaining() * 0.25)
            self.play(FadeIn(note), run_time=self.fit(0.8))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(f, groups, sub, note)), run_time=self.fit(0.6))

    # ── S10 YaRN：三段 + 温度 ───────────────────────────────────────────
    def s10(self) -> None:
        with self.shot("S10"):
            self.play(*self.set_heading("办法二：YaRN——快指针不动，慢指针内插"), run_time=self.fit(0.8))
            keep = D["tiny_keep"]
            turns = D["tiny_turns"]
            bars = VGroup()
            for i, k in enumerate(keep):
                x = -6.3 + 0.36 * i
                col = theme.OUTPUT if k >= 0.999 else (theme.ATTN if k <= 0.001 else theme.PARAM)
                h_keep = 2.2 * k
                full = Rectangle(width=0.28, height=2.2, stroke_color=theme.MUTED, stroke_width=1
                                 ).move_to([x, -0.2, 0], aligned_edge=DOWN)
                b = box(0.28, max(h_keep, 0.03), col).move_to([x, -0.2, 0], aligned_edge=DOWN)
                bars.add(VGroup(full, b, mono(str(i), 14, theme.MUTED).next_to(full, DOWN, 0.08)))
            yl = zh("保留原频率的比例", 20, theme.MUTED).next_to(bars, UP, 0.15)
            xl = VGroup(zh("维度对编号", 18, theme.MUTED),
                        zh("本章小实验：head_dim 32，训练长度 64，s = 4", 16, theme.MUTED)
                        ).arrange(DOWN, buff=0.08).next_to(bars, DOWN, 0.12)
            self.play(FadeIn(yl), LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.05), FadeIn(xl),
                      run_time=self.fit(2))
            legend = VGroup(
                VGroup(box(0.3, 0.25, theme.OUTPUT), zh(f"转了 ≥32 圈：原样保留（第 0 对转了 {turns[0]:.1f} 圈）", 18)
                       ).arrange(RIGHT, buff=0.12),
                VGroup(box(0.3, 0.25, theme.PARAM), zh("中间：线性过渡", 18)).arrange(RIGHT, buff=0.12),
                VGroup(box(0.3, 0.25, theme.ATTN), zh("不满 1 圈：÷ s，和 PI 一样", 18)).arrange(RIGHT, buff=0.12),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([3.2, 1.7, 0])
            self.play(FadeIn(legend), run_time=self.fit(1))
            self.wait(self.remaining() * 0.3)
            temp = MathTex(r"\sqrt{1/t} = 0.1\ln s + 1", font_size=40, color=theme.HIGHLIGHT
                           ).move_to([3.2, -0.1, 0])
            tv = VGroup(zh(f"s = 4 时 = {D['mscale4']:.3f}", 18),
                        zh("logits 乘它的平方，注意力更“尖”", 18)).arrange(DOWN, buff=0.1
                                                                      ).next_to(temp, DOWN, 0.2)
            self.play(Write(temp), FadeIn(tv), run_time=self.fit(1.2))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(bars, yl, xl, legend, temp, tv)), run_time=self.fit(0.6))

    # ── S11 从零实现 + 对拍 ─────────────────────────────────────────────
    def s11(self) -> None:
        with self.shot("S11"):
            self.play(*self.set_heading("十几行代码，与官方实现对拍"), run_time=self.fit(0.8))
            src = """ramp = ((i - low) / (high - low)).clamp(0, 1)
keep = 1 - ramp
new_w = keep * w + (1 - keep) * w / s
mscale = 0.1 * math.log(s) + 1.0"""
            code = code_block(src, 22).move_to([-1.2, 1.2, 0])
            frame = SurroundingRectangle(code, color=theme.MUTED, buff=0.25, corner_radius=0.1)
            self.play(FadeIn(frame), LaggedStart(*[FadeIn(line) for line in code], lag_ratio=0.25),
                      run_time=self.fit(2))
            chain = VGroup(zh("本章 02 脚本", 24, theme.INPUT), zh("=", 28),
                           zh("zero.model", 24, theme.PARAM), zh("=", 28),
                           zh("transformers（Qwen3）", 24, theme.OUTPUT)).arrange(RIGHT, buff=0.3)
            chain.move_to([0, -0.9, 0])
            diff = zh(f"频率最大差 {D['parity']:.0e}（float32 舍入），温度完全相同", 22, theme.MUTED
                      ).move_to([0, -1.7, 0])
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(chain), run_time=self.fit(1))
            self.play(FadeIn(diff), run_time=self.fit(0.8))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(code, frame, chain, diff)), run_time=self.fit(0.6))

    # ── S12 小实验：loss vs 长度 ─────────────────────────────────────────
    def s12(self) -> None:
        with self.shot("S12"):
            self.play(*self.set_heading("小实验：只用长度 64 训练，读 128、256"), run_time=self.fit(0.8))
            badge = self.show_badge()
            C = D["ctx"]

            def panel(key: str, cx: float, title: str, ymin: float, ymax: float, step: float):
                ax = Axes(x_range=[0, 3, 1], y_range=[ymin, ymax, step], x_length=4.4, y_length=3.3,
                          tips=False, axis_config={"color": theme.MUTED, "include_ticks": False}
                          ).move_to([cx, 0.2, 0])
                labs = VGroup(*[mono(str(L), 18, theme.MUTED).next_to(ax.c2p(k + 0.5, ymin), DOWN, 0.12)
                                for k, L in enumerate((64, 128, 256))])
                n = int(round((ymax - ymin) / step))
                labs.add(*[mono(f"{ymin + j * step:.2f}", 15, theme.MUTED).next_to(
                    ax.c2p(0, ymin + j * step), LEFT, 0.1) for j in range(n + 1)])
                t = zh(title, 22).next_to(ax, UP, 0.15)
                lines = VGroup()
                for v in ("none", "pi", "yarn", "abf"):
                    pts = [(k + 0.5, min(C[key][v][str(L)], ymax)) for k, L in enumerate((64, 128, 256))]
                    ln = polyline_in_axes(ax, pts, color=VARIANT_COLOR[v], stroke_width=4)
                    dots = VGroup(*[Dot(ax.c2p(*p), radius=0.06, color=VARIANT_COLOR[v]) for p in pts])
                    lines.add(VGroup(ln, dots))
                return ax, labs, t, lines

            ax1, l1, t1, ln1 = panel("zero_shot", -3.1, "不训练，直接换 RoPE", 1.5, 3.7, 0.5)
            ylab = zh("验证 loss", 18, theme.MUTED).rotate(PI / 2).next_to(ax1, LEFT, 0.75)
            leg = VGroup(*[VGroup(Line(ORIGIN_L, ORIGIN_L + RIGHT * 0.4, color=VARIANT_COLOR[v], stroke_width=4),
                                  zh(VARIANT_NAME[v], 18, VARIANT_COLOR[v])).arrange(RIGHT, buff=0.1)
                           for v in ("none", "pi", "yarn", "abf")]).arrange(RIGHT, buff=0.35)
            leg.move_to([0, -2.35, 0])
            self.play(Create(ax1), FadeIn(l1), FadeIn(t1), FadeIn(ylab), FadeIn(leg), run_time=self.fit(1))
            for g in ln1:
                self.play(Create(g), run_time=self.fit(0.8, reserve=6))
            self.wait(self.remaining() * 0.3)
            ax2, l2, t2, ln2 = panel("finetuned", 3.6, "再用长度 256 微调 150 步", 1.5, 1.75, 0.05)
            self.play(Create(ax2), FadeIn(l2), FadeIn(t2), run_time=self.fit(0.8))
            self.play(LaggedStart(*[Create(g) for g in ln2], lag_ratio=0.2), run_time=self.fit(1.6))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(ax1, l1, t1, ln1, ylab, leg, ax2, l2, t2, ln2, badge)),
                      run_time=self.fit(0.6))

    # ── S13 代价与数据 ───────────────────────────────────────────────────
    def s13(self) -> None:
        with self.shot("S13"):
            self.play(*self.set_heading("长上下文的代价与数据"), run_time=self.fit(0.8))
            F = D["flops"]
            g = VGroup()
            for k, T in enumerate(("4096", "32768")):
                tot, share = F[T]
                y0 = 1.6 - 1.1 * k
                w = tot * 0.2
                mat = box(w * (1 - share), 0.55, theme.PARAM).move_to([-4.3, y0, 0], aligned_edge=LEFT)
                att = box(w * share, 0.55, theme.ATTN).next_to(mat, RIGHT, 0)
                g.add(VGroup(zh(f"{int(T) // 1024}K", 22).move_to([-4.85, y0, 0]), mat, att,
                             zh(f"{tot:.2f} GFLOP/token，注意力占 {share:.0%}", 20).next_to(att, RIGHT, 0.15)))
            cap = zh("主线模型每个训练 token 的运算量", 20, theme.MUTED).move_to([-2.0, 2.45, 0])
            self.play(FadeIn(cap), LaggedStart(*[FadeIn(r) for r in g], lag_ratio=0.4), run_time=self.fit(1.6))
            self.wait(self.remaining() * 0.2)
            data = VGroup(
                zh("• 只在最后训一小段：Llama 3 约 800B token 分 6 段到 128K", 20),
                zh("• 数据要真的长：Qwen3 的长文本 75% 在 16K–32K，另留 25% 较短", 20),
                zh("• 合成长任务：Llama 3 在 SFT 里混 0.1% 长文档问答、摘要", 20),
                zh("• 标准：短文本评测完全恢复，大海捞针全部答对", 20, theme.HIGHLIGHT),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([0, -1.35, 0])
            for p in data:
                self.play(FadeIn(p), run_time=self.fit(0.6, reserve=2))
                self.wait(min(2.0, self.remaining() * 0.15))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(g, cap, data)), run_time=self.fit(0.6))

    # ── S14 怎么评：大海捞针与 RULER ────────────────────────────────────
    def s14(self) -> None:
        with self.shot("S14"):
            self.play(*self.set_heading("怎么评：大海捞针是冒烟测试"), run_time=self.fit(0.8))
            badge = self.show_badge()
            lengths, depths = NIAH_LENGTHS, NIAH_DEPTHS
            grid = VGroup()
            for r, L in enumerate(lengths):
                for c, dd in enumerate(depths):
                    acc = D["niah"][f"{L}_{dd}"][0]
                    cell = Rectangle(width=0.9, height=0.6, stroke_color=theme.MUTED, stroke_width=1,
                                     fill_color=theme.OUTPUT if acc > 0.5 else theme.GRAD,
                                     fill_opacity=0.35)
                    cell.move_to([-4.6 + 0.95 * c, 1.0 - 0.65 * r, 0])
                    grid.add(VGroup(cell, mono(f"{acc:.2f}", 18).move_to(cell)))
            rl = VGroup(*[mono(str(L), 18, theme.MUTED).move_to([-5.65, 1.0 - 0.65 * r, 0])
                          for r, L in enumerate(lengths)])
            cl = VGroup(*[mono(f"{dd:.1f}", 18, theme.MUTED).move_to([-4.6 + 0.95 * c, 1.55, 0])
                          for c, dd in enumerate(depths)])
            ttl = zh("tiny 模型：长度 × 深度的准确率", 20).move_to([-3.7, 2.4, 0])
            hd = VGroup(zh("深度 →", 16, theme.MUTED).move_to([-3.65, 1.95, 0]),
                        zh("长度", 16, theme.MUTED).move_to([-5.65, 1.55, 0]))
            note = zh("约 1.3M 参数，每格 5 题；得分如实报告", 18, theme.MUTED).move_to([-3.7, -0.9, 0])
            self.play(FadeIn(ttl), FadeIn(hd), FadeIn(grid), FadeIn(rl), FadeIn(cl), run_time=self.fit(1.5))
            self.play(FadeIn(note), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.15)
            ruler = VGroup(
                zh("RULER（NVIDIA，2024）：4 类 13 个任务", 22, theme.HIGHLIGHT),
                zh("检索 · 多跳追踪 · 聚合 · 问答", 20),
                zh("17 个模型：NIAH 几乎满分，", 20),
                zh("RULER 随长度明显下降", 20, theme.GRAD),
                zh("声称 ≥32K 的模型，一半在 32K 不达标", 20),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.2, 0.6, 0])
            self.play(LaggedStart(*[FadeIn(x) for x in ruler], lag_ratio=0.3), run_time=self.fit(2.5))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(grid, rl, cl, ttl, hd, note, ruler, badge)), run_time=self.fit(0.6))

    # ── S15 生产级 + 主线进度 ───────────────────────────────────────────
    def s15(self) -> None:
        with self.shot("S15"):
            self.play(*self.set_heading("主线进度：极小配置跑通，GPU 上待训"), run_time=self.fit(0.8))
            badge = self.show_badge()
            cfg = code_block("""[model]
rope_theta = 1000000.0
max_seq_len = 32768
[train]
init_from = "out/main/midtrain/ckpt"
parallel = "fsdp\"""", 18).move_to([-3.8, 1.2, 0])
            ct = zh("configs/main/longctx.toml", 18, theme.MUTED).next_to(cfg, UP, 0.15)
            self.play(FadeIn(ct), FadeIn(cfg), run_time=self.fit(1))
            log = VGroup(zh("tiny 中期训练（zero.train.midtrain）", 20, theme.HIGHLIGHT),
                         *[zh(c, 17) for c in TINY_RUN["mid_changes"]],
                         zh(f"预训练 val {TINY_RUN['pre_val'][-1][1]} → 中期训练 val "
                            f"{TINY_RUN['mid_val'][-1][1]}（长度不同，不能直接比）", 17, theme.MUTED)
                         ).arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([3.0, 1.45, 0])
            self.play(FadeIn(log), run_time=self.fit(1))
            self.wait(self.remaining() * 0.2)
            gate = VGroup(
                zh("闸门 2", 24, theme.HIGHLIGHT),
                zh("□ Base 成绩 vs 闸门 1 的预测，明显偏低先诊断", 19),
                zh("□ 短文本能力完全恢复 · 32K 大海捞针全对 · RULER 如实报", 19),
                zh("□ 预计 $320 + $196，开跑前先批准", 19),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([0, -1.35, 0])
            frame = SurroundingRectangle(gate, color=theme.HIGHLIGHT, buff=0.18, corner_radius=0.1)
            self.play(Create(frame), FadeIn(gate), run_time=self.fit(1.2))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(cfg, ct, log, gate, frame, badge)), run_time=self.fit(0.6))

    # ── S16 小结 + 下一章 ────────────────────────────────────────────────
    def s16(self) -> None:
        with self.shot("S16"):
            self.play(*self.set_heading("小结"), run_time=self.fit(0.8))
            pts = VGroup(
                zh("衰减段：学习率降到 0，换上最好的数据", 26, theme.PARAM),
                zh("读不长：慢指针转到没见过的角度，注意力被摊薄", 26, theme.GRAD),
                zh("调大基频：所有指针变慢；YaRN：只压慢指针 + 调温度", 26, theme.OUTPUT),
                zh("长序列很贵：只在最后训一小段；NIAH 冒烟，RULER 评测", 26, theme.ATTN),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([0, 0.6, 0])
            for p in pts:
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=self.fit(0.7, reserve=3))
                self.wait(min(1.8, self.remaining() * 0.12))
            nxt = zh("下一章：SFT——让 Base 模型学会对话和调用工具", 26, theme.HIGHLIGHT).move_to([0, -2.1, 0])
            self.play(FadeIn(nxt), run_time=self.fit(0.8))
            self.wait(max(0.05, self.remaining() - 0.8))
            self.play(FadeOut(VGroup(pts, nxt)), *self.set_heading(None), run_time=self.fit(0.8))


ORIGIN_L = np.array([0.0, 0.0, 0.0])
