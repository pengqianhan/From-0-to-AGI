"""第 18 章视频：偏好对齐 —— 从 RLHF 到 DPO

画面里的数值由 ../code/ 中的代码真实计算（见 script.md 事实清单）；04、05 较慢，
结果缓存在 video/out/cache.json。冒烟测试数字读 out/smoke/dpo/log.jsonl（极小配置演示）。
渲染：bash chapters/18-preference-alignment/video/build.sh
"""

from __future__ import annotations

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
    Circle,
    Create,
    Cross,
    DashedLine,
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
    Transform,
    VGroup,
    Write,
)

from video_kit import theme
from video_kit.scene import NarratedScene, code_block, polyline_in_axes, zh

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
ROOT = HERE.parents[2]
CACHE = HERE / "out" / "cache.json"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_numbers() -> dict:
    """跑一遍 code/ 里的实验，收集画面要用的数字。"""
    import torch

    torch.set_num_threads(1)
    bt = _load("bt01", "01_bradley_terry.py")
    rl = _load("rlhf02", "02_rlhf_kl.py")
    dv = _load("dpo03", "03_dpo_derivation.py")
    toy = _load("toy04", "04_toy_dpo.py")
    pit = _load("pit05", "05_dpo_pitfalls.py")

    _, hist = bt.train_rm()
    r = rl.proxy_reward()
    betas = [float(b) for b in np.logspace(math.log10(0.02), 2, 40)]
    sweep = []
    for b in betas:
        lp = rl.optimal_policy(r, b)
        p = lp.exp()
        sweep.append({"beta": b, "proxy": float((p * r).sum()), "true": float((p * rl.Q).sum()),
                      "kl": rl.kl(lp, rl.LOGP_REF)})
    probs = {str(b): [float(x) for x in rl.optimal_policy(r, b).exp()] for b in (100.0, 2.0, 0.5, 0.03)}
    kls = {str(b): rl.kl(rl.optimal_policy(r, b), rl.LOGP_REF) for b in (100.0, 2.0, 0.5, 0.03)}
    trues = {str(b): float((rl.optimal_policy(r, b).exp() * rl.Q).sum()) for b in (100.0, 0.5, 0.03)}
    _, trace = rl.ppo(r, 0.5, trace_at=tuple(range(0, 401, 10)))
    star_obj = rl.objective(rl.optimal_policy(r, 0.5), r, 0.5)
    lp_ppo, _ = rl.ppo(r, 0.5)
    grad_rows = dv.step6_gradient()

    R = toy.main_results()
    lrs = pit.lr_sweep()
    nm = pit.near_miss()
    return {
        "rm_w": [hist[-1]["w_q"], hist[-1]["w_len"]],
        "names": rl.NAMES,
        "p_ref": [float(x) for x in rl.LOGP_REF.exp()],
        "probs": probs, "kls": kls, "trues": trues, "sweep": sweep,
        "ppo_trace": [[it, obj, k] for it, obj, k in trace], "ppo_star": star_obj,
        "ppo_kl_star": rl.kl(lp_ppo, rl.optimal_policy(r, 0.5)),
        "grad_rows": grad_rows,
        "toy": {"base": R["base"], "after": R["after"], "hist": R["hist"], "n_pairs": R["n_pairs"]},
        "lr_sweep": lrs, "near": nm,
    }


def numbers() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    d = compute_numbers()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    return d


def smoke_log() -> list[dict]:
    p = ROOT / "out" / "smoke" / "dpo" / "log.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


SHORT = ["简洁对", "详细对", "还行", "一般", "跑题", "错误", "啰嗦", "注水"]


def box(text: str, w: float, h: float, color: str, size: float = 22) -> VGroup:
    r = RoundedRectangle(width=w, height=h, corner_radius=0.12, stroke_color=color, stroke_width=2.5)
    t = zh(text, size, theme.FG)
    if t.width > w - 0.25:
        t.scale_to_fit_width(w - 0.25)
    return VGroup(r, t.move_to(r))


class ChapterScene(NarratedScene):
    chapter_label = "第 18 章"
    chapter_title = "偏好对齐"

    def construct(self) -> None:
        N = numbers()
        smoke = smoke_log()

        # ── S01 片头 ─────────────────────────────────────────────────────
        with self.shot("S01"):
            card = self.chapter_card()
            sub = zh("从 RLHF 到 DPO", 32, theme.HIGHLIGHT).next_to(card, DOWN, 0.6)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(card), FadeOut(sub), run_time=self.fit(0.8))

        # ── S02 判断比写容易 ──────────────────────────────────────────────
        with self.shot("S02"):
            self.play(*self.set_heading("判断比写容易"), run_time=self.fit(0.8))
            prompt = box("提示词：用一句话解释什么是梯度", 8.0, 0.8, theme.INPUT, 26).move_to([0, 2.1, 0])
            a = box("A：函数值上升最快的方向，大小是那个方向的坡度", 6.4, 1.1, theme.MUTED, 22).move_to([-3.4, 0.4, 0])
            b = box("B：梯度就是梯度的意思", 6.4, 1.1, theme.MUTED, 22).move_to([3.4, 0.4, 0])
            self.play(FadeIn(prompt), run_time=self.fit(1))
            self.play(FadeIn(a), FadeIn(b), run_time=self.fit(1))
            self.wait(self.remaining() * 0.25)
            ca = a[0].copy().set_stroke(theme.OUTPUT, 4)
            cb = b[0].copy().set_stroke(theme.GRAD, 4)
            la = zh("chosen（更好）", 26, theme.OUTPUT).next_to(a, DOWN, 0.3)
            lb = zh("rejected", 26, theme.GRAD).next_to(b, DOWN, 0.3)
            self.play(Transform(a[0], ca), Transform(b[0], cb), FadeIn(la), FadeIn(lb), run_time=self.fit(1.2))
            judge = zh("裁判：人，或者一个更强的模型（AI 反馈）", 26, theme.HIGHLIGHT).move_to([0, -1.9, 0])
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(judge), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(prompt, a, b, la, lb, judge)), run_time=self.fit(0.6))

        # ── S03 Bradley–Terry ────────────────────────────────────────────
        with self.shot("S03"):
            self.play(*self.set_heading("Bradley–Terry：分数差 → 赢的概率"), run_time=self.fit(0.8))
            ax = Axes(x_range=[-5, 5, 1], y_range=[0, 1, 0.25], x_length=6.2, y_length=4.0,
                      axis_config={"color": theme.MUTED, "include_numbers": True, "font_size": 20},
                      tips=False).move_to([-3.3, 0.1, 0])
            xl = MathTex(r"r(a)-r(b)", font_size=28, color=theme.MUTED).next_to(ax.x_axis, DOWN, 0.45)
            yl = MathTex(r"P(a \succ b)", font_size=28, color=theme.MUTED).next_to(ax.y_axis, UP, 0.1)
            curve = ax.plot(lambda x: 1 / (1 + math.exp(-x)), x_range=[-5, 5], color=theme.OUTPUT, stroke_width=4)
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=self.fit(1.2))
            f = MathTex(r"P(a \succ b) = \sigma\big(r(a) - r(b)\big)", font_size=40).move_to([3.5, 1.6, 0])
            self.play(Create(curve), Write(f), run_time=self.fit(2))
            marks = VGroup()
            for d in (1, 2):
                p = 1 / (1 + math.exp(-d))
                dot = Dot(ax.c2p(d, p), color=theme.HIGHLIGHT)
                lab = MathTex(f"{p:.2f}", font_size=26, color=theme.HIGHLIGHT).next_to(dot, RIGHT + DOWN * 0.3, 0.1)
                marks.add(dot, lab)
            self.wait(self.remaining() * 0.25)
            self.play(FadeIn(marks), run_time=self.fit(1))
            note = zh("只有差值进公式：\n所有分数同时 +100，概率不变", 24, theme.FG).move_to([3.5, -0.3, 0])
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(note), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(ax, xl, yl, curve, f, marks, note)), run_time=self.fit(0.6))

        # ── S04 奖励模型 ─────────────────────────────────────────────────
        with self.shot("S04"):
            self.play(*self.set_heading("奖励模型：一个二分类"), run_time=self.fit(0.8))
            loss = MathTex(r"\mathcal{L}_{RM} = -\log \sigma\big(r(x, y_w) - r(x, y_l)\big)", font_size=44).move_to([0, 1.9, 0])
            note = zh("= 以 r_w − r_l 为 logit、标签恒为 1 的二分类交叉熵（第 5 章）", 24, theme.MUTED).next_to(loss, DOWN, 0.35)
            self.play(Write(loss), run_time=self.fit(1.5))
            self.play(FadeIn(note), run_time=self.fit(1))
            w = N["rm_w"]
            rows = VGroup(
                zh("系数", 26, theme.MUTED), zh("标注员心里", 26, theme.MUTED), zh("奖励模型学到", 26, theme.MUTED),
                zh("质量 q", 28), MathTex("1.5", font_size=34), MathTex(f"{w[0]:.2f}", font_size=34, color=theme.PARAM),
                zh("长度 ℓ", 28), MathTex("0.4", font_size=34), MathTex(f"{w[1]:.2f}", font_size=34, color=theme.PARAM),
            )
            for i, m in enumerate(rows):
                m.move_to([-3.0 + (i % 3) * 3.0, -0.2 - (i // 3) * 0.75, 0])
            self.wait(self.remaining() * 0.3)
            self.play(LaggedStart(*[FadeIn(m) for m in rows], lag_ratio=0.1), run_time=self.fit(2))
            hl = SurroundingRectangle(VGroup(rows[6], rows[8]), color=theme.GRAD, buff=0.15)
            bias = zh("偏好学会了，偏见也学会了：喜欢长回答", 26, theme.GRAD).move_to([0, -2.35, 0])
            self.wait(self.remaining() * 0.4)
            self.play(Create(hl), FadeIn(bias), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(loss, note, rows, hl, bias)), run_time=self.fit(0.6))

        # ── S05 RLHF 目标 + S06 reward hacking（共用柱状图）──────────────────
        bax = Axes(x_range=[0, 8, 1], y_range=[0, 1, 0.25], x_length=6.2, y_length=3.4,
                   axis_config={"color": theme.MUTED, "include_numbers": False}, tips=False,
                   y_axis_config={"include_numbers": True, "font_size": 18}).move_to([-3.4, 0.2, 0])
        names = VGroup(*[zh(SHORT[i], 16, theme.FG if i != 7 else theme.GRAD).move_to(bax.c2p(i + 0.5, 0) + DOWN * 0.3)
                         for i in range(8)])
        bw = bax.x_length / 8 * 0.36

        def bars(ps, color, shift):
            g = VGroup()
            for i, p in enumerate(ps):
                h = max(p, 1e-3) * bax.y_length
                rect = Rectangle(width=bw, height=h, fill_color=color, fill_opacity=0.85, stroke_width=0)
                rect.move_to(bax.c2p(i + 0.5, 0) + RIGHT * shift, aligned_edge=DOWN)
                g.add(rect)
            return g

        ref_bars = bars(N["p_ref"], theme.MUTED, -bw * 0.55)
        with self.shot("S05"):
            self.play(*self.set_heading("RLHF：追奖励，但拴着缰绳"), run_time=self.fit(0.8))
            obj = MathTex(r"\max_\pi\ \mathbb{E}_{y\sim\pi}[r(y)] - \beta\,\mathrm{KL}(\pi\,\|\,\pi_{ref})",
                          font_size=38).move_to([3.5, 2.2, 0])
            self.play(Create(bax), FadeIn(names), run_time=self.fit(1))
            leg1 = zh("π_ref（SFT）", 20, theme.MUTED).move_to(bax.c2p(1.5, 1.08))
            self.play(FadeIn(ref_bars), FadeIn(leg1), Write(obj), run_time=self.fit(1.5))
            pol = bars(N["probs"]["100.0"], theme.PARAM, bw * 0.55)
            leg2 = zh("π（优化后）", 20, theme.PARAM).move_to(bax.c2p(5.2, 1.08))
            # 缰绳：π_ref 与 π 两个点，中间一根虚线
            anchor = Dot([1.6, 0.4, 0], radius=0.14, color=theme.MUTED)
            a_lab = zh("π_ref", 22, theme.MUTED).next_to(anchor, DOWN, 0.2)
            kl_pos = lambda k: [1.6 + min(k, 5.5) * 0.85, 0.4, 0]  # noqa: E731
            mover = Dot(kl_pos(N["kls"]["100.0"]), radius=0.14, color=theme.PARAM)
            rope = DashedLine(anchor.get_center(), mover.get_center(), color=theme.HIGHLIGHT)
            m_lab = zh("π", 22, theme.PARAM).next_to(mover, UP, 0.2)
            beta_t = MathTex(r"\beta = 100", font_size=34).move_to([3.5, 1.3, 0])
            kl_t = zh(f"KL = {N['kls']['100.0']:.2f}", 24, theme.HIGHLIGHT).move_to([3.5, -0.5, 0])
            self.play(FadeIn(pol), FadeIn(leg2), FadeIn(anchor), FadeIn(a_lab), FadeIn(mover), FadeIn(m_lab),
                      Create(rope), FadeIn(beta_t), FadeIn(kl_t), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.3)
            for b, tex in (("2.0", r"\beta = 2"), ("0.5", r"\beta = 0.5")):
                k = N["kls"][b]
                new_m = Dot(kl_pos(k), radius=0.14, color=theme.PARAM)
                self.play(Transform(pol, bars(N["probs"][b], theme.PARAM, bw * 0.55)),
                          Transform(mover, new_m), m_lab.animate.next_to(new_m, UP, 0.2),
                          Transform(rope, DashedLine(anchor.get_center(), new_m.get_center(), color=theme.HIGHLIGHT)),
                          Transform(beta_t, MathTex(tex, font_size=34).move_to([3.5, 1.3, 0])),
                          Transform(kl_t, zh(f"KL = {k:.2f}", 24, theme.HIGHLIGHT).move_to([3.5, -0.5, 0])),
                          run_time=self.fit(1.5))
                self.wait(self.remaining() * 0.3)
            leash = zh("KL = 缰绳，β = 松紧", 26, theme.HIGHLIGHT).move_to([3.5, -1.3, 0])
            self.play(FadeIn(leash), run_time=self.fit(0.8))

        with self.shot("S06"):
            self.play(*self.set_heading("缰绳太松：reward hacking"),
                      FadeOut(VGroup(obj, anchor, a_lab, mover, m_lab, rope, beta_t, kl_t, leash)),
                      run_time=self.fit(0.8))
            sw = N["sweep"]
            # 横轴是 log10(β) + 2（让纵轴落在最左边）
            cax = Axes(x_range=[0.2, 4, 1], y_range=[-1, 3, 1], x_length=5.6, y_length=3.6,
                       axis_config={"color": theme.MUTED, "font_size": 18}, tips=False,
                       x_axis_config={"include_numbers": False},
                       y_axis_config={"include_numbers": True}).move_to([3.6, 0.2, 0])
            xlab = VGroup(*[MathTex(t, font_size=22, color=theme.MUTED).next_to(cax.c2p(v + 2, -1), DOWN, 0.15)
                            for v, t in ((-1, "0.1"), (0, "1"), (1, "10"), (2, "100"))])
            bl = MathTex(r"\beta", font_size=30, color=theme.MUTED).next_to(cax.c2p(4, -1), RIGHT, 0.2)
            proxy = polyline_in_axes(cax, [(math.log10(s["beta"]) + 2, s["proxy"]) for s in sw], color=theme.PARAM, stroke_width=4)
            true = polyline_in_axes(cax, [(math.log10(s["beta"]) + 2, s["true"]) for s in sw], color=theme.OUTPUT, stroke_width=4)
            lp = zh("奖励模型分", 20, theme.PARAM).move_to(cax.c2p(3.3, 2.8))
            lt = zh("真实质量", 20, theme.OUTPUT).move_to(cax.c2p(3.3, 2.4))
            self.play(Create(cax), FadeIn(xlab), FadeIn(bl), run_time=self.fit(1))
            self.play(Create(proxy), Create(true), FadeIn(lp), FadeIn(lt), run_time=self.fit(2.5))
            t = N["trues"]
            d1 = Dot(cax.c2p(4, t["100.0"]), color=theme.OUTPUT)
            d2 = Dot(cax.c2p(math.log10(0.5) + 2, t["0.5"]), color=theme.HIGHLIGHT)
            d2l = MathTex(f"{t['0.5']:.2f}", font_size=24, color=theme.HIGHLIGHT).next_to(d2, UP, 0.15)
            self.play(FadeIn(d1), FadeIn(d2), FadeIn(d2l), run_time=self.fit(1))
            self.wait(self.remaining() * 0.25)
            d3 = Dot(cax.c2p(math.log10(0.03) + 2, t["0.03"]), color=theme.GRAD)
            d3l = zh(f"β = 0.03：真实质量 {t['0.03']:.2f}，全押注水", 22, theme.GRAD).move_to([3.6, -2.4, 0])
            self.play(Transform(pol, bars(N["probs"]["0.03"], theme.GRAD, bw * 0.55)), FadeIn(d3), FadeIn(d3l),
                      run_time=self.fit(1.5))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(bax, names, ref_bars, pol, leg1, leg2, cax, xlab, bl, proxy, true, lp, lt,
                                     d1, d2, d2l, d3, d3l)), run_time=self.fit(0.6))

        # ── S07 PPO ──────────────────────────────────────────────────────
        with self.shot("S07"):
            self.play(*self.set_heading("PPO：采样，一步步摸过去（铺垫）"), run_time=self.fit(0.8))
            steps = ["采样回答", "奖励 − β·KL", "优势 = 奖励 − 价值", "裁剪比例 ρ", "更新策略"]
            flow = VGroup(*[box(s, 2.4, 0.8, theme.ATTN, 22) for s in steps]).arrange(RIGHT, buff=0.35).move_to([0, 2.0, 0])
            arrows = VGroup(*[Arrow(flow[i].get_right(), flow[i + 1].get_left(), buff=0.05, color=theme.MUTED,
                                    stroke_width=3, max_tip_length_to_length_ratio=0.3) for i in range(4)])
            self.play(LaggedStart(*[FadeIn(m) for m in flow], lag_ratio=0.2), Create(arrows), run_time=self.fit(2.5))
            tr = N["ppo_trace"]
            pax = Axes(x_range=[0, 400, 100], y_range=[0.5, 1.8, 0.5], x_length=5.4, y_length=2.8,
                       axis_config={"color": theme.MUTED, "include_numbers": True, "font_size": 18}, tips=False).move_to([-3.4, -0.6, 0])
            plab = zh("PPO 轮数", 18, theme.MUTED).next_to(pax.x_axis, DOWN, 0.4)
            star = DashedLine(pax.c2p(0, N["ppo_star"]), pax.c2p(400, N["ppo_star"]), color=theme.HIGHLIGHT)
            starl = zh("闭式最优解", 18, theme.HIGHLIGHT).next_to(star, UP, 0.08).shift(RIGHT * 1.3)
            curve = polyline_in_axes(pax, [(p[0], p[1]) for p in tr], color=theme.PARAM, stroke_width=4)
            self.wait(self.remaining() * 0.2)
            self.play(Create(pax), FadeIn(plab), Create(star), FadeIn(starl), run_time=self.fit(1))
            self.play(Create(curve), run_time=self.fit(2))
            ig = VGroup(zh("InstructGPT：SFT → 奖励模型 → PPO", 24, theme.FG),
                        zh("13 亿参数的 InstructGPT", 24, theme.OUTPUT),
                        zh("胜过 1750 亿参数的 GPT-3", 24, theme.OUTPUT)).arrange(DOWN, buff=0.25).move_to([3.5, -0.3, 0])
            self.play(FadeIn(ig), run_time=self.fit(1))
            kl = zh(f"玩具上 400 轮后与最优解 KL = {N['ppo_kl_star']:.4f}", 20, theme.MUTED).move_to([3.5, -1.8, 0])
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(kl), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(flow, arrows, pax, plab, star, starl, curve, ig, kl)), run_time=self.fit(0.6))

        # ── S08 闭式最优解 ───────────────────────────────────────────────
        with self.shot("S08"):
            self.play(*self.set_heading("最优策略可以直接写出来"), run_time=self.fit(0.8))
            l1 = MathTex(r"\mathbb{E}_{\pi}[r] - \beta\,\mathrm{KL}(\pi\|\pi_{ref})", font_size=40)
            l2 = MathTex(r"= \beta \log Z - \beta\,\mathrm{KL}(\pi\,\|\,\pi^*)", font_size=40)
            l3 = MathTex(r"\pi^*(y) = \frac{\pi_{ref}(y)\, e^{r(y)/\beta}}{Z},\quad Z = \sum_y \pi_{ref}(y)\, e^{r(y)/\beta}", font_size=40)
            VGroup(l1, l2, l3).arrange(DOWN, buff=0.55).move_to([0, 0.7, 0])
            self.play(Write(l1), run_time=self.fit(1.5))
            self.wait(self.remaining() * 0.15)
            self.play(Write(l2), run_time=self.fit(1.5))
            note = zh("KL ≥ 0，只在 π = π* 时为 0", 24, theme.MUTED).next_to(l2, RIGHT, 0.3)
            if note.get_right()[0] > 6.9:
                note.next_to(l2, DOWN, 0.1).shift(RIGHT * 3)
            self.play(FadeIn(note), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.25)
            self.play(FadeOut(note), Write(l3), run_time=self.fit(1.5))
            hb = SurroundingRectangle(l3, color=theme.HIGHLIGHT, buff=0.15)
            self.play(Create(hb), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(l1, l2, l3, hb)), run_time=self.fit(0.6))

        # ── S09 反解，Z 消掉 ─────────────────────────────────────────────
        with self.shot("S09"):
            self.play(*self.set_heading("反解奖励，Z 消掉 → DPO"), run_time=self.fit(0.8))
            m1 = MathTex(r"r(y) = \beta \log\frac{\pi^*(y)}{\pi_{ref}(y)}", r"+ \beta\log Z", font_size=40)
            m2 = MathTex(r"r(y_w) - r(y_l) = \beta\log\frac{\pi^*(y_w)}{\pi_{ref}(y_w)}", r"+\beta\log Z",
                         r"- \beta\log\frac{\pi^*(y_l)}{\pi_{ref}(y_l)}", r"-\beta\log Z", font_size=36)
            m3 = MathTex(r"\mathcal{L}_{DPO} = -\log\sigma\Big(\beta\big[\log\tfrac{\pi_\theta(y_w)}{\pi_{ref}(y_w)}"
                         r" - \log\tfrac{\pi_\theta(y_l)}{\pi_{ref}(y_l)}\big]\Big)", font_size=40)
            VGroup(m1, m2, m3).arrange(DOWN, buff=0.6).move_to([0, 0.6, 0])
            m1[1].set_color(theme.MUTED)
            self.play(Write(m1), run_time=self.fit(2))
            self.wait(self.remaining() * 0.2)
            self.play(Write(m2), run_time=self.fit(2))
            c1, c2 = Cross(m2[1], stroke_color=theme.GRAD), Cross(m2[3], stroke_color=theme.GRAD)
            self.play(Create(c1), Create(c2), run_time=self.fit(1))
            self.wait(self.remaining() * 0.25)
            self.play(Write(m3), run_time=self.fit(2))
            hb = SurroundingRectangle(m3, color=theme.OUTPUT, buff=0.15)
            tag = zh("没有奖励模型，没有采样：只有策略和参考模型", 24, theme.OUTPUT).next_to(hb, DOWN, 0.3)
            self.play(Create(hb), FadeIn(tag), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(m1, m2, m3, c1, c2, hb, tag)), run_time=self.fit(0.6))

        # ── S10 梯度 ─────────────────────────────────────────────────────
        with self.shot("S10"):
            self.play(*self.set_heading("梯度：错多少，改多少"), run_time=self.fit(0.8))
            base_y = -1.6
            cb_ = Rectangle(width=1.2, height=2.2, fill_color=theme.OUTPUT, fill_opacity=0.85, stroke_width=0).move_to([-5.0, base_y, 0], aligned_edge=DOWN)
            rb_ = Rectangle(width=1.2, height=2.2, fill_color=theme.GRAD, fill_opacity=0.85, stroke_width=0).move_to([-2.8, base_y, 0], aligned_edge=DOWN)
            cl = zh("chosen", 22, theme.OUTPUT).next_to(cb_, DOWN, 0.2)
            rl_ = zh("rejected", 22, theme.GRAD).next_to(rb_, DOWN, 0.2)
            yl = MathTex(r"\pi_\theta(y)", font_size=30, color=theme.MUTED).move_to([-3.9, 2.2, 0])
            self.play(FadeIn(cb_), FadeIn(rb_), FadeIn(cl), FadeIn(rl_), FadeIn(yl), run_time=self.fit(1))
            up = Arrow(cb_.get_top(), cb_.get_top() + UP * 0.9, color=theme.OUTPUT, buff=0)
            dn = Arrow(rb_.get_top() + UP * 0.9, rb_.get_top(), color=theme.GRAD, buff=0)
            self.play(GrowArrow(up), GrowArrow(dn),
                      cb_.animate.stretch_to_fit_height(3.0, about_edge=DOWN),
                      rb_.animate.stretch_to_fit_height(1.3, about_edge=DOWN), run_time=self.fit(1.5))
            gax = Axes(x_range=[-5, 5, 1], y_range=[0, 1, 0.5], x_length=5.6, y_length=3.0,
                       axis_config={"color": theme.MUTED, "include_numbers": True, "font_size": 18}, tips=False).move_to([3.4, 0.1, 0])
            gl = MathTex(r"\sigma(-h)", font_size=30, color=theme.MUTED).next_to(gax.y_axis, UP, 0.1)
            hl = zh("h = 隐式奖励之差", 20, theme.MUTED).next_to(gax.x_axis, DOWN, 0.4)
            g = gax.plot(lambda h: 1 / (1 + math.exp(h)), x_range=[-5, 5], color=theme.GRAD, stroke_width=4)
            self.play(Create(gax), FadeIn(gl), FadeIn(hl), Create(g), run_time=self.fit(1.5))
            f = MathTex(r"|\nabla| \propto \beta\,\sigma(-h)", font_size=34).move_to([5.3, 2.4, 0])
            self.play(Write(f), run_time=self.fit(1))
            rows = N["grad_rows"]
            dots = VGroup()
            for h, gw, _ in rows:
                if h in (-4.0, 4.0):
                    d = Dot(gax.c2p(h, -gw / 0.1), color=theme.HIGHLIGHT)
                    lab = MathTex(f"{-gw:.4f}", font_size=22, color=theme.HIGHLIGHT).next_to(d, UP if h < 0 else UP, 0.12)
                    dots.add(d, lab)
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(dots), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(cb_, rb_, cl, rl_, yl, up, dn, gax, gl, hl, g, f, dots)), run_time=self.fit(0.6))

        # ── S11 小模型上跑一次 ───────────────────────────────────────────
        with self.shot("S11"):
            self.play(*self.set_heading("在两万参数的小模型上跑 DPO"), run_time=self.fit(0.8))
            T = N["toy"]
            H = T["hist"]
            tax = Axes(x_range=[0, 150, 50], y_range=[0, 1, 0.25], x_length=5.4, y_length=3.4,
                       axis_config={"color": theme.MUTED, "include_numbers": True, "font_size": 18}, tips=False).move_to([-3.5, 0.2, 0])
            tl = zh("DPO 步数", 18, theme.MUTED).next_to(tax.x_axis, DOWN, 0.4)
            task = zh("任务 a+b=？  SFT 示范只有 40% 是对的", 22, theme.FG).move_to([0, 2.5, 0])
            self.play(FadeIn(task), Create(tax), FadeIn(tl), run_time=self.fit(1.2))
            mcur = polyline_in_axes(tax, [(h["step"], h["margin"]) for h in H], color=theme.PARAM, stroke_width=4)
            acur = polyline_in_axes(tax, [(h["step"], h["acc"]) for h in H[1:]], color=theme.OUTPUT, stroke_width=4)
            ml = zh(f"margin → {H[-1]['margin']:.2f}", 20, theme.PARAM).next_to(tax.c2p(150, H[-1]["margin"]), LEFT, 0.1).shift(UP * 0.3)
            al = zh(f"acc {H[-1]['acc']:.2f}", 20, theme.OUTPUT).next_to(tax.c2p(150, 1.0), LEFT, 0.1).shift(DOWN * 0.3)
            # 右边：log 概率柱（向下画，高度 = −log π）
            zero_y = 1.6
            scale = 0.42
            zline = Line([1.6, zero_y, 0], [5.6, zero_y, 0], color=theme.MUTED)
            zl = MathTex("0", font_size=24, color=theme.MUTED).next_to(zline, LEFT, 0.1)
            ylab = MathTex(r"\log\pi", font_size=28, color=theme.MUTED).next_to(zl, LEFT, 0.15)

            def lbar(v, x, color):
                return Rectangle(width=1.0, height=max(-v * scale, 0.02), fill_color=color, fill_opacity=0.85,
                                 stroke_width=0).move_to([x, zero_y, 0], aligned_edge=UP)

            cbar, rbar = lbar(H[0]["logp_w"], 2.8, theme.OUTPUT), lbar(H[0]["logp_l"], 4.4, theme.GRAD)
            cv = MathTex(f"{H[0]['logp_w']:.2f}", font_size=24, color=theme.OUTPUT).next_to(cbar, DOWN, 0.1)
            rv = MathTex(f"{H[0]['logp_l']:.2f}", font_size=24, color=theme.GRAD).next_to(rbar, DOWN, 0.1)
            cn = zh("chosen", 20, theme.OUTPUT).move_to([2.8, zero_y + 0.35, 0])
            rn = zh("rejected", 20, theme.GRAD).move_to([4.4, zero_y + 0.35, 0])
            self.play(FadeIn(zline), FadeIn(zl), FadeIn(ylab), FadeIn(cbar), FadeIn(rbar), FadeIn(cv), FadeIn(rv),
                      FadeIn(cn), FadeIn(rn), run_time=self.fit(1))
            self.play(Create(mcur), Create(acur), run_time=self.fit(2))
            self.play(FadeIn(ml), FadeIn(al), run_time=self.fit(0.6))
            for h in H[1:]:
                nc, nr = lbar(h["logp_w"], 2.8, theme.OUTPUT), lbar(h["logp_l"], 4.4, theme.GRAD)
                self.play(Transform(cbar, nc), Transform(rbar, nr),
                          Transform(cv, MathTex(f"{h['logp_w']:.2f}", font_size=24, color=theme.OUTPUT).next_to(nc, DOWN, 0.1)),
                          Transform(rv, MathTex(f"{h['logp_l']:.2f}", font_size=24, color=theme.GRAD).next_to(nr, DOWN, 0.1)),
                          run_time=self.fit(0.7))
            b, a = T["base"]["p_correct"], T["after"]["p_correct"]
            res = zh(f"留出 30 题上答对的概率：{b:.3f} → {a:.3f}", 24, theme.HIGHLIGHT).move_to([-3.0, -2.45, 0])
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(res), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(task, tax, tl, mcur, acur, ml, al, zline, zl, ylab, cbar, rbar, cv, rv, cn, rn, res)),
                      run_time=self.fit(0.6))

        # ── S12 三个坑 ───────────────────────────────────────────────────
        with self.shot("S12"):
            self.play(*self.set_heading("三个坑"), run_time=self.fit(0.8))
            L = N["lr_sweep"]
            t1 = zh("① 学习率 → 采样格式正确率", 22, theme.FG).move_to([-4.6, 2.3, 0])
            fax = Axes(x_range=[0, 5, 1], y_range=[0, 1, 0.5], x_length=4.0, y_length=2.6,
                       axis_config={"color": theme.MUTED, "font_size": 16}, tips=False,
                       x_axis_config={"include_numbers": False},
                       y_axis_config={"include_numbers": True}).move_to([-4.6, 0.2, 0])
            fbars = VGroup()
            flabs = VGroup()
            for i, r in enumerate(L):
                col = theme.MUTED if i == 0 else (theme.OUTPUT if r["format"] > 0.9 else theme.GRAD)
                rect = Rectangle(width=0.5, height=max(r["format"], 0.01) * fax.y_length, fill_color=col,
                                 fill_opacity=0.85, stroke_width=0).move_to(fax.c2p(i + 0.5, 0), aligned_edge=DOWN)
                fbars.add(rect)
                txt = "SFT" if i == 0 else f"{r['lr']:.0e}".replace("e-0", "e-")
                flabs.add(zh(txt, 15, theme.MUTED).next_to(rect, DOWN, 0.12))
            mnote = zh(f"lr=1e-2：margin {L[3]['margin']:.2f}，格式 {L[3]['format']:.2f}", 18, theme.GRAD).move_to([-4.6, -1.8, 0])
            self.play(FadeIn(t1), Create(fax), FadeIn(fbars), FadeIn(flabs), run_time=self.fit(1.5))
            self.play(FadeIn(mnote), run_time=self.fit(0.6))
            # ③ 冒烟测试（极小配置演示）
            badge = self.show_badge()
            smoke_t = VGroup(zh("③ 主线冒烟测试（tiny）", 22, theme.FG),
                             zh("DPO lr = 5e-4：24 步 margin → 4.8", 18, theme.GRAD),
                             zh("之后 GRPO 格式正确率 0.16 → 0", 18, theme.GRAD),
                             zh("改成 5e-5 才稳住", 18, theme.OUTPUT)).arrange(DOWN, buff=0.2, aligned_edge=LEFT).move_to([4.7, 0.6, 0])
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(smoke_t), run_time=self.fit(1))
            # ② chosen 一起掉
            nm = N["near"]["hist"]
            t2 = zh("② 错答案只差一点", 22, theme.FG).move_to([0.0, 2.3, 0])
            nax = Axes(x_range=[0, 150, 50], y_range=[-7, 0, 2], x_length=3.6, y_length=2.6,
                       axis_config={"color": theme.MUTED, "include_numbers": True, "font_size": 14}, tips=False).move_to([0.1, 0.2, 0])
            wc = polyline_in_axes(nax, [(h["step"], h["logp_w"]) for h in nm], color=theme.OUTPUT, stroke_width=4)
            lc = polyline_in_axes(nax, [(h["step"], h["logp_l"]) for h in nm], color=theme.GRAD, stroke_width=4)
            nb, na = N["near"]["base"]["p_correct"], N["near"]["after"]["p_correct"]
            n2 = zh(f"chosen 也在掉；留出答对 {nb:.2f} → {na:.2f}", 18, theme.GRAD).move_to([0.1, -1.8, 0])
            self.wait(self.remaining() * 0.25)
            self.play(FadeIn(t2), Create(nax), run_time=self.fit(0.8))
            self.play(Create(wc), Create(lc), FadeIn(n2), run_time=self.fit(1.5))
            final = zh("看留出集，别只看训练 margin", 24, theme.HIGHLIGHT).move_to([0, -2.4, 0])
            self.wait(self.remaining() * 0.5)
            self.play(FadeIn(final), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(t1, fax, fbars, flabs, mnote, smoke_t, t2, nax, wc, lc, n2, final, badge)),
                      run_time=self.fit(0.6))

        # ── S13 从极简到生产级 ───────────────────────────────────────────
        with self.shot("S13"):
            self.play(*self.set_heading("从极简到生产级：zero/post/dpo.py"), run_time=self.fit(0.8))
            src = """h = beta * ((pi_w - ref_w)
             - (pi_l - ref_l))
loss = -F.logsigmoid(h).mean()"""
            code = code_block(src, 20).move_to([-3.6, 1.3, 0])
            cl = zh("极简版（code/03、04）", 20, theme.MUTED).next_to(code, UP, 0.25)
            self.play(FadeIn(cl), FadeIn(code), run_time=self.fit(1))
            items = ["encode_pair：只算回复 token", "batch_logps：chosen+rejected 一批前向",
                     "ref_mode = precompute：参考 log 概率预先算好", "dpo_loss：损失 + acc / margin 日志"]
            pipe = VGroup(*[box(t, 5.8, 0.6, theme.INPUT, 20) for t in items]).arrange(DOWN, buff=0.18).move_to([3.4, 0.9, 0])
            self.play(LaggedStart(*[FadeIn(m) for m in pipe], lag_ratio=0.3), run_time=self.fit(3))
            badge = self.show_badge()
            if smoke:
                first, last = smoke[0], smoke[-1]
                txt = (f"tiny 冒烟：{last['step']} 步，loss {first['loss']:.3f} → {last['loss']:.3f}，"
                       f"acc {last['acc']:.2f}，margin {last['margin']:.2f}")
            else:
                txt = "tiny 冒烟：24 步，loss 0.693 → 0.526，acc 0.75，margin 0.39"
            sm = zh(txt, 22, theme.HIGHLIGHT).move_to([0, -1.5, 0])
            todo = zh("主线模型的 DPO：待 GPU 训练后补充", 22, theme.MUTED).move_to([0, -2.2, 0])
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(sm), run_time=self.fit(0.8))
            self.play(FadeIn(todo), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(cl, code, pipe, sm, todo, badge)), run_time=self.fit(0.6))

        # ── S14 谁在用 + 下一章 ──────────────────────────────────────────
        with self.shot("S14"):
            self.play(*self.set_heading("谁在用 DPO"), run_time=self.fit(0.8))
            fams = ["Llama 3", "Qwen2 / 2.5", "DeepSeek LLM", "Tülu 3 / OLMo 2", "Nemotron-4", "SmolLM3（APO）"]
            tags = VGroup(*[box(f, 3.6, 0.8, theme.OUTPUT, 24) for f in fams]).arrange_in_grid(2, 3, buff=0.35).move_to([0, 1.0, 0])
            self.play(LaggedStart(*[FadeIn(t) for t in tags], lag_ratio=0.2), run_time=self.fit(3))
            nxt = zh("下一章：强化学习 —— GRPO 与可验证奖励", 30, theme.HIGHLIGHT).move_to([0, -1.6, 0])
            self.wait(self.remaining() * 0.45)
            self.play(Write(nxt), run_time=self.fit(1.5))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(VGroup(tags, nxt)), *self.set_heading(None), run_time=self.fit(0.8))
