"""第 19 章视频：强化学习 —— 让模型从自己的尝试里学

画面里的数值都由 ../code/ 中的代码真实计算（见 script.md 事实清单）：
  01_reinforce_bandit.py、02_grpo_from_scratch.py、03_reward_hacking.py 现算，结果缓存在 video/out/cache.json；
  删掉缓存会重新计算（03 在繁忙的机器上要几分钟）。
极小配置的冒烟数字（S13）来自 out/smoke/grpo/log.jsonl 与 out/smoke/eval/report.md，写在 SMOKE 里。
渲染：bash chapters/19-reinforcement-learning/video/build.sh
"""

from __future__ import annotations

import importlib.util
import json
import sys
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
    LaggedStart,
    Line,
    MathTex,
    Rectangle,
    RoundedRectangle,
    SurroundingRectangle,
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

# 冒烟测试的真实记录（极小配置演示）：out/smoke/grpo/log.jsonl、out/smoke/eval/report.md
SMOKE = {
    "reward": [-0.1875, -0.14, -0.40, -0.1225, -0.035, -0.3425, -0.1734, -0.175, -0.0163, -0.0725],
    "format": [0.5, 0.625, 0.46875, 0.65625, 0.71875, 0.4375, 0.65625, 0.5625, 0.71875, 0.625],
    "call": [0.34375, 0.625, 0.46875, 0.65625, 0.71875, 0.4375, 0.65625, 0.5625, 0.71875, 0.625],
    "vs_sft": "−0.033  [−0.100, +0.000]  持平",
}


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def compute() -> dict:
    import torch

    torch.set_num_threads(1)
    b = _load("ch19_bandit", "01_reinforce_bandit.py")
    rw = b.REWARDS + 10.0
    curves = {}
    for use_b in (False, True):
        c = np.stack([b.train(rw, 0.5, 300, 8, np.random.default_rng(s), use_b) for s in range(5)])
        curves["base" if use_b else "nobase"] = c.mean(0)[::5].tolist()
    g = _load("ch19_grpo", "02_grpo_from_scratch.py")
    out = g.run(verbose=False)
    ev = [h for h in out["hist"] if "greedy" in h]
    grpo = {
        "teacher": out["teacher_acc"],
        "sft": out["sft"],
        "steps": [h["step"] for h in ev],
        "carry": [h["greedy_carry"] for h in ev],
        "sampled": [h["sampled"] for h in ev],
        "zero_std": {h["step"]: h["zero_std"] for h in out["hist"] if "zero_std" in h},
        "example": out["example"],
    }
    h = _load("ch19_hack", "03_reward_hacking.py")
    hack = {}
    for fixed in (False, True):
        hist, _ = h.train(fixed, verbose=False)
        hack["fixed" if fixed else "naive"] = hist
    return {"bandit": curves, "grpo": grpo, "hack": hack}


def data() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    d = compute()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(d, ensure_ascii=False))
    return d


D = data()


def mono(text: str, size: float = 22, color: str = theme.FG) -> Text:
    return Text(text, font=MONO, font_size=size, color=color)


def fmt(x: float, n: int = 2) -> str:
    return f"{x:.{n}f}"


def small_axes(x_len, y_len, x_max, y_range=(0, 1, 0.5)):
    return Axes(
        x_range=[0, x_max, x_max / 4], y_range=list(y_range), x_length=x_len, y_length=y_len,
        tips=False, axis_config={"color": theme.MUTED, "stroke_width": 2, "include_ticks": True},
    )


class ChapterScene(NarratedScene):
    chapter_label = "第 19 章"
    chapter_title = "强化学习"

    def construct(self):
        G = D["grpo"]

        # ── S01 ──────────────────────────────────────────────────────────
        with self.shot("S01"):
            card = self.chapter_card()
            self.play(FadeIn(card), run_time=self.fit(1.5))
            self.wait(self.remaining() * 0.6)
            sub = zh("会模仿，但不会解题 → 让模型自己去试", 30, theme.HIGHLIGHT).next_to(card, DOWN, buff=0.6)
            self.play(FadeIn(sub), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(card), FadeOut(sub), run_time=0.5)

        # ── S02 模仿的天花板 ─────────────────────────────────────────────
        with self.shot("S02"):
            self.play(*self.set_heading("模仿的天花板"), run_time=self.fit(0.6))
            teacher = VGroup(
                zh("老师（不太会进位）", 26, theme.PARAM),
                mono("3 + 4 → 7   ✓", 24),
                mono("7 + 5 → 2   ✗", 24, theme.GRAD),
                zh("进位题 70% 忘了写进位", 22, theme.MUTED),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([-4.3, 0.6, 0])
            box = SurroundingRectangle(teacher, color=theme.PARAM, buff=0.3, corner_radius=0.1)
            self.play(FadeIn(teacher), Create(box), run_time=self.fit(1.5))
            vals = [("老师示范", G["teacher"], theme.PARAM), ("SFT 采样", G["sft"]["sampled"], theme.INPUT),
                    ("SFT 贪心", G["sft"]["greedy"], theme.INPUT), ("进位题贪心", G["sft"]["greedy_carry"], theme.GRAD)]
            bars = VGroup()
            base_y, H = -2.0, 3.8
            for i, (name, v, col) in enumerate(vals):
                x = 0.3 + i * 1.75
                bar = Rectangle(width=0.9, height=max(H * v, 0.02), fill_color=col, fill_opacity=0.85, stroke_width=0)
                bar.move_to([x, base_y + H * v / 2, 0])
                num = mono(fmt(v, 2), 22).next_to(bar, UP, buff=0.1)
                lab = zh(name, 20, theme.MUTED).move_to([x, base_y - 0.35, 0])
                bars.add(VGroup(bar, num, lab))
            self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.3) for b in bars], lag_ratio=0.5),
                      run_time=self.fit(4))
            self.wait(self.remaining() * 0.4)
            msg = zh("模仿学习的上限 = 示范数据", 30, theme.HIGHLIGHT).move_to([-4.0, -2.0, 0])
            self.play(FadeIn(msg), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(teacher, box, bars, msg)), run_time=0.5)

        # ── S03 对数导数技巧 ─────────────────────────────────────────────
        with self.shot("S03"):
            self.play(*self.set_heading("分数不可导，梯度从哪来"), run_time=self.fit(0.6))
            lines = VGroup(
                MathTex(r"\nabla J = \nabla \sum_y \pi_\theta(y)\, r(y)", font_size=40),
                MathTex(r"= \sum_y r(y)\, \nabla \pi_\theta(y)", font_size=40),
                MathTex(r"= \sum_y \pi_\theta(y)\, r(y)\, \nabla \log \pi_\theta(y)", font_size=40),
                MathTex(r"= \mathbb{E}_{y\sim\pi_\theta}\big[\, r(y)\, \nabla \log \pi_\theta(y) \,\big]",
                        font_size=44, color=theme.HIGHLIGHT),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([0, 0.5, 0])
            hint = MathTex(r"\nabla \pi = \pi \cdot \nabla \log \pi", font_size=34, color=theme.PARAM)
            hint.next_to(lines[2], RIGHT, buff=0.5)
            for i, ln in enumerate(lines):
                self.play(Write(ln), run_time=self.fit(1.6))
                if i == 1:
                    self.play(FadeIn(hint), run_time=self.fit(0.6))
                self.wait(self.remaining() * 0.12)
            note = zh("r 只是一个数：不需要可导，只要能采样、能打分（REINFORCE）", 26, theme.OUTPUT)
            note.move_to([0, -2.25, 0])
            self.play(FadeIn(note), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(lines, hint, note)), run_time=0.5)

        # ── S04 基线 ─────────────────────────────────────────────────────
        with self.shot("S04"):
            self.play(*self.set_heading("基线：期望不变，方差小得多"), run_time=self.fit(0.6))
            f = MathTex(r"\mathbb{E}[(r-b)\nabla\log\pi] = \nabla J", font_size=36).move_to([-3.4, 2.3, 0])
            self.play(Write(f), run_time=self.fit(1.2))
            ax = small_axes(5.6, 3.4, 300).move_to([-3.4, -0.4, 0])
            xl = zh("训练步数", 18, theme.MUTED).next_to(ax, DOWN, buff=0.12)
            yl = zh("π(最好的手臂)", 18, theme.MUTED).next_to(ax, UP, buff=0.1).align_to(ax, LEFT)
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=self.fit(1))
            B = D["bandit"]
            xs = [5 * i + 1 for i in range(len(B["nobase"]))]
            c1 = polyline_in_axes(ax, list(zip(xs, B["nobase"])), color=theme.GRAD, stroke_width=4)
            c2 = polyline_in_axes(ax, list(zip(xs, B["base"])), color=theme.OUTPUT, stroke_width=4)
            l1 = zh(f"无基线 → {B['nobase'][-1]:.2f}", 20, theme.GRAD).move_to(ax.c2p(200, 0.42))
            l2 = zh(f"批均值基线 → {B['base'][-1]:.2f}", 20, theme.OUTPUT).move_to(ax.c2p(190, 0.85))
            self.play(Create(c1), Create(c2), run_time=self.fit(3))
            self.play(FadeIn(l1), FadeIn(l2), run_time=self.fit(0.6))
            tbl = VGroup(
                zh("奖励整体 +10，单样本方差", 24, theme.FG),
                mono("不减基线   84.70", 26, theme.GRAD),
                mono("减 E[r]     0.053", 26, theme.OUTPUT),
                zh("相差 1589 倍", 28, theme.HIGHLIGHT),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([3.6, 0.3, 0])
            self.play(FadeIn(tbl), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(f, ax, xl, yl, c1, c2, l1, l2, tbl)), run_time=0.5)

        # ── S05 PPO ──────────────────────────────────────────────────────
        with self.shot("S05"):
            self.play(*self.set_heading("PPO：价值模型 + 裁剪"), run_time=self.fit(0.6))
            pol = RoundedRectangle(width=2.6, height=1.0, corner_radius=0.15, color=theme.PARAM)
            pol_t = zh("策略模型", 24, theme.PARAM).move_to(pol)
            crit = RoundedRectangle(width=2.6, height=1.0, corner_radius=0.15, color=theme.GRAD)
            crit_t = zh("价值模型", 24, theme.GRAD).move_to(crit)
            g1 = VGroup(pol, pol_t).move_to([-4.6, 1.2, 0])
            g2 = VGroup(crit, crit_t).move_to([-4.6, -0.4, 0])
            same = zh("一样大：多一份显存、多训一个模型", 20, theme.MUTED).move_to([-4.6, -1.4, 0])
            self.play(FadeIn(g1), FadeIn(g2), run_time=self.fit(1))
            self.play(FadeIn(same), run_time=self.fit(0.6))
            # 裁剪函数：A>0 时目标 min(ρA, clip(ρ)A)
            ax = Axes(x_range=[0, 2, 0.5], y_range=[0, 1.6, 0.4], x_length=5.0, y_length=3.0, tips=False,
                      axis_config={"color": theme.MUTED, "stroke_width": 2}).move_to([2.6, 0.2, 0])
            eps = 0.2
            pts = [(x, min(x, 1 + eps)) for x in np.linspace(0, 2, 81)]
            curve = polyline_in_axes(ax, pts, color=theme.OUTPUT, stroke_width=5)
            lo = DashedLine(ax.c2p(1 - eps, 0), ax.c2p(1 - eps, 1.6), color=theme.MUTED)
            hi = DashedLine(ax.c2p(1 + eps, 0), ax.c2p(1 + eps, 1.6), color=theme.MUTED)
            lab_lo = MathTex(r"1-\epsilon", font_size=26, color=theme.MUTED).next_to(ax.c2p(1 - eps, 0), DOWN, buff=0.15)
            lab_hi = MathTex(r"1+\epsilon", font_size=26, color=theme.MUTED).next_to(ax.c2p(1 + eps, 0), DOWN, buff=0.15)
            xl = MathTex(r"\rho = \pi_\theta/\pi_{old}", font_size=28).next_to(ax, RIGHT, buff=0.1).shift(DOWN * 1.2)
            title = MathTex(r"A>0:\ \min(\rho A,\ \mathrm{clip}(\rho)A)", font_size=30).next_to(ax, UP, buff=0.2)
            self.play(Create(ax), FadeIn(title), run_time=self.fit(1))
            self.play(Create(lo), Create(hi), FadeIn(lab_lo), FadeIn(lab_hi), FadeIn(xl), run_time=self.fit(0.8))
            self.play(Create(curve), run_time=self.fit(2))
            flat = zh("超过 1+ε：不再给梯度", 20, theme.HIGHLIGHT).move_to(ax.c2p(1.6, 1.45))
            self.play(FadeIn(flat), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(g1, g2, same, ax, curve, lo, hi, lab_lo, lab_hi, xl, title, flat)), run_time=0.5)

        # ── S06 GRPO 一组样本 ────────────────────────────────────────────
        with self.shot("S06"):
            self.play(*self.set_heading("GRPO：同一道题的其他回答就是基线"), run_time=self.fit(0.6))
            ex = G["example"]
            q = zh(f"题目：{ex['prompt'][0]} + {ex['prompt'][1]} = ?", 30).move_to([0, 2.2, 0])
            self.play(FadeIn(q), run_time=self.fit(0.6))
            cards = VGroup()
            for resp, r in zip(ex["responses"], ex["rewards"]):
                col = theme.OUTPUT if r > 0 else theme.GRAD
                box = RoundedRectangle(width=1.2, height=1.0, corner_radius=0.12, color=col)
                t = mono(resp, 30, col).move_to(box)
                rr = mono(f"r={r:.0f}", 20, theme.FG).next_to(box, DOWN, buff=0.12)
                cards.add(VGroup(box, t, rr))
            cards.arrange(RIGHT, buff=0.35).move_to([0, 0.9, 0])
            self.play(LaggedStart(*[FadeIn(c, shift=DOWN * 0.2) for c in cards], lag_ratio=0.15),
                      run_time=self.fit(2.5))
            rew = np.array(ex["rewards"])
            stat = MathTex(rf"\bar r = {rew.mean():.2f},\quad \mathrm{{std}} = {rew.std(ddof=1):.3f}",
                           font_size=32).move_to([0, -0.35, 0])
            form = MathTex(r"A_i = \frac{r_i - \bar r}{\mathrm{std}}", font_size=34,
                           color=theme.HIGHLIGHT).move_to([-4.6, -0.35, 0])
            self.play(Write(stat), FadeIn(form), run_time=self.fit(1.2))
            advs = VGroup()
            for c, a in zip(cards, ex["adv"]):
                col = theme.OUTPUT if a > 0 else theme.GRAD
                t = mono(f"{a:+.2f}", 22, col).move_to([c.get_center()[0], -1.1, 0])
                arr = Arrow(start=[c.get_center()[0], -1.45, 0],
                            end=[c.get_center()[0], -1.45 + (0.6 if a > 0 else -0.6), 0],
                            color=col, buff=0, stroke_width=5, max_tip_length_to_length_ratio=0.35)
                if a < 0:
                    arr.shift(UP * 0.6)
                advs.add(VGroup(t, arr))
            self.play(LaggedStart(*[FadeIn(a) for a in advs], lag_ratio=0.1), run_time=self.fit(2))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(q, cards, stat, form, advs)), run_time=0.5)

        # ── S07 三个细节 ─────────────────────────────────────────────────
        with self.shot("S07"):
            self.play(*self.set_heading("GRPO 的三个细节"), run_time=self.fit(0.6))
            zs = G["zero_std"]
            z5, z20 = zs.get("5", zs.get(5)), zs.get("20", zs.get(20))
            col1 = VGroup(
                zh("① 零方差组", 26, theme.HIGHLIGHT),
                mono("r = 1 1 1 1 1 1 1 1", 20),
                mono("A = 0 0 0 0 0 0 0 0", 20, theme.GRAD),
                zh("这组没有梯度", 20, theme.MUTED),
                zh(f"第 5 步 {z5:.2f} → 第 20 步 {z20:.2f}", 20, theme.FG),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).move_to([-4.6, 0.3, 0])
            ax = Axes(x_range=[0, 3, 1], y_range=[0, 1.2, 0.5], x_length=3.0, y_length=1.8, tips=False,
                      axis_config={"color": theme.MUTED, "stroke_width": 2})
            k3 = polyline_in_axes(ax, [(x, x - np.log(x) - 1) for x in np.linspace(0.2, 3, 60)],
                                  color=theme.ATTN, stroke_width=4)
            col2 = VGroup(
                zh("② k3 KL（可选）", 26, theme.HIGHLIGHT),
                MathTex(r"x - \log x - 1 \ge 0", font_size=30),
                MathTex(r"x = \pi_{ref}/\pi_\theta", font_size=26, color=theme.MUTED),
                VGroup(ax, k3),
            ).arrange(DOWN, buff=0.25).move_to([0, 0.3, 0])
            short = Rectangle(width=0.8, height=0.35, fill_color=theme.INPUT, fill_opacity=0.8, stroke_width=0)
            long_ = Rectangle(width=2.6, height=0.35, fill_color=theme.INPUT, fill_opacity=0.8, stroke_width=0)
            col3 = VGroup(
                zh("③ token 级平均", 26, theme.HIGHLIGHT),
                short, long_,
                zh("所有回答的 token 一起平均", 20, theme.FG),
                zh("长回答里的 token 不被稀释", 20, theme.MUTED),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).move_to([4.6, 0.3, 0])
            for c in (col1, col2, col3):
                self.play(FadeIn(c), run_time=self.fit(1))
                self.wait(self.remaining() * 0.3)
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(col1, col2, col3)), run_time=0.5)

        # ── S08 学生超过老师 ─────────────────────────────────────────────
        with self.shot("S08"):
            self.play(*self.set_heading("GRPO：学生超过了老师"), run_time=self.fit(0.6))
            steps = G["steps"]
            ax = Axes(x_range=[0, 60, 10], y_range=[0, 1.05, 0.25], x_length=8.5, y_length=4.2, tips=False,
                      axis_config={"color": theme.MUTED, "stroke_width": 2, "include_numbers": True,
                                   "font_size": 20}).move_to([-0.6, 0.0, 0])
            xl = zh("GRPO 步数", 20, theme.MUTED).next_to(ax, DOWN, buff=0.1).shift(RIGHT * 3.2)
            teach = DashedLine(ax.c2p(0, G["teacher"]), ax.c2p(60, G["teacher"]), color=theme.PARAM)
            tl = zh(f"老师 {G['teacher']:.3f}", 20, theme.PARAM).next_to(ax.c2p(60, G["teacher"]), RIGHT, buff=0.15)
            c1 = polyline_in_axes(ax, list(zip(steps, G["carry"])), color=theme.GRAD, stroke_width=5)
            c2 = polyline_in_axes(ax, list(zip(steps, G["sampled"])), color=theme.OUTPUT, stroke_width=5)
            l1 = zh("进位题 · 贪心", 20, theme.GRAD).next_to(ax.c2p(60, G["carry"][-1]), RIGHT, buff=0.15).shift(UP * 0.25)
            l2 = zh("全部 · 采样", 20, theme.OUTPUT).next_to(ax.c2p(60, G["sampled"][-1]), RIGHT, buff=0.15).shift(DOWN * 0.3)
            self.play(Create(ax), FadeIn(xl), run_time=self.fit(1))
            self.play(Create(teach), FadeIn(tl), run_time=self.fit(0.8))
            self.play(Create(c1), Create(c2), run_time=self.fit(3))
            self.play(FadeIn(l1), FadeIn(l2), run_time=self.fit(0.6))
            p0 = zh(f"{G['carry'][0]:.2f} → {G['carry'][1]:.2f}（5 步）", 22, theme.GRAD).move_to(ax.c2p(22, 0.3))
            self.play(FadeIn(p0), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(ax, xl, teach, tl, c1, c2, l1, l2, p0)), run_time=0.5)

        # ── S09 可验证奖励 ───────────────────────────────────────────────
        with self.shot("S09"):
            self.play(*self.set_heading("可验证奖励：让程序判对错"), run_time=self.fit(0.6))
            rows = [("数学", "抽出答案，和标准答案比"), ("代码", "在沙箱里跑测试用例"),
                    ("指令遵循", "每条约束一个检查函数"), ("工具调用", "格式 + 参数 AST / 执行结果比对")]
            tbl = VGroup()
            for i, (k, v) in enumerate(rows):
                y = 1.9 - i * 0.75
                key = zh(k, 26, theme.INPUT)
                key.move_to([-4.2 + key.width / 2, y, 0])
                val = zh(v, 26, theme.FG)
                val.move_to([-1.9 + val.width / 2, y, 0])
                tbl.add(VGroup(key, val))
            self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.3) for r in tbl], lag_ratio=0.4),
                      run_time=self.fit(3.5))
            r1 = zh("DeepSeek-R1：推理任务只用规则奖励（神经奖励模型易被钻空子）", 22, theme.MUTED).move_to([0, -1.2, 0])
            r2 = zh("MiMo 的 RL 环境：可执行测试 · 规则检查 · rubric 逐条判分 · 视觉判分", 22, theme.OUTPUT).move_to([0, -1.9, 0])
            self.play(FadeIn(r1), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(r2), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(tbl, r1, r2)), run_time=0.5)

        # ── S10 真实的作弊 ───────────────────────────────────────────────
        with self.shot("S10"):
            self.play(*self.set_heading("真实的作弊：去掉标签"), run_time=self.fit(0.6))
            badge = self.show_badge()
            tagged = code_block('<tool_call>{"name": "calc",\n "arguments": {"expr": </tool_call>', 18,
                                theme.GRAD).move_to([-3.3, 1.6, 0])
            t1 = zh("标签里 JSON 坏了 → −1", 22, theme.GRAD).next_to(tagged, DOWN, buff=0.2)
            bare = code_block('{"name": "calc",\n "arguments": {"expr": "28 - 44"}}', 18,
                              theme.HIGHLIGHT).move_to([-3.3, -0.6, 0])
            t2 = zh("去掉标签 → 不算调用 → 0（原来的奖励）", 22, theme.HIGHLIGHT).next_to(bare, DOWN, buff=0.2)
            arr = Arrow([-3.3, 0.75, 0], [-3.3, 0.05, 0], color=theme.MUTED, buff=0)
            self.play(FadeIn(tagged), FadeIn(t1), run_time=self.fit(1))
            self.wait(self.remaining() * 0.2)
            self.play(Create(arr), FadeIn(bare), FadeIn(t2), run_time=self.fit(1.2))
            chat = VGroup(zh("闲聊题：\"Thanks a lot.\"", 22, theme.FG),
                          zh("空回复 → 1.0（原来的奖励）", 22, theme.HIGHLIGHT)).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
            chat.move_to([3.4, 1.4, 0])
            self.play(FadeIn(chat), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.4)
            fix = VGroup(zh("守卫 #8", 26, theme.OUTPUT),
                         zh("标签外出现调用样子的 JSON → −1", 22, theme.FG),
                         zh("空回复 → 0", 22, theme.FG)).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
            fix.move_to([3.4, -0.6, 0])
            fbox = SurroundingRectangle(fix, color=theme.OUTPUT, buff=0.25, corner_radius=0.1)
            self.play(FadeIn(fix), Create(fbox), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(tagged, t1, bare, t2, arr, chat, fix, fbox)), run_time=0.5)
        self.remove(badge)

        # ── S11 玩具重演 ─────────────────────────────────────────────────
        with self.shot("S11"):
            self.play(*self.set_heading("重演：奖励涨了，本事没涨"), run_time=self.fit(0.6))
            H = D["hack"]
            panels = VGroup()
            curves_all = []
            for j, (key, title, col) in enumerate((("naive", "天真的奖励", theme.GRAD),
                                                    ("fixed", "加上守卫 #8", theme.OUTPUT))):
                hist = H[key]
                ax = Axes(x_range=[0, 120, 40], y_range=[0, 1.0, 0.5], x_length=5.2, y_length=3.2, tips=False,
                          axis_config={"color": theme.MUTED, "stroke_width": 2, "include_numbers": True,
                                       "font_size": 18}).move_to([-3.5 + 7.0 * j, -0.2, 0])
                tt = zh(title, 24, col).next_to(ax, UP, buff=0.25)
                xs = [h["step"] for h in hist]
                cs = VGroup(
                    polyline_in_axes(ax, list(zip(xs, [h["format_err"] for h in hist])), color=theme.GRAD, stroke_width=4),
                    polyline_in_axes(ax, list(zip(xs, [h["call_rate"] for h in hist])), color=theme.OUTPUT, stroke_width=4),
                    polyline_in_axes(ax, list(zip(xs, [h["success"] for h in hist])), color=theme.HIGHLIGHT, stroke_width=5),
                )
                s0, s1 = hist[0]["success"], hist[-1]["success"]
                lab = zh(f"真实成功率 {s0:.2f} → {s1:.2f}", 20, theme.HIGHLIGHT).next_to(ax, DOWN, buff=0.3)
                panels.add(VGroup(ax, tt, lab))
                curves_all.append(cs)
            legend = VGroup(zh("格式错误率", 18, theme.GRAD), zh("正确格式调用", 18, theme.OUTPUT),
                            zh("真实成功率", 18, theme.HIGHLIGHT)).arrange(RIGHT, buff=0.5).move_to([0, 2.45, 0])
            self.play(FadeIn(panels[0][:2]), FadeIn(panels[1][:2]), FadeIn(legend), run_time=self.fit(1))
            self.play(Create(curves_all[0]), run_time=self.fit(3))
            self.play(FadeIn(panels[0][2]), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.35)
            self.play(Create(curves_all[1]), run_time=self.fit(3))
            self.play(FadeIn(panels[1][2]), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(panels, legend, *curves_all)), run_time=0.5)

        # ── S12 怎么防 ───────────────────────────────────────────────────
        with self.shot("S12"):
            self.play(*self.set_heading("怎么防：别只看平均奖励"), run_time=self.fit(0.6))
            left = VGroup(*[zh(t, 24, theme.FG) for t in (
                "1. 格式要严：宁可误杀，别留缝", "2. 惩罚退化输出：空、超长、伪造结果",
                "3. 训练奖励 ≠ 评估判据；定期抽样本", "4. 能用规则，就别用模型打分", "5. 每个漏洞补一条测试")]
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([-3.2, 0.3, 0])
            right = VGroup(zh("要盯的指标", 24, theme.HIGHLIGHT), *[mono(t, 20) for t in (
                "reward_mean", "format_rate", "call_rate", "resp_len", "kl", "entropy")],
                zh("留出集真实成功率", 22, theme.HIGHLIGHT)).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([4.2, 0.3, 0])
            self.play(LaggedStart(*[FadeIn(t) for t in left], lag_ratio=0.5), run_time=self.fit(5))
            self.play(FadeIn(right), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(left, right)), run_time=0.5)

        # ── S13 主线进度 ─────────────────────────────────────────────────
        with self.shot("S13"):
            self.play(*self.set_heading("主线进度与生产级代码"), run_time=self.fit(0.6))
            badge = self.show_badge()
            ax = Axes(x_range=[0, 10, 2], y_range=[-0.6, 0.8, 0.2], x_length=5.4, y_length=3.2, tips=False,
                      axis_config={"color": theme.MUTED, "stroke_width": 2}).move_to([-3.4, 0.2, 0])
            xs = list(range(1, 11))
            r = polyline_in_axes(ax, list(zip(xs, SMOKE["reward"])), color=theme.PARAM, stroke_width=4)
            fr = polyline_in_axes(ax, list(zip(xs, SMOKE["format"])), color=theme.OUTPUT, stroke_width=4)
            leg = VGroup(zh("平均奖励", 18, theme.PARAM), zh("格式正确率", 18, theme.OUTPUT)).arrange(RIGHT, buff=0.4)
            leg.next_to(ax, UP, buff=0.2)
            xl = zh("tiny GRPO 10 步（每步 32 条回复）", 18, theme.MUTED).next_to(ax, DOWN, buff=0.15)
            self.play(Create(ax), FadeIn(leg), FadeIn(xl), run_time=self.fit(1))
            self.play(Create(r), Create(fr), run_time=self.fit(2))
            info = VGroup(
                zh("奖励 −0.19 → −0.07（噪声大）", 22, theme.FG),
                zh("格式/调用率 0.50/0.34 → 0.62/0.62", 22, theme.FG),
                zh("GRPO vs SFT（工具调用）", 22, theme.FG),
                mono(SMOKE["vs_sft"].replace("持平", "").strip(), 22, theme.HIGHLIGHT),
                zh("→ 持平（CI 跨过 0）", 22, theme.HIGHLIGHT),
                mono("zero/post/grpo.py · envs/tool_env.py", 18, theme.MUTED),
                mono("configs/main/grpo.toml  G=16 × 64 题", 18, theme.MUTED),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([3.4, 0.1, 0])
            self.play(FadeIn(info), run_time=self.fit(1.5))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(ax, r, fr, leg, xl, info)), run_time=0.5)
        self.remove(badge)

        # ── S14 小结 ─────────────────────────────────────────────────────
        with self.shot("S14"):
            self.play(*self.set_heading("小结"), run_time=self.fit(0.6))
            steps = ["采样 G 个回答", "验证器打分", "组内优势", "裁剪的策略梯度"]
            boxes = VGroup()
            for s in steps:
                t = zh(s, 24, theme.FG)
                b = RoundedRectangle(width=t.width + 0.5, height=0.9, corner_radius=0.12, color=theme.INPUT)
                boxes.add(VGroup(b, t))
            boxes.arrange(RIGHT, buff=0.55).move_to([0, 1.0, 0])
            arrows = VGroup(*[Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.08,
                                    color=theme.MUTED) for i in range(3)])
            self.play(LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.3), Create(arrows),
                      run_time=self.fit(2.5))
            warn = zh("奖励决定模型学成什么样 —— 盯住真实成功率", 26, theme.GRAD).move_to([0, -0.5, 0])
            self.play(FadeIn(warn), run_time=self.fit(1))
            self.wait(self.remaining() * 0.6)
            nxt = zh("下一章：发布", 30, theme.HIGHLIGHT).move_to([0, -1.7, 0])
            self.play(FadeIn(nxt), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(boxes, arrows, warn, nxt)), *self.set_heading(None), run_time=0.5)
