"""第 17 章视频：蒸馏 —— 让小模型向大模型学

画面里的数值都由 ../code/ 中的代码真实计算（见 script.md 事实清单）：
  01_soft_labels.py（软标签、温度、梯度）、02_toy_distill.py（玩具实验，结果缓存在 out/toy_cache.json）、
  03_forward_reverse_kl.py（双峰）、04_rejection_sampling.py（漏斗）、05_shared_vocab.py（参数账）；
S11 的数字读自冒烟测试的真实输出 out/smoke/distill/（极小配置演示）。
渲染：bash chapters/17-distillation/video/build.sh --preview
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from functools import lru_cache
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
    FadeIn,
    FadeOut,
    GrowFromEdge,
    LaggedStart,
    Line,
    MathTex,
    Polygon,
    Rectangle,
    RoundedRectangle,
    Text,
    Transform,
    VGroup,
    Write,
)

from video_kit import theme
from video_kit.scene import NarratedScene, polyline_in_axes, zh

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
MONO = "Noto Sans Mono"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


soft = _load("soft_labels", "01_soft_labels.py")


@lru_cache(maxsize=1)
def toy_results() -> dict:
    """02 的玩具实验单线程 CPU 约 2 分钟：第一次跑完缓存到 out/toy_cache.json。"""
    cache = HERE / "out" / "toy_cache.json"
    if cache.exists():
        return json.loads(cache.read_text())
    r = _load("toy", "02_toy_distill.py").run_experiment()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(r, ensure_ascii=False, indent=1))
    return r


@lru_cache(maxsize=1)
def kl_results() -> dict:
    return _load("fr_kl", "03_forward_reverse_kl.py").run_all()


@lru_cache(maxsize=1)
def funnel_results() -> dict:
    rs = _load("rej", "04_rejection_sampling.py")
    tasks = rs.generate_tasks(rs.N_TASKS, seed=17, split="train")
    c = rs.funnel(tasks, rs.SimTeacher(tasks))
    t2 = rs.SimTeacher(tasks)
    kept = 0
    for i, task in enumerate(tasks):
        k, _ = rs.teacher_trajectories(t2, task, rs.K, rs.KEEP, seed=i)
        kept += len(k)
    return {"stages": [c["候选"], c["关1 格式正确"], c["关2 调用正确"], c["关3 执行+回答正确"], kept]}


@lru_cache(maxsize=1)
def vocab_params() -> list[tuple[str, int, float]]:
    from zero.config import load_config
    from zero.model import count_params

    cfg = load_config(REPO / "configs" / "main" / "pretrain.toml").model
    rows = []
    for name, v in [("主线自训 BPE", 65_536), ("Qwen3 分词器", 151_936)]:
        rows.append((name, v, count_params(dataclasses.replace(cfg, vocab_size=v))["total"] / 1e6))
    return rows


@lru_cache(maxsize=1)
def smoke() -> dict:
    """修复判分器之前那次冒烟测试的记录（冻结在 data/ 里；重跑冒烟测试会覆盖 out/smoke）。"""
    return json.loads((HERE / "data" / "smoke_before_fix.json").read_text(encoding="utf-8"))


def mono(text: str, size: float = 24, color: str = theme.FG) -> Text:
    return Text(text, font=MONO, font_size=size, color=color)


def box(w: float, h: float, color: str, opacity: float = 0.85) -> Rectangle:
    return Rectangle(width=max(w, 0.02), height=max(h, 0.02), fill_color=color,
                     fill_opacity=opacity, stroke_width=0)


def bars(values, labels, x0: float, y0: float, width: float, max_h: float, color: str,
         vmax: float = 1.0, fmt: str = "{:.3f}", size: float = 18) -> VGroup:
    """底边在 y0 的竖直柱状图（值都 ≥ 0）。返回 VGroup(柱子们, 标签们, 数值们)。"""
    n = len(values)
    step = width / n
    cols, labs, nums = VGroup(), VGroup(), VGroup()
    for i, (v, lab) in enumerate(zip(values, labels)):
        cx = x0 + step * (i + 0.5)
        h = max_h * v / vmax
        b = box(step * 0.62, h, color).move_to([cx, y0 + max(h, 0.02) / 2, 0])
        cols.add(b)
        labs.add(zh(lab, size, theme.FG).move_to([cx, y0 - 0.25, 0]))
        nums.add(mono(fmt.format(v), size - 4, theme.MUTED).move_to([cx, y0 + max(h, 0.02) + 0.18, 0]))
    return VGroup(cols, labs, nums)


def signed_bars(values, labels, x0: float, y0: float, width: float, scale: float,
                size: float = 18) -> VGroup:
    """以 y0 为零线的正负柱状图：正值（往下压）红色，负值（往上推）绿色。"""
    n = len(values)
    step = width / n
    g = VGroup(Line([x0, y0, 0], [x0 + width, y0, 0], color=theme.MUTED, stroke_width=2))
    for i, (v, lab) in enumerate(zip(values, labels)):
        cx = x0 + step * (i + 0.5)
        h = abs(v) * scale
        c = theme.GRAD if v > 0 else theme.OUTPUT
        b = box(step * 0.6, h, c).move_to([cx, y0 + (h / 2 if v > 0 else -h / 2), 0])
        g.add(b)
        g.add(zh(lab, size, theme.FG).move_to([cx, y0 - scale * 0.62 - 0.25, 0]))
        g.add(mono(f"{v:+.2f}", size - 5, theme.MUTED).move_to(
            [cx, y0 + (h + 0.16 if v > 0 else -h - 0.16), 0]))
    return g


class ChapterScene(NarratedScene):
    chapter_label = "第 17 章"
    chapter_title = "蒸馏"

    def construct(self) -> None:
        for i in range(1, 14):
            getattr(self, f"s{i:02d}")()

    def clear_all(self, *mobs) -> None:
        self.play(FadeOut(VGroup(*mobs)), run_time=self.fit(0.6))

    # ── S01 片头 ─────────────────────────────────────────────────────────
    def s01(self) -> None:
        with self.shot("S01"):
            card = self.chapter_card()
            sub = zh("让小模型向大模型学", 32, theme.HIGHLIGHT).next_to(card, DOWN, 0.6)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(card), FadeOut(sub), run_time=self.fit(0.8))

    # ── S02 大家都这么做 ─────────────────────────────────────────────────
    def s02(self) -> None:
        with self.shot("S02"):
            self.play(*self.set_heading("今天的小模型，几乎都是蒸馏出来的"), run_time=self.fit(0.8))
            t = RoundedRectangle(width=2.6, height=2.2, corner_radius=0.15, color=theme.PARAM,
                                 fill_opacity=0.25).move_to([-4.6, 0.6, 0])
            tl = zh("教师", 30, theme.PARAM).move_to(t)
            s = RoundedRectangle(width=1.3, height=1.1, corner_radius=0.12, color=theme.OUTPUT,
                                 fill_opacity=0.25).move_to([-1.3, 0.6, 0])
            sl = zh("学生", 24, theme.OUTPUT).move_to(s)
            ar = Arrow(t.get_right(), s.get_left(), color=theme.FG, buff=0.1)
            self.play(FadeIn(t), FadeIn(tl), run_time=self.fit(0.6))
            self.play(Create(ar), FadeIn(s), FadeIn(sl), run_time=self.fit(0.8))
            rows = [("Llama 3.2 1B/3B", "8B、70B 的 logits 当目标"),
                    ("Gemma 2 / 3", "用蒸馏代替下一个 token 预测"),
                    ("Qwen3 小模型", "强到弱蒸馏，约 1/10 GPU 时"),
                    ("DeepSeek-R1-Distill", "约 80 万条 R1 数据做 SFT")]
            g = VGroup()
            for k, (a, b) in enumerate(rows):
                y = 2.0 - 0.95 * k
                g.add(VGroup(zh(a, 24, theme.HIGHLIGHT).move_to([0.6, y, 0], aligned_edge=LEFT),
                             zh(b, 20, theme.FG).move_to([0.6, y - 0.38, 0], aligned_edge=LEFT)))
            self.wait(self.remaining() * 0.05)
            for row in g:
                self.play(FadeIn(row, shift=LEFT * 0.2), run_time=self.fit(0.6))
                self.wait(min(3.5, self.remaining() * 0.22))
            self.wait(self.remaining() - 0.6)
            self.clear_all(t, tl, s, sl, ar, g)

    # ── S03 one-hot vs 软标签 ────────────────────────────────────────────
    def s03(self) -> None:
        with self.shot("S03"):
            self.play(*self.set_heading("one-hot 与教师的软标签"), run_time=self.fit(0.8))
            ctx = zh("上文：今天天气很 ＿", 30).move_to([0, 2.35, 0])
            self.play(FadeIn(ctx), run_time=self.fit(0.6))
            onehot = [1.0] + [0.0] * 5
            p_t = soft.softmax(soft.TEACHER_LOGITS).tolist()
            left = bars(onehot, soft.VOCAB, -6.3, -1.2, 5.2, 2.6, theme.INPUT)
            right = bars(p_t, soft.VOCAB, 0.9, -1.2, 5.2, 2.6, theme.PARAM)
            lt = zh("one-hot：熵 0 bit", 22, theme.INPUT).move_to([-3.7, -2.15, 0])
            rt = zh(f"教师 p_T：熵 {soft.entropy_bits(np.array(p_t)):.3f} bit", 22,
                    theme.PARAM).move_to([3.5, -2.15, 0])
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in left[0]], lag_ratio=0.1),
                      FadeIn(left[1]), FadeIn(left[2]), FadeIn(lt), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.3)
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in right[0]], lag_ratio=0.1),
                      FadeIn(right[1]), FadeIn(right[2]), FadeIn(rt), run_time=self.fit(1.2))
            note = zh("次优 ≠ 荒唐：暗知识", 24, theme.HIGHLIGHT).move_to([3.5, 1.75, 0])
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(note), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.clear_all(ctx, left, right, lt, rt, note)

    # ── S04 温度 ─────────────────────────────────────────────────────────
    def s04(self) -> None:
        with self.shot("S04"):
            self.play(*self.set_heading("温度：把暗知识放大"), run_time=self.fit(0.8))
            f = MathTex(r"p_T^{\tau} = \mathrm{softmax}(z_T/\tau)", font_size=40).move_to([3.6, 2.2, 0])
            self.play(Write(f), run_time=self.fit(1))

            def chart(tau: float) -> VGroup:
                p = soft.softmax(soft.TEACHER_LOGITS, tau)
                return bars(p.tolist(), soft.VOCAB, -6.5, -1.3, 6.0, 3.2, theme.PARAM)

            def info(tau: float) -> VGroup:
                p = soft.softmax(soft.TEACHER_LOGITS, tau)
                return VGroup(
                    MathTex(rf"\tau = {tau:g}", font_size=40, color=theme.HIGHLIGHT),
                    zh(f"熵 {soft.entropy_bits(p):.2f} bit", 26),
                    zh(f"p(晴) / p(猫) = {p[3] / p[4]:.1f}", 26),
                ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([3.6, 0.2, 0])

            temps = [0.5, 1.0, 2.0, 4.0]
            cur, inf = chart(temps[0]), info(temps[0])
            self.play(FadeIn(cur), FadeIn(inf), run_time=self.fit(0.8))
            for tau in temps[1:]:
                self.wait(min(3.0, self.remaining() * 0.2))
                self.play(Transform(cur, chart(tau)), Transform(inf, info(tau)), run_time=self.fit(1.2))
            keep = zh("排名不变：好 > 热 > 冷 > 晴 > 猫 > 跑", 22, theme.MUTED).move_to([3.6, -1.7, 0])
            self.play(FadeIn(keep), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.clear_all(f, cur, inf, keep)

    # ── S05 KD 损失与梯度 ────────────────────────────────────────────────
    def s05(self) -> None:
        with self.shot("S05"):
            self.play(*self.set_heading("KD 损失与它的梯度"), run_time=self.fit(0.8))
            loss = MathTex(r"L_{KD} = \tau^2\,\mathrm{KL}(p_T^{\tau}\,\|\,p_S^{\tau})", font_size=40
                           ).move_to([-3.2, 2.25, 0])
            grad = MathTex(r"\frac{\partial L}{\partial z_S} = \tau\,(p_S^{\tau} - p_T^{\tau})",
                           font_size=40, color=theme.GRAD).move_to([3.3, 2.25, 0])
            self.play(Write(loss), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.15)
            self.play(Write(grad), run_time=self.fit(1.2))
            hard = (soft.softmax(soft.STUDENT_LOGITS) - np.eye(6)[0]).tolist()
            kd = soft.kd_grad(soft.STUDENT_LOGITS, soft.TEACHER_LOGITS, 1.0).tolist()
            g1 = signed_bars(hard, soft.VOCAB, -6.4, 0.0, 5.4, 2.2)
            g2 = signed_bars(kd, soft.VOCAB, 0.9, 0.0, 5.4, 2.2)
            t1 = zh("硬标签：p_S − onehot", 22, theme.INPUT).move_to([-3.7, -2.35, 0])
            t2 = zh("KD（τ=1）：p_S − p_T", 22, theme.PARAM).move_to([3.6, -2.35, 0])
            self.play(FadeIn(g1), FadeIn(t1), run_time=self.fit(1))
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(g2), FadeIn(t2), run_time=self.fit(1))
            leg = zh("绿：往上推　红：往下压", 20, theme.MUTED).move_to([0, 1.45, 0])
            self.play(FadeIn(leg), run_time=self.fit(0.5))
            self.wait(self.remaining() - 0.6)
            self.clear_all(loss, grad, g1, g2, t1, t2, leg)

    # ── S06 同一个词表 ───────────────────────────────────────────────────
    def s06(self) -> None:
        with self.shot("S06"):
            self.play(*self.set_heading("logits 蒸馏要求同一个分词器"), run_time=self.fit(0.8))
            cw = 0.62  # 每个汉字的宽度

            def row(tokens, y, color, name):
                g, x = VGroup(), -2.0
                for tok in tokens:
                    w = cw * len(tok)
                    r = RoundedRectangle(width=w - 0.06, height=0.6, corner_radius=0.08, color=color,
                                         fill_opacity=0.2).move_to([x + w / 2, y, 0])
                    g.add(VGroup(r, zh(tok, 26).move_to(r)))
                    x += w
                lab = zh(name, 22, color).move_to([-3.9, y, 0])
                return VGroup(lab, g)

            s_row = row(["今天", "天气", "很", "好"], 2.1, theme.OUTPUT, "学生词表")
            t_row = row(["今天天", "气很", "好"], 1.25, theme.PARAM, "教师词表")
            self.play(FadeIn(s_row), run_time=self.fit(0.8))
            self.play(FadeIn(t_row), run_time=self.fit(0.8))
            x2 = zh("第 2 个位置：学生猜「天气」，教师猜「气很」→ 无法逐位比较", 22,
                    theme.GRAD).move_to([0.9, 0.45, 0])
            self.play(FadeIn(x2), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.3)
            rows = vocab_params()
            hdr = VGroup(zh("分词器", 20, theme.MUTED), zh("词表", 20, theme.MUTED),
                         zh("主线总参数", 20, theme.MUTED))
            tab = VGroup()
            xs = [-3.2, 0.3, 3.2]
            for j, h in enumerate(hdr):
                tab.add(h.move_to([xs[j], -0.4, 0]))
            for k, (name, v, p) in enumerate(rows):
                y = -1.05 - 0.6 * k
                c = theme.OUTPUT if p <= 800 else theme.GRAD
                tab.add(zh(name, 22).move_to([xs[0], y, 0]), mono(f"{v:,}", 22).move_to([xs[1], y, 0]),
                        mono(f"{p:.1f}M", 22, c).move_to([xs[2], y, 0]))
            diff = rows[1][2] - rows[0][2]
            cap = zh(f"换 Qwen3 词表：+{diff:.1f}M 参数，超过 0.8B 上限", 22, theme.GRAD).move_to([0.3, -2.35, 0])
            self.play(FadeIn(tab), run_time=self.fit(1))
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(cap), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.clear_all(s_row, t_row, x2, tab, cap)

    # ── S07 序列级蒸馏 + 玩具实验 ────────────────────────────────────────
    def s07(self) -> None:
        with self.shot("S07"):
            self.play(*self.set_heading("序列级蒸馏：教师写，学生抄"), run_time=self.fit(0.8))
            r = toy_results()
            t = RoundedRectangle(width=1.8, height=0.8, corner_radius=0.1, color=theme.PARAM,
                                 fill_opacity=0.25).move_to([-5.2, 2.0, 0])
            txt = zh("教师写的文本", 20).move_to([-2.4, 2.0, 0])
            s = RoundedRectangle(width=1.4, height=0.7, corner_radius=0.1, color=theme.OUTPUT,
                                 fill_opacity=0.25).move_to([0.6, 2.0, 0])
            flow = VGroup(t, zh("教师", 22, theme.PARAM).move_to(t), txt, s,
                          zh("学生 SFT", 20, theme.OUTPUT).move_to(s),
                          Arrow(t.get_right(), txt.get_left(), buff=0.1, color=theme.FG),
                          Arrow(txt.get_right(), s.get_left(), buff=0.1, color=theme.FG))
            note = zh("对教师只要求能生成文本", 20, theme.MUTED).move_to([4.4, 2.0, 0])
            self.play(FadeIn(flow), FadeIn(note), run_time=self.fit(1))
            self.wait(self.remaining() * 0.25)
            names = [("A 硬标签（2 万真实字符）", theme.INPUT), ("B 序列级（教师写的 2 万字符）", theme.PARAM),
                     ("C logits 蒸馏（τ=2）", theme.OUTPUT), ("D 0.5 硬标签 + 0.5 KD", theme.ATTN),
                     ("R 参照：全部 100 万字符", theme.MUTED)]
            arms = list(r["arms"].values())
            lo, hi = 3.0, 3.8
            x0, W = -1.2, 5.6
            g = VGroup()
            for k, ((name, c), a) in enumerate(zip(names, arms)):
                v = sum(a["val"]) / len(a["val"])
                y = 0.95 - 0.58 * k
                w = W * (v - lo) / (hi - lo)
                b = box(w, 0.36, c).move_to([x0 + w / 2, y, 0])
                g.add(VGroup(zh(name, 19).move_to([x0 - 0.2, y, 0], aligned_edge=RIGHT), b,
                             mono(f"{v:.3f}", 19, c).next_to(b, RIGHT, 0.12)))
            axis = VGroup(Line([x0, -1.95, 0], [x0 + W, -1.95, 0], color=theme.MUTED, stroke_width=2),
                          mono("3.0", 16, theme.MUTED).move_to([x0, -2.15, 0]),
                          mono("3.8", 16, theme.MUTED).move_to([x0 + W, -2.15, 0]),
                          zh("验证 bits/char（越低越好，横轴从 3.0 起）", 18, theme.MUTED
                             ).move_to([x0 + W / 2, -2.5, 0]))
            cap = zh(f"玩具实验：学生 {r['student_params']:,} 参数，3 种子平均", 18,
                     theme.MUTED).move_to([-4.6, -2.5, 0])
            self.play(FadeIn(axis), FadeIn(cap), run_time=self.fit(0.6))
            for row in g:
                self.play(GrowFromEdge(row[1], LEFT), FadeIn(row[0]), FadeIn(row[2]), run_time=self.fit(0.7))
                self.wait(min(2.5, self.remaining() * 0.18))
            self.wait(self.remaining() - 0.6)
            self.clear_all(flow, note, g, axis, cap)

    # ── S08 前向 vs 反向 KL ──────────────────────────────────────────────
    def s08(self) -> None:
        with self.shot("S08"):
            self.play(*self.set_heading("前向 KL 覆盖，反向 KL 挑模式"), run_time=self.fit(0.8))
            r = kl_results()
            ax = Axes(x_range=[-6, 6, 2], y_range=[0, 0.7, 0.2], x_length=8.2, y_length=4.2,
                      tips=False, axis_config={"color": theme.MUTED, "include_ticks": False}
                      ).move_to([-2.3, 0.1, 0])
            grid = np.array(r["grid"])
            idx = slice(None, None, 4)

            def curve(vals, color, width=4):
                return polyline_in_axes(ax, list(zip(grid[idx], np.array(vals)[idx])), color=color,
                                        stroke_width=width)

            pc = curve(r["p"], theme.FG, 5)
            valley = Rectangle(width=ax.c2p(1.0, 0)[0] - ax.c2p(-0.5, 0)[0], height=4.2,
                               fill_color=theme.HIGHLIGHT, fill_opacity=0.10, stroke_width=0)
            valley.move_to([(ax.c2p(-0.5, 0)[0] + ax.c2p(1.0, 0)[0]) / 2, ax.get_center()[1], 0])
            vl = zh("山谷", 18, theme.HIGHLIGHT).next_to(valley, UP, 0.05)
            self.play(Create(ax), run_time=self.fit(0.6))
            self.play(Create(pc), FadeIn(valley), FadeIn(vl), run_time=self.fit(1.2))
            leg_p = zh(f"教师 p（双峰）：山谷里 {r['valley_p']:.1%}", 20).move_to([4.3, 2.1, 0])
            self.play(FadeIn(leg_p), run_time=self.fit(0.5))
            fw = r["fits"][0]
            fc = curve(fw["q"], theme.INPUT)
            leg_f = zh(f"前向 KL(p‖q)：μ={fw['mu']:.2f}，σ={fw['sigma']:.2f}", 20, theme.INPUT
                       ).move_to([4.3, 1.1, 0])
            leg_f2 = zh(f"山谷里 {fw['valley_q']:.1%}：两峰都盖住", 20, theme.INPUT).move_to([4.3, 0.65, 0])
            self.wait(self.remaining() * 0.12)
            self.play(Create(fc), FadeIn(leg_f), FadeIn(leg_f2), run_time=self.fit(1.5))
            self.wait(self.remaining() * 0.45)
            rv = VGroup(*[curve(f["q"], theme.GRAD) for f in r["fits"][1:]])
            mus = "、".join(f"{f['mu']:.2f}" for f in r["fits"][1:])
            leg_r = zh(f"反向 KL(q‖p)：μ = {mus}", 20, theme.GRAD).move_to([4.3, -0.35, 0])
            leg_r2 = zh("缩进一个峰，取决于起点", 20, theme.GRAD).move_to([4.3, -0.8, 0])
            self.play(Create(rv), FadeIn(leg_r), FadeIn(leg_r2), run_time=self.fit(1.5))
            self.wait(self.remaining() - 0.6)
            self.clear_all(ax, pc, valley, vl, leg_p, fc, leg_f, leg_f2, rv, leg_r, leg_r2)

    # ── S09 在线策略蒸馏 ─────────────────────────────────────────────────
    def s09(self) -> None:
        with self.shot("S09"):
            self.play(*self.set_heading("在线策略蒸馏：学生写，教师逐 token 打分"), run_time=self.fit(0.8))
            toks = ["我", "来", "算", "：", "12", "+", "30", "=", "44"]
            shade = [0.05, 0.05, 0.1, 0.05, 0.1, 0.1, 0.15, 0.1, 0.9]  # 示意：最后一个 token 被重罚
            g = VGroup()
            x = -6.2
            for tk, s in zip(toks, shade):
                w = 0.36 + 0.2 * len(tk)
                r = Rectangle(width=w, height=0.62, fill_color=theme.GRAD, fill_opacity=s,
                              stroke_color=theme.MUTED, stroke_width=1).move_to([x + w / 2, 1.9, 0])
                g.add(VGroup(r, zh(tk, 24).move_to(r)))
                x += w + 0.06
            lab1 = zh("学生自己采样", 20, theme.OUTPUT).next_to(g, UP, 0.2)
            lab2 = zh("颜色越深 = 教师扣分越多（示意）", 18, theme.MUTED).next_to(g, DOWN, 0.2)
            self.play(FadeIn(g, lag_ratio=0.1), FadeIn(lab1), run_time=self.fit(1.2))
            self.play(FadeIn(lab2), run_time=self.fit(0.5))
            f = MathTex(r"\text{advantage}_t = -\left(\log p_S(y_t) - \log p_T(y_t)\right)",
                        font_size=34, color=theme.PARAM).move_to([-2.9, 0.1, 0])
            self.play(Write(f), run_time=self.fit(1.2))
            # Qwen3 表 21
            hdr = VGroup(zh("Qwen3-8B", 20, theme.MUTED), zh("AIME'24", 20, theme.MUTED),
                         zh("GPU 时", 20, theme.MUTED))
            data = [("+ 强化学习", "67.6", "17,920", theme.INPUT), ("+ 在线策略蒸馏", "74.4", "1,800", theme.OUTPUT)]
            xs = [2.9, 4.8, 6.1]
            tab = VGroup(*[h.move_to([xs[j], 1.0 - 0.0, 0]) for j, h in enumerate(hdr)])
            tab.shift(DOWN * 0.3)
            for k, (a, b, c, col) in enumerate(data):
                y = 0.1 - 0.55 * k
                tab.add(zh(a, 20, col).move_to([xs[0], y, 0]), mono(b, 20, col).move_to([xs[1], y, 0]),
                        mono(c, 20, col).move_to([xs[2], y, 0]))
            src = zh("Qwen3 技术报告 表 21", 16, theme.MUTED).move_to([4.5, -1.15, 0])
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(tab), FadeIn(src), run_time=self.fit(1))
            who = zh("采用：Qwen3 · Gemma 2 · GLM-5 · MiMo-V2-Flash · DeepSeek-V4", 22, theme.HIGHLIGHT
                     ).move_to([0, -1.65, 0])
            no = zh("同样要求同一个词表 → 主线用不上", 20, theme.GRAD).move_to([0, -2.25, 0])
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(who), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(no), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.clear_all(g, lab1, lab2, f, tab, src, who, no)

    # ── S10 拒绝采样的漏斗 ───────────────────────────────────────────────
    def s10(self) -> None:
        with self.shot("S10"):
            self.play(*self.set_heading("拒绝采样：多采几个，只留验证过的"), run_time=self.fit(0.8))
            st = funnel_results()["stages"]
            names = ["候选（200 任务 × 8 次）", "格式正确", "调用正确", "执行 + 回答正确", "每任务留 1 条"]
            top, h, gap = 2.2, 0.62, 0.1
            maxw = 7.0
            g = VGroup()
            for k, (name, n) in enumerate(zip(names, st)):
                w = maxw * n / st[0]
                w2 = maxw * (st[k + 1] if k + 1 < len(st) else n) / st[0]
                y = top - k * (h + gap)
                poly = Polygon([-w / 2 - 1.8, y + h / 2, 0], [w / 2 - 1.8, y + h / 2, 0],
                               [w2 / 2 - 1.8, y - h / 2, 0], [-w2 / 2 - 1.8, y - h / 2, 0],
                               color=theme.OUTPUT if k == len(st) - 1 else theme.INPUT,
                               fill_opacity=0.35, stroke_width=1.5)
                g.add(VGroup(poly, mono(f"{n}", 22).move_to([-1.8, y, 0]),
                             zh(name, 20, theme.FG).move_to([3.0, y, 0], aligned_edge=LEFT)))
            for row in g:
                self.play(FadeIn(row), run_time=self.fit(0.6))
                self.wait(min(2.8, self.remaining() * 0.17))
            sim = zh("教师是规则模拟器（不是语言模型），按固定概率犯错；筛选用 zero 的生产级代码", 18,
                     theme.MUTED).move_to([0, -1.75, 0])
            self.play(FadeIn(sim), run_time=self.fit(0.5))
            moto = zh("宁可少，不可错", 26, theme.HIGHLIGHT).move_to([0, -2.3, 0])
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(moto), run_time=self.fit(0.5))
            self.wait(self.remaining() - 0.6)
            self.clear_all(g, sim, moto)

    # ── S11 验证器漏洞（极小配置演示）──────────────────────────────────────
    def s11(self) -> None:
        with self.shot("S11"):
            self.play(*self.set_heading("验证器只保证它检查的东西"), run_time=self.fit(0.8))
            badge = self.show_badge()
            sm = smoke()
            nums = VGroup(mono(f"{sm['cand']}", 44, theme.INPUT), zh("个候选", 24),
                          Arrow(LEFT, RIGHT, color=theme.FG).scale(0.6),
                          mono(f"{sm['ok']}", 44, theme.OUTPUT), zh("条通过", 24)
                          ).arrange(RIGHT, buff=0.25).move_to([0, 1.9, 0])
            sub = zh("冒烟测试：tiny SFT 模型自己当替身教师（下载不了开源权重）", 18, theme.MUTED
                     ).next_to(nums, DOWN, 0.2)
            self.play(FadeIn(nums), FadeIn(sub), run_time=self.fit(1))
            u = RoundedRectangle(width=5.2, height=0.7, corner_radius=0.15, color=theme.INPUT,
                                 fill_opacity=0.2).move_to([-2.3, 0.35, 0])
            a = RoundedRectangle(width=5.2, height=0.7, corner_radius=0.15, color=theme.GRAD,
                                 fill_opacity=0.2).move_to([1.9, -0.55, 0])
            ut = zh(f"用户：{sm['user']}", 24).move_to(u)
            at = zh(f"助手：{sm['reply']}", 24).move_to(a)
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(u), FadeIn(ut), run_time=self.fit(0.6))
            self.play(FadeIn(a), FadeIn(at), run_time=self.fit(0.6))
            why = zh("旧版只查“没乱调工具”→ 已修：只给 0.5 分、不算通过；编造数字 0 分", 22,
                     theme.HIGHLIGHT).move_to([0, -1.45, 0])
            loss = MathTex(rf"\text{{loss}}\ {sm['loss']:.3f} = 0.5\times\text{{ce}}\ {sm['ce']:.3f}"
                           rf" + 0.5\times\text{{kd}}\ {sm['kd']:.4f}", font_size=30,
                           color=theme.MUTED).move_to([0, -2.2, 0])
            self.wait(self.remaining() * 0.25)
            self.play(FadeIn(why), run_time=self.fit(0.6))
            self.play(FadeIn(loss), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.clear_all(badge, nums, sub, u, a, ut, at, why, loss)

    # ── S12 许可证 ───────────────────────────────────────────────────────
    def s12(self) -> None:
        with self.shot("S12"):
            self.play(*self.set_heading("教师的许可证：读原文"), run_time=self.fit(0.8))
            rows = [("Qwen3 / Qwen3.5", "Apache-2.0", "无额外限制", True),
                    ("gpt-oss", "Apache-2.0", "遵守适用法律", True),
                    ("DeepSeek-R1 / V4", "MIT", "R1 明写允许蒸馏", True),
                    ("GLM-5 · MiMo-V2-Flash", "MIT", "无额外限制", True),
                    ("Gemma 1–3", "Gemma 条款", "用输出训练的模型也算衍生品", False),
                    ("Llama 3.1–4", "社区许可", "名字须以 Llama 开头", False)]
            xs = [-4.1, -0.9, 2.9, 6.1]
            g = VGroup()
            for k, (a, b, c, ok) in enumerate(rows):
                y = 2.15 - 0.68 * k
                col = theme.OUTPUT if ok else theme.GRAD
                g.add(VGroup(zh(a, 22).move_to([xs[0], y, 0]), zh(b, 22, col).move_to([xs[1], y, 0]),
                             zh(c, 20, theme.MUTED).move_to([xs[2], y, 0]),
                             zh("✓" if ok else "✗", 28, col).move_to([xs[3], y, 0])))
            for row in g:
                self.play(FadeIn(row, shift=RIGHT * 0.15), run_time=self.fit(0.5))
                self.wait(min(2.0, self.remaining() * 0.12))
            rule = zh("主线：只用 Apache-2.0 / MIT 教师，每条数据记录名称、版本、许可证", 22,
                      theme.HIGHLIGHT).move_to([0, -2.2, 0])
            self.play(FadeIn(rule), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.clear_all(g, rule)

    # ── S13 小结 ─────────────────────────────────────────────────────────
    def s13(self) -> None:
        with self.shot("S13"):
            self.play(*self.set_heading("小结"), run_time=self.fit(0.6))
            items = [(["软标签", "+ 温度"], theme.PARAM), (["同一个词表？"], theme.GRAD),
                     (["序列级 / logits", "/ 在线策略"], theme.INPUT), (["执行验证", "+ 许可证"], theme.OUTPUT)]
            g = VGroup()
            for lines, c in items:
                r = RoundedRectangle(width=2.7, height=1.2, corner_radius=0.15, color=c, fill_opacity=0.2)
                txt = VGroup(*[zh(t, 20, c) for t in lines]).arrange(DOWN, buff=0.12).move_to(r)
                g.add(VGroup(r, txt))
            g.arrange(RIGHT, buff=0.55).move_to([0, 1.0, 0])
            arrows = VGroup(*[Arrow(g[i].get_right(), g[i + 1].get_left(), buff=0.05, color=theme.MUTED,
                                    stroke_width=3, max_tip_length_to_length_ratio=0.3)
                              for i in range(3)])
            for i, b in enumerate(g):
                self.play(FadeIn(b), *( [Create(arrows[i - 1])] if i else []), run_time=self.fit(0.6))
                self.wait(min(3.0, self.remaining() * 0.18))
            main = zh("主线：自有分词器 → 序列级蒸馏 + 执行验证", 24, theme.HIGHLIGHT).move_to([0, -0.6, 0])
            self.play(FadeIn(main), run_time=self.fit(0.6))
            nxt = zh("下一章：偏好对齐（RLHF → DPO）", 26, theme.FG).move_to([0, -1.6, 0])
            self.wait(self.remaining() * 0.5)
            self.play(FadeIn(nxt), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(VGroup(g, arrows, main, nxt)), *self.set_heading(None),
                      run_time=self.fit(0.8))
