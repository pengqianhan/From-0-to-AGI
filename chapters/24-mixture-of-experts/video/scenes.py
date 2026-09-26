"""第 24 章视频：混合专家（MoE）—— 参数翻几十倍，每个 token 的算力不变

画面里的所有数值都由 ../code/ 中的代码真实计算（见 script.md 事实清单）。
结果缓存在 video/out/cache.json；删掉它会重新计算。S06、S10、S11 读取 code/out/moe_runs/
（先运行 code/03_train_compare.py）。
渲染：bash chapters/24-mixture-of-experts/video/build.sh
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
    Create,
    DashedLine,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    LaggedStart,
    Line,
    MathTex,
    Rectangle,
    RoundedRectangle,
    SurroundingRectangle,
    Text,
    Transform,
    VGroup,
    Write,
)

from video_kit import theme
from video_kit.scene import NarratedScene, zh

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
    """从 ../code 真实计算视频要用的全部数字。"""
    led = _load("ch24_ledger", "01_param_ledger.py")
    moe = _load("ch24_moe", "02_moe_layer.py")
    exp = _load("ch24_exp", "03_train_compare.py")
    sv = _load("ch24_small", "04_small_vs_large.py")
    d: dict = {}
    d["dense"] = {n: led.dense_breakdown(c) for n, c in led.DENSE.items()}
    d["moe"] = moe_rows = led.moe_rows()
    d["moe"] = [dict(name=r["name"], total=r["off_total"], active=r["off_active"],
                     calc_total=r["total"], calc_active=r["active"]) for r in moe_rows]
    d["skew"] = moe.skewed_router_demo()
    rows = exp.run()
    if len(rows) < len(exp.VARIANTS) * len(exp.SEEDS):
        raise SystemExit("小实验还没跑完：先运行 code/03_train_compare.py")
    d["summary"] = exp.summarize(rows)
    d["loads"] = {r["name"]: r["loads"] for r in rows if r["seed"] == 0 and r["loads"]}
    d["log_every"] = exp.LOG_EVERY
    d["show_layer"] = exp.SHOW_LAYER
    d["touched"] = {n: [sv.touched_fraction(E, K, b) for b in (1, 16, 64)]
                    for n, (E, K, _) in sv.MOE_CFG.items()}
    t5 = sv.QWEN3_TABLE5
    d["table5"] = {k: v for k, v in t5.items()}
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


def bar(width: float, color: str, height: float = 0.36, opacity: float = 0.85) -> Rectangle:
    return Rectangle(width=max(width, 0.03), height=height, fill_color=color,
                     fill_opacity=opacity, stroke_width=0)


def histogram(frac: list[float], width: float = 4.0, height: float = 2.2, color=theme.PARAM,
              top: float = 0.6) -> VGroup:
    """专家负载直方图：底边在 y=0，高度按 top（占比上限）缩放；带一条均匀负载的虚线。"""
    n = len(frac)
    w = width / n
    bars = VGroup()
    for i, f in enumerate(frac):
        h = max(0.02, min(f, top) / top * height)
        r = Rectangle(width=w * 0.75, height=h, fill_color=color, fill_opacity=0.9, stroke_width=0)
        r.move_to([-width / 2 + w * (i + 0.5), h / 2, 0])
        bars.add(r)
    base = Line([-width / 2, 0, 0], [width / 2, 0, 0], color=theme.MUTED, stroke_width=2)
    even = DashedLine([-width / 2, height / n / top, 0], [width / 2, height / n / top, 0],
                      color=theme.HIGHLIGHT, stroke_width=2)
    return VGroup(bars, base, even)


def imb(frac: list[float]) -> float:
    return max(frac) * len(frac)


def summ(name: str) -> dict:
    return next(s for s in D["summary"] if s["name"] == name)


class ChapterScene(NarratedScene):
    chapter_label = "第 24 章"
    chapter_title = "混合专家（MoE）"

    def construct(self) -> None:
        for i in range(1, 15):
            getattr(self, f"s{i:02d}")()

    def hold(self, reserve: float = 0.5) -> None:
        """等到本镜只剩 reserve 秒（剩余不足时不等）。"""
        t = self.remaining() - reserve
        if t > 0.05:
            self.wait(t)

    def clear_all(self, *keep) -> None:
        objs = [m for m in self.mobjects if m is not getattr(self, "_heading", None)
                and m not in keep]
        if objs:
            self.play(*[FadeOut(m) for m in objs], run_time=0.4)

    # ── S01 片头 ─────────────────────────────────────────────────────────
    def s01(self) -> None:
        with self.shot("S01"):
            card = self.chapter_card()
            sub = zh("参数翻几十倍，每个 token 的算力不变", 30, theme.HIGHLIGHT).next_to(card, DOWN, 0.6)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.hold(0.6)
            self.play(FadeOut(card), FadeOut(sub), run_time=0.5)

    # ── S02 FFN 是大头 ───────────────────────────────────────────────────
    def s02(self) -> None:
        with self.shot("S02"):
            self.play(*self.set_heading("参数都去哪了"), run_time=self.fit(0.8))
            rows = VGroup()
            for i, (name, b) in enumerate(D["dense"].items()):
                y = 1.2 - i * 1.6
                total = b["attn"] + b["ffn"]
                W = 8.0
                a = bar(W * b["attn"] / total, theme.ATTN, 0.6).move_to([-3.0, y, 0], aligned_edge=LEFT)
                a.align_to([-3.0, 0, 0], LEFT)
                f = bar(W * b["ffn"] / total, theme.PARAM, 0.6).next_to(a, RIGHT, buff=0)
                lab = zh(name, 26).next_to(a, LEFT, 0.3)
                ta = zh(f"注意力 {b['attn'] / 1e6:.1f}M", 20, theme.ATTN).next_to(a, UP, 0.1)
                ta.align_to(a, LEFT)
                tf = zh(f"FFN {b['ffn'] / 1e6:.1f}M · {b['share']:.1%}", 22, theme.BG).move_to(f)
                rows.add(VGroup(lab, a, f, ta, tf))
            note = zh("每层矩阵参数（≈ 每个 token 的矩阵乘算力）", 22, theme.MUTED).move_to([0.5, -2.2, 0])
            for r in rows:
                self.play(FadeIn(r[0]), GrowFromEdge(r[1], LEFT), GrowFromEdge(r[2], LEFT),
                          run_time=self.fit(1.2))
                self.play(FadeIn(r[3]), FadeIn(r[4]), run_time=self.fit(0.5))
            self.play(FadeIn(note), run_time=self.fit(0.6))
            self.hold(0.5)
            self.clear_all()

    # ── S03 拆成很多专家 ─────────────────────────────────────────────────
    def s03(self) -> None:
        with self.shot("S03"):
            self.play(*self.set_heading("拆成很多专家，每次只用几个"), run_time=self.fit(0.8))
            experts = VGroup()
            for i in range(8):
                box = RoundedRectangle(width=1.3, height=0.42, corner_radius=0.08,
                                       stroke_color=theme.PARAM, fill_color=theme.PARAM,
                                       fill_opacity=0.15)
                box.move_to([0.6, 2.2 - i * 0.6, 0])
                experts.add(VGroup(box, zh(f"专家 {i + 1}", 18).move_to(box)))
            router = RoundedRectangle(width=1.4, height=0.8, corner_radius=0.1,
                                      stroke_color=theme.ATTN).move_to([-2.4, 0.1, 0])
            rlab = zh("路由器", 22, theme.ATTN).move_to(router)
            out = RoundedRectangle(width=1.8, height=0.8, corner_radius=0.1,
                                   stroke_color=theme.OUTPUT).move_to([3.4, 0.1, 0])
            olab = zh("加权求和", 20, theme.OUTPUT).move_to(out)
            self.play(FadeIn(experts), FadeIn(router), FadeIn(rlab), FadeIn(out), FadeIn(olab),
                      run_time=self.fit(1.2))
            info = VGroup(zh("8 选 2：", 24, theme.HIGHLIGHT),
                          zh("总参数 8 份", 24, theme.PARAM),
                          zh("每个 token 只算 2 份", 24, theme.OUTPUT)).arrange(DOWN, aligned_edge=LEFT)
            info.move_to([5.4, -1.4, 0])
            for picks, color, name in (((1, 5), theme.INPUT, "token A"),
                                       ((3, 6), theme.GRAD, "token B")):
                tok = VGroup(*[Rectangle(width=0.28, height=0.28, fill_color=color, fill_opacity=0.8,
                                         stroke_width=0) for _ in range(4)]).arrange(DOWN, buff=0.04)
                tok.move_to([-5.2, 0.1, 0])
                tl = zh(name, 20, color).next_to(tok, DOWN, 0.15)
                a1 = Arrow(tok.get_right(), router.get_left(), buff=0.1, color=color)
                self.play(FadeIn(tok), FadeIn(tl), Create(a1), run_time=self.fit(0.8))
                arrows = VGroup(*[Arrow(router.get_right(), experts[p][0].get_left(), buff=0.08,
                                        color=color, stroke_width=4) for p in picks])
                backs = VGroup(*[Arrow(experts[p][0].get_right(), out.get_left(), buff=0.08,
                                       color=color, stroke_width=4) for p in picks])
                self.play(Create(arrows),
                          *[experts[p][0].animate.set_fill(color, 0.6) for p in picks],
                          run_time=self.fit(1.0))
                self.play(Create(backs), run_time=self.fit(0.7))
                self.wait(min(1.5, self.remaining() * 0.25))
                self.play(FadeOut(tok), FadeOut(tl), FadeOut(a1), FadeOut(arrows), FadeOut(backs),
                          *[experts[p][0].animate.set_fill(theme.PARAM, 0.15) for p in picks],
                          run_time=self.fit(0.5))
            self.play(FadeIn(info), run_time=self.fit(0.8))
            self.hold(0.5)
            self.clear_all()

    # ── S04 路由公式 ─────────────────────────────────────────────────────
    def s04(self) -> None:
        with self.shot("S04"):
            self.play(*self.set_heading("路由：打分 → 选 K 个 → 加权求和"), run_time=self.fit(0.8))
            lines = VGroup(
                MathTex(r"s_i = \sigma(x \cdot e_i)\ \ \text{or}\ \ \mathrm{softmax}_i(x W_r)",
                        font_size=40),
                MathTex(r"\mathcal{S} = \mathrm{TopK}(s + b,\ K)", font_size=40),
                MathTex(r"g_i = \frac{s_i}{\sum_{j\in\mathcal{S}} s_j}", font_size=40),
                MathTex(r"y = \sum_{\text{shared}} \mathrm{FFN}_s(x) + \sum_{i\in\mathcal{S}}"
                        r" g_i\,\mathrm{FFN}_i(x)", font_size=40),
            ).arrange(DOWN, buff=0.4, aligned_edge=LEFT).move_to([-1.2, 0.4, 0])
            notes = VGroup(zh("打分", 22, theme.ATTN), zh("选专家（偏置 b 只管选）", 22, theme.ATTN),
                           zh("门控权重", 22, theme.OUTPUT), zh("只算被选中的专家", 22, theme.OUTPUT))
            for n, line in zip(notes, lines):
                n.next_to(line, RIGHT, 0.5)
                n.set_x(max(n.get_x(), 4.4))
            for line, n in zip(lines, notes):
                self.play(Write(line), FadeIn(n), run_time=self.fit(1.2))
            who = zh("sigmoid：DeepSeek、GLM、Kimi　　softmax：Qwen、Mixtral、gpt-oss", 22,
                     theme.MUTED).move_to([0, -2.3, 0])
            self.play(FadeIn(who), run_time=self.fit(0.8))
            self.hold(0.5)
            self.clear_all()

    # ── S05 真实模型 ─────────────────────────────────────────────────────
    def s05(self) -> None:
        with self.shot("S05"):
            self.play(*self.set_heading("总参数 vs 激活参数（官方数字，单位 B）"), run_time=self.fit(0.8))
            order = ["Mixtral-8x7B", "Qwen3-235B-A22B", "GLM-4.5", "DeepSeek-V3", "Kimi-K2",
                     "gpt-oss-120b"]
            rows = {r["name"]: r for r in D["moe"]}
            scale = 7.0 / 1040
            group = VGroup()
            for i, n in enumerate(order):
                r = rows[n]
                y = 2.1 - i * 0.78
                x0 = -3.0
                tb = bar(r["total"] * scale, theme.PARAM, 0.28, 0.35).move_to([x0, y + 0.13, 0],
                                                                                aligned_edge=LEFT)
                ab = bar(r["active"] * scale, theme.PARAM, 0.28, 1.0).move_to([x0, y - 0.17, 0],
                                                                                aligned_edge=LEFT)
                lab = zh(n, 22).move_to([x0 - 0.2, y, 0], aligned_edge=RIGHT)
                tt = mono(f"{r['total']:g}", 18, theme.MUTED).next_to(tb, RIGHT, 0.12)
                at = mono(f"{r['active']:g}  ({r['active'] / r['total']:.1%})", 18,
                          theme.HIGHLIGHT).next_to(ab, RIGHT, 0.12)
                group.add(VGroup(lab, tb, ab, tt, at))
            legend = VGroup(bar(0.4, theme.PARAM, 0.25, 0.35), zh("总参数", 20, theme.MUTED),
                            bar(0.4, theme.PARAM, 0.25, 1.0), zh("激活参数", 20)).arrange(RIGHT, buff=0.15)
            legend.move_to([3.8, -2.35, 0])
            self.play(LaggedStart(*[FadeIn(g) for g in group], lag_ratio=0.3),
                      run_time=self.fit(3.5))
            self.play(FadeIn(legend), run_time=self.fit(0.6))
            self.hold(0.5)
            self.clear_all()

    # ── S06 坍缩 ─────────────────────────────────────────────────────────
    def s06(self) -> None:
        with self.shot("S06"):
            self.play(*self.set_heading("问题：路由坍缩"), run_time=self.fit(0.8))
            badge = self.demo_badge("极小实验")
            loads = D["loads"]["MoE-无均衡"]
            L = D["show_layer"]
            idxs = [0, 2, 6, len(loads) - 1]
            h = histogram(loads[idxs[0]][L], width=5.5, height=3.0).move_to([-1.5, -0.2, 0])
            h.align_to([0, -2.1, 0], DOWN)
            base_y = h[1].get_y()
            step_lab = zh(f"第 {L} 层 · step {idxs[0] * D['log_every']}", 24).move_to([4.2, 1.2, 0])
            ratio = mono(f"最大/平均 {imb(loads[idxs[0]][L]):.2f}", 24, theme.HIGHLIGHT).next_to(
                step_lab, DOWN, 0.3)
            even = zh("虚线 = 均匀 1/8", 20, theme.HIGHLIGHT).next_to(ratio, DOWN, 0.3)
            self.play(FadeIn(badge), FadeIn(h), FadeIn(step_lab), FadeIn(ratio), FadeIn(even),
                      run_time=self.fit(1.0))
            for i in idxs[1:]:
                nh = histogram(loads[i][L], width=5.5, height=3.0).move_to(h)
                nh.shift(UP * (base_y - nh[1].get_y()))
                ns = zh(f"第 {L} 层 · step {i * D['log_every']}", 24).move_to(step_lab)
                nr = mono(f"最大/平均 {imb(loads[i][L]):.2f}", 24, theme.HIGHLIGHT).move_to(ratio)
                self.play(Transform(h, nh), Transform(step_lab, ns), Transform(ratio, nr),
                          run_time=self.fit(1.5))
                self.wait(min(1.0, self.remaining() * 0.2))
            self.hold(0.5)
            self.clear_all()

    # ── S07 辅助损失 ─────────────────────────────────────────────────────
    def s07(self) -> None:
        with self.shot("S07"):
            self.play(*self.set_heading("办法一：辅助损失"), run_time=self.fit(0.8))
            f = MathTex(r"\mathcal{L}_{aux} = \alpha \sum_i f_i \, P_i", font_size=52).move_to([-3.2, 1.6, 0])
            d1 = zh("f_i：专家 i 实际分到的 token 比例 × N/K", 22).move_to([-3.2, 0.5, 0])
            d2 = zh("P_i：路由器给专家 i 的平均概率", 22).move_to([-3.2, 0.0, 0])
            d3 = zh("完全均衡时 = α；梯度只经过 P", 22, theme.HIGHLIGHT).move_to([-3.2, -0.6, 0])
            ex = VGroup(
                zh("手算：4 个 token，2 个专家，选 1", 22, theme.MUTED),
                MathTex(r"\text{load} = [3,\ 1]\Rightarrow f=[1.5,\ 0.5]", font_size=34),
                MathTex(r"P = [0.7,\ 0.3]", font_size=34),
                MathTex(r"\mathcal{L} = \alpha(1.5\cdot0.7 + 0.5\cdot0.3) = 1.2\,\alpha",
                        font_size=34, color=theme.HIGHLIGHT),
            ).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to([3.6, 0.5, 0])
            self.play(Write(f), run_time=self.fit(1.2))
            self.play(FadeIn(d1), FadeIn(d2), run_time=self.fit(1.0))
            self.play(FadeIn(d3), run_time=self.fit(0.8))
            self.play(LaggedStart(*[FadeIn(e) for e in ex], lag_ratio=0.4), run_time=self.fit(2.5))
            warn = zh("α 太小管不住，太大伤害模型质量", 24, theme.GRAD).move_to([0, -1.9, 0])
            self.wait(max(0.1, self.remaining() * 0.4))
            self.play(FadeIn(warn), run_time=self.fit(0.6))
            self.hold(0.5)
            self.clear_all()

    # ── S08 偏置法 ───────────────────────────────────────────────────────
    def s08(self) -> None:
        with self.shot("S08"):
            self.play(*self.set_heading("办法二：无辅助损失的偏置（DeepSeek-V3）"), run_time=self.fit(0.8))
            rule = MathTex(r"b_i \leftarrow b_i + \gamma\cdot\mathrm{sign}(\bar{c} - c_i)",
                           font_size=40).move_to([3.6, 1.8, 0])
            note = zh("只加在“选谁”上，不进门控、不吃梯度", 22, theme.MUTED).next_to(rule, DOWN, 0.3)
            rows = {r["step"]: r for r in D["skew"]}
            h = histogram(rows[0]["frac"], width=5.0, height=3.0).move_to([-3.2, -0.3, 0])
            h.align_to([0, -2.2, 0], DOWN)
            base_y = h[1].get_y()
            lab = zh("固定的偏心路由器，8 选 2，γ = 0.01", 20, theme.MUTED).next_to(h, UP, 0.25)
            st = zh("更新 0 次", 26).move_to([3.6, 0.1, 0])
            rt = mono(f"最大/平均 {rows[0]['max_over_mean']:.2f}", 26, theme.HIGHLIGHT).next_to(st, DOWN, 0.3)
            self.play(Write(rule), FadeIn(note), run_time=self.fit(1.5))
            self.play(FadeIn(h), FadeIn(lab), FadeIn(st), FadeIn(rt), run_time=self.fit(1.0))
            self.wait(max(0.1, self.remaining() * 0.25))
            for s in (10, 30):
                nh = histogram(rows[s]["frac"], width=5.0, height=3.0).move_to(h)
                nh.shift(UP * (base_y - nh[1].get_y()))
                self.play(Transform(h, nh),
                          Transform(st, zh(f"更新 {s} 次", 26).move_to(st)),
                          Transform(rt, mono(f"最大/平均 {rows[s]['max_over_mean']:.2f}", 26,
                                             theme.HIGHLIGHT).move_to(rt)),
                          run_time=self.fit(1.5))
                self.wait(min(1.5, self.remaining() * 0.3))
            self.hold(0.5)
            self.clear_all()

    # ── S09 容量因子 ─────────────────────────────────────────────────────
    def s09(self) -> None:
        with self.shot("S09"):
            self.play(*self.set_heading("容量因子：超出上限的 token 被丢弃"), run_time=self.fit(0.8))
            rows = {r["step"]: r for r in D["skew"]}

            def buckets(frac, x_center):
                n = len(frac)
                cap = 1.25 / n  # 容量 = 1.25 × 平均
                top = 0.55
                H, W = 3.0, 4.6
                g = VGroup()
                for i, f in enumerate(frac):
                    x = x_center - W / 2 + W / n * (i + 0.5)
                    kept = min(f, cap)
                    hk = max(0.02, kept / top * H)
                    r = Rectangle(width=W / n * 0.7, height=hk, fill_color=theme.OUTPUT,
                                  fill_opacity=0.85, stroke_width=0).move_to([x, -2.1 + hk / 2, 0])
                    g.add(r)
                    if f > cap:
                        hd = min(f, top) / top * H - hk
                        g.add(Rectangle(width=W / n * 0.7, height=hd, fill_color=theme.GRAD,
                                        fill_opacity=0.85, stroke_width=0).next_to(r, UP, 0))
                cy = -2.1 + cap / top * H
                g.add(DashedLine([x_center - W / 2, cy, 0], [x_center + W / 2, cy, 0],
                                 color=theme.HIGHLIGHT))
                g.add(Line([x_center - W / 2, -2.1, 0], [x_center + W / 2, -2.1, 0], color=theme.MUTED))
                return g

            left = buckets(rows[0]["frac"], -3.3)
            right = buckets(rows[30]["frac"], 3.3)
            tl = zh(f"偏心：丢弃 {rows[0]['dropped']:.1%}", 26, theme.GRAD).move_to([-3.3, 1.9, 0])
            tr = zh(f"均衡后：丢弃 {rows[30]['dropped']:.1%}", 26, theme.OUTPUT).move_to([3.3, 1.9, 0])
            cap = zh("虚线 = 容量（1.25 × 平均）", 20, theme.HIGHLIGHT).move_to([0, 2.55, 0])
            self.play(FadeIn(left), FadeIn(tl), FadeIn(cap), run_time=self.fit(1.2))
            self.wait(max(0.1, self.remaining() * 0.35))
            self.play(FadeIn(right), FadeIn(tr), run_time=self.fit(1.2))
            self.hold(0.5)
            self.clear_all()

    # ── S10 小实验：负载 ─────────────────────────────────────────────────
    def s10(self) -> None:
        with self.shot("S10"):
            self.play(*self.set_heading("小实验：三种均衡方式（第 1 层，训练结束）"), run_time=self.fit(0.8))
            badge = self.demo_badge("极小实验")
            names = [("MoE-无均衡", "无均衡", theme.GRAD), ("MoE-辅助损失", "辅助损失", theme.ATTN),
                     ("MoE-无辅助损失", "偏置（无辅助损失）", theme.OUTPUT)]
            groups = VGroup()
            for i, (n, lab, col) in enumerate(names):
                frac = D["loads"][n][-1][D["show_layer"]]
                h = histogram(frac, width=3.4, height=2.6, color=col)
                h.move_to([-4.4 + i * 4.4, 0, 0])
                h.align_to([0, -1.9, 0], DOWN)
                t = zh(lab, 24, col).next_to(h, UP, 0.3)
                r = mono(f"最大/平均 {imb(frac):.2f}", 22, theme.HIGHLIGHT).next_to(h, DOWN, 0.2)
                groups.add(VGroup(h, t, r))
            self.play(FadeIn(badge), run_time=0.3)
            for g in groups:
                self.play(FadeIn(g), run_time=self.fit(1.0))
                self.wait(min(1.5, self.remaining() * 0.2))
            self.hold(0.5)
            self.clear_all()

    # ── S11 小实验：和稠密比 ─────────────────────────────────────────────
    def s11(self) -> None:
        with self.shot("S11"):
            self.play(*self.set_heading("小实验：验证 loss（越低越好，2 个种子）"), run_time=self.fit(0.8))
            badge = self.demo_badge("极小实验")
            ss = D["summary"]
            lo = min(min(s["vals"]) for s in ss) - 0.03
            hi = max(max(s["vals"]) for s in ss) + 0.01
            W = 6.0
            x0 = -1.2
            group = VGroup()
            for i, s in enumerate(ss):
                y = 2.1 - i * 0.72
                col = theme.INPUT if s["name"].startswith("稠密") else theme.PARAM
                w = (s["mean"] - lo) / (hi - lo) * W
                b = bar(w, col, 0.4).move_to([x0, y, 0], aligned_edge=LEFT)
                rng = Line([x0 + (min(s["vals"]) - lo) / (hi - lo) * W, y, 0],
                           [x0 + (max(s["vals"]) - lo) / (hi - lo) * W, y, 0],
                           color=theme.FG, stroke_width=4)
                lab = zh(s["name"], 22).move_to([x0 - 0.25, y, 0], aligned_edge=RIGHT)
                sub = zh(f"FFN 激活 {s['ffn_active'] / 1e3:.0f}K / 总 {s['ffn_total'] / 1e3:.0f}K", 16,
                         theme.MUTED).move_to([x0 - 0.25, y - 0.27, 0], aligned_edge=RIGHT)
                val = mono(f"{s['mean']:.3f}", 20, theme.HIGHLIGHT).next_to(b, RIGHT, 0.15)
                val.set_x(max(val.get_x(), rng.get_right()[0] + 0.5))
                group.add(VGroup(lab, sub, b, rng, val))
            self.play(FadeIn(badge), run_time=0.3)
            self.play(LaggedStart(*[FadeIn(g) for g in group], lag_ratio=0.3), run_time=self.fit(3.0))
            axis = zh(f"横轴从 {lo:.2f} 起（放大差异）；白线 = 两个种子的范围", 18,
                      theme.MUTED).move_to([1.5, -2.35, 0])
            self.play(FadeIn(axis), run_time=self.fit(0.6))
            self.hold(0.5)
            self.clear_all()

    # ── S12 细粒度 + 共享 ────────────────────────────────────────────────
    def s12(self) -> None:
        with self.shot("S12"):
            self.play(*self.set_heading("细粒度专家 + 共享专家"), run_time=self.fit(0.8))

            def grid(n_cols, n_rows, w, h, picks, x, y, shared=False):
                g = VGroup()
                for i in range(n_cols * n_rows):
                    r = Rectangle(width=w, height=h, stroke_color=theme.PARAM, stroke_width=1.5,
                                  fill_color=theme.PARAM, fill_opacity=0.7 if i in picks else 0.1)
                    g.add(r)
                g.arrange_in_grid(n_rows, n_cols, buff=0.06).move_to([x, y, 0])
                if shared:
                    s = Rectangle(width=g.width, height=h, stroke_color=theme.OUTPUT,
                                  fill_color=theme.OUTPUT, fill_opacity=0.7).next_to(g, DOWN, 0.12)
                    g.add(s)
                return g

            coarse = grid(4, 4, 0.55, 0.4, {2, 9}, -4.6, 0.6)
            fine = grid(8, 8, 0.26, 0.17, {1, 12, 19, 30, 37, 44, 50, 61}, -1.6, 0.6, shared=True)
            c1 = MathTex(r"\binom{16}{2}=120", font_size=34).next_to(coarse, DOWN, 0.35)
            c2 = MathTex(r"\binom{64}{8}\approx 4.4\times10^{9}", font_size=34).next_to(fine, DOWN, 0.35)
            c2.set_y(c1.get_y())
            sh = zh("绿色 = 共享专家", 20, theme.OUTPUT).next_to(c2, DOWN, 0.2)
            self.play(FadeIn(coarse), Write(c1), run_time=self.fit(1.2))
            self.play(FadeIn(fine), Write(c2), run_time=self.fit(1.2))
            self.play(FadeIn(sh), run_time=self.fit(0.5))
            yes = VGroup(zh("有共享专家", 24, theme.OUTPUT),
                         *[zh(t, 20) for t in ("DeepSeek-V3 · Kimi K2", "GLM-4.5 · Llama 4",
                                               "Qwen3.5 · Nemotron 3")]).arrange(DOWN, aligned_edge=LEFT)
            no = VGroup(zh("没有", 24, theme.GRAD),
                        *[zh(t, 20) for t in ("Qwen3-MoE · gpt-oss", "Mixtral · MiniMax-M2")]).arrange(
                DOWN, aligned_edge=LEFT)
            yes.move_to([4.2, 1.2, 0])
            no.next_to(yes, DOWN, 0.5, aligned_edge=LEFT)
            self.wait(max(0.1, self.remaining() * 0.3))
            self.play(FadeIn(yes), run_time=self.fit(0.8))
            self.play(FadeIn(no), run_time=self.fit(0.8))
            self.hold(0.5)
            self.clear_all()

    # ── S13 为什么小模型少用 ─────────────────────────────────────────────
    def s13(self) -> None:
        with self.shot("S13"):
            self.play(*self.set_heading("为什么小模型很少用 MoE"), run_time=self.fit(0.8))
            dense = ["0.8B", "2B", "4B", "9B", "27B"]
            moe = ["35B-A3B", "122B-A10B", "397B-A17B"]
            line = VGroup(zh("Qwen3.5：", 22, theme.MUTED),
                          *[zh(t, 22, theme.INPUT) for t in dense], zh("｜", 26, theme.HIGHLIGHT),
                          *[zh(t, 22, theme.PARAM) for t in moe]).arrange(RIGHT, buff=0.25)
            line.move_to([0, 2.2, 0])
            leg = zh("蓝 = 稠密　橙 = MoE（总参数-激活参数）", 18, theme.MUTED).next_to(line, DOWN, 0.2)
            self.play(FadeIn(line), FadeIn(leg), run_time=self.fit(1.2))
            t5 = D["table5"]
            names = [k for k in t5 if k != "benchmarks"]
            tbl = VGroup(zh("Qwen3 报告表 5（同样的数据）", 22, theme.MUTED))
            for n in names:
                r = t5[n]
                tbl.add(zh(f"{n}：MMLU {r['scores'][0]:.2f}  总 {r['total']}B / 激活 {r['active']}B", 20))
            tbl.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([0, 0.75, 0])
            self.play(FadeIn(tbl), run_time=self.fit(1.2))
            tq = D["touched"]["Qwen3-30B-A3B"]
            rd = VGroup(zh("解码一步读取的专家比例（Qwen3-30B-A3B）", 20, theme.MUTED),
                        mono(f"batch 1: {tq[0]:.1%}   16: {tq[1]:.1%}   64: {tq[2]:.1%}", 20,
                             theme.HIGHLIGHT)).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
            rd.move_to([0, -0.75, 0])
            self.play(FadeIn(rd), run_time=self.fit(1.0))
            key = zh("MoE 省的是算力，不是显存", 30, theme.HIGHLIGHT).move_to([0, -1.9, 0])
            self.wait(max(0.1, self.remaining() * 0.4))
            self.play(Write(key), run_time=self.fit(1.0))
            self.hold(0.5)
            self.clear_all()

    # ── S14 谁在用 + 生产级 ──────────────────────────────────────────────
    def s14(self) -> None:
        with self.shot("S14"):
            self.play(*self.set_heading("谁在用 · 从极简到生产级"), run_time=self.fit(0.8))
            free = VGroup(zh("无辅助损失（偏置）", 24, theme.OUTPUT),
                          zh("报告：DeepSeek-V3 · GLM-4.5 · Nemotron 3", 20),
                          zh("配置：Kimi K2/K3 · GLM-5 · MiniMax-M2 · MiMo-V2", 20)).arrange(
                DOWN, aligned_edge=LEFT, buff=0.18).move_to([-3.2, 1.5, 0])
            aux = VGroup(zh("辅助损失", 24, theme.ATTN),
                         zh("Qwen3（全局 batch）· Llama 4", 20),
                         zh("Mixtral · OLMoE", 20)).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
            aux.move_to([3.6, 1.5, 0])
            self.play(FadeIn(free), run_time=self.fit(1.2))
            self.play(FadeIn(aux), run_time=self.fit(1.0))
            a = mono("code/02_moe_layer.py", 22, theme.INPUT).move_to([-3.6, -0.6, 0])
            b = mono("zero/arch/moe.py", 22, theme.OUTPUT).move_to([3.4, -0.6, 0])
            arr = Arrow(a.get_right(), b.get_left(), color=theme.MUTED)
            tst = zh("tests/test_arch_moe.py：1 个专家 = 稠密 SwiGLU；分组 = 朴素循环", 18,
                     theme.MUTED).move_to([0, -1.3, 0])
            self.play(FadeIn(a), Create(arr), FadeIn(b), run_time=self.fit(1.2))
            self.play(FadeIn(tst), run_time=self.fit(0.8))
            nxt = zh("下一章：多 token 预测与推测解码", 26, theme.HIGHLIGHT).move_to([0, -2.1, 0])
            box = SurroundingRectangle(nxt, color=theme.HIGHLIGHT, buff=0.15)
            self.wait(max(0.1, self.remaining() - 2.0))
            self.play(FadeIn(nxt), Create(box), run_time=self.fit(1.0))
            self.hold(0.3)
