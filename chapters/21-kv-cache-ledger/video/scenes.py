"""第 21 章视频：KV cache 的账本 —— 长上下文贵在哪，每个 token 该存多少

画面里的所有数值都由 ../code/ 中的代码真实计算（见 script.md 事实清单）。
结果缓存在 video/out/cache.json；删掉它会重新计算。S11 读取 code/out/results.pt
（先运行 code/04_attention_variants.py）。
渲染：bash chapters/21-kv-cache-ledger/video/build.sh
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
from video_kit.scene import NarratedScene, zh

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
CACHE = HERE / "out" / "cache.json"
MONO = "Noto Sans Mono"
GiB = 2**30


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def compute() -> dict:
    """从 ../code 真实计算视频要用的全部数字。"""
    import torch

    led = _load("kv_ledger", "01_kv_ledger.py")
    pd = _load("prefill_decode", "02_prefill_decode.py")
    d: dict = {}
    m = led.main_model()
    d["main_per_token"] = led.per_token(m)
    d["main_32k"] = led.kv_bytes(m, 32768)
    c, n_params, n_matmul, kv_tok = pd.model_numbers()
    d["weights"] = n_params * 2
    d["prefill"] = []
    for T in (1024, 4096, 32768, 131072):
        lin, att = pd.prefill_flops(c, n_matmul, T)
        d["prefill"].append(dict(T=T, share=att / (lin + att)))
    d["decode"] = {f"{T}_{B}": pd.decode_step(c, n_params, n_matmul, kv_tok, T, B)
                   for T in (4096, 32768) for B in (1, 16)}
    d["ridge"] = pd.PEAK_FLOPS / pd.HBM_BW
    free = pd.HBM_BYTES - d["weights"]
    d["users"] = {name: int(free // (32768 * per)) for name, per in
                  (("MHA", kv_tok * 2), ("GQA", kv_tok), ("MLA", c.n_layers * 576 * 2))}
    names = ["Qwen3-8B", "Llama-3.1-8B", "gpt-oss-120b", "Qwen3.5-9B", "Qwen3.5-397B-A17B",
             "DeepSeek-V3/V3.2", "Kimi-K3"]
    d["public"] = [dict(name=n, gib=led.kv_bytes(led.MODELS[n], 131072) / GiB) for n in names]
    ds = led.MODELS["DeepSeek-V3/V3.2"]
    d["ds_mha"] = led.kv_bytes(dict(layers=ds["layers"], kv_heads=ds["heads"], head_dim=128),
                               131072) / GiB
    rows = torch.load(CODE / "out" / "results.pt", weights_only=False)
    d["exp"] = [dict(name=r["name"], losses=r["losses"], mean=r["mean"], per_layer=r["per_layer"],
                     per_token=r["per_token_bytes"], cache=r["cache"], same=bool(r["same"]))
                for r in rows]
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


def hbar(width: float, color: str, height: float = 0.38, opacity: float = 0.85) -> Rectangle:
    return Rectangle(width=max(width, 0.02), height=height, fill_color=color,
                     fill_opacity=opacity, stroke_width=0)


class ChapterScene(NarratedScene):
    chapter_label = "第 21 章"
    chapter_title = "KV cache 的账本"

    def construct(self) -> None:
        self.s01()
        self.s02()
        self.s03()
        self.s04()
        self.s05()
        self.s06()
        self.s07()
        self.s08()
        self.s09()
        self.s10()
        self.s11()
        self.s12()
        self.s13()

    # ── S01 片头 ─────────────────────────────────────────────────────────
    def s01(self) -> None:
        with self.shot("S01"):
            card = self.chapter_card()
            sub = zh("长上下文贵在哪，每个 token 该存多少", 30, theme.HIGHLIGHT).next_to(card, DOWN, 0.6)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(card), FadeOut(sub), run_time=self.fit(0.8))

    # ── S02 回顾公式 ─────────────────────────────────────────────────────
    def s02(self) -> None:
        with self.shot("S02"):
            self.play(*self.set_heading("回顾：KV cache 的显存公式"), run_time=self.fit(0.8))
            f = MathTex(r"2", r"\times", r"L", r"\times", r"n_{kv}", r"\times", r"d_{head}",
                        r"\times", r"T", r"\times", r"\text{bytes}", font_size=54).move_to([0, 1.9, 0])
            f[4].set_color(theme.PARAM)
            f[8].set_color(theme.INPUT)
            labels = VGroup(*[zh(t, 20, theme.MUTED).next_to(f[i], DOWN, 0.2) for i, t in
                              ((0, "K 和 V"), (2, "层数"), (4, "KV 头数"), (8, "序列长"))])
            self.play(Write(f), run_time=self.fit(2))
            self.play(FadeIn(labels), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.25)
            per = D["main_per_token"]
            sub = zh(f"主线模型：2 × 28 × 8 × 128 × 2 字节 = {per:,} 字节 = {per // 1024} KiB / token",
                     26).move_to([0, 0.5, 0])
            self.play(FadeIn(sub), run_time=self.fit(1))
            scale = 2.0  # 每 GiB 的宽度
            kv_gib, w_gib = D["main_32k"] / GiB, D["weights"] / GiB
            b1 = hbar(kv_gib * scale, theme.ATTN).move_to([-2.0, -0.6, 0], aligned_edge=LEFT)
            b2 = hbar(w_gib * scale, theme.PARAM).move_to([-2.0, -1.4, 0], aligned_edge=LEFT)
            l1 = zh("一条 32K 对话的 KV cache", 22).next_to(b1, LEFT, 0.25)
            l2 = zh("模型权重（689.5M 参数）", 22).next_to(b2, LEFT, 0.25)
            v1 = zh(f"{kv_gib:.2f} GiB", 24, theme.ATTN).next_to(b1, RIGHT, 0.2)
            v2 = zh(f"{w_gib:.2f} GiB", 24, theme.PARAM).next_to(b2, RIGHT, 0.2)
            self.play(FadeIn(l1), FadeIn(l2), GrowFromEdge(b1, LEFT), GrowFromEdge(b2, LEFT),
                      run_time=self.fit(1.5))
            self.play(FadeIn(v1), FadeIn(v2), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(f, labels, sub, b1, b2, l1, l2, v1, v2)),
                      run_time=self.fit(0.6))

    # ── S03 prefill ──────────────────────────────────────────────────────
    def s03(self) -> None:
        with self.shot("S03"):
            self.play(*self.set_heading("prefill：注意力的运算量随 T² 增长"), run_time=self.fit(0.8))
            H, W, base = 4.2, 1.1, -2.2
            xs = [-5.2, -3.0, -0.8, 1.4]
            bars, texts = VGroup(), VGroup()
            for x, p in zip(xs, D["prefill"]):
                lin = hbar(W, theme.PARAM, height=H * (1 - p["share"]))
                lin.move_to([x, base, 0], aligned_edge=DOWN)
                att = hbar(W, theme.ATTN, height=H * p["share"])
                att.move_to([x, base + H * (1 - p["share"]), 0], aligned_edge=DOWN)
                k = p["T"] // 1024
                lab = zh(f"T = {k}K", 24, theme.MUTED).next_to(lin, DOWN, 0.15)
                pct = zh(f"{p['share']:.1%}", 26, theme.ATTN).next_to(att, UP, 0.12)
                bars.add(VGroup(lin, att))
                texts.add(VGroup(lab, pct))
            leg = VGroup(
                VGroup(hbar(0.35, theme.ATTN, 0.3), zh("注意力 ∝ T²", 22)).arrange(RIGHT, buff=0.15),
                VGroup(hbar(0.35, theme.PARAM, 0.3), zh("矩阵乘 ∝ T", 22)).arrange(RIGHT, buff=0.15),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([4.7, 1.2, 0])
            self.play(FadeIn(leg), run_time=self.fit(0.8))
            for b, t in zip(bars, texts):
                self.play(GrowFromEdge(b, DOWN), FadeIn(t), run_time=self.fit(1.2, reserve=3))
                self.wait(min(1.5, self.remaining() * 0.12))
            note = zh("主线模型，前向运算量", 20, theme.MUTED).move_to([4.7, 0.1, 0])
            self.play(FadeIn(note), run_time=self.fit(0.5))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(bars, texts, leg, note)), run_time=self.fit(0.6))

    # ── S04 decode ───────────────────────────────────────────────────────
    def s04(self) -> None:
        with self.shot("S04"):
            self.play(*self.set_heading("decode：每步都要把权重和 KV cache 读一遍"),
                      run_time=self.fit(0.8))
            s = 0.14  # 每 GiB 的宽度
            x0 = -4.3
            groups = VGroup()
            for i, (T, label) in enumerate(((4096, "4K"), (32768, "32K"))):
                d1, d16 = D["decode"][f"{T}_1"], D["decode"][f"{T}_16"]
                y = 1.3 - i * 1.9
                w = hbar(d16["weights"] / GiB * s, theme.PARAM).move_to([x0, y, 0], aligned_edge=LEFT)
                kv = hbar(d16["kv"] / GiB * s, theme.ATTN).next_to(w, RIGHT, buff=0)
                name = zh(f"上下文 {label}\nbatch 16", 22).next_to(w, LEFT, 0.25)
                val = zh(f"权重 {d16['weights'] / GiB:.2f} + KV {d16['kv'] / GiB:.2f} GiB", 20,
                         theme.MUTED).next_to(VGroup(w, kv), DOWN, 0.12, aligned_edge=LEFT)
                gain = zh(f"batch 1 → 16：吞吐 {d1['tok_s']:,.0f} → {d16['tok_s']:,.0f} token/s"
                          f"（{d16['tok_s'] / d1['tok_s']:.1f} 倍）", 22,
                          theme.OUTPUT if i == 0 else theme.GRAD)
                gain.next_to(val, DOWN, 0.12, aligned_edge=LEFT)
                groups.add(VGroup(w, kv, name, val, gain))
            leg = VGroup(
                VGroup(hbar(0.35, theme.PARAM, 0.3), zh("权重：一批请求共享", 20)).arrange(RIGHT, buff=0.15),
                VGroup(hbar(0.35, theme.ATTN, 0.3), zh("KV cache：每条各读各的", 20)).arrange(RIGHT, buff=0.15),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([4.6, 2.3, 0])
            inten = D["decode"]["32768_1"]["intensity"]
            ridge = zh(f"算术强度 ≈ {inten:.1f} 次/字节  ≪  H100 脊点 ≈ {D['ridge']:.0f}", 26,
                       theme.HIGHLIGHT).move_to([0, -2.35, 0])
            self.play(FadeIn(leg), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.12)
            self.play(FadeIn(ridge), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.25)
            for g in groups:
                self.play(GrowFromEdge(g[0], LEFT), GrowFromEdge(g[1], LEFT), FadeIn(g[2]),
                          FadeIn(g[3]), run_time=self.fit(1.2, reserve=2))
                self.play(FadeIn(g[4]), run_time=self.fit(0.6, reserve=1.5))
                self.wait(self.remaining() * 0.3)
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(groups, leg, ridge)), run_time=self.fit(0.6))

    # ── S05 并发 ─────────────────────────────────────────────────────────
    def s05(self) -> None:
        with self.shot("S05"):
            self.play(*self.set_heading("80 GB 能同时服务几条 32K 对话（主线模型）"),
                      run_time=self.fit(0.8))
            u = D["users"]
            unit, base = 4.2 / max(u.values()), -1.9
            items = [("MHA", "不共享 MHA", theme.GRAD), ("GQA", "GQA（主线）", theme.INPUT),
                     ("MLA", "若换 MLA 512+64", theme.ATTN)]
            grp = VGroup()
            for i, (k, name, col) in enumerate(items):
                x = -3.5 + i * 3.5
                b = hbar(1.5, col, height=u[k] * unit).move_to([x, base, 0], aligned_edge=DOWN)
                n = zh(f"{u[k]} 条", 30, col).next_to(b, UP, 0.15)
                lab = zh(name, 24).next_to(b, DOWN, 0.15)
                grp.add(VGroup(b, n, lab))
            for g in grp:
                self.play(GrowFromEdge(g[0], DOWN), FadeIn(g[1]), FadeIn(g[2]),
                          run_time=self.fit(1.2, reserve=2))
                self.wait(self.remaining() * 0.15)
            note = zh("80 GB 扣掉权重全给 KV cache\n不计激活与碎片", 20, theme.MUTED).move_to([-3.5, 1.6, 0])
            self.play(FadeIn(note), run_time=self.fit(0.5))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(grp, note)), run_time=self.fit(0.6))

    # ── S06 按层记账（示意）───────────────────────────────────────────────
    def s06(self) -> None:
        with self.shot("S06"):
            self.play(*self.set_heading("按层记账：四种层，四种涨法（示意）"), run_time=self.fit(0.8))
            t = ValueTracker(0.0)  # 上下文长度，0 → 1（代表 0 → 128K）
            x0, maxw = -2.4, 8.0
            rows = [("全注意力", theme.INPUT, lambda v: v * maxw),
                    ("MLA", theme.ATTN, lambda v: v * maxw * 576 / 2048),
                    ("滑动窗口", theme.OUTPUT, lambda v: min(v, 0.25) * maxw),
                    ("线性注意力", theme.PARAM, lambda v: 0.35)]
            grp = VGroup()
            for i, (name, col, fn) in enumerate(rows):
                y = 1.7 - i * 1.05
                lab = zh(name, 26, col).move_to([x0 - 0.3, y, 0], aligned_edge=RIGHT)
                bar = always_redraw(lambda fn=fn, col=col, y=y: hbar(fn(t.get_value()), col, 0.5)
                                    .move_to([x0, y, 0], aligned_edge=LEFT))
                grp.add(lab, bar)
            axis = Line([x0, -1.7, 0], [x0 + maxw, -1.7, 0], color=theme.MUTED, stroke_width=2)
            tl = always_redraw(lambda: zh(f"上下文 T = {int(t.get_value() * 128)}K", 24,
                                          theme.HIGHLIGHT).move_to([x0 + maxw / 2, -2.2, 0]))
            win = DashedLine([x0 + 0.25 * maxw, 0.0, 0], [x0 + 0.25 * maxw, 2.2, 0],
                             color=theme.MUTED, stroke_width=2)
            wl = zh("窗口", 18, theme.MUTED).next_to(win, UP, 0.05)
            self.play(FadeIn(grp), Create(axis), FadeIn(tl), run_time=self.fit(1))
            self.play(t.animate.set_value(1.0), run_time=self.fit(self.remaining() * 0.55, reserve=1.5))
            self.play(Create(win), FadeIn(wl), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            grp.clear_updaters()
            tl.clear_updaters()
            self.play(FadeOut(VGroup(grp, axis, tl, win, wl)), run_time=self.fit(0.6))

    # ── S07 公开模型 ─────────────────────────────────────────────────────
    def s07(self) -> None:
        with self.shot("S07"):
            self.play(*self.set_heading("128K 上下文的 KV cache（BF16，来自各模型 config.json）"),
                      run_time=self.fit(0.8))
            colors = {"Qwen3-8B": theme.INPUT, "Llama-3.1-8B": theme.INPUT,
                      "gpt-oss-120b": theme.OUTPUT, "Qwen3.5-9B": theme.OUTPUT,
                      "Qwen3.5-397B-A17B": theme.OUTPUT, "DeepSeek-V3/V3.2": theme.ATTN,
                      "Kimi-K3": theme.ATTN}
            x0 = -1.6
            scale = 6.0 / max(r["gib"] for r in D["public"])
            rows = VGroup()
            for i, r in enumerate(D["public"]):
                y = 2.35 - i * 0.62
                b = hbar(r["gib"] * scale, colors[r["name"]], 0.4).move_to([x0, y, 0], aligned_edge=LEFT)
                n = zh(r["name"], 22).move_to([x0 - 0.25, y, 0], aligned_edge=RIGHT)
                v = zh(f"{r['gib']:.2f} GiB", 22, colors[r["name"]]).next_to(b, RIGHT, 0.15)
                rows.add(VGroup(b, n, v))
            self.play(LaggedStart(*[AnimFade(g) for g in rows], lag_ratio=0.25),
                      run_time=self.fit(3, reserve=4))
            ds_row = rows[5]
            mha = zh(f"若同头数 MHA：{D['ds_mha']:.0f} GiB", 20, theme.GRAD).next_to(ds_row[2], RIGHT, 0.3)
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(mha), run_time=self.fit(0.6))
            leg = VGroup(*[VGroup(hbar(0.3, c, 0.26), zh(t, 20)).arrange(RIGHT, buff=0.12) for c, t in
                           ((theme.INPUT, "少存头（GQA）"), (theme.OUTPUT, "少存层（滑动窗口 / 混合线性）"),
                            (theme.ATTN, "每位置存更小（MLA）"))]).arrange(RIGHT, buff=0.5)
            leg.move_to([0, -2.3, 0])
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(leg), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(rows, mha, leg)), run_time=self.fit(0.6))

    # ── S08 MHA → GQA → MQA ──────────────────────────────────────────────
    def s08(self) -> None:
        with self.shot("S08"):
            self.play(*self.set_heading("第一条路：少存头"), run_time=self.fit(0.8))
            cols = VGroup()
            for ci, (name, n_kv, ratio) in enumerate((("MHA", 8, "缓存 1×"), ("GQA", 2, "缓存 1/4"),
                                                     ("MQA", 1, "缓存 1/8"))):
                cx = -4.6 + ci * 4.6
                qs = VGroup(*[Rectangle(width=0.36, height=0.36, fill_color=theme.INPUT,
                                        fill_opacity=0.85, stroke_width=0) for _ in range(8)])
                qs.arrange(RIGHT, buff=0.1).move_to([cx, 1.5, 0])
                kvs = VGroup(*[Rectangle(width=0.36, height=0.36, fill_color=theme.PARAM,
                                         fill_opacity=0.9, stroke_width=0) for _ in range(n_kv)])
                kvs.arrange(RIGHT, buff=0.1 if n_kv == 8 else 0.9).move_to([cx, -0.4, 0])
                lines = VGroup(*[Line(q.get_bottom(), kvs[qi * n_kv // 8].get_top(),
                                      color=theme.MUTED, stroke_width=1.5) for qi, q in enumerate(qs)])
                title = zh(name, 30).next_to(qs, UP, 0.3)
                lab = zh(ratio, 24, theme.PARAM).next_to(kvs, DOWN, 0.3)
                cols.add(VGroup(title, qs, lines, kvs, lab))
            key = VGroup(
                VGroup(hbar(0.3, theme.INPUT, 0.3), zh("查询头", 20)).arrange(RIGHT, buff=0.12),
                VGroup(hbar(0.3, theme.PARAM, 0.3), zh("KV 头（要缓存）", 20)).arrange(RIGHT, buff=0.12),
            ).arrange(RIGHT, buff=0.6).move_to([0, -2.1, 0])
            self.play(FadeIn(key), run_time=self.fit(0.5))
            for g in (cols[0], cols[2], cols[1]):
                self.play(FadeIn(g[0]), FadeIn(g[1]), run_time=self.fit(0.6, reserve=3))
                self.play(Create(g[2]), FadeIn(g[3]), FadeIn(g[4]), run_time=self.fit(1.2, reserve=2))
                self.wait(self.remaining() * 0.2)
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(cols, key)), run_time=self.fit(0.6))

    # ── S09 MLA 压缩 ─────────────────────────────────────────────────────
    def s09(self) -> None:
        with self.shot("S09"):
            self.play(*self.set_heading("第二条路：MLA 把 K、V 压成一个潜向量"), run_time=self.fit(0.8))
            x = hbar(0.45, theme.INPUT, height=4.2).move_to([-5.6, 0.2, 0])
            xl = MathTex("x", font_size=40).next_to(x, UP, 0.12)
            xd = zh("7168 维", 20, theme.MUTED).next_to(x, DOWN, 0.12)
            c = hbar(0.45, theme.HIGHLIGHT, height=0.9).move_to([-2.6, 0.9, 0])
            cl = MathTex(r"c^{KV}", font_size=38, color=theme.HIGHLIGHT).next_to(c, UP, 0.12)
            cd = zh("512 维", 20, theme.MUTED).next_to(c, DOWN, 0.1)
            kr = hbar(0.45, theme.OUTPUT, height=0.3).move_to([-2.6, -1.1, 0])
            krl = MathTex(r"k^{R}", font_size=34, color=theme.OUTPUT).next_to(kr, UP, 0.1)
            krd = zh("64 维，带 RoPE", 18, theme.MUTED).next_to(kr, DOWN, 0.1)
            a1 = Arrow(x.get_right() + UP * 0.7, c.get_left(), buff=0.1, color=theme.MUTED)
            a1l = MathTex(r"W^{DKV}", font_size=30, color=theme.PARAM).next_to(a1, DOWN, 0.08).shift(LEFT * 0.35)
            a2 = Arrow(x.get_right() + DOWN * 1.3, kr.get_left(), buff=0.1, color=theme.MUTED)
            cache_box = SurroundingRectangle(VGroup(c, cd, kr, krd, cl, krl), color=theme.HIGHLIGHT,
                                             buff=0.18)
            cache_l = zh("只缓存这两样", 22, theme.HIGHLIGHT).next_to(cache_box, DOWN, 0.12)
            self.play(FadeIn(x), FadeIn(xl), FadeIn(xd), run_time=self.fit(0.8))
            self.play(GrowFromEdge(c, LEFT), Create(a1), FadeIn(a1l), FadeIn(cl), FadeIn(cd),
                      run_time=self.fit(1.2))
            self.play(Create(a2), FadeIn(kr), FadeIn(krl), FadeIn(krd), run_time=self.fit(0.8))
            self.play(Create(cache_box), FadeIn(cache_l), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.2)
            heads = VGroup()
            for i in range(4):
                y = 2.1 - i * 1.25
                k = hbar(1.1, theme.ATTN, 0.32).move_to([2.3, y + 0.2, 0])
                v = hbar(1.1, theme.OUTPUT, 0.32).move_to([2.3, y - 0.22, 0])
                kl = MathTex(rf"k_{i + 1}", font_size=28).next_to(k, RIGHT, 0.1)
                vl = MathTex(rf"v_{i + 1}", font_size=28).next_to(v, RIGHT, 0.1)
                arr = Arrow(c.get_right(), VGroup(k, v).get_left(), buff=0.12, color=theme.MUTED,
                            stroke_width=2, max_tip_length_to_length_ratio=0.08)
                heads.add(VGroup(arr, k, v, kl, vl))
            up = MathTex(r"W^{UK}_i,\ W^{UV}_i", font_size=30, color=theme.PARAM).move_to([4.9, 2.6, 0])
            up_l = zh("每个头自己的上投影", 20, theme.MUTED).next_to(up, DOWN, 0.1)
            self.play(LaggedStart(*[FadeIn(h) for h in heads], lag_ratio=0.3), FadeIn(up), FadeIn(up_l),
                      run_time=self.fit(2))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(x, xl, xd, c, cl, cd, kr, krl, krd, a1, a1l, a2, cache_box,
                                     cache_l, heads, up, up_l)), run_time=self.fit(0.6))

    # ── S10 吸收与解耦 RoPE ──────────────────────────────────────────────
    def s10(self) -> None:
        with self.shot("S10"):
            self.play(*self.set_heading("吸收：推理时不用还原 K、V"), run_time=self.fit(0.8))
            e1 = MathTex(r"q_i^{\top} k_j", r"=", r"(q_i^{C})^{\top}\,", r"W^{UK}_i", r"\,c^{KV}_j",
                         font_size=46).move_to([0, 2.0, 0])
            e2 = MathTex(r"=", r"\big(", r"W^{UK\top}_i", r"q_i^{C}", r"\big)^{\top}", r"c^{KV}_j",
                         font_size=46).next_to(e1, DOWN, 0.45, aligned_edge=LEFT).shift(RIGHT * 1.0)
            for e, idx in ((e1, 3), (e2, 2)):
                e[idx].set_color(theme.PARAM)
            e1[4].set_color(theme.HIGHLIGHT)
            e2[5].set_color(theme.HIGHLIGHT)
            n2 = zh("先把 W_UK 并进 query，直接和缓存的潜向量点积", 22, theme.MUTED).next_to(e2, DOWN, 0.25)
            self.play(Write(e1), run_time=self.fit(1.5))
            self.wait(self.remaining() * 0.08)
            self.play(Write(e2), FadeIn(n2), run_time=self.fit(1.5))
            self.wait(self.remaining() * 0.2)
            rope = VGroup(
                MathTex(r"+\ (q_i^{R})^{\top} k^{R}_j", font_size=42, color=theme.OUTPUT),
                zh("位置交给另外 64 维，所有头共享一份", 22, theme.OUTPUT),
            ).arrange(RIGHT, buff=0.35).move_to([0, -0.9, 0])
            self.play(FadeIn(rope), run_time=self.fit(1))
            self.wait(self.remaining() * 0.3)
            cmp = zh("每层每位置：512 + 64 = 576 个数　vs　同头数 MHA 40,960 个（1.4%）", 24,
                     theme.HIGHLIGHT).move_to([0, -1.8, 0])
            self.play(FadeIn(cmp), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.35)
            last = zh("训练时像 MHA，推理时像 MQA", 28).move_to([0, -2.4, 0])
            self.play(FadeIn(last), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(e1, e2, n2, rope, cmp, last)), run_time=self.fit(0.6))

    # ── S11 小实验 ───────────────────────────────────────────────────────
    def s11(self) -> None:
        with self.shot("S11"):
            self.play(*self.set_heading("小实验：只换注意力，600 步 × 3 个种子"), run_time=self.fit(0.8))
            badge = self.demo_badge("极小实验")
            self.play(FadeIn(badge), run_time=0.3)
            exp = D["exp"]
            cols = [theme.INPUT, theme.INPUT, theme.INPUT, theme.ATTN, theme.ATTN]
            # 左：每 token 缓存（字节）
            x0, maxw = -4.6, 2.6
            mx = max(r["per_token"] for r in exp)
            left = VGroup()
            for i, r in enumerate(exp):
                y = 1.9 - i * 0.85
                b = hbar(r["per_token"] / mx * maxw, cols[i], 0.42).move_to([x0, y, 0], aligned_edge=LEFT)
                n = zh(r["name"], 22).move_to([x0 - 0.2, y, 0], aligned_edge=RIGHT)
                v = zh(f"{r['per_token']} B", 20, cols[i]).next_to(b, RIGHT, 0.12)
                left.add(VGroup(b, n, v))
            lt = zh("每 token 缓存（FP32，4 层）", 20, theme.MUTED).move_to([-3.6, 2.65, 0])
            # 右：验证 loss 点图
            allv = [x for r in exp for x in r["losses"]]
            lo, hi = min(allv), max(allv)
            pad = (hi - lo) * 0.15 + 1e-3
            lo, hi = lo - pad, hi + pad
            ax0, ax1 = 1.6, 6.4

            def xpos(v: float) -> float:
                return ax0 + (v - lo) / (hi - lo) * (ax1 - ax0)

            axis = Line([ax0, -2.2, 0], [ax1, -2.2, 0], color=theme.MUTED, stroke_width=2)
            ticks = VGroup()
            for v in (lo + pad, (lo + hi) / 2, hi - pad):
                ticks.add(zh(f"{v:.3f}", 18, theme.MUTED).move_to([xpos(v), -2.45, 0]))
            right = VGroup()
            for i, r in enumerate(exp):
                y = 1.9 - i * 0.85
                dots = VGroup(*[Dot([xpos(v), y, 0], radius=0.07, color=cols[i]) for v in r["losses"]])
                mean = Line([xpos(r["mean"]), y - 0.25, 0], [xpos(r["mean"]), y + 0.25, 0],
                            color=theme.HIGHLIGHT, stroke_width=4)
                right.add(VGroup(dots, mean))
            rt = zh("验证 loss（点 = 种子，黄线 = 均值）", 20, theme.MUTED).move_to([4.0, 2.65, 0])
            self.play(FadeIn(lt), LaggedStart(*[FadeIn(g) for g in left], lag_ratio=0.2),
                      run_time=self.fit(2, reserve=4))
            self.wait(self.remaining() * 0.25)
            self.play(FadeIn(rt), Create(axis), FadeIn(ticks),
                      LaggedStart(*[FadeIn(g) for g in right], lag_ratio=0.2), run_time=self.fit(2))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(left, lt, right, rt, axis, ticks, badge)), run_time=self.fit(0.6))

    # ── S12 采用与代价 ───────────────────────────────────────────────────
    def s12(self) -> None:
        with self.shot("S12"):
            self.play(*self.set_heading("谁在用 MLA，代价是什么"), run_time=self.fit(0.8))
            fam = VGroup(*[zh(t, 24) for t in (
                "DeepSeek：V2 / V3 / V3.2", "Kimi：K2（K3 的全注意力层）", "GLM：GLM-5",
                "Mistral：Mistral Large 3")]).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
            fam.move_to([-3.6, 0.7, 0])
            ft = zh("采用 MLA（均为大 MoE 旗舰）", 24, theme.ATTN).next_to(fam, UP, 0.35, aligned_edge=LEFT)
            cost = VGroup(*[zh(t, 22) for t in (
                "decode 时每头在 576 维上点积，算力更高", "与 QK-Norm 不兼容（Kimi K2 报告）",
                "GLM-5：起初不如 GQA-8，改优化器后追平", "需要专门 kernel（如 FlashMLA）")])
            cost.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([3.3, 0.7, 0])
            ct = zh("代价", 24, theme.GRAD).next_to(cost, UP, 0.35, aligned_edge=LEFT)
            concl = zh("稠密小模型几乎都用 GQA → 主线模型继续用 GQA", 26, theme.HIGHLIGHT).move_to([0, -2.2, 0])
            self.play(FadeIn(ft), LaggedStart(*[FadeIn(t) for t in fam], lag_ratio=0.3),
                      run_time=self.fit(2.5, reserve=5))
            self.wait(self.remaining() * 0.25)
            self.play(FadeIn(ct), LaggedStart(*[FadeIn(t) for t in cost], lag_ratio=0.4),
                      run_time=self.fit(3, reserve=2))
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(concl), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(fam, ft, cost, ct, concl)), run_time=self.fit(0.6))

    # ── S13 从极简到生产级 ───────────────────────────────────────────────
    def s13(self) -> None:
        with self.shot("S13"):
            self.play(*self.set_heading("从极简到生产级"), run_time=self.fit(0.8))
            pairs = [("code/01_kv_ledger.py", "zero/tools/kv_cache_calc.py"),
                     ("code/03_mla.py", "zero/arch/mla.py")]
            rows = VGroup()
            for i, (a, b) in enumerate(pairs):
                y = 1.6 - i * 1.2
                ta = mono(a, 24, theme.INPUT).move_to([-3.6, y, 0])
                tb = mono(b, 24, theme.OUTPUT).move_to([3.4, y, 0])
                ar = Arrow(ta.get_right(), tb.get_left(), buff=0.25, color=theme.MUTED)
                rows.add(VGroup(ta, ar, tb))
            tests = zh("对拍：与 KVCache.nbytes() 逐字节相等；MLA 带缓存 = 不带缓存", 22,
                       theme.MUTED).move_to([0, -0.5, 0])
            ind = zh("上线：vLLM 分页 KV cache · FlashMLA kernel", 24).move_to([0, -1.3, 0])
            nxt = zh("下一章：局部与稀疏注意力", 30, theme.HIGHLIGHT).move_to([0, -2.2, 0])
            for r in rows:
                self.play(FadeIn(r[0]), Create(r[1]), FadeIn(r[2]), run_time=self.fit(1, reserve=4))
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(tests), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(ind), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.5)
            self.play(FadeIn(nxt), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(rows, tests, ind, nxt)), *self.set_heading(None),
                      run_time=self.fit(0.6))


def AnimFade(m):  # noqa: N802 - 与 Manim 动画类同风格
    return FadeIn(m, shift=RIGHT * 0.2)
