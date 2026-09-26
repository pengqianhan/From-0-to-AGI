"""第 23 章视频：线性注意力与混合架构 —— 把 KV cache 压成一个固定大小的矩阵

画面里的数字都由 ../code/ 中的代码真实计算（见 script.md 事实清单），结果缓存在 video/out/cache.json。
渲染：bash chapters/23-linear-attention-hybrid/video/build.sh
"""

from __future__ import annotations

import importlib.util
import json
import math
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
    MathTex,
    Rectangle,
    RoundedRectangle,
    Square,
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
    """从 ../code 真实计算视频要用的全部数字。"""
    import torch

    m1 = _load("ch23_linear", "01_linear_attention.py")
    m2 = _load("ch23_chunked", "02_chunked.py")
    m3 = _load("ch23_delta", "03_delta_rule.py")
    m4 = _load("ch23_hybrid_lm", "04_hybrid_lm.py")
    m5 = _load("ch23_recall", "05_associative_recall.py")
    F = torch.nn.functional
    d: dict = {}

    # 01：结合律、KV vs 状态、解码一步
    torch.manual_seed(0)
    q, k, v = torch.randn(256, 16), torch.randn(256, 16), torch.randn(256, 16)
    par, rec = m1.linear_attention_parallel(q, k, v), m1.linear_attention_recurrent(q, k, v)
    d["assoc_rel"] = float((par - rec).abs().max() / par.abs().max())
    d["kv_vs_state"] = m1.kv_cache_vs_state()
    d["decode"] = [(T, *m1.decode_step_time(T)) for T in (1024, 8192, 65536, 262144)]

    # 02：分块一致、衰减曲线、速度
    torch.manual_seed(0)
    T, dd = 512, 64
    q = F.normalize(torch.randn(T, dd), dim=-1)
    k = F.normalize(torch.randn(T, dd), dim=-1)
    v, g = torch.randn(T, dd), -torch.rand(T) * 0.1
    r = m2.recurrent(q, k, v, g)
    d["chunk_err"] = float((r - m2.chunked(q, k, v, g)).abs().max())
    qq = torch.zeros(T, dd)
    qq[:, 0] = 1.0
    kk, vv = torch.zeros(T, dd), torch.zeros(T, 1)
    kk[0, 0], vv[0, 0] = 1.0, 1.0
    d["decay"] = {}
    for a in (1.0, 0.99, 0.9):
        o = m2.recurrent(qq, kk, vv, torch.full((T,), math.log(a)))
        d["decay"][str(a)] = [float(x) for x in o[:, 0]]
    speed = []
    for T in (1024, 4096):
        q = F.normalize(torch.randn(T, dd), dim=-1)
        k = F.normalize(torch.randn(T, dd), dim=-1)
        v, g = torch.randn(T, dd), -torch.rand(T) * 0.1
        speed.append((T, m2.timed(m2.recurrent, q, k, v, g, reps=1)[1],
                      m2.timed(m2.chunked, q, k, v, g)[1], m2.timed(m2.parallel, q, k, v, g)[1]))
    d["speed"] = speed

    # 03：覆盖、容量、分块一致
    torch.manual_seed(0)
    kk = F.normalize(torch.randn(64), dim=0)
    v1, v2 = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    d["overwrite"] = {
        rule: [round(x, 3) for x in (m3.write_stream(torch.stack([kk, kk]), torch.stack([v1, v2]),
                                                     rule) @ kk).tolist()]
        for rule in ("linear", "delta")
    }
    cap = []
    for N in (16, 32, 64, 128, 256):
        keys = F.normalize(torch.randn(N, 64), dim=1)
        vals = torch.randn(N, 64)
        cap.append((N, m3.read_error(m3.write_stream(keys, vals, "linear"), keys, vals),
                    m3.read_error(m3.write_stream(keys, vals, "delta"), keys, vals),
                    m3.attention_read_error(keys, vals)))
    d["capacity"] = cap
    torch.manual_seed(1)
    q = F.normalize(torch.randn(2, 3, 128, 16), dim=-1)
    k = F.normalize(torch.randn(2, 3, 128, 16), dim=-1)
    v, g, beta = torch.randn(2, 3, 128, 16), -torch.rand(2, 3, 128) * 0.2, torch.rand(2, 3, 128)
    o1, _ = m3.gated_delta_recurrent(q, k, v, g, beta)
    o2, _ = m3.gated_delta_chunked(q, k, v, g, beta, C=32)
    d["gdn_chunk_err"] = float((o1 - o2).abs().max())

    # 04：Qwen3.5-0.8B 的层类型与缓存；小语言模型的验证 loss
    c = m4.QWEN35_08B
    d["qwen_layers"] = c["layer_types"] * c["repeat"]
    d["qwen_cache"] = [(T, *m4.qwen35_cache_mib(T), sum(m4.qwen35_cache_mib(T, all_full=True)))
                       for T in (4096, 32768, 262144)]
    d["lm"] = [(p, m4.val_loss(m4.train_lm(p)[0])) for p in m4.PATTERNS]

    # 05：联想回忆
    d["recall"] = m5.results()
    return d


def get_data() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    d = compute()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


D = get_data()
NAMES = {"AAAA": "纯注意力", "LLLL": "纯线性", "GGGG": "纯 GDN", "GGGA": "3:1 混合",
         "AA": "纯注意力", "LL": "纯线性", "GG": "纯 GDN", "GA": "混合"}
PCOLORS = {"AAAA": theme.INPUT, "LLLL": theme.MUTED, "GGGG": theme.ATTN, "GGGA": theme.OUTPUT,
           "AA": theme.INPUT, "LL": theme.MUTED, "GG": theme.ATTN, "GA": theme.OUTPUT}


def mono(text: str, size: float = 24, color: str = theme.FG) -> Text:
    return Text(text, font=MONO, font_size=size, color=color)


def grid(n: int, cell: float, color: str, opacity: float = 0.25) -> VGroup:
    """n×n 的小方格矩阵（状态 S 的示意）。"""
    g = VGroup(*[Square(cell, stroke_width=1, stroke_color=color, fill_color=color,
                        fill_opacity=opacity) for _ in range(n * n)])
    return g.arrange_in_grid(n, n, buff=0)


def cells(n: int, w: float, h: float, color: str, opacity: float = 0.35) -> VGroup:
    g = VGroup(*[Rectangle(width=w, height=h, stroke_width=1, stroke_color=color,
                           fill_color=color, fill_opacity=opacity) for _ in range(n)])
    return g.arrange(RIGHT, buff=0.04)


def table(rows: list[tuple], col_w: list[float], size: float = 22,
          colors: list[str] | None = None, row_h: float = 0.46) -> VGroup:
    """简单表格：第一行是表头（灰色）。每格一个中文或等宽文本。"""
    out = VGroup()
    for r, row in enumerate(rows):
        x = 0.0
        line = VGroup()
        for c, (cell, w) in enumerate(zip(row, col_w)):
            color = theme.MUTED if r == 0 else (colors[c] if colors else theme.FG)
            t = zh(str(cell), size, color)
            t.move_to([x + w / 2, -r * row_h, 0])
            line.add(t)
            x += w
        out.add(line)
    return out


class ChapterScene(NarratedScene):
    chapter_label = "第 23 章"
    chapter_title = "线性注意力与混合架构"

    def construct(self) -> None:
        for i in range(1, 15):
            getattr(self, f"s{i:02d}")()

    # ── S01 片头 ─────────────────────────────────────────────────────────
    def s01(self):
        with self.shot("S01"):
            card = self.chapter_card()
            sub = zh("把 KV cache 压成一个固定大小的矩阵", 32, theme.HIGHLIGHT).next_to(card, DOWN, 0.6)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.wait(max(0.1, self.remaining() - 0.8))
            self.play(FadeOut(card), FadeOut(sub), run_time=self.fit(0.8))

    # ── S02 KV cache 越长越大 ────────────────────────────────────────────
    def s02(self):
        with self.shot("S02"):
            self.play(*self.set_heading("问题：KV cache 随上下文增长"), run_time=self.fit(0.8))
            left_x = -6.3
            k_lbl = MathTex("K", color=theme.PARAM, font_size=36).move_to([left_x, 1.4, 0])
            v_lbl = MathTex("V", color=theme.OUTPUT, font_size=36).move_to([left_x, 0.6, 0])
            t_lbl = zh("KV cache（softmax 注意力）", 26, theme.FG).move_to([-3.4, 2.4, 0])
            self.play(FadeIn(k_lbl), FadeIn(v_lbl), FadeIn(t_lbl), run_time=self.fit(0.8))
            krow, vrow = VGroup(), VGroup()
            n_max = 16
            per = max(0.12, (self.remaining() * 0.45) / n_max)
            for i in range(n_max):
                x = left_x + 0.55 + i * 0.36
                kc = Rectangle(width=0.3, height=0.55, stroke_width=1, stroke_color=theme.PARAM,
                               fill_color=theme.PARAM, fill_opacity=0.5).move_to([x, 1.4, 0])
                vc = Rectangle(width=0.3, height=0.55, stroke_width=1, stroke_color=theme.OUTPUT,
                               fill_color=theme.OUTPUT, fill_opacity=0.5).move_to([x, 0.6, 0])
                krow.add(kc)
                vrow.add(vc)
                self.play(FadeIn(kc), FadeIn(vc), run_time=self.fit(per))
            rows = D["kv_vs_state"]
            kv_txt = VGroup(*[
                zh(f"T = {T:,}：{kv / 1024:,.0f} KB", 24, theme.FG) for T, kv, _ in
                (rows[1], rows[3], rows[4])
            ]).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([-3.6, -1.3, 0])
            self.play(FadeIn(kv_txt), run_time=self.fit(0.8))
            S = grid(8, 0.26, theme.ATTN).move_to([3.8, 0.9, 0])
            s_lbl = zh("线性注意力：状态 S", 26, theme.FG).move_to([3.8, 2.4, 0])
            s_txt = zh(f"任何 T：{rows[0][2] / 1024:.0f} KB", 24, theme.ATTN).move_to([3.8, -1.3, 0])
            note = zh("单层单头，d = 64，BF16", 20, theme.MUTED).move_to([0.2, -2.35, 0])
            self.play(FadeIn(S), FadeIn(s_lbl), run_time=self.fit(1.0))
            self.play(FadeIn(s_txt), FadeIn(note), run_time=self.fit(0.6))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(k_lbl, v_lbl, t_lbl, krow, vrow, kv_txt, S, s_lbl, s_txt,
                                     note)), run_time=self.fit(0.6))

    # ── S03 结合律 ───────────────────────────────────────────────────────
    def s03(self):
        with self.shot("S03"):
            self.play(*self.set_heading("去掉 softmax：矩阵乘法换个顺序"), run_time=self.fit(0.8))
            top = MathTex(r"(Q K^\top)\,V", font_size=46).move_to([-4.2, 1.3, 0])
            big = Square(2.2, stroke_color=theme.GRAD, fill_color=theme.GRAD, fill_opacity=0.25)
            big.move_to([0.3, 1.0, 0])
            big_lbl = MathTex(r"T\times T", font_size=36, color=theme.GRAD).move_to(big)
            bot = MathTex(r"Q\,(K^\top V)", font_size=46).move_to([-4.2, -1.3, 0])
            small = Square(0.6, stroke_color=theme.ATTN, fill_color=theme.ATTN, fill_opacity=0.4)
            small.move_to([0.3, -1.5, 0])
            small_lbl = MathTex(r"d\times d", font_size=36, color=theme.ATTN).next_to(small, RIGHT)
            self.play(Write(top), run_time=self.fit(1.0))
            self.play(FadeIn(big), FadeIn(big_lbl), run_time=self.fit(1.0))
            self.wait(self.remaining() * 0.2)
            self.play(Write(bot), run_time=self.fit(1.0))
            self.play(FadeIn(small), FadeIn(small_lbl), run_time=self.fit(1.0))
            eq = MathTex(r"=", font_size=60).move_to([-4.2, 0, 0])
            self.play(FadeIn(eq), run_time=self.fit(0.5))
            self.wait(self.remaining() * 0.35)
            sm = MathTex(r"\mathrm{softmax}(QK^\top)\,V", font_size=38, color=theme.MUTED)
            sm.move_to([4.4, 1.3, 0])
            cross = zh("拆不开", 28, theme.GRAD).next_to(sm, DOWN, 0.25)
            self.play(FadeIn(sm), FadeIn(cross), run_time=self.fit(0.8))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(top, big, big_lbl, bot, small, small_lbl, eq, sm, cross)),
                      run_time=self.fit(0.6))

    # ── S04 递推形式 ─────────────────────────────────────────────────────
    def s04(self):
        with self.shot("S04"):
            self.play(*self.set_heading("递推形式：注意力变成了 RNN"), run_time=self.fit(0.8))
            S = grid(6, 0.34, theme.ATTN, 0.2).move_to([-3.4, 0.4, 0])
            s_lbl = MathTex("S", font_size=44, color=theme.ATTN).next_to(S, UP, 0.2)
            f1 = MathTex(r"S_t = S_{t-1} + v_t k_t^\top", font_size=42).move_to([3.3, 1.2, 0])
            f2 = MathTex(r"o_t = S_t\, q_t", font_size=42).move_to([3.3, 0.1, 0])
            w1 = zh("写入", 24, theme.PARAM).next_to(f1, LEFT, 0.3)
            w2 = zh("读出", 24, theme.OUTPUT).next_to(f2, LEFT, 0.3)
            self.play(FadeIn(S), FadeIn(s_lbl), run_time=self.fit(0.8))
            self.play(Write(f1), FadeIn(w1), run_time=self.fit(1.0))
            for step in range(3):
                outer = grid(6, 0.34, theme.PARAM, 0.55).move_to([-3.4, -2.0 + 0.0 * step, 0])
                outer.scale(0.5).move_to([-5.6, 0.4, 0])
                lbl = MathTex(r"v_t k_t^\top", font_size=30, color=theme.PARAM).next_to(outer, UP)
                self.play(FadeIn(outer), FadeIn(lbl), run_time=self.fit(0.5))
                self.play(outer.animate.scale(2).move_to(S).set_opacity(0), FadeOut(lbl),
                          S.animate.set_fill(opacity=0.2 + 0.15 * (step + 1)),
                          run_time=self.fit(0.7))
                self.remove(outer)
            self.play(Write(f2), FadeIn(w2), run_time=self.fit(1.0))
            ok = zh(f"递推 vs 一次算 T×T：最大相对误差 {D['assoc_rel']:.1e}", 24,
                    theme.HIGHLIGHT).move_to([2.2, -1.6, 0])
            self.play(FadeIn(ok), run_time=self.fit(0.8))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(S, s_lbl, f1, f2, w1, w2, ok)), run_time=self.fit(0.6))

    # ── S05 解码一步 ─────────────────────────────────────────────────────
    def s05(self):
        with self.shot("S05"):
            self.play(*self.set_heading("解码一步的开销（单头，CPU）"), run_time=self.fit(0.8))
            ax = Axes(x_range=[0, 4, 1], y_range=[-2, 2, 1], x_length=7.5, y_length=4.0,
                      axis_config={"color": theme.MUTED, "include_ticks": False}).move_to([-1.2, 0.0, 0])
            ylab = VGroup(*[mono(t, 18, theme.MUTED).next_to(ax.c2p(0, y), LEFT, 0.1)
                            for y, t in ((-2, "0.01 ms"), (0, "1 ms"), (2, "100 ms"))])
            self.play(Create(ax), FadeIn(ylab), run_time=self.fit(1.0))
            bars = VGroup()
            for i, (T, ts, tl) in enumerate(D["decode"]):
                x = i + 0.5
                for val, col, dx in ((ts, theme.INPUT, -0.17), (tl, theme.ATTN, 0.17)):
                    y = max(-2.0, min(2.0, math.log10(val)))
                    base, top = ax.c2p(x + dx, -2), ax.c2p(x + dx, y)
                    r = Rectangle(width=0.3, height=max(0.02, top[1] - base[1]), stroke_width=0,
                                  fill_color=col, fill_opacity=0.9)
                    r.move_to([base[0], (base[1] + top[1]) / 2, 0])
                    bars.add(r)
                lab = mono(f"{T // 1024}K", 20, theme.FG).next_to(ax.c2p(x, -2), DOWN, 0.12)
                bars.add(lab)
            leg = VGroup(zh("softmax 注意力", 24, theme.INPUT), zh("线性注意力", 24, theme.ATTN))
            leg.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([4.9, 1.2, 0])
            self.play(LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.08), FadeIn(leg),
                      run_time=self.fit(2.5))
            T, ts, tl = D["decode"][-1]
            note = zh(f"256K 上下文：{ts:.0f} ms vs {tl:.2f} ms", 24, theme.HIGHLIGHT)
            note.move_to([4.6, -0.4, 0])
            self.play(FadeIn(note), run_time=self.fit(0.6))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(ax, ylab, bars, leg, note)), run_time=self.fit(0.6))

    # ── S06 分块形式 ─────────────────────────────────────────────────────
    def s06(self):
        with self.shot("S06"):
            self.play(*self.set_heading("训练：块内并行 + 块间递推"), run_time=self.fit(0.8))
            chunks = VGroup()
            for c in range(4):
                box = Rectangle(width=2.2, height=1.4, stroke_color=theme.INPUT, stroke_width=2)
                box.move_to([-4.6 + c * 2.9, 1.2, 0])
                tri = MathTex(r"QK^\top\!\odot M", font_size=26, color=theme.INPUT).move_to(box)
                chunks.add(VGroup(box, tri))
            arrows = VGroup(*[Arrow(chunks[c][0].get_right(), chunks[c + 1][0].get_left(), buff=0.05,
                                    color=theme.ATTN, stroke_width=4) for c in range(3)])
            s_lbls = VGroup(*[MathTex("S", font_size=30, color=theme.ATTN).next_to(a, UP, 0.05)
                              for a in arrows])
            self.play(LaggedStart(*[FadeIn(c) for c in chunks], lag_ratio=0.2), run_time=self.fit(1.2))
            self.play(LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.3), FadeIn(s_lbls),
                      run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.25)
            rows = [("T", "逐 token 递推", "分块 C=64", "完全并行")]
            for T, tr, tc, tp in D["speed"]:
                rows.append((f"{T:,}", f"{tr:,.0f} ms", f"{tc:,.0f} ms", f"{tp:,.0f} ms"))
            tb = table(rows, [1.6, 2.6, 2.3, 2.3], size=24).move_to([0, -1.1, 0])
            same = zh(f"三种算法数值一致（最大差 {D['chunk_err']:.0e}）", 22, theme.MUTED)
            same.move_to([0, -2.3, 0])
            self.play(FadeIn(tb), FadeIn(same), run_time=self.fit(1.0))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(chunks, arrows, s_lbls, tb, same)), run_time=self.fit(0.6))

    # ── S07 容量 ─────────────────────────────────────────────────────────
    def s07(self):
        with self.shot("S07"):
            self.play(*self.set_heading("代价：固定大小的记忆会满"), run_time=self.fit(0.8))
            S = grid(8, 0.3, theme.ATTN, 0.3).move_to([-4.6, 0.3, 0])
            s_lbl = zh("64×64 的状态", 24, theme.ATTN).next_to(S, UP, 0.25)
            self.play(FadeIn(S), FadeIn(s_lbl), run_time=self.fit(0.8))
            rows = [("写入 N 个", "线性注意力", "softmax 注意力")]
            for N, el, _, ea in D["capacity"]:
                if N in (16, 64, 256):
                    rows.append((str(N), f"{el:.2f}", f"{ea:.3f}"))
            tb = table(rows, [2.0, 2.6, 2.8], size=26, row_h=0.62,
                       colors=[theme.FG, theme.GRAD, theme.OUTPUT]).move_to([1.8, 0.6, 0])
            cap = zh("读回的相对误差（0 = 完美，1 ≈ 和信号一样大的噪声）", 22, theme.MUTED)
            cap.move_to([1.2, -1.4, 0])
            self.play(FadeIn(tb[0]), FadeIn(cap), run_time=self.fit(0.8))
            for row in tb[1:]:
                self.play(FadeIn(row), run_time=self.fit(0.8))
                self.wait(self.remaining() * 0.15)
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(S, s_lbl, tb, cap)), run_time=self.fit(0.6))

    # ── S08 衰减门 ───────────────────────────────────────────────────────
    def s08(self):
        with self.shot("S08"):
            self.play(*self.set_heading("衰减门：学会遗忘"), run_time=self.fit(0.8))
            f = MathTex(r"S_t = \alpha_t\, S_{t-1} + v_t k_t^\top", font_size=40).move_to([3.4, 2.2, 0])
            ax = Axes(x_range=[0, 500, 100], y_range=[0, 1.1, 0.5], x_length=6.8, y_length=3.6,
                      axis_config={"color": theme.MUTED}).move_to([-2.4, -0.3, 0])
            xl = zh("写入之后过了多少步", 20, theme.MUTED).next_to(ax, DOWN, 0.12)
            self.play(Write(f), Create(ax), FadeIn(xl), run_time=self.fit(1.2))
            labs = VGroup()
            for a, col, y in (("1.0", theme.MUTED, 1.2), ("0.99", theme.ATTN, 0.5), ("0.9", theme.GRAD, -0.2)):
                curve = D["decay"][a]
                line = polyline_in_axes(ax, [(t, curve[t]) for t in range(0, 500, 2)], color=col,
                                        stroke_width=4)
                lab = MathTex(rf"\alpha={a}", font_size=32, color=col).move_to([3.3, y, 0])
                labs.add(line, lab)
                self.play(Create(line), FadeIn(lab), run_time=self.fit(1.0))
            c99 = D["decay"]["0.99"]
            vals = zh(f"α = 0.99：{c99[10]:.3f} → {c99[100]:.3f} → {c99[500 - 1]:.3f}", 22, theme.ATTN)
            vals.move_to([3.4, -1.1, 0])
            fam = zh("RetNet · GLA · Mamba-2 的共同骨架", 22, theme.MUTED).move_to([3.4, -1.8, 0])
            self.play(FadeIn(vals), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(fam), run_time=self.fit(0.6))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(f, ax, xl, labs, vals, fam)), run_time=self.fit(0.6))

    # ── S09 delta 规则 ───────────────────────────────────────────────────
    def s09(self):
        with self.shot("S09"):
            self.play(*self.set_heading("delta 规则：覆盖，而不是累加"), run_time=self.fit(0.8))
            seq = zh("同一个 k：先写 v1 = [1, 0]，再写 v2 = [0, 1]", 26, theme.FG).move_to([0, 2.2, 0])
            self.play(FadeIn(seq), run_time=self.fit(0.8))
            ov = D["overwrite"]
            left = VGroup(zh("线性注意力读出", 26, theme.GRAD),
                          mono(str(ov["linear"]), 30, theme.GRAD)).arrange(DOWN, buff=0.25)
            right = VGroup(zh("delta 规则读出", 26, theme.OUTPUT),
                           mono(str(ov["delta"]), 30, theme.OUTPUT)).arrange(DOWN, buff=0.25)
            left.move_to([-3.3, 0.9, 0])
            right.move_to([3.3, 0.9, 0])
            self.play(FadeIn(left), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(right), run_time=self.fit(0.8))
            f = MathTex(r"S_t = S_{t-1} + \beta_t\,(", r"v_t - S_{t-1}k_t", r")\,k_t^\top",
                        font_size=40).move_to([0, -0.7, 0])
            f[1].set_color(theme.HIGHLIGHT)
            f2 = MathTex(r"= S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top", font_size=40)
            f2.move_to([0.4, -1.7, 0])
            self.play(Write(f), run_time=self.fit(1.2))
            note = zh("新值 − 旧答案", 22, theme.HIGHLIGHT).next_to(f[1], UP, 0.12)
            self.play(FadeIn(note), run_time=self.fit(0.5))
            self.wait(self.remaining() * 0.3)
            self.play(Write(f2), run_time=self.fit(1.0))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(seq, left, right, f, f2, note)), run_time=self.fit(0.6))

    # ── S10 Gated DeltaNet ───────────────────────────────────────────────
    def s10(self):
        with self.shot("S10"):
            self.play(*self.set_heading("Gated DeltaNet = 衰减 × delta 规则"), run_time=self.fit(0.8))
            f = MathTex(r"S_t = ", r"\alpha_t", r"\, S_{t-1}(I - ", r"\beta_t", r" k_t k_t^\top) + ",
                        r"\beta_t", r" v_t k_t^\top", font_size=44).move_to([0, 1.4, 0])
            f[1].set_color(theme.ATTN)
            f[3].set_color(theme.PARAM)
            f[5].set_color(theme.PARAM)
            self.play(Write(f), run_time=self.fit(1.5))
            tags = VGroup(
                VGroup(zh("α：衰减", 28, theme.ATTN), zh("忘掉过时的", 22, theme.MUTED)),
                VGroup(zh("β：delta 写入", 28, theme.PARAM), zh("覆盖旧值", 22, theme.MUTED)),
                VGroup(zh("分块形式", 28, theme.INPUT), zh("可以并行训练", 22, theme.MUTED)),
            )
            for t in tags:
                t.arrange(DOWN, buff=0.15)
            tags.arrange(RIGHT, buff=1.2).move_to([0, -0.2, 0])
            self.play(LaggedStart(*[FadeIn(t) for t in tags], lag_ratio=0.4), run_time=self.fit(2.0))
            ok = zh(f"分块 == 递推：最大差 {D['gdn_chunk_err']:.1e}", 24, theme.HIGHLIGHT)
            ok.move_to([0, -1.7, 0])
            self.play(FadeIn(ok), run_time=self.fit(0.6))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(f, tags, ok)), run_time=self.fit(0.6))

    # ── S11 Qwen3.5 的 3:1 堆叠 ──────────────────────────────────────────
    def s11(self):
        with self.shot("S11"):
            self.play(*self.set_heading("混合：Qwen3.5-0.8B 的 24 层"), run_time=self.fit(0.8))
            blocks = VGroup()
            for t in D["qwen_layers"]:
                col = theme.INPUT if t == "full_attention" else theme.ATTN
                b = RoundedRectangle(width=0.4, height=0.9, corner_radius=0.06, stroke_color=col,
                                     fill_color=col, fill_opacity=0.7 if t == "full_attention" else 0.35)
                blocks.add(b)
            blocks.arrange(RIGHT, buff=0.06).move_to([0, 1.6, 0])
            idx = VGroup(*[mono(str(i), 14, theme.MUTED).next_to(b, DOWN, 0.06)
                           for i, b in enumerate(blocks) if i % 4 == 3])
            leg = VGroup(zh("Gated DeltaNet", 22, theme.ATTN), zh("全注意力（gated attention）", 22,
                                                                  theme.INPUT)).arrange(RIGHT, buff=1.0)
            leg.move_to([0, 2.55, 0])
            self.play(LaggedStart(*[FadeIn(b) for b in blocks], lag_ratio=0.05), FadeIn(leg),
                      run_time=self.fit(2.0))
            self.play(FadeIn(idx), run_time=self.fit(0.5))
            src = zh("来源：Qwen/Qwen3.5-0.8B 的 config.json（layer_types）", 18, theme.MUTED)
            src.move_to([0, 0.75, 0])
            self.play(FadeIn(src), run_time=self.fit(0.5))
            self.wait(self.remaining() * 0.15)
            T, kv, st, full = D["qwen_cache"][-1]
            vmax = full
            rows = [("24 层全注意力（假想）", full, theme.GRAD), ("3:1 混合：KV（6 层）", kv, theme.INPUT),
                    ("3:1 混合：线性状态（18 层）", st, theme.ATTN)]
            bars = VGroup()
            for j, (name, val, col) in enumerate(rows):
                y = -0.1 - j * 0.72
                lab = zh(name, 22, theme.FG).move_to([-4.4, y, 0])
                w = max(0.04, val / vmax * 6.0)
                r = Rectangle(width=w, height=0.42, stroke_width=0, fill_color=col, fill_opacity=0.85)
                r.move_to([-1.9 + w / 2, y, 0])
                num = mono(f"{val:,.1f} MiB" if val < 100 else f"{val:,.0f} MiB", 20, theme.FG)
                num.next_to(r, RIGHT, 0.15)
                bars.add(VGroup(lab, r, num))
            cap = zh(f"一条 {T:,} token 的序列（KV 用 BF16，状态用 FP32）", 20, theme.MUTED)
            cap.move_to([0, -2.35, 0])
            self.play(LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.4), FadeIn(cap),
                      run_time=self.fit(2.0))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(blocks, idx, leg, src, bars, cap)), run_time=self.fit(0.6))

    # ── S12 小实验 ───────────────────────────────────────────────────────
    def s12(self):
        with self.shot("S12"):
            self.play(*self.set_heading("小实验：混合把回忆找回来"), run_time=self.fit(0.8))
            badge = self.demo_badge("极小规模实验")
            self.play(FadeIn(badge), run_time=0.3)
            # 左：验证 loss
            lm = D["lm"]
            lo = min(v for _, v in lm) - 0.1
            hi = max(v for _, v in lm) + 0.05
            bars = VGroup()
            for i, (p, v) in enumerate(lm):
                h = (v - lo) / (hi - lo) * 2.8
                r = Rectangle(width=0.7, height=h, stroke_width=0, fill_color=PCOLORS[p],
                              fill_opacity=0.85).move_to([-5.6 + i * 1.0, -1.9 + h / 2, 0])
                num = mono(f"{v:.3f}", 18, theme.FG).next_to(r, UP, 0.08)
                lab = zh(NAMES[p], 18, theme.FG).next_to(r, DOWN, 0.1)
                bars.add(VGroup(r, num, lab))
            lt = zh("语言模型验证 loss（越低越好）", 22, theme.MUTED).move_to([-4.1, 1.9, 0])
            self.play(FadeIn(lt), LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.2),
                      run_time=self.fit(2.0))
            self.wait(self.remaining() * 0.25)
            # 右：联想回忆准确率
            rc = D["recall"]
            Ns = rc["N"]
            ax = Axes(x_range=[0, max(Ns) + 4, 16], y_range=[0, 1.05, 0.5], x_length=5.2,
                      y_length=3.2, axis_config={"color": theme.MUTED}).move_to([2.9, -0.35, 0])
            ylab = VGroup(*[mono(t, 16, theme.MUTED).next_to(ax.c2p(0, y), LEFT, 0.08)
                            for y, t in ((0, "0"), (0.5, "50%"), (1.0, "100%"))])
            xlab = VGroup(*[mono(str(n), 16, theme.MUTED).next_to(ax.c2p(n, 0), DOWN, 0.08)
                            for n in Ns])
            rt = zh("联想回忆准确率 vs 键值对个数", 22, theme.MUTED).move_to([2.9, 1.9, 0])
            self.play(Create(ax), FadeIn(ylab), FadeIn(xlab), FadeIn(rt), run_time=self.fit(1.0))
            lines = VGroup()
            for p in rc["patterns"]:
                pts = list(zip(Ns, rc["acc"][p]))
                line = polyline_in_axes(ax, pts, color=PCOLORS[p], stroke_width=4)
                end = zh(NAMES[p], 18, PCOLORS[p]).next_to(ax.c2p(*pts[-1]), RIGHT, 0.1)
                lines.add(VGroup(line, end))
            self.play(LaggedStart(*[Create(g) for g in lines], lag_ratio=0.3), run_time=self.fit(2.5))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(badge, bars, lt, ax, ylab, xlab, rt, lines)),
                      run_time=self.fit(0.6))

    # ── S13 谁在用 ───────────────────────────────────────────────────────
    def s13(self):
        with self.shot("S13"):
            self.play(*self.set_heading("谁在用：少量全注意力 + 大量线性层"), run_time=self.fit(0.8))
            rows = [
                ("模型", "线性层", "全注意力", "比例"),
                ("Qwen3-Next / Qwen3.5", "Gated DeltaNet", "gated attention", "3:1"),
                ("Kimi Linear", "KDA", "MLA", "3:1"),
                ("蚂蚁 Ling-3.0", "KDA", "MLA", "3:1"),
                ("Nemotron-H / Nemotron 3", "Mamba-2", "GQA", "少量注意力"),
                ("IBM Granite 4.0-H", "Mamba-2", "GQA", "9:1"),
                ("Falcon-H1", "Mamba-2", "同层并联", "—"),
            ]
            tb = table(rows, [4.2, 2.8, 2.8, 2.2], size=22, row_h=0.5).move_to([0, 0.6, 0])
            self.play(LaggedStart(*[FadeIn(r) for r in tb], lag_ratio=0.25), run_time=self.fit(3.0))
            self.wait(self.remaining() * 0.35)
            mm = zh("反例：MiniMax-01 线性注意力 7:1 混合 → MiniMax-M2 回到全注意力", 22, theme.GRAD)
            mm.move_to([0, -2.35, 0])
            self.play(FadeIn(mm), run_time=self.fit(0.8))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(tb, mm)), run_time=self.fit(0.6))

    # ── S14 从极简到生产级 ───────────────────────────────────────────────
    def s14(self):
        with self.shot("S14"):
            self.play(*self.set_heading("从极简到生产级"), run_time=self.fit(0.8))
            src = code_block(
                """S = torch.zeros(d_v, d_k)
for t in range(T):
    S = S + torch.outer(v[t], k[t])
    o[t] = S @ q[t]""",
                size=20,
            ).move_to([-3.6, 0.6, 0])
            lt = zh("极简版：写入 + 读出", 22, theme.MUTED).next_to(src, UP, 0.3)
            items = VGroup(*[zh(s, 22, theme.FG) for s in (
                "zero/arch/linear_attention.py",
                "· 短卷积 + L2 归一化 q、k",
                "· 数据相关的 α、β；门控 RMSNorm",
                "· 分块（训练）/ 递推（解码）两套算子",
                "· HybridTransformer：每 4 层 1 层全注意力",
            )]).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.2, 0.6, 0])
            items[0].set_color(theme.PARAM)
            self.play(FadeIn(lt), FadeIn(src), run_time=self.fit(1.0))
            self.play(LaggedStart(*[FadeIn(i) for i in items], lag_ratio=0.3), run_time=self.fit(2.0))
            ok = zh("与 Hugging Face 的 Qwen3.5 GatedDeltaNet 权重对拍：输出一致", 22, theme.OUTPUT)
            ok.move_to([0, -1.5, 0])
            self.play(FadeIn(ok), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.5)
            nxt = zh("下一章：混合专家（MoE）", 30, theme.HIGHLIGHT).move_to([0, -2.3, 0])
            self.play(FadeIn(nxt), run_time=self.fit(0.8))
            self.wait(max(0.1, self.remaining() - 0.8))
            self.play(FadeOut(VGroup(lt, src, items, ok, nxt)), *self.set_heading(None),
                      run_time=self.fit(0.8))
