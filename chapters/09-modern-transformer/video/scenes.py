"""第 9 章视频：现代 Transformer —— 把注意力搭成一个会写字的模型

画面里的所有数值都由 ../code/ 中的代码真实计算（见 script.md 事实清单 F2–F12）。
训练结果读自 code/out/tiny_transformer.pt（先运行 code/02_tiny_transformer.py）。
渲染：bash chapters/09-modern-transformer/video/build.sh
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    UR,
    Arc,
    Arrow,
    Axes,
    Circle,
    Create,
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
    Text,
    Transform,
    ValueTracker,
    VGroup,
    Write,
    always_redraw,
)

from video_kit import theme
from video_kit.scene import NarratedScene, polyline_in_axes, zh

CODE = Path(__file__).resolve().parent.parent / "code"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):  # 这些脚本在导入时会打印结果
        spec.loader.exec_module(mod)
    return mod


tiny = _load("tiny", "02_tiny_transformer.py")
pos = _load("position", "01_position.py")
qkn = _load("qk_norm", "05_qk_norm.py")
shp = _load("shapes", "03_shapes.py")

MONO = "Noto Sans Mono"


def mono(text: str, size: float = 22, color: str = theme.FG) -> Text:
    return Text(text, font=MONO, font_size=size, color=color)


def box(label: str, color: str, w: float = 3.0, h: float = 0.62, size: float = 24) -> VGroup:
    r = RoundedRectangle(width=w, height=h, corner_radius=0.1, stroke_color=color,
                         fill_color=color, fill_opacity=0.15, stroke_width=2.5)
    t = zh(label, size, theme.FG).move_to(r)
    return VGroup(r, t)


# ── 由代码算出的数字 ────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def position_numbers():
    """S04：打乱前文前后，最后一个位置的输出差；S05：不同 (m, n) 的点积。"""
    diffs = []
    for rope in (False, True):
        oa = pos.causal_attention(pos.a, pos.wq, pos.wk, pos.wv, rope)[-1]
        ob = pos.causal_attention(pos.b, pos.wq, pos.wk, pos.wv, rope)[-1]
        diffs.append((oa - ob).abs().max().item())
    dots = [(m, n, (pos.rope_at(pos.q0, m) @ pos.rope_at(pos.k0, n)).item())
            for m, n in [(3, 1), (10, 8), (50, 48), (5, 1), (40, 36)]]
    return diffs, dots


@lru_cache(maxsize=1)
def qk_rows():
    """S08：放大 s 倍时第 0 行的注意力权重（不加 / 加 QK-Norm），以及平均最大权重。"""
    out = {}
    for s in (1, 4, 16):
        q, k = qkn.q0 * s, qkn.k0 * s
        p_raw = (q @ k.T / math.sqrt(qkn.D)).softmax(-1)
        qn, kn = qkn.rms_norm(q), qkn.rms_norm(k)
        p_norm = (qn @ kn.T / math.sqrt(qkn.D)).softmax(-1)
        out[s] = (p_raw[0].tolist(), p_norm[0].tolist(),
                  qkn.stats(q, k)[1], qkn.stats(qn, kn)[1])
    return out


@lru_cache(maxsize=1)
def trained():
    model, ck = tiny.load_trained()
    return model, ck


@lru_cache(maxsize=1)
def parity_numbers():
    """S12：极简 vs zero 的 logits 差、贪心生成是否一致（与 code/04_parity_with_zero.py 相同）。"""
    from zero.config import ModelConfig
    from zero.generate import generate as zero_generate
    from zero.model import Transformer

    mini, _ = trained()
    c = mini.cfg
    zcfg = ModelConfig(vocab_size=c.vocab_size, dim=c.dim, n_layers=c.n_layers, n_heads=c.n_heads,
                       n_kv_heads=c.n_heads, head_dim=c.dim // c.n_heads, ffn_dim=c.ffn_dim,
                       rope_theta=c.rope_theta, max_seq_len=c.seq_len, norm_eps=c.eps,
                       qk_norm=True, tie_embeddings=True)
    prod = Transformer(zcfg).eval()
    prod.load_state_dict(mini.state_dict())
    text = b"ROMEO:\nBut soft, what light through yonder window breaks?\nJULIET:\n"
    tokens = torch.tensor([list(text)])
    with torch.no_grad():
        diff = (mini(tokens) - prod(tokens)).abs().max().item()
        prompt = list(b"ROMEO:\n")
        ids = torch.tensor([prompt])
        for _ in range(80):
            ids = torch.cat([ids, mini(ids)[:, -1].argmax(-1, keepdim=True)], dim=1)
    same = ids[0, len(prompt):].tolist() == zero_generate(prod, prompt, 80, temperature=0.0)
    return diff, same


def printable(s: str, width: int, lines: int) -> list[str]:
    """把采样文本整理成可显示的若干行：不可打印的字符（非法字节、控制字符）显示成 ·。"""
    out = []
    for raw in s.split("\n"):
        clean = "".join(ch if 32 <= ord(ch) < 127 else "·" for ch in raw)
        while len(clean) > width:
            out.append(clean[:width])
            clean = clean[width:]
        out.append(clean)
    return out[:lines]


def sci(x: float) -> str:
    m, e = f"{x:.2e}".split("e")
    return rf"{m}\times 10^{{{int(e)}}}"


class ChapterScene(NarratedScene):
    chapter_label = "第 9 章"
    chapter_title = "现代 Transformer"

    def construct(self) -> None:
        # ── S01 片头 ─────────────────────────────────────────────────────
        with self.shot("S01"):
            card = self.chapter_card()
            sub = zh("把注意力搭成一个会写字的模型", 32, theme.HIGHLIGHT).next_to(card, DOWN, 0.6)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(card), FadeOut(sub), run_time=self.fit(0.8))

        # ── S02 全景 ─────────────────────────────────────────────────────
        with self.shot("S02"):
            self.play(*self.set_heading("全景：查表 → N 个 Block → 打分"), run_time=self.fit(0.8))
            specs = [("token id：R O M E O …", theme.INPUT), ("Embedding 查表", theme.PARAM),
                     ("Block", theme.ATTN), ("Block", theme.ATTN), ("Block", theme.ATTN),
                     ("RMSNorm", theme.MUTED), ("LM head 打分", theme.PARAM),
                     ("下一个字节的概率", theme.OUTPUT)]
            stack = VGroup(*[box(t, c, w=4.2, h=0.5, size=22) for t, c in specs])
            stack.arrange(UP, buff=0.14).move_to([-2.2, 0.05, 0])
            n_lbl = zh("× N", 30, theme.ATTN).next_to(VGroup(*stack[2:5]), RIGHT, 0.25)
            steps = VGroup(
                zh("① 查表：编号 → 向量", 26, theme.PARAM),
                zh("② N 个一模一样的 Block", 26, theme.ATTN),
                zh("③ 打分：向量 → 256 个分数", 26, theme.OUTPUT),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to([4.2, 0.2, 0])
            per = self.remaining() / 12
            self.play(FadeIn(stack[0], shift=UP * 0.2), FadeIn(stack[1], shift=UP * 0.2),
                      run_time=self.fit(per * 1.5))
            self.play(FadeIn(steps[0]), run_time=self.fit(per))
            self.wait(per)
            self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in stack[2:5]], lag_ratio=0.4),
                      FadeIn(n_lbl), run_time=self.fit(per * 2))
            self.play(FadeIn(steps[1]), run_time=self.fit(per))
            self.wait(per)
            self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in stack[5:]], lag_ratio=0.4),
                      run_time=self.fit(per * 2))
            self.play(FadeIn(steps[2]), run_time=self.fit(per))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(stack), FadeOut(n_lbl), FadeOut(steps), run_time=self.fit(0.6))

        # ── S03 Block 内部 ───────────────────────────────────────────────
        with self.shot("S03"):
            self.play(*self.set_heading("Block：Pre-Norm + 残差"), run_time=self.fit(0.8))
            y0 = -1.3
            stream = Arrow([-6.6, y0, 0], [6.6, y0, 0], color=theme.INPUT, stroke_width=8,
                           buff=0, max_tip_length_to_length_ratio=0.03)
            s_lbl = zh("残差流：形状 (B, T, d) 从头到尾不变", 22, theme.INPUT).next_to(stream, DOWN, 0.2)

            def branch(x0: float, x1: float, name: str, color: str):
                n1 = box("RMSNorm", theme.MUTED, w=1.9, h=0.6, size=22).move_to([x0 + 1.1, 0.55, 0])
                n2 = box(name, color, w=2.0, h=0.6, size=22).move_to([x0 + 3.35, 0.55, 0])
                up = Line([x0, y0, 0], [x0, 0.55, 0], color=theme.MUTED)
                a1 = Line([x0, 0.55, 0], n1.get_left(), color=theme.MUTED)
                a2 = Arrow(n1.get_right(), n2.get_left(), buff=0.05, color=theme.MUTED, stroke_width=3)
                down = Arrow([x1, 0.55, 0], [x1, y0 + 0.25, 0], buff=0, color=color, stroke_width=4)
                h = Line(n2.get_right(), [x1, 0.55, 0], color=color)
                plus = VGroup(Circle(0.24, color=theme.HIGHLIGHT, fill_color=theme.BG, fill_opacity=1),
                              MathTex("+", color=theme.HIGHLIGHT, font_size=36)).move_to([x1, y0, 0])
                return VGroup(up, a1, n1, a2, n2, h, down, plus)

            b1 = branch(-5.9, -0.9, "注意力", theme.ATTN)
            b2 = branch(0.4, 5.4, "FFN", theme.OUTPUT)
            l1 = zh("位置之间交换信息", 20, theme.ATTN).next_to(b1[4], UP, 0.15)
            l2 = zh("每个位置各自加工", 20, theme.OUTPUT).next_to(b2[4], UP, 0.15)
            f = MathTex(r"x \leftarrow x + \mathrm{Attn}(\mathrm{RMSNorm}(x))", r"\qquad",
                        r"x \leftarrow x + \mathrm{FFN}(\mathrm{RMSNorm}(x))",
                        font_size=34).move_to([0, 2.35, 0])
            self.play(GrowArrow(stream), FadeIn(s_lbl), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.08)
            self.play(Create(b1), FadeIn(l1), run_time=self.fit(2))
            self.play(Create(b2), FadeIn(l2), run_time=self.fit(2))
            self.wait(self.remaining() * 0.25)
            self.play(Write(f), run_time=self.fit(2))
            hl = VGroup(SurroundingRectangle(b1[2], color=theme.HIGHLIGHT, buff=0.08),
                        SurroundingRectangle(b2[2], color=theme.HIGHLIGHT, buff=0.08))
            pn = zh("Pre-Norm：归一化放在子层入口，主干道上不做归一化", 22,
                    theme.HIGHLIGHT).move_to([0, -2.35, 0])
            self.wait(self.remaining() * 0.45)
            self.play(Create(hl), FadeIn(pn), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in (stream, s_lbl, b1, b2, l1, l2, f, hl, pn)],
                      run_time=self.fit(0.6))

        # ── S04 注意力分不清顺序 ─────────────────────────────────────────
        (d_none, d_rope), dots = position_numbers()
        with self.shot("S04"):
            self.play(*self.set_heading("问题：注意力分不清顺序"), run_time=self.fit(0.8))

            def row(chars: str, y: float) -> VGroup:
                cells = VGroup(*[box(ch, theme.INPUT, w=0.8, h=0.8, size=34) for ch in chars])
                cells.arrange(RIGHT, buff=0.25).move_to([-3.6, y, 0])
                arcs = VGroup(*[
                    Arrow(cells[i].get_top(), cells[-1].get_top(), path_arc=-1.2, buff=0.05,
                          color=theme.ATTN, stroke_width=2.5, max_tip_length_to_length_ratio=0.12)
                    for i in range(3)])
                return VGroup(cells, arcs)

            r1, r2 = row("狗咬人了", 1.2), row("人咬狗了", -1.2)
            self.play(FadeIn(r1[0]), FadeIn(r2[0]), run_time=self.fit(1))
            self.play(Create(r1[1]), Create(r2[1]), run_time=self.fit(1.5))
            t1 = zh("“了”的输出，两句的最大差", 24, theme.FG).move_to([3.6, 1.3, 0])
            # 两句的差别在浮点误差以内；有的机器上恰好是 0（sci(0) 会显示成 0.00×10⁰）
            v1 = VGroup(zh("无位置信息：", 24, theme.MUTED),
                        MathTex(sci(d_none) if d_none > 0 else "0", font_size=38,
                                color=theme.GRAD)).arrange(RIGHT, buff=0.2)
            v1.move_to([3.6, 0.4, 0])
            note = zh("浮点误差以内：完全一样", 22, theme.GRAD).next_to(v1, DOWN, 0.2)
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(t1), run_time=self.fit(0.8))
            self.play(FadeIn(v1), run_time=self.fit(1))
            self.play(FadeIn(note), run_time=self.fit(0.8))
            v2 = VGroup(zh("加 RoPE：", 24, theme.MUTED),
                        MathTex(f"{d_rope:.3f}", font_size=38, color=theme.OUTPUT)).arrange(RIGHT, buff=0.2)
            v2.move_to([3.6, -1.0, 0])
            self.wait(self.remaining() * 0.5)
            self.play(FadeIn(v2), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in (r1, r2, t1, v1, note, v2)], run_time=self.fit(0.6))

        # ── S05 RoPE：旋转 ───────────────────────────────────────────────
        with self.shot("S05"):
            self.play(*self.set_heading("RoPE：按位置把 q、k 旋转"), run_time=self.fit(0.8))
            center = np.array([-3.6, -0.1, 0])
            R = 2.1
            circ = Circle(R, color=theme.MUTED, stroke_width=2).move_to(center)
            omega = 0.32
            m0, n0 = 3, 1
            phi_q, phi_k = 0.35, -0.1
            t = ValueTracker(0)

            def vec(phi, color):
                return Arrow(center, center + R * np.array([np.cos(phi), np.sin(phi), 0]), buff=0,
                             color=color, stroke_width=6, max_tip_length_to_length_ratio=0.12)

            q_arrow = always_redraw(lambda: vec(phi_q + (m0 + t.get_value()) * omega, theme.PARAM))
            k_arrow = always_redraw(lambda: vec(phi_k + (n0 + t.get_value()) * omega, theme.INPUT))
            ang = always_redraw(lambda: Arc(
                radius=0.7, start_angle=phi_k + (n0 + t.get_value()) * omega,
                angle=(phi_q - phi_k) + (m0 - n0) * omega, arc_center=center,
                color=theme.HIGHLIGHT, stroke_width=4))
            labels = always_redraw(lambda: VGroup(
                MathTex(f"q\\ @\\ m={m0 + int(round(t.get_value()))}", font_size=30, color=theme.PARAM),
                MathTex(f"k\\ @\\ n={n0 + int(round(t.get_value()))}", font_size=30, color=theme.INPUT),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to(center + np.array([0, R + 0.55, 0])))
            rot = MathTex(r"(a,b)\to(a\cos\theta-b\sin\theta,\ b\cos\theta+a\sin\theta)",
                          font_size=30).move_to([3.4, 2.3, 0])
            self.play(Create(circ), run_time=self.fit(0.8))
            self.play(Write(rot), run_time=self.fit(1.5))
            self.play(GrowArrow(q_arrow), GrowArrow(k_arrow), FadeIn(labels), run_time=self.fit(1))
            self.add(q_arrow, k_arrow, labels)
            self.play(Create(ang), run_time=self.fit(0.6))
            self.add(ang)
            key = MathTex(r"\mathrm{RoPE}(q,m)\cdot\mathrm{RoPE}(k,n)=f(m-n)",
                          font_size=34, color=theme.HIGHLIGHT).move_to([3.4, 1.4, 0])
            self.wait(self.remaining() * 0.12)
            self.play(Write(key), run_time=self.fit(1.2))
            self.play(t.animate.set_value(7), run_time=self.fit(3))
            rows = VGroup(*[
                VGroup(MathTex(f"({m},{n})", font_size=30), MathTex(f"m-n={m - n}", font_size=30,
                                                                     color=theme.MUTED),
                       MathTex(f"{d:.4f}", font_size=30,
                               color=theme.HIGHLIGHT if m - n == 2 else theme.FG)
                       ).arrange(RIGHT, buff=0.45)
                for m, n, d in dots])
            rows.arrange(DOWN, buff=0.18, aligned_edge=LEFT).move_to([3.4, -0.75, 0])
            head = zh("位置 (m, n)　　相隔　　点积 q_m·k_n", 20, theme.MUTED).next_to(rows, UP, 0.2)
            self.play(FadeIn(head), LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.3),
                      run_time=self.fit(2.5))
            self.play(t.animate.set_value(47), run_time=self.fit(3))
            self.wait(self.remaining() - 0.6)
            for mob in (q_arrow, k_arrow, ang, labels):
                mob.clear_updaters()
            self.play(*[FadeOut(m) for m in (circ, rot, key, rows, head, q_arrow, k_arrow, ang, labels)],
                      run_time=self.fit(0.6))

        # ── S06 不同的转速 ───────────────────────────────────────────────
        with self.shot("S06"):
            self.play(*self.set_heading("不同维度对，转速不同"), run_time=self.fit(0.8))
            inv = (10000.0 ** (-torch.arange(0, pos.D, 2).float() / pos.D)).tolist()
            p = ValueTracker(0)
            dials = VGroup()
            needles = []
            for i, w in enumerate(inv):
                c = np.array([-5.1 + 3.4 * i, 0.2, 0])
                dial = Circle(1.1, color=theme.MUTED, stroke_width=2).move_to(c)
                lbl = MathTex(rf"\omega_{i}={w:g}", font_size=30, color=theme.PARAM).next_to(dial, DOWN, 0.25)
                per_turn = zh(f"一圈 ≈ {2 * math.pi / w:.0f} 个位置" if w < 1 else "一圈 ≈ 6.3 个位置",
                              20, theme.MUTED).next_to(lbl, DOWN, 0.15)
                dials.add(VGroup(dial, lbl, per_turn))
                needles.append(always_redraw(lambda c=c, w=w: Line(
                    c, c + 1.0 * np.array([np.cos(math.pi / 2 - w * p.get_value()),
                                           np.sin(math.pi / 2 - w * p.get_value()), 0]),
                    color=theme.PARAM, stroke_width=5)))
            counter = always_redraw(lambda: MathTex(f"m = {p.get_value():.0f}", font_size=36,
                                                    color=theme.HIGHLIGHT).move_to([0, 2.1, 0]))
            formula = MathTex(r"\omega_i = 10000^{-2i/d}", font_size=34).move_to([4.6, 2.1, 0])
            self.play(FadeIn(dials), *[Create(n) for n in needles], FadeIn(counter), Write(formula),
                      run_time=self.fit(1.5))
            self.add(*needles, counter)
            self.play(p.animate.set_value(60), run_time=self.fit(self.remaining() - 1.0),
                      rate_func=lambda x: x)
            for mob in (*needles, counter):
                mob.clear_updaters()
            self.play(FadeOut(dials), FadeOut(formula), *[FadeOut(n) for n in needles], FadeOut(counter),
                      run_time=self.fit(0.6))

        # ── S07 SwiGLU ───────────────────────────────────────────────────
        with self.shot("S07"):
            self.play(*self.set_heading("FFN：SwiGLU（门控前馈）"), run_time=self.fit(0.8))
            f = MathTex(r"\mathrm{FFN}(x)=W_{down}\big(\mathrm{SiLU}(W_{gate}x)\odot W_{up}x\big)",
                        font_size=36).move_to([0, 2.3, 0])
            xb = box("x", theme.INPUT, w=0.9, h=0.7, size=28).move_to([-6.1, 0.2, 0])
            gate = box("W_gate", theme.PARAM, w=1.7, h=0.6, size=22).move_to([-4.0, 1.1, 0])
            upb = box("W_up", theme.PARAM, w=1.7, h=0.6, size=22).move_to([-4.0, -0.7, 0])
            silu = box("SiLU（门）", theme.GRAD, w=2.0, h=0.6, size=22).move_to([-1.7, 1.1, 0])
            mul = VGroup(Circle(0.3, color=theme.HIGHLIGHT), MathTex(r"\odot", font_size=40,
                                                                      color=theme.HIGHLIGHT)).move_to([0.2, 0.2, 0])
            down = box("W_down", theme.PARAM, w=1.8, h=0.6, size=22).move_to([2.0, 0.2, 0])
            arrows = VGroup(
                Arrow(xb.get_right(), gate.get_left(), buff=0.08, stroke_width=3, color=theme.MUTED),
                Arrow(xb.get_right(), upb.get_left(), buff=0.08, stroke_width=3, color=theme.MUTED),
                Arrow(gate.get_right(), silu.get_left(), buff=0.08, stroke_width=3, color=theme.MUTED),
                Arrow(silu.get_right(), mul.get_top(), buff=0.08, stroke_width=3, color=theme.MUTED),
                Arrow(upb.get_right(), mul.get_bottom(), buff=0.08, stroke_width=3, color=theme.MUTED),
                Arrow(mul.get_right(), down.get_left(), buff=0.08, stroke_width=3, color=theme.MUTED),
            )
            cont = zh("内容", 20, theme.PARAM).next_to(upb, DOWN, 0.12)
            # 右侧：8 个维度上的门、内容、相乘（随机向量真实计算）
            g = torch.Generator().manual_seed(3)
            z, u = torch.randn(8, generator=g) * 2, torch.randn(8, generator=g)
            gv = torch.nn.functional.silu(z)
            pv = gv * u
            grid = VGroup()
            for name, vals, col in (("门", gv, theme.GRAD), ("内容", u, theme.PARAM), ("相乘", pv, theme.OUTPUT)):
                cells = VGroup(*[
                    Rectangle(width=0.26, height=0.26, stroke_width=1, stroke_color=theme.MUTED,
                              fill_color=col, fill_opacity=min(1.0, abs(v) / 2.2))
                    for v in vals.tolist()]).arrange(RIGHT, buff=0.04)
                grid.add(VGroup(zh(name, 20, col), cells).arrange(RIGHT, buff=0.2))
            grid.arrange(DOWN, aligned_edge=RIGHT, buff=0.2).move_to([5.25, 0.2, 0])
            gnote = zh("门≈0 的维度被关掉", 20, theme.MUTED).next_to(grid, DOWN, 0.2)
            params = zh("d = 1280：4d 的 MLP 13.11M 参数　vs　8/3·d 的 SwiGLU 13.11M 参数",
                        22, theme.HIGHLIGHT).move_to([0, -2.2, 0])
            self.play(Write(f), run_time=self.fit(1.5))
            self.play(FadeIn(xb), FadeIn(gate), FadeIn(upb), Create(arrows[0]), Create(arrows[1]),
                      FadeIn(cont), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.1)
            self.play(FadeIn(silu), Create(arrows[2]), run_time=self.fit(1))
            self.play(FadeIn(mul), Create(arrows[3]), Create(arrows[4]), FadeIn(down), Create(arrows[5]),
                      run_time=self.fit(1.2))
            self.play(FadeIn(grid), FadeIn(gnote), run_time=self.fit(1.5))
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(params), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in (f, xb, gate, upb, silu, mul, down, arrows, cont, grid,
                                             gnote, params)], run_time=self.fit(0.6))

        # ── S08 QK-Norm ──────────────────────────────────────────────────
        rowsd = qk_rows()
        with self.shot("S08"):
            self.play(*self.set_heading("QK-Norm：给注意力分数上保险"), run_time=self.fit(0.8))

            def bars(vals, y_base, color):
                w = 0.42
                g = VGroup()
                for i, v in enumerate(vals):
                    h = max(0.02, 1.7 * v)
                    r = Rectangle(width=w, height=h, stroke_width=0, fill_color=color, fill_opacity=0.9)
                    r.move_to([-6.0 + i * (w + 0.1) + w / 2, y_base + h / 2, 0])
                    g.add(r)
                return g

            top_base, bot_base = 0.55, -2.35
            lab_t = zh("不加 QK-Norm", 22, theme.GRAD).move_to([-4.6, 2.55, 0])
            lab_b = zh("加 QK-Norm", 22, theme.OUTPUT).move_to([-4.8, -0.35, 0])
            base_t = Line([-6.1, top_base, 0], [2.3, top_base, 0], color=theme.MUTED, stroke_width=1)
            base_b = Line([-6.1, bot_base, 0], [2.3, bot_base, 0], color=theme.MUTED, stroke_width=1)
            bt = bars(rowsd[1][0], top_base, theme.ATTN)
            bb = bars(rowsd[1][1], bot_base, theme.ATTN)

            def info(s):
                _, _, mr, mn = rowsd[s]
                return VGroup(
                    zh(f"放大 {s} 倍", 28, theme.HIGHLIGHT),
                    zh(f"平均最大权重 {mr:.3f}", 24, theme.GRAD),
                    zh(f"平均最大权重 {mn:.3f}", 24, theme.OUTPUT),
                ).arrange(DOWN, buff=0.9, aligned_edge=LEFT).move_to([4.8, 0.0, 0])

            inf = info(1)
            self.play(FadeIn(lab_t), FadeIn(lab_b), Create(base_t), Create(base_b), FadeIn(bt), FadeIn(bb),
                      FadeIn(inf), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.2)
            for s in (4, 16):
                self.play(Transform(bt, bars(rowsd[s][0], top_base, theme.ATTN)),
                          Transform(bb, bars(rowsd[s][1], bot_base, theme.ATTN)),
                          Transform(inf, info(s)), run_time=self.fit(1.5))
                self.wait(self.remaining() * 0.25)
            cap = zh("点积之前，对每个头的 q、k 做 RMSNorm", 24, theme.HIGHLIGHT).move_to([0.8, -0.35, 0])
            cap.next_to(lab_b, RIGHT, 0.6)
            self.play(FadeIn(cap), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in (lab_t, lab_b, base_t, base_b, bt, bb, inf, cap)],
                      run_time=self.fit(0.6))

        # ── S09 共享 embedding ───────────────────────────────────────────
        with self.shot("S09"):
            self.play(*self.set_heading("共享 embedding：一张表，两头用"), run_time=self.fit(0.8))
            emb = VGroup(Rectangle(width=1.6, height=3.0, color=theme.PARAM, fill_opacity=0.25),
                         zh("输入表\nV × d", 22, theme.FG)).move_to([-5.4, 0.6, 0])
            emb[1].move_to(emb[0])
            head = VGroup(Rectangle(width=3.0, height=1.2, color=theme.PARAM, fill_opacity=0.25),
                          zh("输出头 d × V", 22, theme.FG)).move_to([-2.4, 0.6, 0])
            head[1].move_to(head[0])
            self.play(FadeIn(emb), FadeIn(head), run_time=self.fit(1))
            self.wait(self.remaining() * 0.15)
            same = zh("同一个矩阵：lm_head.weight = tok_emb.weight", 22, theme.HIGHLIGHT).move_to([-3.6, -1.6, 0])
            self.play(Transform(head[0], emb[0].copy()), FadeOut(head[1]), run_time=self.fit(1.2))
            self.play(FadeIn(same), run_time=self.fit(0.8))
            # 右侧：参数账本（由 zero.model.count_params 计算）
            rows = []
            for name, cfg in (("Qwen3-0.6B", shp.qwen3_06b), ("主线 configs/main", shp.m)):
                cfg.tie_embeddings = True
                c = shp.count_params(cfg)
                rows.append((name, c["total"] / 1e6, c["embedding"] / 1e6))
            scale = 4.2 / 800
            chart = VGroup()
            for i, (name, tot, e) in enumerate(rows):
                y = 1.3 - i * 1.5
                full = Rectangle(width=tot * scale, height=0.45, stroke_width=0, fill_color=theme.MUTED,
                                 fill_opacity=0.5)
                full.move_to([1.0 + tot * scale / 2, y, 0])
                part = Rectangle(width=e * scale, height=0.45, stroke_width=0, fill_color=theme.PARAM,
                                 fill_opacity=1)
                part.move_to([1.0 + e * scale / 2, y, 0])
                t1 = zh(name, 22, theme.FG).next_to(full, UP, 0.1, aligned_edge=LEFT)
                t2 = zh(f"词表矩阵 {e:.1f}M / 总 {tot:.1f}M = {e / tot:.1%}", 20, theme.PARAM)
                t2.next_to(full, DOWN, 0.1, aligned_edge=LEFT)
                chart.add(VGroup(full, part, t1, t2))
            self.wait(self.remaining() * 0.15)
            self.play(LaggedStart(*[FadeIn(r) for r in chart], lag_ratio=0.5), run_time=self.fit(2))
            rule = zh("小模型普遍共享，大模型大多不共享", 24, theme.HIGHLIGHT).move_to([3.1, -1.9, 0])
            self.wait(self.remaining() * 0.5)
            self.play(FadeIn(rule), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in (emb, head[0], same, chart, rule)], run_time=self.fit(0.6))

        # ── S10 张量形状数据流 ───────────────────────────────────────────
        with self.shot("S10"):
            self.play(*self.set_heading("主线模型一次前向（B = 8, T = 4096）"), run_time=self.fit(0.8))
            flow = [(a.strip(), b.split("  ←")[0]) for a, b in shp.flow]
            lines = VGroup()
            for a, b in flow:
                lines.add(VGroup(zh(a, 20, theme.MUTED), mono(b, 20, theme.FG)))
            for ln in lines:
                ln[1].move_to([-1.9, 0, 0], aligned_edge=LEFT)
                ln[0].move_to([-2.4, 0, 0], aligned_edge=RIGHT)
            ys = np.linspace(2.45, -2.4, len(lines))
            for ln, y in zip(lines, ys):
                ln.shift(UP * (y - ln.get_center()[1]))
            notes = {4: ("GQA：8 个 K/V 头（第 10 章）", theme.ATTN),
                     6: ("T × T 的方阵", theme.GRAD),
                     12: ("≈ 21.5 亿个数，fp32 约 8 GiB", theme.GRAD)}
            per = (self.remaining() - 1.5) / len(lines)
            cursor = SurroundingRectangle(lines[0], color=theme.HIGHLIGHT, buff=0.06)
            self.play(FadeIn(lines[0]), Create(cursor), run_time=self.fit(min(0.6, per)))
            shown = []
            for i in range(1, len(lines)):
                anims = [FadeIn(lines[i]), Transform(cursor, SurroundingRectangle(lines[i], color=theme.HIGHLIGHT,
                                                                                    buff=0.06))]
                if i in notes:
                    txt, col = notes[i]
                    nt = zh(txt, 18, col).next_to(lines[i][1], RIGHT, 0.3)
                    if nt.get_right()[0] > 7.0:
                        nt.shift(LEFT * (nt.get_right()[0] - 7.0))
                    shown.append(nt)
                    anims.append(FadeIn(nt))
                self.play(*anims, run_time=self.fit(min(0.8, per * 0.5)))
                self.wait(max(0.05, per * 0.5))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(lines), FadeOut(cursor), *[FadeOut(n) for n in shown], run_time=self.fit(0.6))

        # ── S11 训练 ─────────────────────────────────────────────────────
        _, ck = trained()
        hist = ck["history"]
        with self.shot("S11"):
            self.play(*self.set_heading(f"训练：{ck['n_params'] / 1e4:.0f} 万参数 · 1200 步 · 字节级"),
                      run_time=self.fit(0.8))
            ax = Axes(x_range=[0, 1200, 400], y_range=[0, 9, 2], x_length=5.4, y_length=3.8,
                      axis_config={"color": theme.MUTED, "include_numbers": True, "font_size": 18},
                      tips=False).move_to([-3.5, -0.1, 0])
            xl = zh("步数", 18, theme.MUTED).next_to(ax.x_axis, DOWN, 0.35)
            yl = zh("验证集 bit/字节", 18, theme.MUTED).next_to(ax.y_axis, UP, 0.15)
            yl.shift(RIGHT * (ax.y_axis.get_left()[0] + 0.1 - yl.get_left()[0]))
            pts = [(h["step"], h["val_bpb"]) for h in hist]
            curve = polyline_in_axes(ax, pts, color=theme.GRAD, stroke_width=4)
            dots = VGroup(*[Dot(ax.c2p(x, y), radius=0.05, color=theme.GRAD) for x, y in pts])
            lbls = VGroup(*[zh(f"{y:.2f}", 16, theme.FG).next_to(ax.c2p(x, y), UR, 0.08)
                            for x, y in pts if x in (0, 200, 1200)])
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=self.fit(1))
            self.play(Create(curve), FadeIn(dots), run_time=self.fit(3))
            self.play(FadeIn(lbls), run_time=self.fit(0.8))
            before = VGroup(*[mono(s or ".", 16, theme.MUTED if s else theme.BG)
                              for s in printable(ck["before"], 34, 6)])
            before.arrange(DOWN, aligned_edge=LEFT, buff=0.08)
            btitle = zh("训练前（· = 非法字节 / 控制字符）", 20, theme.GRAD)
            bgrp = VGroup(btitle, before).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.5, 0.9, 0])
            self.wait(self.remaining() * 0.12)
            self.play(FadeIn(bgrp), run_time=self.fit(1))
            self.wait(self.remaining() * 0.2)
            after = VGroup(*[mono(s or ".", 16, theme.FG if s else theme.BG)
                             for s in printable(ck["after"], 44, 12)])
            after.arrange(DOWN, aligned_edge=LEFT, buff=0.06)
            atitle = zh("训练后（温度 0.8 采样）", 20, theme.OUTPUT)
            agrp = VGroup(atitle, after).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([3.4, 0.05, 0])
            self.play(FadeOut(bgrp), FadeIn(agrp), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in (ax, xl, yl, curve, dots, lbls, agrp)], run_time=self.fit(0.6))

        # ── S12 两层对拍 ─────────────────────────────────────────────────
        diff, same = parity_numbers()
        with self.shot("S12"):
            self.play(*self.set_heading("两层对拍：极简版 = zero = Qwen3"), run_time=self.fit(0.8))
            b1 = VGroup(box("极简模型", theme.INPUT, w=3.0, h=1.0, size=26),
                        mono("code/02_tiny_transformer.py", 14, theme.MUTED))
            b2 = VGroup(box("zero.Transformer", theme.PARAM, w=3.0, h=1.0, size=26),
                        mono("zero/model.py", 14, theme.MUTED))
            b3 = VGroup(box("官方 Qwen3", theme.OUTPUT, w=3.0, h=1.0, size=26),
                        mono("transformers", 14, theme.MUTED))
            for b, x in ((b1, -4.8), (b2, 0.0), (b3, 4.8)):
                b[0].move_to([x, 1.4, 0])
                b[1].next_to(b[0], DOWN, 0.12)
            a1 = Arrow(b1[0].get_right(), b2[0].get_left(), buff=0.1, color=theme.MUTED)
            a2 = Arrow(b2[0].get_right(), b3[0].get_left(), buff=0.1, color=theme.MUTED)
            t1 = zh("权重原样搬运", 16, theme.MUTED).next_to(a1, UP, 0.15)
            t2 = zh("导出 HF 格式", 16, theme.MUTED).next_to(a2, UP, 0.15)
            self.play(FadeIn(b1), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.08)
            self.play(GrowArrow(a1), FadeIn(t1), FadeIn(b2), run_time=self.fit(1.2))
            res = VGroup(
                VGroup(zh("同一段文字的 logits 最大差", 24, theme.FG),
                       MathTex(sci(diff), font_size=36, color=theme.HIGHLIGHT)).arrange(RIGHT, buff=0.3),
                zh(f"贪心生成 80 个字节：{'完全相同' if same else '不同'}", 24, theme.OUTPUT if same else theme.GRAD),
            ).arrange(DOWN, buff=0.3).move_to([0, -0.5, 0])
            self.play(FadeIn(res[0]), run_time=self.fit(1))
            self.play(FadeIn(res[1]), run_time=self.fit(1))
            self.wait(self.remaining() * 0.15)
            self.play(GrowArrow(a2), FadeIn(t2), FadeIn(b3), run_time=self.fit(1.2))
            tests = zh("tests/test_model_hf_parity.py：8 passed", 22, theme.OUTPUT).move_to([0, -1.55, 0])
            self.play(FadeIn(tests), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.3)
            cmd = mono("uv run python -m zero.train.pretrain --config configs/tiny/pretrain.toml", 18,
                       theme.MUTED).move_to([0, -2.3, 0])
            self.play(FadeIn(cmd), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in (b1, b2, b3, a1, a2, t1, t2, res, tests, cmd)],
                      run_time=self.fit(0.6))

        # ── S13 历史对照与下一章 ─────────────────────────────────────────
        with self.shot("S13"):
            self.play(*self.set_heading("GPT-2（2019）→ 现代 Transformer"), run_time=self.fit(0.8))
            pairs = [("LayerNorm", "RMSNorm"), ("学出来的位置表（1024）", "RoPE"),
                     ("4d GELU MLP", "SwiGLU"), ("—", "QK-Norm")]
            table = VGroup()
            for a, b in pairs:
                table.add(VGroup(zh(a, 26, theme.MUTED), zh("→", 26, theme.FG), zh(b, 26, theme.OUTPUT)))
            for r in table:
                r[0].move_to([-2.2, 0, 0], aligned_edge=RIGHT)
                r[1].move_to([-1.2, 0, 0])
                r[2].move_to([-0.2, 0, 0], aligned_edge=LEFT)
            for r, y in zip(table, np.linspace(2.0, 0.2, len(table))):
                r.shift(UP * (y - r.get_center()[1]))
            same_bone = zh("骨架不变：查表 → N 个 Block → 打分", 24, theme.HIGHLIGHT).move_to([0, -0.6, 0])
            total = self.remaining()
            self.play(LaggedStart(*[FadeIn(r) for r in table], lag_ratio=0.6),
                      run_time=self.fit(total * 0.3))
            self.play(FadeIn(same_bone), run_time=self.fit(0.8))
            self.wait(max(0.1, self.remaining() - total * 0.45))
            self.play(FadeOut(table), FadeOut(same_bone), *self.set_heading("为什么生成这么慢？"),
                      run_time=self.fit(0.6))
            # 生成：每出一个字节都把整段重算一遍
            text = "ROMEO:  And yield for what be"
            line = mono(text.replace(" ", "·"), 30, theme.FG).move_to([0, 0.8, 0])
            chars = VGroup(*line.submobjects)
            pre = 7
            self.add(*chars[:pre])
            rec = zh("每生成 1 个字节，整段重新算一遍", 24, theme.GRAD).move_to([0, -0.3, 0])
            self.play(FadeIn(rec), run_time=self.fit(0.6))
            steps = len(chars) - pre
            per = max(0.1, (self.remaining() - 2.2) / steps)
            hl = SurroundingRectangle(chars[:pre], color=theme.GRAD, buff=0.08)
            self.play(Create(hl), run_time=self.fit(min(0.3, per)))
            for i in range(pre, len(chars)):
                self.play(FadeIn(chars[i]), Transform(hl, SurroundingRectangle(chars[:i + 1], color=theme.GRAD,
                                                                                buff=0.08)),
                          run_time=self.fit(per * 0.8))
            nxt = zh("下一章：推理 —— KV cache 与 GQA", 30, theme.HIGHLIGHT).move_to([0, -1.5, 0])
            self.play(FadeIn(nxt), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(chars), FadeOut(hl), FadeOut(rec), FadeOut(nxt), *self.set_heading(None),
                      run_time=self.fit(0.6))
