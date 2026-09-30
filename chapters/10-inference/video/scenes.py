"""第 10 章视频：推理 —— 采样、KV cache、GQA

画面里的所有数值都由 ../code/ 中的代码真实计算（见 script.md 事实清单）。
较慢的计算（测速、GQA 对比）结果缓存在 video/out/cache.json；删掉它会重新计算。
渲染：bash chapters/10-inference/video/build.sh
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
from video_kit.scene import NarratedScene, code_block, zh

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

    tiny = _load("tiny_model", "01_tiny_model.py")
    samp = _load("sampling", "02_sampling.py")
    kvc = _load("kv_cache_demo", "03_kv_cache.py")
    mem = _load("kv_memory", "04_kv_memory.py")
    gqa = _load("gqa", "05_gqa.py")
    model, data = tiny.load_or_train(4), tiny.CharData()
    d: dict = {"params": sum(p.numel() for p in model.parameters())}

    d["prompt"] = samp.PROMPT
    d["greedy"] = samp.generate(model, data, samp.PROMPT, 200, seed=0, temperature=0)
    d["topp"] = samp.generate(model, data, samp.PROMPT, 200, seed=0, temperature=1.0, top_p=0.9)
    d["distinct"] = {name: samp.distinct_4gram(samp.generate(model, data, samp.PROMPT, 200,
                                                               seed=0, **kw))
                     for name, kw in samp.STRATEGIES}

    logits = samp.next_logits(model, data, samp.PROMPT)
    order = torch.argsort(logits, descending=True)[:8].tolist()
    d["dist_chars"] = [data.chars[i] for i in order]
    d["dist"] = {str(T): [float(samp.filtered_probs(logits, T)[i]) for i in order]
                 for T in (0.5, 1.0, 1.5)}

    d["contexts"] = {}
    for name, ctx in samp.CONTEXTS.items():
        lg = samp.next_logits(model, data, ctx)
        p = torch.softmax(lg, -1)
        o = torch.argsort(p, descending=True)[:20].tolist()
        n90 = int((samp.filtered_probs(lg, 1.0, top_p=0.9) > 0).sum())
        d["contexts"][name] = dict(ctx=ctx, chars=[data.chars[i] for i in o],
                                   probs=[float(p[i]) for i in o], n90=n90)

    prompt = data.encode(kvc.PROMPT)
    a, _ = kvc.generate_naive(model, prompt, 200, seed=0, temperature=0)
    b, _, _ = kvc.generate_cached(model, prompt, 200, seed=0, temperature=0)
    a2, _ = kvc.generate_naive(model, prompt, 200, seed=0, temperature=1.0, top_p=0.9)
    b2, _, _ = kvc.generate_cached(model, prompt, 200, seed=0, temperature=1.0, top_p=0.9)
    d["same"] = bool(a == b and a2 == b2)
    d["prompt_len"] = len(prompt)
    d["speed"] = kvc.speed_table(model, data)
    d["pd"] = kvc.prefill_vs_decode(model, data)

    m = mem.main_config()
    d["main"] = dict(L=m.n_layers, Hq=m.n_heads, Hkv=m.n_kv_heads, D=m.head_dim)
    d["public"] = [list(r) for r in mem.PUBLIC]

    rows, noise = gqa.run()
    d["gqa"] = rows
    d["gqa_noise"] = noise
    return d


def get_data() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    d = compute()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


D = get_data()
GIB = 2**30


def kv_bytes(L, H, Dh, T, b=2):
    return 2 * L * H * Dh * T * b


def mono(text: str, size: float = 24, color: str = theme.FG) -> Text:
    return Text(text, font=MONO, font_size=size, color=color)


def show(ch: str) -> str:
    return {"\n": "↵", " ": "␣"}.get(ch, ch)


def token_box(ch: str, color: str, size: float = 0.42) -> VGroup:
    box = RoundedRectangle(width=size, height=size * 1.2, corner_radius=0.06, stroke_color=color,
                           stroke_width=2, fill_color=color, fill_opacity=0.18)
    t = mono(show(ch), 22, theme.FG).move_to(box)
    return VGroup(box, t)


def bars(values, labels, width=0.5, max_h=3.0, vmax=None, color=theme.OUTPUT, gap=0.15,
         label_size=22, base_y=-2.2) -> VGroup:
    """竖直柱状图：返回 VGroup(柱子组, 标签组)，柱底在 base_y。"""
    vmax = vmax or max(values)
    rects, labs = VGroup(), VGroup()
    for i, (v, lab) in enumerate(zip(values, labels)):
        h = max(0.02, v / vmax * max_h)
        r = Rectangle(width=width, height=h, stroke_width=0, fill_color=color, fill_opacity=0.9)
        r.move_to([i * (width + gap), base_y + h / 2, 0])
        rects.add(r)
        labs.add(mono(lab, label_size, theme.FG).next_to([i * (width + gap), base_y, 0], DOWN, 0.1))
    return VGroup(rects, labs)


class ChapterScene(NarratedScene):
    chapter_label = "第 10 章"
    chapter_title = "推理"

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
        self.s14()
        self.s15()
        self.s16()

    # ── S01 片头 ─────────────────────────────────────────────────────────
    def s01(self):
        with self.shot("S01"):
            card = self.chapter_card()
            sub = zh("采样、KV cache 与 GQA", 32, theme.HIGHLIGHT).next_to(card, DOWN, 0.6)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(card), FadeOut(sub), run_time=self.fit(0.8))

    # ── S02 自回归生成 ───────────────────────────────────────────────────
    def s02(self):
        with self.shot("S02"):
            self.play(*self.set_heading("自回归生成：一次一个字"), run_time=self.fit(0.8))
            prompt = D["prompt"]
            gen = D["greedy"][:11]
            n_total = len(prompt) + len(gen)
            step = 0.5
            x0 = -(n_total - 1) * step / 2
            boxes = VGroup(*[token_box(c, theme.INPUT).move_to([x0 + i * step, 1.6, 0])
                             for i, c in enumerate(prompt)])
            p_lbl = zh("提示词", 22, theme.INPUT).next_to(boxes, UP, 0.2)
            self.play(LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.05), FadeIn(p_lbl),
                      run_time=self.fit(1.5))
            model = RoundedRectangle(width=3.6, height=1.0, corner_radius=0.15,
                                     stroke_color=theme.PARAM, fill_color=theme.PARAM,
                                     fill_opacity=0.12).move_to([0, -0.3, 0])
            m_lbl = zh(f"小模型（{D['params'] / 1e6:.2f}M 参数）", 26, theme.PARAM).move_to(model)
            loop = zh("logits → 挑一个字 → 接到末尾 → 再喂回去", 24, theme.MUTED).move_to([0, -1.7, 0])
            self.play(FadeIn(model), FadeIn(m_lbl), run_time=self.fit(0.8))
            self.play(FadeIn(loop), run_time=self.fit(0.8))
            per = max(0.3, (self.remaining() - 1.0) / len(gen))
            for i, c in enumerate(gen):
                pos = [x0 + (len(prompt) + i) * step, 1.6, 0]
                down = Arrow([0, 1.1, 0], model.get_top(), color=theme.MUTED, buff=0.05,
                             stroke_width=3)
                up = Arrow(model.get_top() + RIGHT * 1.2, [pos[0], 1.1, 0], color=theme.OUTPUT,
                           buff=0.05, stroke_width=3)
                nb = token_box(c, theme.OUTPUT).move_to(pos)
                self.play(FadeIn(down), run_time=self.fit(per * 0.3))
                self.play(FadeOut(down), FadeIn(up), FadeIn(nb, shift=UP * 0.2),
                          run_time=self.fit(per * 0.4))
                self.play(FadeOut(up), run_time=self.fit(per * 0.3))
                boxes.add(nb)
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(boxes, p_lbl, model, m_lbl, loop)), run_time=self.fit(0.6))

    # ── S03 贪心的问题 ───────────────────────────────────────────────────
    def s03(self):
        with self.shot("S03"):
            self.play(*self.set_heading("贪心：每一步都挑概率最大的"), run_time=self.fit(0.8))

            def panel(title, text, stat, color, y):
                lines = [ln for ln in text[:96].split("\n") if ln.strip()]
                body = VGroup(*[mono(ln, 22, theme.FG) for ln in lines[:2]]).arrange(
                    DOWN, aligned_edge=LEFT, buff=0.12)
                head = zh(title, 26, color)
                st = zh(stat, 24, color)
                g = VGroup(head, body, st).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
                box = SurroundingRectangle(g, buff=0.2, color=color, corner_radius=0.1)
                return VGroup(box, g).move_to([0, y, 0])

            dg = D["distinct"]
            greedy = panel("贪心（T = 0）", D["prompt"].split("\n")[1] + D["greedy"],
                           f"不重复的 4 字符片段占比：{dg['贪心 (T=0)']:.2f}", theme.GRAD, 1.15)
            self.play(FadeIn(greedy), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.55)
            topp = panel("随机抽样（T = 1.0，top-p = 0.9）", D["prompt"].split("\n")[1] + D["topp"],
                         f"不重复的 4 字符片段占比：{dg['T=1.0, top-p=0.9']:.2f}", theme.OUTPUT, -1.35)
            self.play(FadeIn(topp), run_time=self.fit(1.2))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(greedy), FadeOut(topp), run_time=self.fit(0.6))

    # ── S04 温度 ─────────────────────────────────────────────────────────
    def s04(self):
        with self.shot("S04"):
            self.play(*self.set_heading("温度：先把分布调尖或调平"), run_time=self.fit(0.8))
            formula = MathTex(r"p_i=\mathrm{softmax}(z/T)_i", font_size=40).move_to([4.4, 2.3, 0])
            ctx = zh("“I will ” 之后下一个字符的真实分布（前 8 名）", 22,
                     theme.MUTED).move_to([-2.6, 2.3, 0])
            chars = [show(c) for c in D["dist_chars"]]
            vmax = max(D["dist"]["0.5"])

            def chart(T, color):
                vals = D["dist"][str(T)]
                g = bars(vals, chars, width=0.7, max_h=3.4, vmax=vmax, color=color, gap=0.35)
                nums = VGroup(*[mono(f"{v:.2f}", 20, theme.FG).next_to(r, UP, 0.08)
                                for v, r in zip(vals, g[0])])
                g.add(nums)
                return g.move_to([-1.2, -0.45, 0], aligned_edge=DOWN).shift(DOWN * 1.7)

            cur = chart(1.0, theme.OUTPUT)
            tlabel = MathTex("T=1.0", font_size=44, color=theme.OUTPUT).move_to([4.6, 0.8, 0])
            self.play(FadeIn(ctx), Write(formula), run_time=self.fit(1.2))
            self.play(FadeIn(cur), FadeIn(tlabel), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.2)
            for T, col, note in ((0.5, theme.HIGHLIGHT, "更尖：更保守"), (1.5, theme.ATTN, "更平：更大胆")):
                new = chart(T, col)
                nl = MathTex(f"T={T}", font_size=44, color=col).move_to([4.6, 0.8, 0])
                nt = zh(note, 26, col).move_to([4.6, 0.0, 0])
                self.play(Transform(cur, new), Transform(tlabel, nl), run_time=self.fit(1.5))
                self.play(FadeIn(nt), run_time=self.fit(0.5))
                self.wait(self.remaining() * 0.35)
                self.play(FadeOut(nt), run_time=self.fit(0.4))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(cur, tlabel, formula, ctx)), run_time=self.fit(0.6))

    # ── S05 top-k 与 top-p ───────────────────────────────────────────────
    def s05(self):
        with self.shot("S05"):
            self.play(*self.set_heading("截掉长尾：top-k 与 top-p"), run_time=self.fit(0.8))
            groups = VGroup()
            for (name, c), cx in zip(D["contexts"].items(), (-3.55, 3.55)):
                probs, n90 = c["probs"], c["n90"]
                g = bars(probs, [show(ch) for ch in c["chars"]], width=0.22, max_h=2.6, vmax=0.6,
                         color=theme.MUTED, gap=0.07, label_size=14, base_y=-1.9)
                g.move_to([cx, -0.55, 0])
                title = zh(f"{name}：“{c['ctx'].split(chr(10))[-1] or c['ctx'].strip()}”", 24,
                           theme.FG).move_to([cx, 2.35, 0])
                groups.add(VGroup(g, title))
                g.n90, g.cx = n90, cx
            self.play(*[FadeIn(g) for g in groups], run_time=self.fit(1.5))
            self.wait(self.remaining() * 0.25)
            anims, marks = [], VGroup()
            for grp in groups:
                g = grp[0]
                for i, r in enumerate(g[0]):
                    if i < g.n90:
                        anims.append(r.animate.set_fill(theme.OUTPUT))
                right = g[0][g.n90 - 1].get_right()[0] + 0.04
                cut = DashedLine([right, -1.9, 0], [right, 1.3, 0], color=theme.HIGHLIGHT)
                lbl = zh(f"top-p 0.9：留 {g.n90} 个", 24, theme.OUTPUT).move_to([g.cx, 1.65, 0])
                k_right = g[0][4].get_right()[0] + 0.04
                kcut = DashedLine([k_right, -1.9, 0], [k_right, 0.7, 0], color=theme.PARAM)
                klbl = zh("top-k 5", 20, theme.PARAM).next_to(kcut.get_top(), LEFT, 0.1)
                marks.add(cut, lbl, kcut, klbl)
            self.play(*anims, run_time=self.fit(1.2))
            self.play(*[Create(m) for m in marks[0::4]], *[FadeIn(m) for m in marks[1::4]],
                      run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.4)
            self.play(*[Create(m) for m in marks[2::4]], *[FadeIn(m) for m in marks[3::4]],
                      run_time=self.fit(1.0))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(groups), FadeOut(marks), run_time=self.fit(0.6))

    # ── S06 默认配置 ─────────────────────────────────────────────────────
    def s06(self):
        with self.shot("S06"):
            self.play(*self.set_heading("标准做法：温度 + top-p"), run_time=self.fit(0.8))
            rows = [("模型（generation_config.json）", "温度", "top-p", "top-k"),
                    ("Qwen3-8B", "0.6", "0.95", "20"),
                    ("Llama-3.1-8B-Instruct", "0.6", "0.9", "—"),
                    ("SmolLM3-3B", "0.6", "0.95", "—"),
                    ("Gemma-3-27B-it", "—", "0.95", "64")]
            table = VGroup()
            for r, row in enumerate(rows):
                col = theme.MUTED if r == 0 else theme.FG
                cells = VGroup(zh(row[0], 22, col) if r == 0 else mono(row[0], 22, col),
                               *[zh(x, 22, col) if r == 0 else mono(x, 22, col) for x in row[1:]])
                for j, (cell, x) in enumerate(zip(cells, (-4.6, -1.2, 0.0, 1.2))):
                    cell.move_to([x, 1.8 - r * 0.55, 0])
                    if j == 0:
                        cell.align_to([-6.6, 0, 0], LEFT)
                table.add(cells)
            self.play(LaggedStart(*[FadeIn(r) for r in table], lag_ratio=0.3), run_time=self.fit(2.5))
            self.wait(self.remaining() * 0.35)
            names = ["贪心 (T=0)", "T=0.5", "T=1.0", "T=1.0, top-p=0.9", "T=1.5"]
            short = ["贪心", "T0.5", "T1.0", "top-p", "T1.5"]
            vals = [D["distinct"][n] for n in names]
            g = bars(vals, short, width=0.5, max_h=2.2, vmax=1.0, color=theme.OUTPUT, gap=0.25,
                     label_size=18, base_y=-2.2)
            g[0][0].set_fill(theme.GRAD)
            nums = VGroup(*[mono(f"{v:.2f}", 18).next_to(r, UP, 0.06) for v, r in zip(vals, g[0])])
            chart = VGroup(g, nums).move_to([4.5, -0.6, 0])
            cap = zh("不重复 4-gram 占比（小模型实测）", 20, theme.MUTED).next_to(chart, UP, 0.2)
            if cap.get_right()[0] > 6.8:
                cap.shift(LEFT * (cap.get_right()[0] - 6.8))
            self.play(FadeIn(chart), FadeIn(cap), run_time=self.fit(1.5))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(table), FadeOut(chart), FadeOut(cap), run_time=self.fit(0.6))

    # ── S07 重复计算的三角形 ─────────────────────────────────────────────
    def s07(self):
        with self.shot("S07"):
            self.play(*self.set_heading("朴素生成：每一步都从头重算"), run_time=self.fit(0.8))
            P, N, s = 4, 12, 0.3
            cells = []
            grid = VGroup()
            for step in range(N):
                row = VGroup()
                for pos in range(P + step):
                    r = Rectangle(width=s * 0.9, height=s * 0.9, stroke_width=0,
                                  fill_color=theme.INPUT, fill_opacity=0.85)
                    r.move_to([-5.5 + pos * s, 2.2 - step * s, 0])
                    row.add(r)
                    cells.append((step, pos, r))
                grid.add(row)
            xl = zh("位置 →", 20, theme.MUTED).next_to(grid, UP, 0.1).align_to(grid, LEFT)
            yl = zh("步 ↓", 20, theme.MUTED).next_to(grid, LEFT, 0.15).align_to(grid, UP)
            note = zh("示意：提示词 4 个，生成 12 个", 20, theme.MUTED).next_to(grid, DOWN, 0.2)\
                .align_to(grid, LEFT)
            self.play(FadeIn(xl), FadeIn(yl), run_time=self.fit(0.5))
            self.play(LaggedStart(*[FadeIn(r) for r in grid], lag_ratio=0.15),
                      run_time=self.fit(3))
            self.play(FadeIn(note), run_time=self.fit(0.5))
            last = D["speed"][-1]
            txt = VGroup(
                zh(f"真实：提示词 {D['prompt_len']} 个字符，生成 {last['n']} 个", 26, theme.FG),
                zh(f"朴素版共处理 {last['proc_naive']:,} 个位置", 28, theme.GRAD),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.2, 1.2, 0])
            self.play(FadeIn(txt), run_time=self.fit(1.0))
            self.wait(self.remaining() * 0.4)
            anims = []
            for step, pos, r in cells:
                if pos == P + step - 1:
                    anims.append(r.animate.set_fill(theme.HIGHLIGHT))
                else:
                    anims.append(r.animate.set_fill(theme.MUTED, opacity=0.35))
            self.play(*anims, run_time=self.fit(1.2))
            rep = VGroup(zh("灰：过去位置，结果不会变 → 重复劳动", 24, theme.MUTED),
                         zh("黄：每步真正新的位置", 24, theme.HIGHLIGHT)).arrange(
                DOWN, aligned_edge=LEFT, buff=0.25).move_to([3.2, -0.8, 0])
            self.play(FadeIn(rep), run_time=self.fit(0.8))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(grid, xl, yl, note, txt, rep)), run_time=self.fit(0.6))

    # ── S08 KV cache 原理 ────────────────────────────────────────────────
    def s08(self):
        with self.shot("S08"):
            self.play(*self.set_heading("KV cache：把算过的 K、V 存起来"), run_time=self.fit(0.8))
            s = 0.46
            k_lbl = MathTex("K", font_size=38, color=theme.INPUT).move_to([-6.3, 0.6, 0])
            v_lbl = MathTex("V", font_size=38, color=theme.OUTPUT).move_to([-6.3, -0.3, 0])
            cache_title = zh("缓存（每层一份，只存 K 和 V）", 22, theme.MUTED).move_to([-3.6, -1.0, 0])

            def cell(i, row_y, color):
                return Rectangle(width=s * 0.9, height=s * 0.9, stroke_color=color, stroke_width=2,
                                 fill_color=color, fill_opacity=0.35).move_to([-5.7 + i * s, row_y, 0])

            n0 = 6
            ks = VGroup(*[cell(i, 0.6, theme.INPUT) for i in range(n0)])
            vs = VGroup(*[cell(i, -0.3, theme.OUTPUT) for i in range(n0)])
            self.play(FadeIn(k_lbl), FadeIn(v_lbl), FadeIn(cache_title), FadeIn(ks), FadeIn(vs),
                      run_time=self.fit(1.2))
            info = VGroup(
                zh("每步只喂 1 个新 token：", 24, theme.FG),
                zh("① 算它自己的 q、k、v", 24, theme.FG),
                zh("② k、v 追加到缓存末尾", 24, theme.FG),
                zh("③ q 和缓存里全部 K 做注意力", 24, theme.ATTN),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([3.7, 1.0, 0])
            self.play(FadeIn(info[0]), run_time=self.fit(0.6))
            steps = 3
            per = (self.remaining() - 4.0) / steps
            for t in range(steps):
                i = n0 + t
                x = -5.7 + i * s
                tok = token_box("?", theme.HIGHLIGHT, 0.46).move_to([x, 2.3, 0])
                q = MathTex("q", font_size=34, color=theme.HIGHLIGHT).move_to([x, 1.55, 0])
                nk, nv = cell(i, 0.6, theme.HIGHLIGHT), cell(i, -0.3, theme.HIGHLIGHT)
                self.play(FadeIn(tok, shift=DOWN * 0.2), *([FadeIn(info[1])] if t == 0 else []),
                          run_time=self.fit(per * 0.25))
                self.play(FadeIn(q), FadeIn(nk, shift=DOWN * 0.3), FadeIn(nv, shift=DOWN * 0.3),
                          *([FadeIn(info[2])] if t == 0 else []), run_time=self.fit(per * 0.25))
                lines = VGroup(*[Line(q.get_bottom(), c.get_top(), color=theme.ATTN, stroke_width=1.5,
                                      stroke_opacity=0.7) for c in list(ks) + [nk]])
                self.play(Create(lines), *([FadeIn(info[3])] if t == 0 else []),
                          run_time=self.fit(per * 0.3))
                self.play(FadeOut(lines), FadeOut(tok), FadeOut(q),
                          nk.animate.set_color(theme.INPUT).set_fill(opacity=0.35),
                          nv.animate.set_color(theme.OUTPUT).set_fill(opacity=0.35),
                          run_time=self.fit(per * 0.2))
                ks.add(nk)
                vs.add(nv)
            last = D["speed"][-1]
            res = zh(f"生成 {last['n']} 个：只处理 {last['proc_cache']} 个位置"
                     f"（朴素版 {last['proc_naive']:,}）", 26, theme.HIGHLIGHT).move_to([0, -2.2, 0])
            noq = zh("不缓存 Q：它只在当前这一步用一次", 22, theme.MUTED).move_to([3.7, -1.0, 0])
            self.play(FadeIn(res), run_time=self.fit(0.8))
            self.play(FadeIn(noq), run_time=self.fit(0.6))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(ks, vs, k_lbl, v_lbl, cache_title, info, res, noq)),
                      run_time=self.fit(0.6))

    # ── S09 实测速度 ─────────────────────────────────────────────────────
    def s09(self):
        with self.shot("S09"):
            self.play(*self.set_heading("实测：缓存版快多少（单线程 CPU）"), run_time=self.fit(0.8))
            same = zh("对拍：贪心与采样，缓存版与朴素版逐字相同 ✓" if D["same"] else "对拍失败",
                      24, theme.OUTPUT).move_to([0, 2.4, 0])
            self.play(FadeIn(same), run_time=self.fit(0.8))
            sp = D["speed"]
            vmax = max(r["naive"] for r in sp)
            base = -2.0
            group = VGroup()
            for i, r in enumerate(sp):
                x = -4.5 + i * 3.0
                hn, hc = r["naive"] / vmax * 3.4, max(0.03, r["cached"] / vmax * 3.4)
                bn = Rectangle(width=0.8, height=hn, stroke_width=0, fill_color=theme.GRAD,
                               fill_opacity=0.85).move_to([x - 0.45, base + hn / 2, 0])
                bc = Rectangle(width=0.8, height=hc, stroke_width=0, fill_color=theme.OUTPUT,
                               fill_opacity=0.9).move_to([x + 0.45, base + hc / 2, 0])
                tn = mono(f"{r['naive']:.2f}s", 18).next_to(bn, UP, 0.06)
                tc = mono(f"{r['cached']:.2f}s", 18).next_to(bc, UP, 0.06)
                lab = zh(f"生成 {r['n']} 个", 20, theme.MUTED).move_to([x, base - 0.3, 0])
                spd = zh(f"{r['speedup']:.1f}×", 30, theme.HIGHLIGHT).move_to(
                    [x + 0.45, max(bc.get_top()[1] + 0.75, base + 1.0), 0])
                group.add(VGroup(bn, bc, tn, tc, lab, spd))
            legend = VGroup(
                VGroup(Rectangle(width=0.3, height=0.2, fill_color=theme.GRAD, fill_opacity=0.85,
                                 stroke_width=0), zh("朴素（整段重算）", 20)).arrange(RIGHT, buff=0.12),
                VGroup(Rectangle(width=0.3, height=0.2, fill_color=theme.OUTPUT, fill_opacity=0.9,
                                 stroke_width=0), zh("KV cache", 20)).arrange(RIGHT, buff=0.12),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([-5.2, 1.4, 0])
            self.play(FadeIn(legend), run_time=self.fit(0.5))
            self.wait(self.remaining() * 0.15)
            per = (self.remaining() - 1.5) / len(group)
            for g in group:
                self.play(GrowFromEdge(g[0], DOWN), GrowFromEdge(g[1], DOWN), FadeIn(g[4]),
                          run_time=self.fit(per * 0.4))
                self.play(FadeIn(g[2]), FadeIn(g[3]), FadeIn(g[5]), run_time=self.fit(per * 0.3))
                self.wait(per * 0.3)
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(group, legend, same)), run_time=self.fit(0.6))

    # ── S10 prefill 与 decode ────────────────────────────────────────────
    def s10(self):
        with self.shot("S10"):
            self.play(*self.set_heading("两个阶段：prefill 与 decode"), run_time=self.fit(0.8))
            pd = D["pd"]
            scale = 11.0 / (pd["decode"] * 1000)
            x_left = -5.6
            p_w = max(0.12, pd["prefill"] * 1000 * scale)
            pre = Rectangle(width=p_w, height=0.7, stroke_width=0, fill_color=theme.INPUT,
                            fill_opacity=0.9).move_to([x_left + p_w / 2, 1.2, 0])
            pre_l = zh(f"prefill：{pd['n']} 个位置一次并行算完", 24, theme.INPUT).move_to([0, 2.2, 0])
            pre_t = mono(f"{pd['prefill'] * 1000:.1f} ms", 22).next_to(pre, RIGHT, 0.2)
            self.play(FadeIn(pre_l), GrowFromEdge(pre, LEFT), FadeIn(pre_t), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.25)
            n_seg = 64
            d_w = pd["decode"] * 1000 * scale
            segs = VGroup(*[Rectangle(width=d_w / n_seg * 0.8, height=0.7, stroke_width=0,
                                      fill_color=theme.PARAM, fill_opacity=0.9).move_to(
                [x_left + (k + 0.5) * d_w / n_seg, -0.6, 0]) for k in range(n_seg)])
            dec_l = zh(f"decode：{pd['n']} 个位置，一次一个", 24, theme.PARAM).move_to([0, 0.4, 0])
            dec_t = mono(f"{pd['decode'] * 1000:.0f} ms", 22).next_to(segs, DOWN, 0.2).align_to(
                segs, RIGHT)
            self.play(FadeIn(dec_l), LaggedStart(*[FadeIn(s) for s in segs], lag_ratio=0.05),
                      run_time=self.fit(3))
            self.play(FadeIn(dec_t), run_time=self.fit(0.5))
            ratio = zh(f"吞吐相差 {pd['prefill_tps'] / pd['decode_tps']:.0f} 倍 → 推理服务把很多请求拼成一批一起 decode",
                       24, theme.HIGHLIGHT).move_to([0, -2.2, 0])
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(ratio), run_time=self.fit(0.8))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(pre, pre_l, pre_t, segs, dec_l, dec_t, ratio)),
                      run_time=self.fit(0.6))

    # ── S11 显存账 ───────────────────────────────────────────────────────
    def s11(self):
        with self.shot("S11"):
            self.play(*self.set_heading("代价：KV cache 占显存"), run_time=self.fit(0.8))
            m = D["main"]
            f = MathTex(r"\text{bytes}=2\times L\times n_{kv}\times d_{head}\times T\times b",
                        font_size=42).move_to([0, 2.3, 0])
            self.play(Write(f), run_time=self.fit(1.5))
            per_tok = kv_bytes(m["L"], m["Hkv"], m["D"], 1)
            sub = MathTex(rf"2\times{m['L']}\times{m['Hkv']}\times{m['D']}\times 2"
                          rf"={per_tok:,}".replace(",", "{,}"), font_size=38).move_to([-1.3, 1.3, 0])
            sub_r = zh(f"= {per_tok // 1024} KiB / token", 30, theme.HIGHLIGHT).next_to(sub, RIGHT, 0.3)
            who = zh(f"主线模型：{m['L']} 层、{m['Hkv']} 个 KV 头、head_dim {m['D']}、BF16", 22,
                     theme.MUTED).move_to([0, 0.6, 0])
            self.wait(self.remaining() * 0.15)
            self.play(FadeIn(who), Write(sub), run_time=self.fit(1.5))
            self.play(FadeIn(sub_r), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.2)
            w = 689.5e6 * 2 / GIB
            kv32 = kv_bytes(m["L"], m["Hkv"], m["D"], 32768) / GIB
            scale = 7.0 / kv32
            items = [("模型权重（689.5M 参数）", w, theme.PARAM),
                     ("一条 32K 对话的 KV cache", kv32, theme.GRAD)]
            rows = VGroup()
            for i, (name, v, col) in enumerate(items):
                y = -0.4 - i * 0.9
                lab = zh(name, 22, theme.FG).move_to([-4.3, y, 0])
                bar = Rectangle(width=v * scale, height=0.5, stroke_width=0, fill_color=col,
                                fill_opacity=0.9).move_to([-2.2 + v * scale / 2, y, 0])
                num = mono(f"{v:.2f} GiB", 22).next_to(bar, RIGHT, 0.15)
                rows.add(VGroup(lab, bar, num))
            for r in rows:
                self.play(FadeIn(r[0]), GrowFromEdge(r[1], LEFT), FadeIn(r[2]), run_time=self.fit(1.0))
            b16 = zh(f"同时服务 16 条 32K 对话：{kv32 * 16:.0f} GiB", 24, theme.HIGHLIGHT).move_to(
                [3.0, -2.3, 0])
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(b16), run_time=self.fit(0.8))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(f, sub, sub_r, who, rows, b16)), run_time=self.fit(0.6))

    # ── S12 GQA 思路 ─────────────────────────────────────────────────────
    def s12(self):
        with self.shot("S12"):
            self.play(*self.set_heading("GQA：几个查询头共享一组 K、V"), run_time=self.fit(0.8))
            nq = 8
            cols = VGroup()
            for cx, name, nkv in ((-4.6, "MHA", 8), (0.0, "GQA", 2), (4.6, "MQA", 1)):
                qs = VGroup(*[Rectangle(width=0.36, height=0.36, stroke_width=0,
                                        fill_color=theme.HIGHLIGHT, fill_opacity=0.85)
                              for _ in range(nq)]).arrange(RIGHT, buff=0.1).move_to([cx, 1.2, 0])
                kvs = VGroup(*[Rectangle(width=0.36, height=0.36, stroke_width=0,
                                         fill_color=theme.INPUT, fill_opacity=0.9)
                               for _ in range(nkv)]).arrange(RIGHT, buff=0.1 + (nq - nkv) * 0.46 / max(nkv, 1)
                                                            if nkv > 1 else 0.1).move_to([cx, -0.3, 0])
                links = VGroup(*[Line(q.get_bottom(), kvs[i * nkv // nq].get_top(), color=theme.MUTED,
                                      stroke_width=1.5) for i, q in enumerate(qs)])
                title = zh(f"{name}：{nkv} 组 K/V", 24, theme.FG).move_to([cx, 2.1, 0])
                cols.add(VGroup(title, qs, kvs, links))
            ql = zh("黄：查询头 Q　蓝：K/V 头（示意：8 个查询头）", 20, theme.MUTED).move_to([0, -1.0, 0])
            for c in cols:
                self.play(FadeIn(c[0]), FadeIn(c[1]), FadeIn(c[2]), Create(c[3]),
                          run_time=self.fit(1.2))
                self.wait(self.remaining() * 0.12)
            self.play(FadeIn(ql), run_time=self.fit(0.5))
            m = D["main"]
            vals = [(f"MHA {m['Hq']} 头", m["Hq"]), (f"GQA {m['Hkv']} 头（主线）", m["Hkv"]),
                    ("MQA 1 头", 1)]
            bars_g = VGroup()
            mx = kv_bytes(m["L"], m["Hq"], m["D"], 32768) / GIB
            for i, (name, h) in enumerate(vals):
                v = kv_bytes(m["L"], h, m["D"], 32768) / GIB
                y = -1.55 - i * 0.42
                lab = zh(name, 20, theme.FG).move_to([-4.6, y, 0])
                w = max(0.05, v / mx * 6.5)
                bar = Rectangle(width=w, height=0.3, stroke_width=0, fill_color=theme.INPUT,
                                fill_opacity=0.9).move_to([-2.7 + w / 2, y, 0])
                num = mono(f"{v:.2f} GiB", 20).next_to(bar, RIGHT, 0.15)
                bars_g.add(VGroup(lab, bar, num))
            cap = zh("主线模型 32K 上下文", 20, theme.MUTED).move_to([4.6, -2.1, 0])
            self.wait(self.remaining() * 0.2)
            self.play(FadeOut(ql), FadeIn(cap), *[FadeIn(b) for b in bars_g], run_time=self.fit(1.5))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(cols, bars_g, cap)), run_time=self.fit(0.6))

    # ── S13 GQA 小实验 ───────────────────────────────────────────────────
    def s13(self):
        with self.shot("S13"):
            self.play(*self.set_heading("小实验：同样训练 600 步"), run_time=self.fit(0.8))
            hdr = ["方案", "KV 头", "验证 loss", "KV cache", "注意力参数"]
            xs = [-5.0, -3.0, -0.8, 1.9, 4.8]
            table = VGroup(VGroup(*[zh(h, 22, theme.MUTED).move_to([x, 1.9, 0])
                                    for h, x in zip(hdr, xs)]))
            base = D["gqa"][0]["cache"]
            for i, r in enumerate(D["gqa"]):
                y = 1.2 - i * 0.65
                vals = [r["name"], str(r["kv"]), f"{r['val']:.3f}",
                        f"{r['cache'] / 1024:.0f} KiB ({r['cache'] / base:.2f}×)", f"{r['attn']:,}"]
                table.add(VGroup(*[mono(v, 22).move_to([x, y, 0]) for v, x in zip(vals, xs)]))
            self.play(FadeIn(table[0]), run_time=self.fit(0.6))
            for row in table[1:]:
                self.play(FadeIn(row), run_time=self.fit(0.8))
                self.wait(self.remaining() * 0.12)
            noise = D["gqa_noise"]
            diff = abs(noise - D["gqa"][0]["val"])
            nz = zh(f"对照：MHA 只换随机种子重训，loss {noise:.3f}（差 {diff:.3f}）", 24,
                    theme.HIGHLIGHT).move_to([0, -1.2, 0])
            self.wait(self.remaining() * 0.25)
            self.play(FadeIn(nz), run_time=self.fit(0.8))
            concl = zh("缓存按 KV 头数成比例缩小；loss 差别要和种子噪声比着看", 24,
                       theme.FG).move_to([0, -2.0, 0])
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(concl), run_time=self.fit(0.8))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(table, nz, concl)), run_time=self.fit(0.6))

    # ── S14 公开模型 ─────────────────────────────────────────────────────
    def s14(self):
        with self.shot("S14"):
            self.play(*self.set_heading("主流开源模型都在用 GQA"), run_time=self.fit(0.8))
            pick = ["Llama-3.3-70B", "Qwen3-8B", "Llama-3.1-8B", "Qwen2.5-7B", "gpt-oss-20b"]
            pub = {r[0]: r for r in D["public"]}
            mx = max(kv_bytes(pub[n][1], pub[n][2], pub[n][4], 32768) for n in pick) / GIB
            rows = VGroup()
            for i, n in enumerate(pick):
                _, L, hq, hkv, dh, _ = pub[n]
                mha = kv_bytes(L, hq, dh, 32768) / GIB
                gqa = kv_bytes(L, hkv, dh, 32768) / GIB
                y = 1.9 - i * 0.82
                lab = mono(n, 20).move_to([-5.3, y + 0.05, 0])
                sub = zh(f"{hq} Q 头 / {hkv} KV 头", 16, theme.MUTED).move_to([-5.3, y - 0.27, 0])
                wm, wg = mha / mx * 7.0, max(0.05, gqa / mx * 7.0)
                bm = Rectangle(width=wm, height=0.24, stroke_width=0, fill_color=theme.MUTED,
                               fill_opacity=0.5).move_to([-3.4 + wm / 2, y + 0.13, 0])
                bg = Rectangle(width=wg, height=0.24, stroke_width=0, fill_color=theme.INPUT,
                               fill_opacity=0.95).move_to([-3.4 + wg / 2, y - 0.15, 0])
                nm = mono(f"{mha:g}", 16, theme.MUTED).next_to(bm, RIGHT, 0.1)
                ng = mono(f"{gqa:.2f} GiB", 16).next_to(bg, RIGHT, 0.1)
                rows.add(VGroup(lab, sub, bm, bg, nm, ng))
            leg = zh("灰：若不共享（MHA）　蓝：实际配置　32K 上下文、BF16、按公式计", 18,
                     theme.MUTED).move_to([0.8, -2.3, 0])
            self.play(FadeIn(leg), run_time=self.fit(0.5))
            per = (self.remaining() * 0.6) / len(rows)
            for r in rows:
                self.play(FadeIn(r[0]), FadeIn(r[1]), GrowFromEdge(r[2], LEFT), GrowFromEdge(r[3], LEFT),
                          FadeIn(r[4]), FadeIn(r[5]), run_time=self.fit(per))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(VGroup(rows, leg)), run_time=self.fit(0.6))

    # ── S15 从极简到生产级 ───────────────────────────────────────────────
    def s15(self):
        with self.shot("S15"):
            self.play(*self.set_heading("从极简到生产级"), run_time=self.fit(0.8))
            src = """# zero/generate.py
nxt = sample_next(logits, temperature, top_p)
logits = model(nxt, kv_cache=cache, start_pos=pos)
# zero/kv_cache.py  预分配，不拼接
k = zeros(n_layers, B, n_kv_heads, max_len, head_dim)
# zero/model.py  GQA 不复制 K/V
F.scaled_dot_product_attention(q, k, v,
    enable_gqa=n_kv_heads != n_heads)"""
            code = code_block(src, 19).to_edge(LEFT, buff=0.5).shift(UP * 0.6)
            for i in (0, 3, 5):
                code[i].set_color(theme.MUTED)
            self.play(LaggedStart(*[FadeIn(ln) for ln in code], lag_ratio=0.2),
                      run_time=self.fit(3))
            test = zh("tests/test_kv_cache.py：7 项通过", 22, theme.OUTPUT)
            test2 = zh("缓存 = 不缓存（贪心、采样、YaRN）", 20, theme.OUTPUT)
            vllm = zh("上线服务：vLLM", 24, theme.HIGHLIGHT)
            vllm2 = zh("PagedAttention + 连续批处理", 20, theme.HIGHLIGHT)
            side = VGroup(test, test2, vllm, vllm2).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
            side[2].shift(DOWN * 0.3)
            side[3].shift(DOWN * 0.3)
            side.to_edge(RIGHT, buff=0.4).shift(UP * 0.4)
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(side[:2]), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(side[2:]), run_time=self.fit(0.8))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.play(FadeOut(code), FadeOut(side), run_time=self.fit(0.6))

    # ── S16 小结与下一章 ─────────────────────────────────────────────────
    def s16(self):
        with self.shot("S16"):
            self.play(*self.set_heading("小结"), run_time=self.fit(0.8))
            items = [("生成循环", "算分布 → 挑一个 → 接上", theme.FG),
                     ("温度 + top-p", "在稳和活之间取舍", theme.OUTPUT),
                     ("KV cache", "只算新 token，O(T²) → O(T)", theme.INPUT),
                     ("GQA", "共享 K/V 头，缓存成比例缩小", theme.ATTN)]
            boxes = VGroup()
            for name, desc, col in items:
                b = VGroup(zh(name, 28, col), zh(desc, 20, theme.MUTED)).arrange(DOWN, buff=0.15)
                r = SurroundingRectangle(b, buff=0.25, color=col, corner_radius=0.1)
                boxes.add(VGroup(r, b))
            boxes.arrange(RIGHT, buff=0.3).move_to([0, 0.9, 0])
            if boxes.width > 13.6:
                boxes.scale_to_fit_width(13.6)
            self.play(LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.5),
                      run_time=self.fit(4))
            self.wait(self.remaining() * 0.5)
            nxt = zh("下一章：评测——先定考卷", 32, theme.HIGHLIGHT).move_to([0, -1.3, 0])
            self.play(FadeIn(nxt, shift=UP * 0.2), run_time=self.fit(1.0))
            self.wait(max(0.1, self.remaining() - 0.8))
            self.play(FadeOut(boxes), FadeOut(nxt), *self.set_heading(None), run_time=self.fit(0.8))
