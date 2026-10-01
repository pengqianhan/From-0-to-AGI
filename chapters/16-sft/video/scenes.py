"""第 16 章视频：SFT —— 对话模板、loss mask、打包

画面里的数值由 ../code/ 中的代码真实计算（见 script.md 事实清单）；S13 的冒烟测试数字读自
video/data/smoke_before_fix.json：修复工具调用判分器之前那次冒烟测试的真实输出，与 README「主线进度」同一次运行
（重跑冒烟测试会覆盖 out/smoke，所以冻结在这里，来源写在 json 里）。
较慢的计算（小模型 SFT、生成）结果缓存在 video/out/cache.json；删掉它会重新计算。
渲染：bash chapters/16-sft/video/build.sh
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
    Axes,
    Create,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    LaggedStart,
    MathTex,
    Rectangle,
    RoundedRectangle,
    SurroundingRectangle,
    Text,
    VGroup,
    Write,
)

from video_kit import theme
from video_kit.scene import NarratedScene, polyline_in_axes, zh

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
CACHE = HERE / "out" / "cache.json"
MONO = "Noto Sans Mono"

ROLE_COLOR = {"system": theme.MUTED, "user": theme.INPUT, "assistant": theme.OUTPUT,
              "tool": theme.ATTN}


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def compute() -> dict:
    """从 ../code 真实计算视频要用的全部数字。"""
    tmpl = _load("ch16_chat_template", "01_chat_template.py")
    sft = _load("ch16_sft_tiny", "03_sft_tiny.py")
    pack = _load("ch16_packing", "04_packing.py")
    d: dict = {}
    segs = tmpl.render(tmpl.MESSAGES, tmpl.TOOLS)
    d["segs"] = [[s, r, t] for s, r, t in segs]
    text = "".join(s for s, _, _ in segs)
    d["n_chars"] = len(text)
    d["n_train"] = sum(len(s) for s, _, t in segs if t)
    d["sft"] = sft.run(log=lambda _: None)
    d["pack"] = pack.run()
    frozen = json.loads((HERE / "data" / "smoke_before_fix.json").read_text(encoding="utf-8"))
    d["smoke"] = {"sft": frozen["sft"], "eval": frozen["eval"]}
    d["smoke_samples"] = [frozen["sample"]]
    return d


def get_data() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    d = compute()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


D = get_data()
S = D["sft"]
P = D["pack"]


def mono(text: str, size: float = 20, color: str = theme.FG) -> Text:
    return Text(text, font=MONO, font_size=size, color=color)


def show_ws(s: str) -> str:
    return s.replace("\n", "↵")


def sample_lines() -> list[tuple[str, str, bool]]:
    """把真实样本压缩成适合上屏的若干行：(文本, 角色, 是否算 loss)。system 的工具说明缩成 3 行。"""
    lines: list[tuple[str, str, bool]] = [
        ("<|im_start|>system", "system", False),
        ("你是一个会使用工具的助手……", "system", False),
        ("# Tools … <tools>…</tools> …<|im_end|>", "system", False),
    ]
    for s, role, train in D["segs"]:
        if role == "system":
            continue
        for ln in s.split("\n"):
            if ln.strip():
                cut = 44 if role in ("user", "assistant") and "response" not in ln else 38
                lines.append((ln if len(ln) <= cut else ln[: cut - 1] + "…", role, train))
    return lines


def role_panel(lines, x_left: float, y_top: float, size: float = 17) -> VGroup:
    g = VGroup()
    for i, (t, role, _train) in enumerate(lines):
        m = mono(t, size, ROLE_COLOR[role])
        m.move_to([x_left, y_top - i * 0.27, 0], aligned_edge=LEFT)
        g.add(m)
    return g


def table(rows, col_w, x0: float, y0: float, size: float = 22, row_h: float = 0.5,
          colors=None) -> VGroup:
    """简单表格：rows 是字符串二维表，第一行是表头。"""
    g = VGroup()
    for i, row in enumerate(rows):
        x = x0
        for j, cell in enumerate(row):
            c = theme.MUTED if i == 0 else (colors[i][j] if colors else theme.FG)
            t = zh(cell, size, c)
            t.move_to([x, y0 - i * row_h, 0], aligned_edge=LEFT)
            g.add(t)
            x += col_w[j]
    return g


class ChapterScene(NarratedScene):
    chapter_label = "第 16 章"
    chapter_title = "SFT"

    def construct(self) -> None:
        for i in range(1, 15):
            getattr(self, f"s{i:02d}")()

    def badge_in(self):
        return self.show_badge()

    # ── S01 片头 ─────────────────────────────────────────────────────────
    def s01(self):
        with self.shot("S01"):
            card = self.chapter_card()
            sub = zh("把只会续写的底座，教成会调工具的助手", 30, theme.HIGHLIGHT).next_to(card, DOWN, 0.6)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.wait(max(0.1, self.remaining() - 0.8))
            self.play(FadeOut(card), FadeOut(sub), run_time=self.fit(0.8))

    # ── S02 底座只会续写 ─────────────────────────────────────────────────
    def s02(self):
        with self.shot("S02"):
            self.play(*self.set_heading("底座只会续写"), run_time=self.fit(0.8))
            badge = self.badge_in()
            q = mono(S["question"], 26, theme.INPUT).move_to([0, 2.4, 0])
            self.play(FadeIn(q), run_time=self.fit(0.8))
            base_txt = S["base_continue"].replace("\n", "↵")
            left = VGroup(zh("底座（第 10 章，0.86M 参数）", 22, theme.GRAD),
                          mono(base_txt[:24], 18), mono(base_txt[24:48], 18),
                          mono(base_txt[48:72], 18)).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
            ex = next(e for e in S["examples"] if e["q"] == S["question"])
            rep = ex["reply"].split("\n")
            right = VGroup(zh("SFT 之后", 22, theme.OUTPUT),
                           *[mono(r_, 14) for r_ in rep]).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
            left.move_to([-4.4, 0.2, 0])
            right.move_to([2.6, 0.2, 0])
            box_l = SurroundingRectangle(left, color=theme.GRAD, buff=0.25)
            box_r = SurroundingRectangle(right, color=theme.OUTPUT, buff=0.25)
            self.play(FadeIn(left), Create(box_l), run_time=self.fit(1.5))
            self.wait(self.remaining() * 0.45)
            self.play(FadeIn(right), Create(box_r), run_time=self.fit(1.5))
            self.wait(max(0.1, self.remaining() - 0.8))
            self.play(*[FadeOut(m) for m in (q, left, right, box_l, box_r, badge)],
                      run_time=self.fit(0.8))

    # ── S03 同一个 loss ───────────────────────────────────────────────────
    def s03(self):
        with self.shot("S03"):
            self.play(*self.set_heading("同一个 loss，换一种数据"), run_time=self.fit(0.8))
            pre = MathTex(r"\mathcal{L}_{\mathrm{pre}} = -\frac{1}{T}\sum_{t=1}^{T}"
                          r"\log p_\theta(x_t \mid x_{<t})", font_size=40, color=theme.MUTED)
            sft = MathTex(r"\mathcal{L}_{\mathrm{SFT}} = -\frac{1}{|A|}\sum_{t\in A}"
                          r"\log p_\theta(x_t \mid x_{<t})", font_size=46)
            pre.move_to([0, 2.0, 0])
            sft.move_to([0, 0.7, 0])
            self.play(FadeIn(pre), run_time=self.fit(1.0))
            self.play(Write(sft), run_time=self.fit(1.5))
            parts = [("system", 2.8, False), ("user", 2.0, False), ("assistant 头", 2.0, False),
                     ("助手输出（A）", 3.0, True)]
            roles = ["system", "user", "assistant", "assistant"]
            strip = VGroup()
            x = -4.9
            for (label, w, train), role in zip(parts, roles):
                col = ROLE_COLOR[role]
                r = Rectangle(width=w, height=0.7, stroke_color=col, stroke_width=2,
                              fill_color=col, fill_opacity=0.55 if train else 0.12)
                r.move_to([x + w / 2, -1.0, 0])
                t = zh(label, 18, theme.FG).move_to(r)
                strip.add(VGroup(r, t))
                x += w + 0.1
            self.play(LaggedStart(*[FadeIn(s) for s in strip], lag_ratio=0.2), run_time=self.fit(1.5))
            note = zh("只有 A 里的位置参与求和", 24, theme.OUTPUT).move_to([2.3, -2.0, 0])
            self.play(FadeIn(note), run_time=self.fit(0.8))
            self.wait(max(0.1, self.remaining() - 0.8))
            self.play(*[FadeOut(m) for m in (pre, sft, strip, note)], run_time=self.fit(0.8))

    # ── S04 ChatML ────────────────────────────────────────────────────────
    def s04(self):
        with self.shot("S04"):
            self.play(*self.set_heading("ChatML：角色标记"), run_time=self.fit(0.8))
            msgs = [("system", "你是一个会使用工具的助手。"), ("user", "成都和广州今天哪个更热？"),
                    ("assistant", "广州更热（成都 21°C，广州 31°C）。")]
            blocks = VGroup()
            for i, (role, content) in enumerate(msgs):
                col = ROLE_COLOR[role]
                head = mono(f"<|im_start|>{role}", 22, col)
                body = VGroup(zh(content, 22, theme.FG), mono("<|im_end|>", 22, col)).arrange(RIGHT, buff=0.05)
                blk = VGroup(head, body).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
                bg = RoundedRectangle(width=7.6, height=blk.height + 0.3, corner_radius=0.1,
                                      stroke_color=col, stroke_width=2, fill_color=col, fill_opacity=0.08)
                blk.move_to(bg).align_to(bg, LEFT).shift(RIGHT * 0.25)
                g = VGroup(bg, blk).move_to([-1.6, 1.8 - i * 1.45, 0])
                blocks.add(g)
            for b in blocks:
                self.play(FadeIn(b, shift=RIGHT * 0.3), run_time=self.fit(0.9))
            fam = VGroup(zh("官方模板都用这套标记", 22, theme.MUTED),
                         *[mono(n, 22, theme.HIGHLIGHT) for n in
                           ("Qwen3", "Qwen3.5", "SmolLM3", "Hermes 3", "OLMo 3")]
                         ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([4.8, 0.6, 0])
            self.play(FadeIn(fam), run_time=self.fit(1.0))
            self.wait(max(0.1, self.remaining() - 0.8))
            self.play(FadeOut(blocks), FadeOut(fam), run_time=self.fit(0.8))

    # ── S05 工具调用样本 ─────────────────────────────────────────────────
    def s05(self):
        with self.shot("S05"):
            self.play(*self.set_heading("工具调用：一条真实样本"), run_time=self.fit(0.8))
            lines = sample_lines()
            split = next(i for i, (_, r, _) in enumerate(lines) if r == "tool")
            left = role_panel(lines[:split], -6.8, 2.5)
            right = role_panel(lines[split:], 0.6, 2.5)
            self.lines_obj = VGroup(*left, *right)
            self.lines_meta = lines[:split] + lines[split:]
            legend = VGroup(*[VGroup(Rectangle(width=0.3, height=0.2, fill_color=c, fill_opacity=1,
                                               stroke_width=0), zh(n, 18, c)).arrange(RIGHT, buff=0.1)
                              for n, c in (("system", theme.MUTED), ("user", theme.INPUT),
                                           ("assistant", theme.OUTPUT), ("工具返回", theme.ATTN))]
                            ).arrange(RIGHT, buff=0.4).move_to([2.8, -2.35, 0])
            self.legend = legend
            self.play(FadeIn(legend), run_time=self.fit(0.6))
            self.play(LaggedStart(*[FadeIn(m) for m in left], lag_ratio=0.15),
                      run_time=self.fit(self.remaining() * 0.4))
            self.play(LaggedStart(*[FadeIn(m) for m in right], lag_ratio=0.15),
                      run_time=self.fit(self.remaining() * 0.4))
            self.wait(max(0.1, self.remaining() - 0.2))

    # ── S06 loss mask ─────────────────────────────────────────────────────
    def s06(self):
        with self.shot("S06"):
            self.play(*self.set_heading("loss mask：只学助手的输出"), run_time=self.fit(0.8))
            dims, hi = [], VGroup()
            for m, (_, _role, train) in zip(self.lines_obj, self.lines_meta):
                if train:
                    hi.add(SurroundingRectangle(m, color=theme.HIGHLIGHT, buff=0.04, stroke_width=2))
                else:
                    dims.append(m.animate.set_opacity(0.25))
            self.play(*dims, FadeOut(self.legend), run_time=self.fit(1.2))
            self.play(Create(hi), run_time=self.fit(1.0))
            pct1 = 100 * D["n_train"] / D["n_chars"]
            pct2 = 100 * 67191 / 502756  # 冒烟测试全部 1500 条，用 zero 分词器编码（README 3.2）
            bars = VGroup()
            for i, (lab, p) in enumerate((("这一条", pct1), ("冒烟测试全部", pct2))):
                full = Rectangle(width=4.0, height=0.35, stroke_color=theme.MUTED, stroke_width=1.5)
                part = Rectangle(width=4.0 * p / 100, height=0.35, stroke_width=0,
                                 fill_color=theme.OUTPUT, fill_opacity=0.9)
                full.move_to([3.6, -0.9 - i * 0.75, 0])
                part.align_to(full, LEFT).align_to(full, DOWN)
                t = zh(f"{lab} {p:.1f}%", 20, theme.FG).next_to(full, UP, 0.08).align_to(full, LEFT)
                bars.add(VGroup(full, part, t))
            self.play(FadeIn(bars), run_time=self.fit(1.0))
            self.wait(max(0.1, self.remaining() - 0.8))
            self.play(FadeOut(self.lines_obj), FadeOut(hi), FadeOut(bars), run_time=self.fit(0.8))

    # ── S07 小实验结果 ───────────────────────────────────────────────────
    def s07(self):
        with self.shot("S07"):
            self.play(*self.set_heading("小实验：把小底座 SFT 成会调工具"), run_time=self.fit(0.8))
            self.badge_in()
            task = VGroup(zh("两个工具", 22, theme.MUTED),
                          mono("get_weather(city)", 20, theme.OUTPUT),
                          mono("add(a, b)", 20, theme.OUTPUT),
                          zh("测试城市训练中从未出现", 20, theme.HIGHLIGHT),
                          zh("300 步 · batch 8 · lr 1e-3", 20, theme.MUTED)
                          ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([-5.0, 0.6, 0])
            self.play(FadeIn(task), run_time=self.fit(1.0))
            be, m, u = S["base_eval"], S["masked"], S["unmasked"]
            rows = [["", "格式", "函数名", "参数", "验证 loss"],
                    ["底座", f"{be['format']:.2f}", f"{be['name']:.2f}", f"{be['args']:.2f}",
                     f"{S['base_val_asst_loss']:.3f}"],
                    ["有 mask", f"{m['eval']['format']:.2f}", f"{m['eval']['name']:.2f}",
                     f"{m['eval']['args']:.2f}", f"{m['val_asst_loss']:.3f}"],
                    ["无 mask", f"{u['eval']['format']:.2f}", f"{u['eval']['name']:.2f}",
                     f"{u['eval']['args']:.2f}", f"{u['val_asst_loss']:.3f}"]]
            colors = [[theme.FG] * 5] + [[theme.FG, theme.OUTPUT, theme.OUTPUT, theme.GRAD, theme.FG]] * 3
            colors[1] = [theme.FG, theme.GRAD, theme.GRAD, theme.GRAD, theme.FG]
            tab = table(rows, [1.6, 1.1, 1.3, 1.1, 1.4], -2.2, 1.8, size=24, row_h=0.75, colors=colors)
            self.play(LaggedStart(*[FadeIn(t) for t in tab], lag_ratio=0.03), run_time=self.fit(2.0))
            args_col = VGroup(*[tab[i * 5 + 3] for i in range(1, 4)])
            self.wait(self.remaining() * 0.3)
            self.play(Create(SurroundingRectangle(args_col, color=theme.GRAD, buff=0.12)),
                      run_time=self.fit(0.8))
            self.wait(max(0.1, self.remaining() - 0.8))
            self.play(*[FadeOut(x) for x in self.mobjects if x is not self._heading],
                      run_time=self.fit(0.8))

    # ── S08 难在参数 ──────────────────────────────────────────────────────
    def s08(self):
        with self.shot("S08"):
            self.play(*self.set_heading("loss 堆在参数上"), run_time=self.fit(0.8))
            badge = self.badge_in()
            sp = S["masked"]["split"]
            ax_base = -1.8
            vals = [(f"格式、函数名（{sp['n_rest']} 个）", sp["rest"], theme.OUTPUT),
                    (f"参数值（{sp['n_args']} 个）", sp["args"], theme.GRAD)]
            bars = VGroup()
            for i, (lab, v, c) in enumerate(vals):
                h = max(0.04, v / 1.3 * 3.4)
                r = Rectangle(width=1.4, height=h, stroke_width=0, fill_color=c, fill_opacity=0.9)
                r.move_to([-4.3 + i * 2.8, ax_base + h / 2, 0])
                num = mono(f"{v:.3f}", 24, c).next_to(r, UP, 0.12)
                lb = zh(lab, 18, theme.FG).next_to([-4.3 + i * 2.8, ax_base, 0], DOWN, 0.15)
                bars.add(VGroup(r, num, lb))
            base = Rectangle(width=5.6, height=0.01, stroke_color=theme.MUTED).move_to([-2.9, ax_base, 0])
            self.play(FadeIn(base), *[GrowFromEdge(b[0], DOWN) for b in bars],
                      *[FadeIn(b[1:]) for b in bars], run_time=self.fit(1.5))
            ex = next(e for e in S["examples"] if "Nairobi" in e["q"])
            q = mono(ex["q"], 20, theme.INPUT)
            city = json.loads(ex["reply"].split("\n")[1])["arguments"]["city"]
            out = mono(f'get_weather  {{"city": "{city}"}}', 18, theme.OUTPUT)
            grp = VGroup(zh("问", 20, theme.MUTED), q, zh("答", 20, theme.MUTED), out
                         ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([3.6, 0.6, 0])
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(grp), run_time=self.fit(1.0))
            hl = zh("没照抄：写了训练里常见的 Denver", 20, theme.GRAD).next_to(grp, DOWN, 0.35)
            self.play(FadeIn(hl), run_time=self.fit(0.8))
            self.wait(max(0.1, self.remaining() - 0.8))
            self.play(*[FadeOut(x) for x in (bars, base, grp, hl, badge)], run_time=self.fit(0.8))

    # ── S09 打包 ──────────────────────────────────────────────────────────
    def s09(self):
        with self.shot("S09"):
            self.play(*self.set_heading("打包：把几条对话装进一个窗口"), run_time=self.fit(0.8))
            W = P["window"]
            scale = 5.2 / W
            flat = [n for b in P["bins_preview"] for n in b]
            cols = [theme.INPUT, theme.OUTPUT, theme.PARAM, theme.ATTN]
            left, right = VGroup(), VGroup()
            for i, n in enumerate(flat[:6]):
                y = 1.9 - i * 0.55
                frame = Rectangle(width=W * scale, height=0.4, stroke_color=theme.MUTED, stroke_width=1)
                frame.move_to([-3.6, y, 0])
                seg = Rectangle(width=n * scale, height=0.4, stroke_width=0,
                                fill_color=cols[i % 4], fill_opacity=0.85)
                seg.align_to(frame, LEFT).align_to(frame, DOWN)
                left.add(VGroup(frame, seg))
            k = 0
            for i, b in enumerate(P["bins_preview"][:6]):
                y = 1.9 - i * 0.55
                frame = Rectangle(width=W * scale, height=0.4, stroke_color=theme.MUTED, stroke_width=1)
                frame.move_to([3.4, y, 0])
                row = VGroup(frame)
                x = frame.get_left()[0]
                for n in b:
                    seg = Rectangle(width=n * scale - 0.03, height=0.4, stroke_width=0,
                                    fill_color=cols[k % 4], fill_opacity=0.85)
                    seg.move_to([x + n * scale / 2, y, 0])
                    row.add(seg)
                    x += n * scale
                    k += 1
                right.add(row)
            tl = zh(f"不打包：真实 token {P['pad_pct']:.0f}%", 24, theme.GRAD).move_to([-3.6, 2.6, 0])
            tr = zh(f"首次适配打包：{P['pack_pct']:.0f}%", 24, theme.OUTPUT).move_to([3.4, 2.6, 0])
            self.play(FadeIn(tl), LaggedStart(*[FadeIn(r) for r in left], lag_ratio=0.1),
                      run_time=self.fit(1.5))
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(tr), LaggedStart(*[FadeIn(r) for r in right], lag_ratio=0.1),
                      run_time=self.fit(1.5))
            note = zh(f"{P['n']} 条对话 · 窗口 {W} · 一条对话绝不切开", 22, theme.MUTED).move_to([0, -1.75, 0])
            self.play(FadeIn(note), run_time=self.fit(0.8))
            self.wait(max(0.1, self.remaining() - 0.8))
            self.play(*[FadeOut(x) for x in (left, right, tl, tr, note)], run_time=self.fit(0.8))

    # ── S10 串门与文档 mask ───────────────────────────────────────────────
    def s10(self):
        with self.shot("S10"):
            self.play(*self.set_heading("串门：打包的代价"), run_time=self.fit(0.8))
            badge = self.badge_in()
            na, nb = 4, 6
            n = na + nb
            cell = 0.3

            def grid(doc_mask: bool, cx: float) -> VGroup:
                g = VGroup()
                for i in range(n):
                    for j in range(n):
                        if j > i:
                            col, op = theme.MUTED, 0.05
                        elif i >= na and j < na:
                            col, op = (theme.MUTED, 0.05) if doc_mask else (theme.GRAD, 0.85)
                        else:
                            col, op = (theme.INPUT if i < na else theme.OUTPUT), 0.7
                        r = Rectangle(width=cell, height=cell, stroke_color=theme.BG, stroke_width=1,
                                      fill_color=col, fill_opacity=op)
                        r.move_to([cx + (j - n / 2 + 0.5) * cell, 0.9 - (i - n / 2 + 0.5) * cell, 0])
                        g.add(r)
                return g

            g1, g2 = grid(False, -3.3), grid(True, 3.3)
            t1 = zh("普通因果 mask：B 能看到 A", 22, theme.GRAD).next_to(g1, UP, 0.25)
            t2 = zh("文档 mask：各看各的", 22, theme.OUTPUT).next_to(g2, UP, 0.25)
            self.play(FadeIn(g1), FadeIn(t1), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.25)
            nums = VGroup(
                zh(f"B 单独：{P['loss_alone']:.4f}", 22, theme.FG),
                zh(f"串门：{P['loss_mixed']:.4f}", 22, theme.GRAD),
                zh(f"隔离：{P['loss_isolated']:.4f}", 22, theme.OUTPUT),
            ).arrange(RIGHT, buff=0.8).move_to([0, -1.75, 0])
            self.play(FadeIn(nums[0]), FadeIn(nums[1]), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.25)
            self.play(FadeIn(g2), FadeIn(t2), FadeIn(nums[2]), run_time=self.fit(1.2))
            self.wait(max(0.1, self.remaining() - 0.8))
            self.play(*[FadeOut(x) for x in (g1, g2, t1, t2, nums, badge)], run_time=self.fit(0.8))

    # ── S11 过拟合 ────────────────────────────────────────────────────────
    def s11(self):
        with self.shot("S11"):
            self.play(*self.set_heading("过拟合：40 条数据训 60 个 epoch"), run_time=self.fit(0.8))
            badge = self.badge_in()
            ax = Axes(x_range=[0, 300, 50], y_range=[0, 0.6, 0.1], x_length=8.5, y_length=4.2,
                      axis_config={"color": theme.MUTED, "include_numbers": True, "font_size": 20},
                      tips=False).move_to([-0.6, 0.2, 0])
            xl = zh("步数", 20, theme.MUTED).next_to(ax.x_axis, DOWN, 0.35).shift(RIGHT * 3.8)
            hist = S["small40"]["hist"]
            tr = polyline_in_axes(ax, [(h["step"], h["train_loss"]) for h in hist],
                                  color=theme.INPUT, stroke_width=4)
            va = polyline_in_axes(ax, [(h["step"], h["val_asst_loss"]) for h in hist],
                                  color=theme.GRAD, stroke_width=4)
            last = hist[-1]
            lt = zh(f"训练 {last['train_loss']:.3f}", 22, theme.INPUT).next_to(
                ax.c2p(300, last["train_loss"]), RIGHT, 0.15)
            lv = zh(f"验证 {last['val_asst_loss']:.3f}", 22, theme.GRAD).next_to(
                ax.c2p(300, last["val_asst_loss"]), RIGHT, 0.15)
            self.play(Create(ax), FadeIn(xl), run_time=self.fit(1.0))
            self.play(Create(tr), Create(va), run_time=self.fit(2.0))
            self.play(FadeIn(lt), FadeIn(lv), run_time=self.fit(0.6))
            self.wait(max(0.1, self.remaining() - 0.8))
            self.play(*[FadeOut(x) for x in (ax, xl, tr, va, lt, lv, badge)], run_time=self.fit(0.8))

    # ── S12 数据与超参数 ─────────────────────────────────────────────────
    def s12(self):
        with self.shot("S12"):
            self.play(*self.set_heading("数据与超参数"), run_time=self.fit(0.8))
            data = [["数据集", "许可证"],
                    ["Tülu 3 SFT mixture", "ODC-BY（部分子集非商用）"],
                    ["xLAM 函数调用 60k", "CC-BY-4.0"],
                    ["ToolACE", "Apache-2.0"],
                    ["Hermes FC v1", "Apache-2.0"]]
            t1 = table(data, [3.0, 3.6], -6.8, 2.3, size=20, row_h=0.55)
            hp = [["", "学习率", "epoch"],
                  ["Llama 2", "2e-5", "2"],
                  ["Tülu 3 8B", "5e-6", "2"],
                  ["Tülu 3 70B", "2e-6", "2"]]
            t2 = table(hp, [1.9, 1.3, 1.0], 2.0, 2.3, size=20, row_h=0.55)
            self.play(FadeIn(t1), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(t2), run_time=self.fit(1.0))
            self.wait(self.remaining() * 0.4)
            lora = zh("LoRA：只训一对低秩矩阵 ΔW = B·A；主线 0.7B → 全参数微调", 22,
                      theme.HIGHLIGHT).move_to([0, -1.6, 0])
            self.play(FadeIn(lora), run_time=self.fit(0.8))
            self.wait(max(0.1, self.remaining() - 0.8))
            self.play(FadeOut(t1), FadeOut(t2), FadeOut(lora), run_time=self.fit(0.8))

    # ── S13 主线进度：极小配置演示 ───────────────────────────────────────
    def s13(self):
        with self.shot("S13"):
            self.play(*self.set_heading("主线进度：zero 冒烟测试"), run_time=self.fit(0.8))
            badge = self.badge_in()
            sm = D["smoke"]
            if sm is None:
                msg = zh("没有 out/smoke：待运行冒烟测试", 28, theme.GRAD)
                self.play(FadeIn(msg), run_time=self.fit(1.0))
                self.wait(max(0.1, self.remaining() - 0.8))
                self.play(FadeOut(msg), FadeOut(badge), run_time=self.fit(0.8))
                return
            s, e = sm["sft"], sm["eval"]
            cards = VGroup()
            for lab, val, c in (("SFT 步数", f"{s['steps']}", theme.FG),
                                ("训练 / 验证 loss", f"{s['loss']:.3f} / {s['val_loss']:.3f}", theme.FG),
                                ("格式正确", f"{e['format_ok']:.2f}", theme.OUTPUT),
                                ("调用完全正确", f"{e['call_exact']:.2f}", theme.GRAD)):
                box = RoundedRectangle(width=3.0, height=1.2, corner_radius=0.1,
                                       stroke_color=c, stroke_width=2)
                t = VGroup(zh(lab, 20, theme.MUTED), mono(val, 26, c)).arrange(DOWN, buff=0.12).move_to(box)
                cards.add(VGroup(box, t))
            cards.arrange(RIGHT, buff=0.3).move_to([0, 1.7, 0])
            self.play(LaggedStart(*[FadeIn(c) for c in cards], lag_ratio=0.2), run_time=self.fit(1.5))
            smp = D["smoke_samples"][0]
            gold = ", ".join(f"{g['name']}({g['arguments']['city']})" for g in smp["gold"])
            got_s = ", ".join(f"{g['name']}({g['arguments'].get('city')})" for g in smp["calls"])
            ex = VGroup(zh("问：" + smp["q"], 22, theme.INPUT),
                        zh("标准：" + gold, 22, theme.OUTPUT),
                        zh("模型：" + got_s, 22, theme.GRAD)
                        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([0, -0.6, 0])
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(ex), run_time=self.fit(1.2))
            tag = zh("约 1.3M 参数 · 格式全对，参数是编的 · 主线结果待 GPU 训练后补充", 20,
                     theme.MUTED).move_to([0, -2.2, 0])
            self.play(FadeIn(tag), run_time=self.fit(0.8))
            self.wait(max(0.1, self.remaining() - 0.8))
            self.play(*[FadeOut(x) for x in (cards, ex, tag, badge)], run_time=self.fit(0.8))

    # ── S14 结尾 ──────────────────────────────────────────────────────────
    def s14(self):
        with self.shot("S14"):
            self.play(*self.set_heading("小结"), run_time=self.fit(0.8))
            items = [("对话模板", "训练与推理逐字一致", theme.INPUT),
                     ("loss mask", "只算助手的输出", theme.OUTPUT),
                     ("打包", "省算力，但会串门", theme.PARAM)]
            cards = VGroup()
            for t, sub, c in items:
                box = RoundedRectangle(width=3.8, height=1.6, corner_radius=0.12, stroke_color=c,
                                       stroke_width=2.5)
                txt = VGroup(zh(t, 30, c), zh(sub, 20, theme.FG)).arrange(DOWN, buff=0.2).move_to(box)
                cards.add(VGroup(box, txt))
            cards.arrange(RIGHT, buff=0.4).move_to([0, 0.8, 0])
            self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in cards], lag_ratio=0.3),
                      run_time=self.fit(2.0))
            nxt = zh("下一章：蒸馏", 30, theme.HIGHLIGHT).move_to([0, -1.4, 0])
            self.wait(self.remaining() * 0.5)
            self.play(FadeIn(nxt), run_time=self.fit(0.8))
            self.wait(max(0.1, self.remaining() - 1.0))
            self.play(FadeOut(cards), FadeOut(nxt), *self.set_heading(None), run_time=self.fit(0.8))
