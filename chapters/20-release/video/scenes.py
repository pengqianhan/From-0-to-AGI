"""Video for Chapter 20: release.

Hand in the exam as preregistered, and put the model on a laptop.

Sources of the numbers on screen (see the fact list in script.md):
- Quantization error, values of the first block, loss table of the small model, memory calculation:
  the scripts in ../code/ calculate them. The cache video/out/cache.json keeps the results.
- Decision table, export results: we read video/data/smoke_before_fix.json. This file is the real output
  of the smoke test before the fix of the tool-call grader (`uv run python -m zero.smoke`,
  tiny-configuration demo). It is the same run as in README Section 1.3.
  A new smoke test overwrites out/smoke (after the fix, all 4 rows are "tie").
  Thus we freeze the output here. The json file gives the source.
- Parity check with llama.cpp: we measured it when we wrote this chapter (README section
  "From minimal code to production code"). LLAMA_CHECK below keeps the result without changes.
Render: bash chapters/20-release/video/build.sh
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
    Axes,
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
    RoundedRectangle,
    SurroundingRectangle,
    Text,
    VGroup,
    Write,
)

from video_kit import theme
from video_kit.scene import NarratedScene, zh

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
CACHE = HERE / "out" / "cache.json"
MONO = "Noto Sans Mono"

# Result that we measured with out/smoke/gguf and llama.cpp (commit 81bc6b8)
# when we wrote this chapter. The README has the full record.
LLAMA_CHECK = {
    "prompt": "3 * (4 + 5) 等于多少？",
    "n_ids": 20,
    "rows": [
        ("zero（fp32）", '<tool_call>↵{"name":233 = 2 = 14 = 24 = 18)', True),
        ("GGUF f16", '<tool_call>↵{"name":233 = 2 = 14 = 24 = 18)', True),
        ("GGUF Q8_0", '<tool_call>↵{"name":233 = 2 = 14 = 24 = 18)', True),
        ("更低位（Q5_0+Q8_0 混合）", '<tool_call>↵{"name":233 =）。', False),
    ],
}


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def compute() -> dict:
    import numpy as np

    q = _load("ch20_q", "01_blockwise_quant.py")
    w = q.make_weight()
    x = np.random.default_rng(1).normal(size=(64, w.shape[1])).astype(np.float32)
    d: dict = {"block": q.show_block(w), "table": q.table(w, x)}
    d["first32"] = w[0, :32].tolist()

    calc = _load("ch20_calc", "03_memory_calculator.py")
    m = calc.load_model_section(calc.DEFAULT_CFG)
    d["params"] = sum(k for _, _, k in calc.tensors(m))
    d["sizes"] = {f: calc.file_bytes(m, f) for f in ("bf16", "q8_0", "q4_k_m")}
    d["kv32k"] = calc.kv_cache_bytes(m, 32768)

    tm = _load("ch20_tiny", "02_quantize_tiny_model.py")
    d["tiny"] = tm.run()
    d["tiny_params"] = sum(p.numel() for p in tm.tiny.load_or_train(4).parameters())

    frozen = json.loads((HERE / "data" / "smoke_before_fix.json").read_text(encoding="utf-8"))
    d["comparisons"] = [c for c in frozen["comparisons"] if c["task"] in ("toy_mc", "tool_dev")]
    d["smoke"] = {"export (HF + GGUF)": frozen["export"]}
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


def mono(text: str, size: float = 24, color: str = theme.FG) -> Text:
    return Text(text, font=MONO, font_size=size, color=color)


def box(label: str, w: float, h: float, color: str, size: float = 24) -> VGroup:
    r = RoundedRectangle(width=w, height=h, corner_radius=0.12, stroke_color=color,
                         fill_color=color, fill_opacity=0.12)
    t = zh(label, size, theme.FG)
    if t.width > w - 0.2:
        t.scale_to_fit_width(w - 0.2)
    return VGroup(r, t.move_to(r))


DECISION_COLOR = {"超过": theme.OUTPUT, "持平": theme.HIGHLIGHT, "落后": theme.GRAD}


class ChapterScene(NarratedScene):
    chapter_label = "第 20 章"
    chapter_title = "发布"

    def construct(self) -> None:
        for i in range(1, 15):
            getattr(self, f"s{i:02d}")()

    def clear_all(self, *mobs, badge=None):
        items = list(mobs) + ([badge] if badge is not None else [])
        self.play(*[FadeOut(m) for m in items], run_time=self.fit(0.6))

    # ── S01 Opening ──────────────────────────────────────────────────────
    def s01(self):
        with self.shot("S01"):
            card = self.chapter_card()
            sub = zh("按预注册交卷，把模型装进笔记本", 32, theme.HIGHLIGHT).next_to(card, DOWN, 0.6)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.wait(max(0.1, self.remaining() - 0.8))
            self.play(FadeOut(card), FadeOut(sub), run_time=self.fit(0.8))

    # ── S02 Pipeline ─────────────────────────────────────────────────────
    def s02(self):
        with self.shot("S02"):
            self.play(*self.set_heading("从 checkpoint 到笔记本"), run_time=self.fit(0.8))
            specs = [("checkpoint", theme.MUTED), ("HF 目录\nsafetensors", theme.INPUT),
                     ("GGUF\nQ8_0 / Q4_K_M", theme.PARAM), ("笔记本\nllama.cpp / Ollama", theme.OUTPUT)]
            xs = [-5.2, -1.75, 1.75, 5.2]
            boxes = VGroup(*[box(t, 2.8, 1.3, c, 24).move_to([x, -0.6, 0])
                             for (t, c), x in zip(specs, xs)])
            arrows = VGroup(*[Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.08,
                                    color=theme.MUTED, stroke_width=3) for i in range(3)])
            ev = box("最终评测（闸门 3）", 3.4, 0.9, theme.ATTN, 24).move_to([-1.75, 1.7, 0])
            card = box("模型卡", 2.4, 0.9, theme.HIGHLIGHT, 24).move_to([1.75, 1.7, 0])
            a1 = Arrow(boxes[1].get_top(), ev.get_bottom(), buff=0.08, color=theme.ATTN,
                       stroke_width=3)
            a2 = Arrow(ev.get_right(), card.get_left(), buff=0.08, color=theme.ATTN, stroke_width=3)
            for i in range(4):
                self.play(FadeIn(boxes[i]), *( [Create(arrows[i - 1])] if i else []),
                          run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.3)
            self.play(Create(a1), FadeIn(ev), run_time=self.fit(0.9))
            self.play(Create(a2), FadeIn(card), run_time=self.fit(0.9))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.clear_all(boxes, arrows, ev, card, a1, a2)

    # ── S03 Gate 3 ───────────────────────────────────────────────────────
    def s03(self):
        with self.shot("S03"):
            self.play(*self.set_heading("闸门 3：照着考卷跑，一个字不改"), run_time=self.fit(0.8))
            lines = ["基准与版本", "评测框架与版本", "模板、解码参数", "对手清单、冻结日期", "“超过”的判定标准"]
            items = VGroup(*[zh("· " + t, 24) for t in lines]).arrange(DOWN, aligned_edge=LEFT,
                                                                       buff=0.22)
            title = zh("eval/PREREGISTRATION.md", 22, theme.MUTED)
            grp = VGroup(title, items).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
            frame = SurroundingRectangle(grp, buff=0.3, color=theme.INPUT, corner_radius=0.1)
            card = VGroup(frame, grp).move_to([-4.3, 0.75, 0])
            stamp = zh("训练前冻结", 30, theme.GRAD).rotate(0.12).next_to(frame, DOWN, 0.35)
            stamp_box = SurroundingRectangle(stamp, buff=0.12, color=theme.GRAD)
            self.play(FadeIn(card), run_time=self.fit(1.0))
            self.play(FadeIn(VGroup(stamp, stamp_box), scale=1.4), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.25)

            x0, scale = 2.4, 9.0  # Number line: difference d → x
            axis = Line([-0.3, -1.9, 0], [6.8, -1.9, 0], color=theme.MUTED)
            zero = DashedLine([x0, 2.0, 0], [x0, -1.9, 0], color=theme.FG)
            zlab = MathTex("0", font_size=34).next_to(zero, DOWN, 0.1)
            self.play(Create(axis), Create(zero), FadeIn(zlab), run_time=self.fit(0.8))
            intervals = [(0.03, 0.27, "超过：下界 > 0", 1.3), (-0.10, 0.12, "持平：跨过 0", 0.2),
                         (-0.35, -0.08, "落后：上界 < 0", -0.9)]
            objs = VGroup()
            for lo, hi, text, y in intervals:
                col = DECISION_COLOR[text[:2]]
                seg = Line([x0 + lo * scale, y, 0], [x0 + hi * scale, y, 0], color=col,
                           stroke_width=8)
                ends = VGroup(*[Line([x0 + v * scale, y - 0.15, 0], [x0 + v * scale, y + 0.15, 0],
                                     color=col, stroke_width=4) for v in (lo, hi)])
                lab = zh(text, 22, col).next_to(seg, UP, 0.12)
                if lo < 0 < hi:  # The interval crosses 0: put the label on the right, off the 0 line
                    lab.next_to(seg, RIGHT, 0.2)
                objs.add(VGroup(seg, ends, lab))
                self.play(Create(seg), FadeIn(ends), FadeIn(lab), run_time=self.fit(0.8))
            note = zh("95% 置信区间（配对 bootstrap）", 22, theme.MUTED).move_to([4.9, -2.35, 0])
            self.play(FadeIn(note), run_time=self.fit(0.5))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.clear_all(card, stamp, stamp_box, axis, zero, zlab, objs, note)

    # ── S04 Decision table (tiny-configuration demo) ─────────────────────
    def s04(self):
        with self.shot("S04"):
            self.play(*self.set_heading("一张判定表：全部列出"), run_time=self.fit(0.8))
            badge = self.show_badge()
            head = ["模型", "对照", "任务", "差值", "95% CI", "判定"]
            xs = [-5.6, -4.0, -2.2, 0.2, 2.6, 5.4]
            hdr = VGroup(*[zh(h, 24, theme.MUTED).move_to([x, 1.9, 0]) for h, x in zip(head, xs)])
            rule = Line([-6.5, 1.55, 0], [6.5, 1.55, 0], color=theme.MUTED, stroke_width=1.5)
            self.play(FadeIn(hdr), Create(rule), run_time=self.fit(0.8))
            rows = VGroup()
            task_name = {"toy_mc": "选择题", "tool_dev": "工具调用"}
            for i, c in enumerate(D["comparisons"]):
                y = 1.0 - i * 0.75
                col = DECISION_COLOR[c["decision"]]
                cells = [mono(c["model"], 24), mono(c["baseline"], 24),
                         zh(task_name[c["task"]], 24),
                         mono(f"{c['diff']:+.3f}", 24),
                         mono(f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]", 22),
                         zh(c["decision"], 26, col)]
                row = VGroup(*[m.move_to([x, y, 0]) for m, x in zip(cells, xs)])
                rows.add(row)
            self.play(LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.3),
                      run_time=self.fit(2.0))
            win = next(i for i, c in enumerate(D["comparisons"]) if c["decision"] == "超过")
            hl = SurroundingRectangle(rows[win][5], color=theme.OUTPUT, buff=0.1)
            self.play(Create(hl), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.3)
            tool = next(i for i, c in enumerate(D["comparisons"])
                        if c["model"] == D["comparisons"][win]["model"] and c["task"] == "tool_dev")
            hl2 = SurroundingRectangle(rows[tool], color=theme.HIGHLIGHT, buff=0.1)
            msg = zh("硬目标是工具调用：这一行是“持平”，不能只报上面那一格", 24,
                     theme.HIGHLIGHT).move_to([0, -2.3, 0])
            self.play(Create(hl2), FadeIn(msg), run_time=self.fit(0.8))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.clear_all(hdr, rule, rows, hl, hl2, msg, badge=badge)

    # ── S05 Standard format ──────────────────────────────────────────────
    def s05(self):
        with self.shot("S05"):
            self.play(*self.set_heading("标准格式：别人的工具直接能用"), run_time=self.fit(0.8))
            badge = self.show_badge()
            files = [("config.json", "架构 Qwen3ForCausalLM、层数、RoPE/YaRN"),
                     ("model.safetensors", "头 + 原始字节，不能藏代码"),
                     ("tokenizer.json", "分词器"),
                     ("tokenizer_config.json", "chat template（与训练逐字一致）"),
                     ("generation_config.json", "结束符 <|im_end|>")]
            rows = VGroup()
            for name, desc in files:
                rows.add(VGroup(mono(name, 22, theme.INPUT), zh(desc, 20, theme.MUTED)).arrange(
                    DOWN, aligned_edge=LEFT, buff=0.06))
            rows.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([-3.3, 0.35, 0])
            frame = SurroundingRectangle(rows, buff=0.25, color=theme.INPUT, corner_radius=0.1)
            self.play(Create(frame), LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.2),
                      run_time=self.fit(2.0))
            fws = VGroup(*[box(t, 2.6, 0.75, theme.OUTPUT, 24) for t in
                           ("transformers", "vLLM", "llama.cpp")]).arrange(DOWN, buff=0.35)
            fws.move_to([4.4, 0.9, 0])
            arrs = VGroup(*[Arrow(frame.get_right(), f.get_left(), buff=0.1, color=theme.MUTED,
                                  stroke_width=3) for f in fws])
            self.play(LaggedStart(*[Create(a) for a in arrs], lag_ratio=0.2), FadeIn(fws),
                      run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.35)
            ex = D["smoke"]["export (HF + GGUF)"]
            proof = VGroup(zh(f"logits 最大误差 {ex['hf_logits_maxdiff']}", 22, theme.OUTPUT),
                           zh("chat template 逐字一致" if ex["hf_template_identical"] else "模板不一致",
                              22, theme.OUTPUT)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            proof.move_to([4.4, -1.6, 0])
            self.play(FadeIn(proof), run_time=self.fit(0.8))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.clear_all(frame, rows, fws, arrs, proof, badge=badge)

    # ── S06 Memory = parameters × bits ───────────────────────────────────
    def s06(self):
        with self.shot("S06"):
            self.play(*self.set_heading("内存 ≈ 参数量 × bit 数"), run_time=self.fit(0.8))
            f = MathTex(r"\text{memory} \approx N \times \frac{\text{bits}}{8}", font_size=44)
            f.move_to([2.8, 2.3, 0])
            n = zh(f"主线模型 {D['params'] / 1e6:.1f}M 参数（暂定形状）", 24, theme.MUTED)
            n.move_to([-3.2, 2.3, 0])
            self.play(Write(f), FadeIn(n), run_time=self.fit(1.2))
            items = [("bf16", D["sizes"]["bf16"], theme.INPUT, "16 bit"),
                     ("Q8_0", D["sizes"]["q8_0"], theme.PARAM, "8.5 bit"),
                     ("Q4_K_M", D["sizes"]["q4_k_m"], theme.OUTPUT, "≈5 bit"),
                     ("KV cache 32K", D["kv32k"], theme.GRAD, "fp16")]
            vmax, x0, wmax = D["kv32k"], -4.2, 8.6
            bars = VGroup()
            for i, (name, b, col, bits) in enumerate(items):
                y = 1.2 - i * 0.9
                w = max(0.05, b / vmax * wmax)
                r = Rectangle(width=w, height=0.55, stroke_width=0, fill_color=col,
                              fill_opacity=0.85 if i < 3 else 0.35)
                r.move_to([x0 + w / 2, y, 0])
                lab = mono(name, 22, theme.FG).next_to([x0, y, 0], LEFT, 0.2)
                val = zh(f"{b / GIB:.2f} GiB（{bits}）", 22, col).next_to(r, RIGHT, 0.15)
                if val.get_right()[0] > 6.9:
                    val.next_to(r.get_right(), LEFT, 0.15).set_color(theme.FG)
                bars.add(VGroup(r, lab, val))
            for i in range(3):
                self.play(GrowFromEdge(bars[i][0], LEFT), FadeIn(bars[i][1]), FadeIn(bars[i][2]),
                          run_time=self.fit(0.9))
            self.wait(self.remaining() * 0.35)
            self.play(GrowFromEdge(bars[3][0], LEFT), FadeIn(bars[3][1]), FadeIn(bars[3][2]),
                      run_time=self.fit(1.0))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.clear_all(f, n, bars)

    # ── S07 Block-wise quantization ──────────────────────────────────────
    def s07(self):
        with self.shot("S07"):
            self.play(*self.set_heading("分块量化：每 32 个数一个 scale"), run_time=self.fit(0.8))
            w = D["first32"]
            amax = max(abs(v) for v in w)
            cells = VGroup()
            for i, v in enumerate(w):
                col = theme.INPUT if v >= 0 else theme.GRAD
                sq = Rectangle(width=0.32, height=0.5, stroke_color=theme.MUTED, stroke_width=1,
                               fill_color=col, fill_opacity=0.15 + 0.85 * abs(v) / amax)
                sq.move_to([-5.1 + i * 0.33, 1.6, 0])
                cells.add(sq)
            blab = zh("一块：32 个权重（颜色深浅 = 大小）", 22, theme.MUTED).next_to(cells, UP, 0.15)
            sc = box("scale（fp16）", 2.4, 0.55, theme.PARAM, 22).next_to(cells, RIGHT, 0.2)
            if sc.get_right()[0] > 7.0:
                sc.move_to([5.9, 2.4, 0])
            self.play(LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=0.02), FadeIn(blab),
                      run_time=self.fit(1.2))
            self.play(FadeIn(sc), run_time=self.fit(0.5))
            eqs = VGroup(MathTex(r"s=\frac{\max|w|}{q_{\max}}", font_size=38),
                         MathTex(r"q=\mathrm{round}(w/s)", font_size=38),
                         MathTex(r"\hat w = s\cdot q", font_size=38)).arrange(RIGHT, buff=0.8)
            eqs.move_to([0, 0.45, 0])
            self.play(Write(eqs), run_time=self.fit(1.5))
            self.wait(self.remaining() * 0.2)
            b = D["block"]
            xs = [-4.9 + j * 1.5 for j in range(8)]
            r_w = VGroup(zh("w", 22, theme.MUTED).move_to([-6.3, -0.55, 0]),
                         *[mono(f"{v:+.4f}", 19).move_to([x, -0.55, 0]) for v, x in zip(b["w"], xs)])
            r8 = VGroup(zh("INT8 q", 22, theme.PARAM).move_to([-6.3, -1.2, 0]),
                        *[mono(str(v), 21, theme.PARAM).move_to([x, -1.2, 0])
                          for v, x in zip(b["int8"]["q"], xs)])
            r4 = VGroup(zh("INT4 q", 22, theme.GRAD).move_to([-6.3, -1.85, 0]),
                        *[mono(str(v), 21, theme.GRAD).move_to([x, -1.85, 0])
                          for v, x in zip(b["int4"]["q"], xs)])
            self.play(FadeIn(r_w), run_time=self.fit(0.6))
            self.play(FadeIn(r8), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(r4), run_time=self.fit(0.6))
            zeros = VGroup(*[SurroundingRectangle(r4[1 + j], color=theme.HIGHLIGHT, buff=0.06)
                             for j, v in enumerate(b["int4"]["q"]) if v == 0])
            note = zh("INT4 只有 15 个格子：小数被压成 0", 22, theme.HIGHLIGHT).move_to([0, -2.4, 0])
            self.play(Create(zeros), FadeIn(note), run_time=self.fit(0.8))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.clear_all(cells, blab, sc, eqs, r_w, r8, r4, zeros, note)

    # ── S08 Why blocks ───────────────────────────────────────────────────
    def s08(self):
        with self.shot("S08"):
            self.play(*self.set_heading("为什么分块：一个离群值撑大整组 scale"), run_time=self.fit(0.8))
            t = {r["scheme"]: r for r in D["table"]}
            items = [("整张一个 scale", "int4-tensor", theme.GRAD),
                     ("每行一个 scale", "int4-row", theme.PARAM),
                     ("每 32 个一块", "int4-block32", theme.OUTPUT),
                     ("两级 scale（Q4_K 思路）", "int4-kquant", theme.OUTPUT)]
            x0, wmax = -2.0, 7.4
            cap = zh("INT4 的权重相对误差（1024×1024，含 0.1% 离群值）", 22, theme.MUTED)
            cap.move_to([0.3, 2.3, 0])
            self.play(FadeIn(cap), run_time=self.fit(0.6))
            bars = VGroup()
            for i, (name, key, col) in enumerate(items):
                y = 1.3 - i * 0.95
                e = t[key]["w_err"]
                wdt = max(0.05, e * wmax)
                r = Rectangle(width=wdt, height=0.6, stroke_width=0, fill_color=col,
                              fill_opacity=0.85).move_to([x0 + wdt / 2, y, 0])
                lab = zh(name, 22).next_to([x0, y, 0], LEFT, 0.2)
                val = mono(f"{e:.1%}  ({t[key]['bpw']:.3g} bit)", 22, col).next_to(r, RIGHT, 0.15)
                if val.get_right()[0] > 6.9:
                    val.next_to(r.get_right(), LEFT, 0.15).set_color(theme.BG)
                bars.add(VGroup(r, lab, val))
                self.play(GrowFromEdge(r, LEFT), FadeIn(lab), FadeIn(val),
                          run_time=self.fit(0.9, reserve=1.0))
                self.wait(self.remaining() * 0.12)
            note = zh("分块多花 0.5 bit 存 scale，误差降一个数量级", 24, theme.HIGHLIGHT)
            note.move_to([0.3, -2.35, 0])
            self.play(FadeIn(note), run_time=self.fit(0.6))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.clear_all(cap, bars, note)

    # ── S09 The blocks of GGUF ───────────────────────────────────────────
    def s09(self):
        with self.shot("S09"):
            self.play(*self.set_heading("GGUF 的块：Q8_0 与 Q4_K"), run_time=self.fit(0.8))

            def strip(parts, unit, y, title):
                g = VGroup()
                x = -6.0
                for nbytes, col, lab in parts:
                    w = nbytes * unit
                    r = Rectangle(width=w, height=0.6, stroke_color=theme.BG, stroke_width=1,
                                  fill_color=col, fill_opacity=0.8).move_to([x + w / 2, y, 0])
                    t = zh(lab or " ", 18, theme.FG).next_to(r, DOWN, 0.1)
                    if t.get_left()[0] < -6.9:
                        t.align_to(r, LEFT)
                    g.add(VGroup(r, t))
                    x += w
                head = zh(title, 24, theme.FG).move_to([-6.0, y + 0.65, 0], aligned_edge=LEFT)
                return VGroup(head, g)

            q8 = strip([(2, theme.PARAM, "scale 2B"), (32, theme.INPUT, "32 个 int8 = 32B")],
                       0.3, 1.2, "Q8_0：32 个数 → 34 字节 = 8.5 bit/权重")
            q4 = strip([(4, theme.PARAM, ""), (12, theme.ATTN, "d,dmin 4B + 子块 scale/min 12B"),
                        (128, theme.INPUT, "256 个 4 bit 整数 = 128B")],
                       0.07, -0.6, "Q4_K：256 个数 → 144 字节 = 4.5 bit/权重")
            self.play(FadeIn(q8), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.25)
            self.play(FadeIn(q4), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.3)
            m = zh("Q4_K_M：输出层、部分 attn_v / ffn_down 用 Q6_K（6.56 bit），其余 Q4_K", 22,
                   theme.HIGHLIGHT).move_to([0, -1.9, 0])
            chk = zh("计算器 vs llama-quantize 实测：张量字节数逐字节一致", 22, theme.OUTPUT)
            chk.move_to([0, -2.4, 0])
            self.play(FadeIn(m), run_time=self.fit(0.8))
            self.play(FadeIn(chk), run_time=self.fit(0.6))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.clear_all(q8, q4, m, chk)

    # ── S10 Trade quality for size ───────────────────────────────────────
    def s10(self):
        with self.shot("S10"):
            self.play(*self.set_heading("质量换大小：在小模型上量一量"), run_time=self.fit(0.8))
            rows = {r["scheme"]: r for r in D["tiny"]}
            ref = rows["fp32"]["loss"]
            cap = zh(f"第 10 章的小模型（{D['tiny_params'] / 1e6:.2f}M 参数），验证集", 22,
                     theme.MUTED).move_to([-3.4, 2.35, 0])
            sel = [("fp32", "fp32"), ("INT8 分块", "int8-block32"), ("INT4 分块", "int4-block32"),
                   ("INT4 两级", "int4-kquant"), ("INT3 分块", "int3-block32"),
                   ("INT2 分块", "int2-block32")]
            xs = [-5.6, -3.7, -2.1, -0.6]
            hdr = VGroup(*[zh(h, 20, theme.MUTED).move_to([x, 1.8, 0])
                           for h, x in zip(["方案", "KiB", "Δloss", "一致率"], xs)])
            tbl = VGroup()
            for i, (name, key) in enumerate(sel):
                r = rows[key]
                y = 1.25 - i * 0.55
                dl = r["loss"] - ref
                col = theme.GRAD if dl > 0.03 else (theme.HIGHLIGHT if dl > 0.001 else theme.OUTPUT)
                tbl.add(VGroup(zh(name, 20).move_to([xs[0], y, 0]),
                               mono(f"{r['kib']:.0f}", 20).move_to([xs[1], y, 0]),
                               mono(f"{dl:+.4f}", 20, col).move_to([xs[2], y, 0]),
                               mono(f"{r['agree']:.1%}", 20).move_to([xs[3], y, 0])))
            self.play(FadeIn(cap), FadeIn(hdr), run_time=self.fit(0.6))
            self.play(LaggedStart(*[FadeIn(t) for t in tbl], lag_ratio=0.25),
                      run_time=self.fit(2.0))
            # The data sets the top of the y-axis. The INT2 Δloss is 0.51–0.58
            # on base models that different machines trained.
            top = max(0.55, max(r["loss"] for r in D["tiny"]) - ref + 0.08)
            ax = Axes(x_range=[2, 8, 1], y_range=[0, top, 0.1], x_length=4.6, y_length=3.2,
                      axis_config={"color": theme.MUTED, "include_numbers": True,
                                   "font_size": 20}, tips=False).move_to([4.3, 0.2, 0])
            xl = zh("bit 数（分块 32）", 18, theme.MUTED).next_to(ax, DOWN, 0.15)
            yl = zh("Δloss", 18, theme.MUTED).next_to(ax, UP, 0.1).shift(LEFT * 1.9)
            pts = [(b, rows[k]["loss"] - ref) for b, k in
                   ((2, "int2-block32"), (3, "int3-block32"), (4, "int4-block32"), (8, "int8-block32"))]
            line = VGroup()
            dots = VGroup(*[Dot(ax.coords_to_point(x, max(0.0, y)), color=theme.GRAD, radius=0.07)
                            for x, y in pts])
            for a, b in zip(dots[:-1], dots[1:]):
                line.add(Line(a.get_center(), b.get_center(), color=theme.GRAD, stroke_width=3))
            self.wait(self.remaining() * 0.25)
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=self.fit(0.8))
            self.play(FadeIn(dots), Create(line), run_time=self.fit(1.0))
            ref8b = zh("Llama 3 8B（llama.cpp 文档）：f16 6.233 · Q8_0 6.234 · Q4_K_M 6.407", 20,
                       theme.MUTED).move_to([0, -2.4, 0])
            self.play(FadeIn(ref8b), run_time=self.fit(0.6))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.clear_all(cap, hdr, tbl, ax, xl, yl, dots, line, ref8b)

    # ── S11 Parity check ─────────────────────────────────────────────────
    def s11(self):
        with self.shot("S11"):
            self.play(*self.set_heading("对拍：导出的还是同一个模型吗"), run_time=self.fit(0.8))
            badge = self.show_badge()
            top = zh(f"提示词“{LLAMA_CHECK['prompt']}”：llama-tokenize 的 {LLAMA_CHECK['n_ids']} 个 "
                     "token id 与我们的分词器完全相同 ✓", 22, theme.OUTPUT).move_to([0, 2.2, 0])
            self.play(FadeIn(top), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.15)
            sub = zh("贪心生成 24 个 token", 22, theme.MUTED).move_to([0, 1.5, 0])
            self.play(FadeIn(sub), run_time=self.fit(0.4))
            rows = VGroup()
            for i, (name, text, ok) in enumerate(LLAMA_CHECK["rows"]):
                y = 0.8 - i * 0.75
                col = theme.OUTPUT if ok else theme.GRAD
                lab = zh(name, 20, col).move_to([-6.8, y, 0], aligned_edge=LEFT)
                txt = mono(text, 18, theme.FG).move_to([-3.0, y, 0], aligned_edge=LEFT)
                mark = zh("逐字相同" if ok else "第 10 个 token 分叉", 20, col)
                mark.move_to([6.9, y, 0], aligned_edge=RIGHT)
                rows.add(VGroup(lab, txt, mark))
            for r in rows[:3]:
                self.play(FadeIn(r), run_time=self.fit(0.7, reserve=2.0))
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(rows[3]), run_time=self.fit(0.7))
            note = zh("输出是乱码（1.3M 参数），对拍只关心：两边算的是同一个东西", 22,
                      theme.HIGHLIGHT).move_to([0, -2.4, 0])
            self.play(FadeIn(note), run_time=self.fit(0.6))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.clear_all(top, sub, rows, note, badge=badge)

    # ── S12 Run locally and demo ─────────────────────────────────────────
    def s12(self):
        with self.shot("S12"):
            self.play(*self.set_heading("本地运行：笔记本与服务器"), run_time=self.fit(0.8))
            lap = VGroup(box("笔记本", 2.6, 0.7, theme.OUTPUT, 26),
                         zh("llama.cpp / Ollama", 22), zh("吃 GGUF（Q4_K_M）", 20, theme.MUTED)
                         ).arrange(DOWN, buff=0.15).move_to([-3.6, 1.6, 0])
            srv = VGroup(box("服务器", 2.6, 0.7, theme.INPUT, 26),
                         zh("vLLM", 22), zh("safetensors + hermes 工具解析器", 20, theme.MUTED)
                         ).arrange(DOWN, buff=0.15).move_to([3.6, 1.6, 0])
            self.play(FadeIn(lap), FadeIn(srv), run_time=self.fit(1.0))
            self.wait(self.remaining() * 0.35)
            badge = self.show_badge()
            lines = VGroup(
                mono("$ python -m zero.demo.cli --once ...", 18,
                     theme.MUTED),
                zh("你：3 * (4 + 5) 等于多少？", 22),
                zh("助手：（空）", 22, theme.GRAD),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
            term = VGroup(SurroundingRectangle(lines, buff=0.25, color=theme.MUTED,
                                               corner_radius=0.08), lines).move_to([-1.2, -1.3, 0])
            side = VGroup(zh("生成 → 解析 → 执行 → 喂回：链路已通", 20, theme.OUTPUT),
                          zh("1.3M 参数：还不会用工具", 20, theme.GRAD),
                          zh("search_files 只在 --root 之内", 20, theme.HIGHLIGHT)
                          ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            term.move_to([-3.2, -1.3, 0])
            side.next_to(term, RIGHT, 0.4)
            if side.get_right()[0] > 6.9:
                side.shift(LEFT * (side.get_right()[0] - 6.9))
            self.play(FadeIn(term), run_time=self.fit(1.0))
            self.play(FadeIn(side), run_time=self.fit(0.8))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.clear_all(lap, srv, term, side, badge=badge)

    # ── S13 Model card and license ───────────────────────────────────────
    def s13(self):
        with self.shot("S13"):
            self.play(*self.set_heading("模型卡与许可证"), run_time=self.fit(0.8))
            secs = ["YAML：license / language / datasets", "训练数据与许可证", "各阶段配方与花费",
                    "预注册", "全部评测结果（含落后）", "去污染检查", "已知局限"]
            items = VGroup(*[zh(s, 22) for s in secs]).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
            todo = zh("没有来源的数 → “待训练”", 22, theme.HIGHLIGHT)
            card = VGroup(items, todo).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
            frame = SurroundingRectangle(card, buff=0.25, color=theme.HIGHLIGHT, corner_radius=0.1)
            left = VGroup(frame, card).move_to([-3.5, 0.2, 0])
            self.play(Create(frame), LaggedStart(*[FadeIn(i) for i in items], lag_ratio=0.15),
                      run_time=self.fit(2.0))
            self.play(FadeIn(todo), run_time=self.fit(0.5))
            self.wait(self.remaining() * 0.2)
            lic = VGroup(zh("权重许可证（作者决定）", 22, theme.MUTED),
                         zh("Apache-2.0：Qwen3、OLMo 2、SmolLM3", 21),
                         zh("MIT", 21),
                         zh("社区许可证：Llama、Gemma", 21),
                         zh("数据署名义务（照办）", 22, theme.MUTED),
                         zh("ODC-By：FineWeb-Edu 等", 21, theme.OUTPUT),
                         zh("CC-BY-4.0：DCLM", 21, theme.OUTPUT),
                         ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([3.6, 0.3, 0])
            self.play(LaggedStart(*[FadeIn(x) for x in lic], lag_ratio=0.2),
                      run_time=self.fit(2.0))
            self.wait(self.remaining() * 0.4)
            ck = zh("中间 checkpoint 一起发布：读者可从我们的 Base 开始", 22, theme.INPUT)
            ck.move_to([0, -2.4, 0])
            self.play(FadeIn(ck), run_time=self.fit(0.6))
            self.wait(max(0.1, self.remaining() - 0.6))
            self.clear_all(left, lic, ck)

    # ── S14 After the release ────────────────────────────────────────────
    def s14(self):
        with self.shot("S14"):
            self.play(*self.set_heading("发布之后：收反馈，不夸大"), run_time=self.fit(0.8))
            pts = VGroup(zh("1. 收集失败的例子：输入、工具定义、输出", 26),
                         zh("2. 复现不一致：先对齐框架版本、模板、解码参数", 26),
                         zh("3. 只在“超过”的格子上说“超过”", 26, theme.HIGHLIGHT)
                         ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to([0, 0.6, 0])
            for p in pts:
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=self.fit(0.8, reserve=2.0))
                self.wait(self.remaining() * 0.15)
            nxt = zh("下一章：KV cache 的账本", 30, theme.INPUT).move_to([0, -1.8, 0])
            self.play(FadeIn(nxt), run_time=self.fit(0.8))
            self.wait(max(0.1, self.remaining() - 0.8))
            self.play(FadeOut(pts), FadeOut(nxt), *self.set_heading(None), run_time=self.fit(0.8))
