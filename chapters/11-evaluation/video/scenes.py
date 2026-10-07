"""第 11 章视频：评测：先定考卷 —— 考什么、怎么判、差多少才算赢

画面里的数值都由 ../code/ 中的代码真实计算（见 script.md 事实清单）；已核实的外部数字（基准题量、
BFCL 权重、Qwen3.5-0.8B 模型卡分数）写在本文件的常量里并注明出处。
较慢的模拟（04 的 simulations()，单线程约 10 秒）缓存到 video/out/cache.json。
渲染：bash chapters/11-evaluation/video/build.sh
"""

from __future__ import annotations

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
    Circle,
    Create,
    DashedLine,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    LaggedStart,
    Line,
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
CACHE_VERSION = 1


def _load(name: str):  # noqa: ANN202
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, CODE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=1)
def data() -> dict:
    """所有画面数字：优先读缓存，否则调用 code/ 里的函数现算。"""
    if CACHE.exists():
        d = json.loads(CACHE.read_text(encoding="utf-8"))
        if d.get("version") == CACHE_VERSION:
            return d
    pick = _load("01_pick_on_test")
    toy = _load("02_loglik_vs_generate")
    sens = _load("03_prompt_sensitivity")
    boot = _load("04_paired_bootstrap")
    cont = _load("05_contamination")

    world = toy.build_world()
    lm = toy.SuffixLM(world.corpus)
    cloze = toy.score_mc_loglik(lm, world.items, toy.cloze_prompt)
    diff_id = next(p["id"] for p in cloze["items"] if p["correct"] != p["correct_norm"])
    it = next(x for x in world.items if x.id == diff_id)
    example = {
        "stem": it.stem,
        "answer": it.answer,
        "choices": it.choices,
        "lps": [lm.logprob(it.stem, c) for c in it.choices],
        "bytes": [len(c.encode("utf-8")) for c in it.choices],
    }

    x, y = boot.smoke_scores("grpo", "toy_mc"), boot.smoke_scores("sft", "toy_mc")
    diff, lo, hi, dec, stats = boot.paired_bootstrap(x, y, n_boot=2000, seed=0)
    vals, counts = np.unique(np.round(stats, 6), return_counts=True)
    a = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
    b = [1, 0, 0, 1, 0, 1, 0, 0, 1, 1]
    rng = np.random.default_rng(1)  # 与 04 的手算例子相同的种子
    resamples = [rng.integers(0, 10, size=10).tolist() for _ in range(2)]

    sims = boot.simulations()
    found = cont.find_leaks(world.docs, [q.question for q in world.items], n=13)
    truth = {i for i, q in enumerate(world.items) if q.leaked}
    variants = [(name, cont.overlap(text, cont.ITEM, 13)) for name, text in cont.VARIANTS.items()]

    d = {
        "version": CACHE_VERSION,
        "pick": pick.pick_table(),
        "example": example,
        "formats": sens.run_formats(),
        "boot": {"diff": diff, "lo": lo, "hi": hi, "decision": dec,
                 "vals": vals.tolist(), "counts": counts.tolist(),
                 "n_diff_items": int((x != y).sum())},
        "hand": {"a": a, "b": b, "resamples": resamples},
        "sims": {"by_n": sims["by_n"], "any_hit": sims["any_hit"], "single_hit": sims["single_hit"]},
        "leak": {"found": len(found), "truth": len(truth), "hit": len(found & truth)},
        "variants": variants,
        "item_tokens": cont.tokens(cont.ITEM),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


# code/ 里的格式名、判定结论、变体名已改成英文；视频仍显示原来的中文。
# 旧缓存里存的是中文，所以查不到时原样返回。
NAME_ZH = {
    "Cloze": "完形填空",
    "Cloze + trailing space": "完形填空 + 末尾空格",
    "QA (workbook format)": "问答（练习册格式）",
    "QA (English labels)": "问答（英文标签）",
    "QA (colon → space)": "问答（冒号换空格）",
    "Letter choice": "字母选择题",
    "ahead": "超过",
    "tie": "持平",
    "behind": "落后",
    "Exact copy": "原文照抄",
    "Change case, punctuation, and whitespace": "改大小写、标点和空白",
    "Copy of the first half only": "只抄了前半句",
    "Paraphrase (other words, other word order)": "改写（换词换语序）",
    "Translation into Chinese": "翻译成中文",
}


def name_zh(s: str) -> str:
    return NAME_ZH.get(s, s)


# 已核实的外部数字（出处见 README 与 script.md 事实清单）
QWEN35 = {"MMLU-Redux": (48.5, 59.5), "IFEval": (52.1, 44.0)}  # (非思考, 思考)，Qwen3.5-0.8B 模型卡
BFCL_WEIGHTS = [("Non-Live", 10), ("Live", 10), ("相关性", 10), ("Multi-Turn", 30), ("Agentic", 40)]
BENCH_CARDS = [
    ("MMLU-Redux", "英文知识（修过错）", "5,700 题"),
    ("MMLU-Pro", "知识 + 推理，10 选 1", "12,032 题"),
    ("C-Eval", "中文学科", "val 1,346 题"),
    ("CMMLU", "中文 67 个主题", "约 1.15 万题"),
    ("GSM8K", "小学数学应用题", "1,319 题"),
    ("MATH-500", "竞赛数学", "500 题"),
    ("HumanEval+ / MBPP+", "写代码、跑测试", "164 / 378 题"),
    ("IFEval", "可程序检查的指令", "541 条"),
]


def card(w: float, h: float, color: str = theme.MUTED) -> RoundedRectangle:
    return RoundedRectangle(width=w, height=h, corner_radius=0.15, stroke_color=color,
                            stroke_width=2, fill_color=theme.BG, fill_opacity=1)


class ChapterScene(NarratedScene):
    chapter_label = "第 11 章"
    chapter_title = "评测：先定考卷"

    def construct(self):  # noqa: C901, PLR0915
        d = data()

        # ── S01 片头 ─────────────────────────────────────────────────────────
        with self.shot("S01"):
            c = self.chapter_card()
            self.play(FadeIn(c), run_time=self.fit(1.2))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(c), run_time=self.fit(0.6))

        # ── S02 凭什么说"超过" ──────────────────────────────────────────────
        with self.shot("S02"):
            self.play(*self.set_heading("凭什么说“超过”？"), run_time=self.fit(0.6))
            ours = VGroup(card(3.4, 1.1, theme.OUTPUT), zh("我们的模型", 30, theme.OUTPUT))
            them = VGroup(card(3.4, 1.1, theme.INPUT), zh("Qwen3.5-0.8B", 30, theme.INPUT))
            ours.move_to([-3.6, 1.9, 0])
            them.move_to([3.6, 1.9, 0])
            q = zh("?", 72, theme.HIGHLIGHT).move_to([0, 1.9, 0])
            self.play(FadeIn(ours), FadeIn(them), run_time=self.fit(0.8))
            self.play(Write(q), run_time=self.fit(0.6))
            qs = ["考哪几张卷子？", "用谁的对话模板？", "对手开不开“思考”？",
                  "差 1 分算不算赢？", "考题有没有混进训练数据？"]
            lines = VGroup(*[zh(s, 28) for s in qs]).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
            lines.move_to([0, -0.85, 0])
            for ln in lines:
                self.play(FadeIn(ln, shift=RIGHT * 0.2), run_time=self.fit(0.5, reserve=1.0))
                self.wait(self.remaining() * 0.12)
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(ours, them, q, lines)), run_time=self.fit(0.5))

        # ── S03 时间线 ───────────────────────────────────────────────────────
        with self.shot("S03"):
            self.play(*self.set_heading("先定考卷，再开始训练"), run_time=self.fit(0.6))
            axis = Arrow([-6.4, 0.9, 0], [6.4, 0.9, 0], buff=0, stroke_width=3, color=theme.MUTED)
            xs = [-5.2, -1.8, 1.8, 5.2]
            names = ["预注册", "预训练", "后训练", "最终评测"]
            subs = ["git commit = 登记时间", "第 12–15 章", "第 16–19 章", "第 20 章（闸门 3）"]
            nodes = VGroup()
            for x, n, s in zip(xs, names, subs):
                dot = Circle(radius=0.14, color=theme.PARAM, fill_opacity=1).move_to([x, 0.9, 0])
                t = zh(n, 30, theme.PARAM if n == "预注册" else theme.FG).move_to([x, 1.75, 0])
                st = zh(s, 20, theme.MUTED).move_to([x, 1.3, 0])
                nodes.add(VGroup(dot, t, st))
            self.play(Create(axis), run_time=self.fit(0.8))
            self.play(LaggedStart(*[FadeIn(nd) for nd in nodes], lag_ratio=0.35),
                      run_time=self.fit(2.0))
            dev = Rectangle(width=6.6, height=0.55, stroke_width=0, fill_color=theme.INPUT,
                            fill_opacity=0.35).move_to([0, -0.2, 0])
            dev_t = zh("开发集：挑 checkpoint、调超参、选提示词，随时用", 22).move_to(dev)
            self.play(FadeIn(dev), FadeIn(dev_t), run_time=self.fit(0.8))
            box = card(4.6, 1.2, theme.GRAD).move_to([3.4, -1.75, 0])
            box_t = VGroup(zh("测试集：锁住", 26, theme.GRAD),
                           zh("只在闸门和最终评测时打开", 20, theme.MUTED)).arrange(DOWN, buff=0.12)
            box_t.move_to(box)
            arrow = Arrow(box.get_top() + RIGHT * 1.8, [5.2, 0.75, 0], buff=0.05, color=theme.GRAD,
                          stroke_width=3)
            self.wait(self.remaining() * 0.25)
            self.play(FadeIn(box), FadeIn(box_t), run_time=self.fit(0.8))
            self.play(Create(arrow), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(axis, nodes, dev, dev_t, box, box_t, arrow)),
                      run_time=self.fit(0.5))

        # ── S04 在测试集上挑，分数会虚高 ─────────────────────────────────────
        with self.shot("S04"):
            self.play(*self.set_heading("在测试集上挑：分数会虚高"), run_time=self.fit(0.6))
            ax = Axes(x_range=[0, 4, 1], y_range=[0.45, 0.58, 0.05], x_length=8, y_length=4.2,
                      axis_config={"color": theme.MUTED, "include_ticks": False},
                      y_axis_config={"include_ticks": True}).move_to([-1.3, -0.2, 0])
            ylabels = VGroup(*[zh(f"{v:.2f}", 20, theme.MUTED).next_to(ax.c2p(0, v), LEFT, buff=0.15)
                               for v in (0.45, 0.50, 0.55)])
            base = DashedLine(ax.c2p(0, 0.5), ax.c2p(4, 0.5), color=theme.MUTED)
            base_t = zh("虚线 = 真实水平 0.5", 20, theme.MUTED).move_to([5.3, 0.25, 0])
            self.play(Create(ax), FadeIn(ylabels), Create(base), FadeIn(base_t),
                      run_time=self.fit(1.0))
            bars = VGroup()
            for i, (k, on_test, on_dev) in enumerate(d["pick"]):
                for j, (v, col) in enumerate(((on_test, theme.PARAM), (on_dev, theme.OUTPUT))):
                    x0 = i + 0.1 + j * 0.42
                    r = Rectangle(width=ax.x_length / 4 * 0.26, height=(ax.c2p(0, v) - ax.c2p(0, 0.45))[1],
                                  stroke_width=0, fill_color=col, fill_opacity=0.9)
                    r.move_to(ax.c2p(x0 + 0.13, 0.45), aligned_edge=DOWN)
                    val = zh(f"{v:.3f}", 16, col).next_to(r, UP, buff=0.06)
                    bars.add(VGroup(r, val))
                klab = zh(f"K = {k}", 22).next_to(ax.c2p(i + 0.5, 0.45), DOWN, buff=0.2)
                bars.add(klab)
            legend = VGroup(
                VGroup(Rectangle(width=0.3, height=0.3, stroke_width=0, fill_color=theme.PARAM,
                                 fill_opacity=0.9), zh("看着测试分挑", 22)).arrange(RIGHT, buff=0.15),
                VGroup(Rectangle(width=0.3, height=0.3, stroke_width=0, fill_color=theme.OUTPUT,
                                 fill_opacity=0.9), zh("开发集上挑", 22)).arrange(RIGHT, buff=0.15),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([5.3, 1.0, 0])
            self.play(FadeIn(legend), LaggedStart(*[FadeIn(b_) for b_ in bars], lag_ratio=0.15),
                      run_time=self.fit(3.0))
            k10 = next(r for r in d["pick"] if r[0] == 10)
            note = zh(f"K = 10：虚高 {k10[1] - 0.5:+.3f}", 26, theme.HIGHLIGHT).move_to([5.3, -0.6, 0])
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(note), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(ax, ylabels, base, base_t, bars, legend, note)),
                      run_time=self.fit(0.5))

        # ── S05 通用组 ───────────────────────────────────────────────────────
        with self.shot("S05"):
            self.play(*self.set_heading("通用组：如实报告"), run_time=self.fit(0.6))
            cards = VGroup()
            for i, (name, what, n) in enumerate(BENCH_CARDS):
                x = [-5.1, -1.7, 1.7, 5.1][i % 4]
                y = 1.25 if i < 4 else -1.0
                box = card(3.2, 1.95, theme.INPUT).move_to([x, y, 0])
                size = 26 if len(name) < 12 else 22
                txt = VGroup(zh(name, size, theme.INPUT), zh(what, 20), zh(n, 20, theme.MUTED))
                txt.arrange(DOWN, buff=0.16).move_to(box)
                cards.add(VGroup(box, txt))
            self.play(LaggedStart(*[FadeIn(c_) for c_ in cards], lag_ratio=0.3),
                      run_time=self.fit(self.remaining() * 0.55))
            lock = zh("C-Eval：test 答案不公开 → 只能用 val（1,346 题）", 22, theme.HIGHLIGHT)
            lock.move_to([0, -2.4, 0])
            hl = SurroundingRectangle(cards[2][0], color=theme.HIGHLIGHT, buff=0.04)
            self.play(Create(hl), FadeIn(lock), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(cards, hl, lock)), run_time=self.fit(0.5))

        # ── S06 专项组 ───────────────────────────────────────────────────────
        with self.shot("S06"):
            self.play(*self.set_heading("专项组：决定“超过”与否"), run_time=self.fit(0.6))
            title = zh("BFCL V4 总分的权重", 24, theme.MUTED).move_to([0, 2.45, 0])
            x0, total_w = -6.0, 12.0
            segs = VGroup()
            colors = [theme.INPUT, theme.INPUT, theme.INPUT, theme.ATTN, theme.MUTED]
            for (name, w), col in zip(BFCL_WEIGHTS, colors):
                width = total_w * w / 100
                r = Rectangle(width=width, height=0.8, stroke_color=theme.BG, stroke_width=3,
                              fill_color=col, fill_opacity=0.35 if name == "Agentic" else 0.8)
                r.move_to([x0 + width / 2, 1.3, 0])
                pct = zh(f"{w}%", 22).move_to(r)
                lab = zh(name, 18, theme.FG).next_to(r, UP, buff=0.08)
                segs.add(VGroup(r, pct, lab))
                x0 += width
            self.play(FadeIn(title), LaggedStart(*[FadeIn(s) for s in segs], lag_ratio=0.25),
                      run_time=self.fit(2.0))
            ast_t = zh("单轮：AST 匹配", 22, theme.INPUT).move_to([-4.2, 0.45, 0])
            mt_t = zh("多轮：执行后比状态", 22, theme.ATTN).move_to([-0.6, 0.45, 0])
            ag_t = zh("web search 不可复现 → 只报告", 22, theme.MUTED).move_to([3.6, 0.45, 0])
            self.play(FadeIn(ast_t), run_time=self.fit(0.5))
            self.wait(self.remaining() * 0.12)
            self.play(FadeIn(mt_t), run_time=self.fit(0.5))
            self.wait(self.remaining() * 0.15)
            self.play(FadeIn(ag_t), run_time=self.fit(0.5))
            self.wait(self.remaining() * 0.15)
            ace = VGroup(card(6.4, 1.75, theme.OUTPUT),
                         VGroup(zh("ACEBench（中文）", 26, theme.OUTPUT),
                                zh("Normal + Special：规则判分 → 硬目标", 20),
                                zh("Agent：GPT-4o 扮用户 → 只报告", 20, theme.MUTED)).arrange(DOWN, buff=0.12))
            ace[1].move_to(ace[0])
            ace.move_to([-3.0, -1.55, 0])
            tau = VGroup(card(4.6, 1.75, theme.MUTED),
                         VGroup(zh("τ²-bench", 26, theme.FG),
                                zh("大模型模拟用户 → 只报告", 20, theme.MUTED)).arrange(DOWN, buff=0.15))
            tau[1].move_to(tau[0])
            tau.move_to([3.9, -1.55, 0])
            self.play(FadeIn(ace), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(tau), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(title, segs, ast_t, mt_t, ag_t, ace, tau)), run_time=self.fit(0.5))

        # ── S07 对数似然 vs 生成 ─────────────────────────────────────────────
        with self.shot("S07"):
            self.play(*self.set_heading("两种判分：对数似然 vs 生成"), run_time=self.fit(0.6))
            ex = d["example"]
            stem = zh(f"{ex['stem']}____", 34).move_to([-2.2, 2.3, 0])
            hdr = zh("log P(选项｜题干)", 22, theme.MUTED).move_to([1.4, 1.65, 0])
            self.play(FadeIn(stem), FadeIn(hdr), run_time=self.fit(0.8))
            lps, nbytes = ex["lps"], ex["bytes"]
            per_byte = [lp / nb for lp, nb in zip(lps, nbytes)]
            ys = [1.0, 0.25, -0.5, -1.25]

            def bars_for(values, floor, scale):  # noqa: ANN001, ANN202
                best = int(np.argmax(values))
                g = VGroup()
                for i, (v, y) in enumerate(zip(values, ys)):
                    col = theme.OUTPUT if i == ex["answer"] else theme.MUTED
                    if i == best and i != ex["answer"]:
                        col = theme.GRAD
                    r = Rectangle(width=max(0.05, (v - floor) * scale), height=0.42, stroke_width=0,
                                  fill_color=col, fill_opacity=0.85)
                    r.move_to([-2.4, y, 0], aligned_edge=LEFT)
                    val = zh(f"{v:.2f}" if abs(v) > 5 else f"{v:.3f}", 20, col).next_to(r, RIGHT, buff=0.12)
                    g.add(VGroup(r, val))
                return g

            labels = VGroup(*[zh(ch + ("（正确）" if i == ex["answer"] else ""), 24,
                                 theme.OUTPUT if i == ex["answer"] else theme.FG).move_to([-2.6, y, 0],
                                                                                         aligned_edge=RIGHT)
                              for i, (ch, y) in enumerate(zip(ex["choices"], ys))])
            b1 = bars_for(lps, min(lps) - 0.8, 1.2)
            self.play(FadeIn(labels), LaggedStart(*[GrowFromEdge(b[0], LEFT) for b in b1], lag_ratio=0.2),
                      *[FadeIn(b[1]) for b in b1], run_time=self.fit(1.6))
            self.wait(self.remaining() * 0.35)
            hdr2 = zh("÷ 字节数（acc_norm）", 22, theme.HIGHLIGHT).move_to(hdr)
            b2 = bars_for(per_byte, min(per_byte) - 0.3, 2.0)
            self.play(FadeOut(hdr), FadeIn(hdr2), FadeOut(b1), FadeIn(b2), run_time=self.fit(1.0))
            self.wait(self.remaining() * 0.45)
            foot = zh("Base 模型 → 对数似然        对话模型 → 生成 + 精确匹配", 24).move_to([0, -2.25, 0])
            self.play(FadeIn(foot), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(stem, hdr2, labels, b2, foot)), run_time=self.fit(0.5))

        # ── S08 提示词敏感性 ─────────────────────────────────────────────────
        with self.shot("S08"):
            self.play(*self.set_heading("只换提示词格式，分数差几倍"), run_time=self.fit(0.6))
            rows = d["formats"]
            x0, scale = -1.6, 8.5
            ys = [2.05, 1.35, 0.65, -0.05, -0.75, -1.45]
            chance = DashedLine([x0 + 0.25 * scale, 2.5, 0], [x0 + 0.25 * scale, -1.85, 0],
                                color=theme.MUTED)
            chance_t = zh("随机猜 0.25", 18, theme.MUTED).next_to(chance, UP, buff=0.05)
            items = VGroup()
            for r, y in zip(rows, ys):
                col = theme.GRAD if "空格" in name_zh(r["name"]) else theme.INPUT
                lab = zh(name_zh(r["name"]), 22, col).move_to([x0 - 0.2, y, 0], aligned_edge=RIGHT)
                bar = Rectangle(width=r["acc"] * scale, height=0.45, stroke_width=0, fill_color=col,
                                fill_opacity=0.85).move_to([x0, y, 0], aligned_edge=LEFT)
                note = f"{r['acc']:.3f}"
                if r["most_common_pred_share"] >= 0.999:
                    note += "  （32 题全选同一个字母）"
                val = zh(note, 20, col).next_to(bar, RIGHT, buff=0.12)
                items.add(VGroup(lab, bar, val))
            self.play(Create(chance), FadeIn(chance_t), run_time=self.fit(0.6))
            for it in items:
                self.play(FadeIn(it[0]), GrowFromEdge(it[1], LEFT), FadeIn(it[2]),
                          run_time=self.fit(0.6, reserve=2.0))
                self.wait(self.remaining() * 0.08)
            ref = zh("真模型：Sclar 等（ICLR 2024）发现同义格式之间最多差 76 个百分点", 22,
                     theme.HIGHLIGHT).move_to([0, -2.35, 0])
            self.play(FadeIn(ref), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(chance, chance_t, items, ref)), run_time=self.fit(0.5))

        # ── S09 思考模式 ─────────────────────────────────────────────────────
        with self.shot("S09"):
            self.play(*self.set_heading("对手两种模式都测，取较高分"), run_time=self.fit(0.6))
            ax = Axes(x_range=[0, 2, 1], y_range=[0, 70, 10], x_length=7, y_length=3.8,
                      axis_config={"color": theme.MUTED, "include_ticks": False}).move_to([-1.2, 0.1, 0])
            src = zh("数据：Qwen3.5-0.8B 模型卡", 20, theme.MUTED).move_to([-1.2, 2.45, 0])
            bars = VGroup()
            for i, (bench, (nt, th)) in enumerate(QWEN35.items()):
                for j, (v, col, name) in enumerate(((nt, theme.INPUT, "非思考"), (th, theme.ATTN, "思考"))):
                    r = Rectangle(width=1.1, height=(ax.c2p(0, v) - ax.c2p(0, 0))[1], stroke_width=0,
                                  fill_color=col, fill_opacity=0.9)
                    r.move_to(ax.c2p(i + 0.28 + j * 0.44, 0), aligned_edge=DOWN)
                    hi_ = v == max(nt, th)
                    val = zh(f"{v:.1f}" + ("  较高" if hi_ else ""), 20,
                             theme.HIGHLIGHT if hi_ else theme.FG).next_to(r, UP, buff=0.08)
                    mode = zh(name, 18, col).next_to(r, DOWN, buff=0.1)
                    bars.add(VGroup(r, val, mode))
                bars.add(zh(bench, 24).next_to(ax.c2p(i + 0.5, 0), DOWN, buff=0.55))
            rule = VGroup(zh("两种都测", 26, theme.FG), zh("逐个基准", 26, theme.FG),
                          zh("取对手较高者", 26, theme.HIGHLIGHT)).arrange(DOWN, buff=0.35)
            rule.move_to([4.9, 0.2, 0])
            self.play(Create(ax), FadeIn(src), run_time=self.fit(0.8))
            self.play(FadeIn(rule), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.3)
            self.play(LaggedStart(*[FadeIn(b_) for b_ in bars], lag_ratio=0.25), run_time=self.fit(2.0))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(ax, src, bars, rule)), run_time=self.fit(0.5))

        # ── S10 配对 bootstrap ───────────────────────────────────────────────
        with self.shot("S10"):
            self.play(*self.set_heading("配对 bootstrap：差几分才算赢"), run_time=self.fit(0.6))
            a, b = d["hand"]["a"], d["hand"]["b"]
            xs = [-4.4 + 0.85 * i for i in range(10)]

            def row(vals, y, name):  # noqa: ANN001, ANN202
                g = VGroup(zh(name, 24).move_to([-5.7, y, 0]))
                for x, v in zip(xs, vals):
                    g.add(Circle(radius=0.22, stroke_width=0, fill_opacity=1,
                                 fill_color=theme.OUTPUT if v else theme.MUTED).move_to([x, y, 0]))
                return g

            ra, rb = row(a, 2.0, "A"), row(b, 1.25, "B")
            nums = VGroup(*[zh(str(i + 1), 16, theme.MUTED).move_to([x, 2.55, 0]) for i, x in enumerate(xs)])
            dvals = [ai - bi for ai, bi in zip(a, b)]
            dr = VGroup(zh("A−B", 22).move_to([-5.7, 0.5, 0]),
                        *[zh(f"{v:+d}" if v else "0", 22, theme.HIGHLIGHT if v else theme.FG).move_to([x, 0.5, 0])
                          for x, v in zip(xs, dvals)])
            legend = zh("绿 = 答对   灰 = 答错", 18, theme.MUTED).move_to([4.9, 2.55, 0])
            self.play(FadeIn(nums), FadeIn(ra), FadeIn(rb), FadeIn(legend), run_time=self.fit(1.0))
            self.play(FadeIn(dr), run_time=self.fit(0.8))
            mean_t = zh(f"平均分差 {np.mean(dvals):+.2f}", 24).move_to([4.9, 0.5, 0])
            self.play(FadeIn(mean_t), run_time=self.fit(0.5))
            self.wait(self.remaining() * 0.12)
            for k, idx in enumerate(d["hand"]["resamples"]):
                marks = VGroup()
                count = {}
                for i in idx:
                    count[i] = count.get(i, 0) + 1
                for i, c_ in count.items():
                    marks.add(SurroundingRectangle(VGroup(ra[i + 1], rb[i + 1]), color=theme.PARAM, buff=0.06))
                    if c_ > 1:
                        marks.add(zh(f"×{c_}", 18, theme.PARAM).move_to([xs[i], -0.1, 0]))
                ds = np.mean([dvals[i] for i in idx])
                txt = zh(f"重抽第 {k + 1} 次：题号 {', '.join(str(i + 1) for i in idx)}  →  d* = {ds:+.2f}",
                         22, theme.PARAM).move_to([0, -0.75, 0])
                self.play(FadeIn(marks), FadeIn(txt), run_time=self.fit(0.6, reserve=3.0))
                self.wait(self.remaining() * 0.15)
                self.play(FadeOut(marks), FadeOut(txt), run_time=self.fit(0.3, reserve=2.5))
            rule = VGroup(zh("重复 10000 次 → 取中间 95% → 置信区间", 24),
                          VGroup(zh("全在 0 右边：超过", 24, theme.OUTPUT), zh("全在 0 左边：落后", 24, theme.GRAD),
                                 zh("跨过 0：持平", 24, theme.MUTED)).arrange(RIGHT, buff=0.7)
                          ).arrange(DOWN, buff=0.35).move_to([0, -1.7, 0])
            self.play(FadeIn(rule[0]), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(rule[1]), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(nums, ra, rb, legend, dr, mean_t, rule)), run_time=self.fit(0.5))

        # ── S11 极小配置演示：bootstrap 直方图 ───────────────────────────────
        with self.shot("S11"):
            self.play(*self.set_heading("冒烟测试：grpo vs sft（toy_mc，30 题）"), run_time=self.fit(0.6))
            badge = self.show_badge()
            bt = d["boot"]
            vals, counts = np.array(bt["vals"]), np.array(bt["counts"])
            ax = Axes(x_range=[-0.1, 0.4, 0.1], y_range=[0, float(counts.max()) * 1.15, 100],
                      x_length=9, y_length=3.6, axis_config={"color": theme.MUTED, "include_ticks": True},
                      y_axis_config={"include_ticks": False}).move_to([-0.9, -0.1, 0])
            xl = VGroup(*[zh(f"{v:+.1f}" if v else "0", 18, theme.MUTED).next_to(ax.c2p(v, 0), DOWN, buff=0.15)
                          for v in (-0.1, 0.0, 0.1, 0.2, 0.3, 0.4)])
            xt = zh("重抽 2000 次的平均分差 d*", 20, theme.MUTED).next_to(ax, DOWN, buff=0.45)
            bars = VGroup()
            w = (ax.c2p(1 / 30, 0) - ax.c2p(0, 0))[0] * 0.8
            for v, c_ in zip(vals, counts):
                inside = bt["lo"] <= v <= bt["hi"]
                r = Rectangle(width=w, height=(ax.c2p(0, c_) - ax.c2p(0, 0))[1], stroke_width=0,
                              fill_color=theme.INPUT if inside else theme.MUTED, fill_opacity=0.85)
                r.move_to(ax.c2p(v, 0), aligned_edge=DOWN)
                bars.add(r)
            zero = DashedLine(ax.c2p(0, 0), ax.c2p(0, float(counts.max()) * 1.1), color=theme.FG)
            self.play(Create(ax), FadeIn(xl), FadeIn(xt), run_time=self.fit(0.8))
            self.play(LaggedStart(*[GrowFromEdge(r, DOWN) for r in bars], lag_ratio=0.05),
                      Create(zero), run_time=self.fit(1.6))
            ytop = float(counts.max()) * 1.05
            ci = VGroup(Line(ax.c2p(bt["lo"], ytop), ax.c2p(bt["hi"], ytop), color=theme.HIGHLIGHT, stroke_width=5),
                        Line(ax.c2p(bt["lo"], ytop * 0.93), ax.c2p(bt["lo"], ytop), color=theme.HIGHLIGHT, stroke_width=5),
                        Line(ax.c2p(bt["hi"], ytop * 0.93), ax.c2p(bt["hi"], ytop), color=theme.HIGHLIGHT, stroke_width=5))
            ci_t = zh(f"95% 区间 [{bt['lo']:+.3f}, {bt['hi']:+.3f}]", 22, theme.HIGHLIGHT)
            ci_t.next_to(ci, UP, buff=0.1)
            dec = VGroup(zh(f"差值 {bt['diff']:+.3f}", 24), zh(f"下界 > 0 → {name_zh(bt['decision'])}", 26, theme.OUTPUT)
                         ).arrange(DOWN, buff=0.25).move_to([5.3, 0.6, 0])
            self.wait(self.remaining() * 0.12)
            self.play(Create(ci), FadeIn(ci_t), run_time=self.fit(0.8))
            self.play(FadeIn(dec), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.25)
            warn = VGroup(zh(f"可这个差距只来自 {bt['n_diff_items']} 道题", 30, theme.HIGHLIGHT),
                          zh(f"水平相同的模型比 6 次：{bt_pct(d['sims']['any_hit'])} 至少一次误判", 28),
                          zh("→ 主终点必须事先指定", 28, theme.OUTPUT)).arrange(DOWN, buff=0.4).move_to([0, 0, 0])
            self.play(FadeOut(VGroup(ax, xl, xt, bars, zero, ci, ci_t, dec)), run_time=self.fit(0.5))
            for w_ in warn:
                self.play(FadeIn(w_), run_time=self.fit(0.5, reserve=1.0))
                self.wait(self.remaining() * 0.2)
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(warn), FadeOut(badge), run_time=self.fit(0.5))

        # ── S12 题数与区间宽度 ───────────────────────────────────────────────
        with self.shot("S12"):
            self.play(*self.set_heading("题数决定能看清多小的差距"), run_time=self.fit(0.6))
            lo_v, hi_v, x_lo, x_hi = -0.15, 0.2, -3.0, 6.3

            def xpos(v: float) -> float:
                return x_lo + (v - lo_v) / (hi_v - lo_v) * (x_hi - x_lo)

            zero = DashedLine([xpos(0), 2.2, 0], [xpos(0), -2.0, 0], color=theme.FG)
            zero_t = zh("0", 20).next_to(zero, UP, buff=0.05)
            true_l = Line([xpos(0.03), 2.2, 0], [xpos(0.03), -2.0, 0], color=theme.OUTPUT, stroke_width=3)
            true_t = zh("真实差距 +0.03", 20, theme.OUTPUT).next_to(true_l, UP, buff=0.05).shift(RIGHT * 0.9)
            self.play(Create(zero), FadeIn(zero_t), Create(true_l), FadeIn(true_t), run_time=self.fit(0.8))
            rows = VGroup()
            for (n, width, win), y in zip(d["sims"]["by_n"], (1.3, 0.0, -1.3)):
                seg = Line([xpos(0.03 - width / 2), y, 0], [xpos(0.03 + width / 2), y, 0],
                           color=theme.HIGHLIGHT, stroke_width=8)
                lab = VGroup(zh(f"{n} 题", 26), zh(f"区间宽 {width:.3f}", 20, theme.HIGHLIGHT),
                             zh(f"判出“超过” {win:.2f}", 20, theme.MUTED)).arrange(DOWN, buff=0.08,
                                                                                aligned_edge=LEFT)
                lab.move_to([-5.4, y, 0])
                rows.add(VGroup(seg, lab))
            for r in rows:
                self.play(FadeIn(r[1]), Create(r[0]), run_time=self.fit(0.8, reserve=2.0))
                self.wait(self.remaining() * 0.18)
            note = zh("示意：区间画在真实差距处，宽度是 100 次模拟的平均值", 18, theme.MUTED).move_to([1.2, -2.4, 0])
            self.play(FadeIn(note), run_time=self.fit(0.5))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(zero, zero_t, true_l, true_t, rows, note)), run_time=self.fit(0.5))

        # ── S13 污染：13-gram ────────────────────────────────────────────────
        with self.shot("S13"):
            self.play(*self.set_heading("数据污染：13-gram 重叠检查"), run_time=self.fit(0.6))
            toks = d["item_tokens"]
            words = VGroup()
            x, y, x_left, x_right = -6.6, 1.9, -6.6, 0.6
            for t in toks:
                m = Text(t, font="Noto Sans Mono", font_size=20, color=theme.FG)
                if x + m.width > x_right:
                    x, y = x_left, y - 0.5
                m.move_to([x, y, 0], aligned_edge=LEFT)
                x += m.width + 0.16
                words.add(m)
            cap = zh("考题（规范化后：小写、去标点，一个词一格）", 20, theme.MUTED).move_to([-3.0, 2.5, 0])
            self.play(FadeIn(cap), LaggedStart(*[FadeIn(w_) for w_ in words], lag_ratio=0.02),
                      run_time=self.fit(1.2))
            n = 13
            starts = [0, 5, 10, 15, len(toks) - n]
            win_t = zh("窗口 = 连续 13 个词；训练文档里出现任意一个 → 撞了", 18, theme.GRAD)
            win_t.move_to([-6.5, y - 0.6, 0], aligned_edge=LEFT)
            self.play(FadeIn(win_t), run_time=self.fit(0.5))
            for s in starts:
                box = SurroundingRectangle(VGroup(*words[s:s + n]), color=theme.GRAD, buff=0.05)
                self.play(*[w_.animate.set_color(theme.GRAD if s <= i < s + n else theme.FG)
                            for i, w_ in enumerate(words)], FadeIn(box), run_time=self.fit(0.4, reserve=4.0))
                self.play(FadeOut(box), run_time=self.fit(0.2, reserve=3.8))
            table = VGroup(zh("变体", 20, theme.MUTED), zh("共享 13-gram", 20, theme.MUTED))
            table.arrange(RIGHT, buff=0.8)
            rows = VGroup()
            for name, o in d["variants"]:
                rows.add(VGroup(zh(name_zh(name), 20), zh(str(o), 22, theme.GRAD if o else theme.OUTPUT)))
            tab = VGroup()
            for i, (a_, b_) in enumerate([tuple(table)] + [tuple(r) for r in rows]):
                a_.move_to([1.3, 1.9 - 0.55 * i, 0], aligned_edge=LEFT)
                b_.move_to([6.2, 1.9 - 0.55 * i, 0])
                tab.add(a_, b_)
            self.play(*[w_.animate.set_color(theme.FG) for w_ in words], FadeIn(tab), run_time=self.fit(1.0))
            lk = d["leak"]
            toy = zh(f"玩具世界：查出 {lk['found']} 道 = 真泄漏 {lk['truth']} 道；改写、翻译：0 → 抓不到",
                     22, theme.HIGHLIGHT).move_to([0, -2.35, 0])
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(toy), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(cap, words, win_t, tab, toy)), run_time=self.fit(0.5))

        # ── S14 预注册 ───────────────────────────────────────────────────────
        with self.shot("S14"):
            self.play(*self.set_heading("预注册：把考卷锁进 git"), run_time=self.fit(0.6))
            doc = card(7.6, 4.6, theme.PARAM).move_to([-2.7, 0.0, 0])
            title = zh("eval/PREREGISTRATION.md（草案）", 24, theme.PARAM).move_to([-2.7, 1.9, 0])
            items = ["1  基准及版本", "2  评测框架及版本号", "3  提示词、模板、解码参数",
                     "4  对手清单与冻结日期", "5  “超过”的判定标准"]
            lines = VGroup(*[zh(s, 26) for s in items]).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
            lines.move_to([-2.7, -0.3, 0])
            self.play(Create(doc), FadeIn(title), run_time=self.fit(0.8))
            for ln in lines:
                self.play(FadeIn(ln, shift=RIGHT * 0.2), run_time=self.fit(0.45, reserve=3.0))
                self.wait(self.remaining() * 0.05)
            side = VGroup(zh("git commit", 26, theme.HIGHLIGHT), zh("= 登记时间", 24),
                          zh("之后只能追加修订", 20, theme.MUTED)).arrange(DOWN, buff=0.18).move_to([4.2, 1.3, 0])
            opp = VGroup(zh("对手：参数量 0.7–1.3 倍", 22), zh("+ Qwen3.5-0.8B 必比", 22, theme.INPUT),
                         zh("每个对手 × 每个主终点都超过", 20, theme.OUTPUT)).arrange(DOWN, buff=0.15).move_to([4.2, -0.5, 0])
            cost = zh("第二步重跑 ≈ $60–200", 22, theme.PARAM).move_to([4.2, -1.9, 0])
            self.play(FadeIn(side), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(opp), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(cost), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(doc, title, lines, side, opp, cost)), run_time=self.fit(0.5))

        # ── S15 小结 ─────────────────────────────────────────────────────────
        with self.shot("S15"):
            self.play(*self.set_heading("小结"), run_time=self.fit(0.6))
            kws = ["先定考卷", "看清考什么", "写死提示词", "配对 bootstrap", "防污染", "锁进 git"]
            grid = VGroup()
            for i, k in enumerate(kws):
                box = VGroup(card(3.6, 1.1, theme.INPUT), zh(k, 28))
                box[1].move_to(box[0])
                box.move_to([[-4.0, 0.0, 4.0][i % 3], [1.3, -0.3][i // 3], 0])
                grid.add(box)
            for g in grid:
                self.play(FadeIn(g), run_time=self.fit(0.5, reserve=2.0))
                self.wait(self.remaining() * 0.1)
            nxt = zh("下一章：Scaling Law 与闸门 1", 30, theme.HIGHLIGHT).move_to([0, -1.9, 0])
            self.play(FadeIn(nxt), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(VGroup(grid, nxt)), *self.set_heading(None), run_time=self.fit(0.6))


def bt_pct(x: float) -> str:
    return f"{x * 100:.0f}%"
