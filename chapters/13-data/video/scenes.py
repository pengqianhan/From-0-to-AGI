"""第 13 章视频：数据 —— 从一堆网页到一份能训练的数据集

画面里的数值都由 ../code/ 中的脚本真实计算（见 script.md 事实清单）：
- 01–05 的漏斗、S 曲线、分类器阈值表、去污染表：现场计算（约 1 分钟），结果缓存到 video/out/cache.json；
- 06/07 的消融：读 video/out/ablation_quality.json、ablation_mixture.json（由脚本的 --json 写出；没有就现场跑）；
- 08 的词表测量：读 video/out/vocab_big.json（`08_vocab_size.py --corpus ... --json`；没有就用 tiny 语料现场测）；
- 极小配置流水线：读 out/tiny/data_pipeline/manifest.json（没有就现场跑 zero.data.pipeline）。
渲染：bash chapters/13-data/video/build.sh
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tomllib
from pathlib import Path

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Axes,
    Circle,
    Create,
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
from video_kit.scene import NarratedScene, polyline_in_axes, zh

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
REPO = HERE.parents[2]
OUT = HERE / "out"
CACHE = OUT / "cache.json"
sys.path.insert(0, str(REPO))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"ch13_{name}", CODE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_data() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text("utf-8"))
    c01, c02, c03 = _load("01_noisy_crawl"), _load("02_heuristic_filter"), _load("03_minhash")
    c04, c05 = _load("04_quality_classifier"), _load("05_decontam")
    crawl = c01.build_crawl()
    docs = crawl["docs"]
    d: dict = {"kinds": dict(c01.kind_table(docs)), "n_raw": len(docs)}
    kept, by_rule, by_kind = c02.heuristic_filter(docs)
    d["heur_removed"] = dict(by_kind)
    d["n_heur"] = len(kept)
    d["s_curve"] = c03.s_curve_check(n_pairs=1000, seed=3)
    ex = [kept[i] for i in c03.exact_dedup([x["text"] for x in kept])]
    keep2, clusters = c03.near_dedup([x["text"] for x in ex])
    near = [ex[i] for i in keep2]
    d["dedup_n"] = [len(kept), len(ex), len(near)]
    d["redundant"] = [c03.redundant(kept), c03.redundant(ex), c03.redundant(near)]
    r = c04.classify(near)
    rest = r["rest"]
    rows = []
    for th in [0.3, 0.5, 0.7, 0.9]:
        tp = sum(r["p"][i] < th and r["label"][i] == 0 for i in rest)
        fp = sum(r["p"][i] < th and r["label"][i] == 1 for i in rest)
        fn = sum(r["p"][i] >= th and r["label"][i] == 0 for i in rest)
        rows.append([th, int(tp), int(fn), int(fp), tp / max(tp + fp, 1), tp / max(tp + fn, 1)])
    d["clf"] = rows
    qdocs = [x for x, p in zip(near, r["p"]) if p >= 0.5]
    d["n_quality"] = len(qdocs)
    variants = {"verbatim": 0, "case_punct": 0, "paraphrase": 0}
    for x in qdocs:
        if x["kind"] == "contaminated":
            variants[x["leak"]["variant"]] += 1
    dec = []
    for n in [5, 8, 13, 20]:
        hits = c05.find_contaminated(qdocs, crawl["eval"], n)
        flagged = [qdocs[i] for i in hits]
        caught = {k: sum(f["kind"] == "contaminated" and f["leak"]["variant"] == k for f in flagged)
                  for k in variants}
        dec.append([n, len(flagged), caught, sum(f["kind"] != "contaminated" for f in flagged)])
    d["decontam"] = {"variants": variants, "rows": dec}
    hits13 = c05.find_contaminated(qdocs, crawl["eval"], 13)
    d["n_final"] = len(qdocs) - len(hits13)
    # 一个泄漏例子：考题与文档共享的 13-gram
    ev = {e["id"]: e for e in crawl["eval"]}
    leak_doc = next(x for x in qdocs if x["kind"] == "contaminated"
                    and x["leak"]["variant"] == "verbatim" and x["lang"] == "en")
    q = ev[leak_doc["leak"]["eval_id"]]["question"]
    d["leak_example"] = {"question": q}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(d, ensure_ascii=False), "utf-8")
    return d


def load_json_or_run(name: str, script: str, args: list[str]) -> dict:
    p = OUT / name
    if not p.exists():
        subprocess.run([sys.executable, str(CODE / script), "--json", str(p), *args], check=True,
                       cwd=REPO)
    return json.loads(p.read_text("utf-8"))


def load_manifest() -> dict:
    p = REPO / "out" / "tiny" / "data_pipeline" / "manifest.json"
    if not p.exists():
        subprocess.run([sys.executable, "-m", "zero.data.pipeline", "--config",
                        "configs/tiny/data.toml"], check=True, cwd=REPO)
    return json.loads(p.read_text("utf-8"))


D = compute_data()
ABL1 = load_json_or_run("ablation_quality.json", "06_quality_ablation.py", [])
ABL2 = load_json_or_run("ablation_mixture.json", "07_mixture_ablation.py", [])
VOCAB = load_json_or_run("vocab_big.json", "08_vocab_size.py", [])
MANIFEST = load_manifest()
with open(REPO / "configs" / "main" / "data.toml", "rb") as _f:
    VOCAB_CHOICE = tomllib.load(_f)["tokenizer"]["vocab_size"]  # 主线词表的选择（正文第 10 节）

MONO = "DejaVu Sans Mono"
KIND_ZH = {"good": "好文档", "contaminated": "夹带考题", "exact_dup": "原样转载", "near_dup": "改写转载",
           "nav": "导航页", "spam": "广告", "garbled": "乱码", "salad": "乱序"}
KIND_COLOR = {"good": theme.OUTPUT, "contaminated": theme.ATTN, "exact_dup": theme.PARAM,
              "near_dup": "#D9A35E", "nav": theme.GRAD, "spam": "#C94C73", "garbled": theme.MUTED,
              "salad": theme.INPUT}


def mono(text: str, size: float = 24, color: str = theme.FG) -> Text:
    return Text(text, font=MONO, font_size=size, color=color)


def card(lines: list[tuple[str, float, str]], w: float, h: float, color: str) -> VGroup:
    box = RoundedRectangle(width=w, height=h, corner_radius=0.12, color=color, stroke_width=2,
                           fill_color=color, fill_opacity=0.08)
    txt = VGroup(*[zh(t, s, c) for t, s, c in lines]).arrange(DOWN, buff=0.1)
    if txt.width > w - 0.25:
        txt.scale_to_fit_width(w - 0.25)
    txt.move_to(box)
    return VGroup(box, txt)


def table(rows: list[list[str]], widths: list[float], size: float = 20,
          header_color: str = theme.MUTED, row_h: float = 0.42) -> VGroup:
    """简单表格：每格一个居中的文字，第一行是表头。"""
    out = VGroup()
    for r, row in enumerate(rows):
        line = VGroup()
        x = 0.0
        for cell, w in zip(row, widths):
            t = zh(cell, size, header_color if r == 0 else theme.FG)
            if t.width > w - 0.1:
                t.scale_to_fit_width(w - 0.1)
            t.move_to([x + w / 2, -r * row_h, 0])
            line.add(t)
            x += w
        out.add(line)
    return out


class ChapterScene(NarratedScene):
    chapter_label = "第 13 章"
    chapter_title = "数据"

    def fade_all(self, *mobs) -> None:  # noqa: ANN002
        self.play(*[FadeOut(m) for m in mobs], run_time=self.fit(0.6))

    def construct(self) -> None:
        # ── S01 片头 ─────────────────────────────────────────────────────
        with self.shot("S01"):
            c = self.chapter_card()
            sub = zh("从一堆网页到一份能训练的数据集", 32, theme.HIGHLIGHT).next_to(c, DOWN, 0.6)
            self.play(FadeIn(c, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(c), FadeOut(sub), run_time=self.fit(0.8))

        # ── S02 数据是最大的杠杆 ─────────────────────────────────────────
        with self.shot("S02"):
            self.play(*self.set_heading("同样的尺寸，数据决定上限"), run_time=self.fit(0.8))
            scale = 9.0 / 36.0
            x0 = -4.6
            lab1 = zh("Qwen3 小模型", 26, theme.MUTED).move_to([x0 - 1.3, 1.6, 0])
            bar1 = Rectangle(width=36 * scale, height=0.55, fill_color=theme.MUTED, fill_opacity=0.6,
                             stroke_width=0).move_to([x0 + 36 * scale / 2, 1.6, 0])
            v1 = zh("36T token", 24).next_to(bar1, DOWN, 0.12).align_to(bar1, RIGHT)
            lab2 = zh("MobileLLM-R1", 26, theme.OUTPUT).move_to([x0 - 1.3, 0.35, 0])
            bar2 = Rectangle(width=4.2 * scale, height=0.55, fill_color=theme.OUTPUT, fill_opacity=0.8,
                             stroke_width=0).move_to([x0 + 4.2 * scale / 2, 0.35, 0])
            v2 = zh("4.2T token（11.7%）→ 推理追平 Qwen3-0.6B", 24, theme.OUTPUT).next_to(bar2, RIGHT, 0.3)
            self.play(FadeIn(lab1), GrowFromEdge(bar1, LEFT), run_time=self.fit(1.5))
            self.play(FadeIn(v1), run_time=self.fit(0.5))
            self.wait(self.remaining() * 0.15)
            self.play(FadeIn(lab2), GrowFromEdge(bar2, LEFT), run_time=self.fit(1.2))
            self.play(FadeIn(v2), run_time=self.fit(0.8))
            puro = card([("Puro-2B：只用公开数据 + 代理实验", 28, theme.FG),
                         ("约 $4.4K 算力 ≈ Qwen2-1.5B", 30, theme.HIGHLIGHT)], 9.0, 1.4, theme.PARAM)
            puro.move_to([0, -1.5, 0])
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(puro, shift=UP * 0.2), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.fade_all(lab1, bar1, v1, lab2, bar2, v2, puro)

        # ── S03 开放数据集与许可证 ───────────────────────────────────────
        with self.shot("S03"):
            self.play(*self.set_heading("开放数据集：每一份都要核对许可证"), run_time=self.fit(0.8))
            ds = [
                ("FineWeb-Edu", "英 · 1.3T token", "ODC-By 1.0", False),
                ("DCLM-baseline", "英 · 4T token", "CC-BY-4.0 · 卡片写仅供研究", True),
                ("FineWeb-2 中文", "中 · 6.36 亿篇", "ODC-By 1.0", False),
                ("Ultra-FineWeb", "中 120B · 英 1T", "Apache-2.0 · 上游不一", True),
                ("Stack-Edu", "代码 · 125B token", "只含文件 id · 看 Stack v2", True),
                ("FineMath-3+", "数学 · 34B token", "ODC-By 1.0", False),
            ]
            cards = VGroup()
            for name, size, lic, warn in ds:
                col = theme.HIGHLIGHT if warn else theme.INPUT
                cards.add(card([(name, 26, theme.FG), (size, 22, theme.MUTED), (lic, 20, col)],
                               4.2, 1.6, col))
            cards.arrange_in_grid(rows=2, cols=3, buff=(0.35, 0.4)).move_to([0, 0.35, 0])
            self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.15) for c in cards], lag_ratio=0.25),
                      run_time=self.fit(4))
            note = zh("黄色 = 待核实：下载器默认拒绝", 26, theme.HIGHLIGHT).move_to([0, -2.2, 0])
            self.wait(self.remaining() * 0.55)
            self.play(FadeIn(note), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.fade_all(cards, note)

        # ── S04 流水线与一份脏网页 ───────────────────────────────────────
        with self.shot("S04"):
            self.play(*self.set_heading("一条流水线：便宜的步骤放前面"), run_time=self.fit(0.8))
            steps = ["抽取", "语言识别", "启发式规则", "去重", "模型打分", "去污染", "分词分片"]
            boxes = VGroup(*[card([(s, 22, theme.FG)], 1.62, 0.7, theme.INPUT) for s in steps])
            boxes.arrange(RIGHT, buff=0.3).move_to([0, 1.9, 0])
            arrows = VGroup(*[Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.02,
                                    stroke_width=3, max_tip_length_to_length_ratio=0.35,
                                    color=theme.MUTED) for i in range(len(steps) - 1)])
            self.play(LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.2), Create(arrows),
                      run_time=self.fit(2.5))
            cheap = zh("几微秒 / 篇", 20, theme.OUTPUT).next_to(boxes[2], DOWN, 0.15)
            dear = zh("要跑神经网络", 20, theme.GRAD).next_to(boxes[4], DOWN, 0.15)
            self.play(FadeIn(cheap), FadeIn(dear), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.25)
            badge = self.show_badge()
            kinds = ["good", "contaminated", "exact_dup", "near_dup", "nav", "spam", "garbled", "salad"]
            total = D["n_raw"]
            W = 12.5
            segs, labels = VGroup(), VGroup()
            x = -W / 2
            for k in kinds:
                n = D["kinds"][k]
                w = W * n / total
                seg = Rectangle(width=w, height=0.6, fill_color=KIND_COLOR[k], fill_opacity=0.85,
                                stroke_width=1, stroke_color=theme.BG).move_to([x + w / 2, -0.2, 0])
                segs.add(seg)
                x += w
            for k in kinds:
                lab = zh(f"{KIND_ZH[k]} {D['kinds'][k]}", 20, KIND_COLOR[k])
                labels.add(lab)
            labels.arrange_in_grid(rows=2, cols=4, buff=(0.5, 0.18)).move_to([0, -1.45, 0])
            title = zh(f"自己造的脏网页：{total} 篇，每篇都带标签", 26).move_to([0, 0.6, 0])
            self.play(FadeIn(title), LaggedStart(*[GrowFromEdge(s, LEFT) for s in segs], lag_ratio=0.1),
                      run_time=self.fit(2))
            self.play(FadeIn(labels), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.fade_all(boxes, arrows, cheap, dear, segs, labels, title, badge)

        # ── S05 启发式规则 ───────────────────────────────────────────────
        with self.shot("S05"):
            self.play(*self.set_heading("启发式规则：抓格式，抓不住“不像话”"), run_time=self.fit(0.8))
            badge = self.show_badge()
            rules = VGroup(
                zh("Gopher：词数 50–100,000", 22), zh("Gopher：停用词 ≥ 2 个", 22),
                zh("Gopher：重复行 ≤ 30%", 22), zh("C4：lorem ipsum、花括号", 22),
                zh("FineWeb：以标点结尾的行 > 12%", 22),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).move_to([-4.2, 0.6, 0])
            self.play(LaggedStart(*[FadeIn(r) for r in rules], lag_ratio=0.3), run_time=self.fit(3))
            show = ["nav", "spam", "garbled", "salad", "good"]
            bars = VGroup()
            for i, k in enumerate(show):
                frac = D["heur_removed"].get(k, 0) / D["kinds"][k]
                y = 1.9 - i * 0.75
                lab = zh(KIND_ZH[k], 22, KIND_COLOR[k]).move_to([0.2, y, 0])
                back = Rectangle(width=4.0, height=0.42, stroke_color=theme.MUTED, stroke_width=1)
                back.move_to([3.2, y, 0])
                col = theme.GRAD if k == "good" else KIND_COLOR[k]
                fill = Rectangle(width=max(4.0 * frac, 0.02), height=0.42, fill_color=col,
                                 fill_opacity=0.85, stroke_width=0)
                fill.align_to(back, LEFT).set_y(y)
                pct = zh(f"删 {frac:.0%}", 20).next_to(back, RIGHT, 0.15)
                bars.add(VGroup(lab, back, fill, pct))
            head = zh(f"{D['n_raw']} → {D['n_heur']} 篇", 24, theme.HIGHLIGHT).move_to([3.2, 2.6, 0])
            self.wait(self.remaining() * 0.1)
            self.play(FadeIn(head), LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.25),
                      run_time=self.fit(3))
            ex = VGroup(mono("QUEEN ELIZABETH:", 20, theme.GRAD), mono("O thou well skill'd in curses,", 18),
                        mono("QUEEN MARGARET:", 20, theme.GRAD), mono("Forbear to sleep the nights,", 18),
                        mono("QUEEN ELIZABETH:", 20, theme.GRAD))
            ex.arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to([-4.0, -1.85, 0])
            exl = zh("好文档被误杀：人名行 = 重复行", 22, theme.GRAD).next_to(ex, RIGHT, 0.5)
            self.wait(self.remaining() * 0.45)
            self.play(FadeOut(rules), run_time=self.fit(0.5))
            self.play(FadeIn(ex), FadeIn(exl), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.fade_all(bars, head, ex, exl, badge)

        # ── S06 去重：Jaccard ────────────────────────────────────────────
        with self.shot("S06"):
            self.play(*self.set_heading("去重：两篇文档有多像？"), run_time=self.fit(0.8))
            a = VGroup(zh("原文", 22, theme.MUTED), mono("Now is the winter of", 22),
                       mono("our discontent made", 22), mono("glorious summer ...", 22))
            b = VGroup(zh("转载", 22, theme.MUTED), mono("Now is the winter of", 22),
                       mono("our discontent made", 22), mono("glorious autumn ...", 22))
            b[3][9:15].set_color(theme.GRAD)
            a.arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([-3.6, 1.4, 0])
            b.arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([3.0, 1.4, 0])
            self.play(FadeIn(a), FadeIn(b), run_time=self.fit(1.2))
            hash_a = mono("sha1 → 3f9c…", 22, theme.PARAM).next_to(a, DOWN, 0.3)
            hash_b = mono("sha1 → a07e…", 22, theme.PARAM).next_to(b, DOWN, 0.3)
            self.wait(self.remaining() * 0.25)
            self.play(FadeIn(hash_a), FadeIn(hash_b), run_time=self.fit(1))
            diff = zh("改一个词，哈希全变", 24, theme.GRAD).move_to([0, -0.4, 0])
            self.play(FadeIn(diff), run_time=self.fit(0.8))
            ca = Circle(radius=0.95, color=theme.INPUT, fill_opacity=0.15).move_to([-2.35, -1.75, 0])
            cb = Circle(radius=0.95, color=theme.PARAM, fill_opacity=0.15).move_to([-1.25, -1.75, 0])
            la = zh("A 的 5-gram", 20, theme.INPUT).next_to(ca, LEFT, 0.2)
            lb = zh("B 的 5-gram", 20, theme.PARAM).next_to(cb, RIGHT, 0.2)
            form = MathTex(r"J(A,B)=\frac{|A\cap B|}{|A\cup B|}", font_size=40).move_to([3.9, -1.75, 0])
            self.wait(self.remaining() * 0.25)
            self.play(FadeOut(diff), Create(ca), Create(cb), FadeIn(la), FadeIn(lb), run_time=self.fit(1.2))
            self.play(Write(form), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.fade_all(a, b, hash_a, hash_b, ca, cb, la, lb, form)

        # ── S07 MinHash ─────────────────────────────────────────────────
        with self.shot("S07"):
            self.play(*self.set_heading("MinHash：只看哈希值最小的那个元素"), run_time=self.fit(0.8))
            union = ["a", "b", "c", "d", "e", "f", "g", "h"]
            A, B = {"a", "b", "c", "d", "e", "f"}, {"c", "d", "e", "f", "g", "h"}
            orders = [["e", "a", "g", "b", "c", "h", "d", "f"], ["h", "c", "a", "f", "b", "d", "g", "e"],
                      ["a", "h", "d", "e", "g", "b", "f", "c"]]
            rows = VGroup()
            for k, order in enumerate(orders):
                cells = VGroup()
                for i, e in enumerate(order):
                    inA, inB = e in A, e in B
                    col = theme.OUTPUT if (inA and inB) else (theme.INPUT if inA else theme.PARAM)
                    box = RoundedRectangle(width=0.62, height=0.55, corner_radius=0.06, color=col,
                                           fill_color=col, fill_opacity=0.15)
                    cells.add(VGroup(box, mono(e, 22)).move_to([-2.5 + i * 0.7, 1.9 - k * 0.95, 0]))
                lab = zh(f"哈希 {k + 1}：从小到大", 20, theme.MUTED).next_to(cells, LEFT, 0.3)
                minA = next(i for i, e in enumerate(order) if e in A)
                minB = next(i for i, e in enumerate(order) if e in B)
                ma = zh(f"A 最小 {order[minA]}", 20, theme.INPUT)
                mb = zh(f"B 最小 {order[minB]}", 20, theme.PARAM)
                same = order[minA] == order[minB]
                verdict = zh("相等" if same else "不等", 22, theme.OUTPUT if same else theme.GRAD)
                info = VGroup(ma, mb, verdict).arrange(RIGHT, buff=0.25).next_to(cells, RIGHT, 0.35)
                rows.add(VGroup(lab, cells, info))
            legend = VGroup(zh("只在 A", 20, theme.INPUT), zh("A 和 B 都有", 20, theme.OUTPUT),
                            zh("只在 B", 20, theme.PARAM)).arrange(RIGHT, buff=0.5).move_to([0, -0.9, 0])
            self.play(FadeIn(legend), run_time=self.fit(0.6))
            for row in rows:
                self.play(FadeIn(row), run_time=self.fit(1.2))
                self.wait(self.remaining() * 0.08)
            j = len(A & B) / len(set(union))
            form = MathTex(r"P\big(\min h(A)=\min h(B)\big)=J(A,B)", font_size=38).move_to([0, -1.65, 0])
            jv = zh(f"这里 J = 4/8 = {j:.1f}", 22, theme.HIGHLIGHT).next_to(form, RIGHT, 0.4)
            self.wait(self.remaining() * 0.2)
            self.play(Write(form), FadeIn(jv), run_time=self.fit(1.2))
            sig = zh("128 个哈希函数 → 128 位签名；相等的位数占比 ≈ J", 24).move_to([0, -2.35, 0])
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(sig), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.fade_all(rows, legend, form, jv, sig)

        # ── S08 LSH 的 S 曲线 ────────────────────────────────────────────
        with self.shot("S08"):
            self.play(*self.set_heading("LSH：切成 16 段，任一段相同就是候选"), run_time=self.fit(0.8))
            badge = self.show_badge()
            segs = VGroup(*[Rectangle(width=0.32, height=0.5, color=theme.ATTN, fill_color=theme.ATTN,
                                      fill_opacity=0.2 + 0.03 * (i % 2), stroke_width=1.5)
                            for i in range(16)]).arrange(RIGHT, buff=0.05).move_to([-3.6, 2.2, 0])
            sl = zh("16 段 × 8 行", 20, theme.MUTED).next_to(segs, DOWN, 0.15)
            self.play(FadeIn(segs), FadeIn(sl), run_time=self.fit(1))
            ax = Axes(x_range=[0, 1, 0.2], y_range=[0, 1, 0.25], x_length=5.6, y_length=3.4,
                      axis_config={"color": theme.MUTED, "include_numbers": True, "font_size": 20},
                      tips=False).move_to([2.9, 0.15, 0])
            xl = zh("相似度 s", 20, theme.MUTED).next_to(ax, DOWN, 0.1)
            yl = zh("成为候选的概率", 20, theme.MUTED).rotate(math.pi / 2).next_to(ax, LEFT, 0.15)
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=self.fit(1))
            pts = [(s / 100, 1 - (1 - (s / 100) ** 8) ** 16) for s in range(0, 101)]
            curve = polyline_in_axes(ax, pts, color=theme.ATTN, stroke_width=4)
            form = MathTex(r"P(s)=1-(1-s^{8})^{16}", font_size=34, color=theme.ATTN).move_to([-3.6, 0.9, 0])
            self.play(Write(form), Create(curve), run_time=self.fit(2))
            knee = (1 / 16) ** (1 / 8)
            kl = Line(ax.c2p(knee, 0), ax.c2p(knee, 1), color=theme.HIGHLIGHT, stroke_width=2)
            kt = zh(f"拐点 ≈ {knee:.2f}", 20, theme.HIGHLIGHT).next_to(ax.c2p(knee, 1), UP, 0.08)
            self.play(Create(kl), FadeIn(kt), run_time=self.fit(0.8))
            dots = VGroup(*[Dot(ax.c2p(s, emp), color=theme.OUTPUT, radius=0.07) for s, emp, _ in D["s_curve"]])
            dl = zh("实测（每档 1000 对）", 20, theme.OUTPUT).move_to([-3.6, 0.1, 0])
            self.wait(self.remaining() * 0.2)
            self.play(LaggedStart(*[FadeIn(d, scale=1.5) for d in dots], lag_ratio=0.2), FadeIn(dl),
                      run_time=self.fit(1.5))
            r = D["redundant"]
            red = VGroup(zh("多余副本", 22, theme.MUTED),
                         zh(f"{r[0]} → {r[1]} → {r[2]}", 34, theme.HIGHLIGHT),
                         zh("（去重前 → 精确 → MinHash）", 18, theme.MUTED)).arrange(DOWN, buff=0.12)
            red.move_to([-3.6, -1.4, 0])
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(red), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.fade_all(segs, sl, ax, xl, yl, curve, form, kl, kt, dots, dl, red, badge)

        # ── S09 基于模型的质量过滤 ───────────────────────────────────────
        with self.shot("S09"):
            self.play(*self.set_heading("模型打分：大模型标一小部分，小分类器打全部"), run_time=self.fit(0.8))
            flow = VGroup(card([("Llama-3-70B", 22, theme.FG), ("标 46 万个网页 0–5 分", 20, theme.MUTED)],
                               3.6, 1.0, theme.ATTN),
                          card([("小分类器", 22, theme.FG), ("学会这个打分", 20, theme.MUTED)], 3.0, 1.0,
                               theme.PARAM),
                          card([("给 15T token 打分", 22, theme.FG), ("留 ≥3 分：删掉 92%", 20, theme.HIGHLIGHT)],
                               3.6, 1.0, theme.OUTPUT)).arrange(RIGHT, buff=0.55).move_to([0, 1.9, 0])
            arr = VGroup(*[Arrow(flow[i].get_right(), flow[i + 1].get_left(), buff=0.05, color=theme.MUTED,
                                 stroke_width=3) for i in range(2)])
            tag = zh("FineWeb-Edu", 22, theme.MUTED).next_to(flow, UP, 0.12)
            self.play(FadeIn(tag), LaggedStart(*[FadeIn(f) for f in flow], lag_ratio=0.4), Create(arr),
                      run_time=self.fit(3))
            self.wait(self.remaining() * 0.25)
            badge = self.show_badge()
            rows = [["阈值", "抓到差文档", "漏掉", "误伤好文档", "精确率", "召回率"]]
            for th, tp, fn, fp, prec, rec in D["clf"]:
                rows.append([f"{th}", str(tp), str(fn), str(fp), f"{prec:.2f}", f"{rec:.2f}"])
            tb = table(rows, [1.2, 1.9, 1.1, 1.9, 1.3, 1.3], size=22).move_to([0, -0.9, 0])
            cap = zh("玩具版：评审员标 250 篇 → 逻辑回归（最有用的特征：相邻词对有多眼熟）", 20,
                     theme.MUTED).next_to(tb, UP, 0.2)
            self.play(FadeIn(cap), FadeIn(tb), run_time=self.fit(1.2))
            hl = SurroundingRectangle(tb[2], color=theme.HIGHLIGHT, buff=0.06)
            self.wait(self.remaining() * 0.3)
            self.play(Create(hl), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.6)
            self.fade_all(flow, arr, tag, tb, cap, hl, badge)

        # ── S10 合成改写 ─────────────────────────────────────────────────
        with self.shot("S10"):
            self.play(*self.set_heading("合成改写：让模型把文本换个写法重写"), run_time=self.fit(0.8))
            data = [("原文 × 10 遍", 23.76, theme.MUTED), ("改写 1 次 × 10 遍", 27.39, theme.PARAM),
                    ("改写 10 次 × 1 遍", 28.94, theme.OUTPUT)]
            base_y = -0.9
            bars = VGroup()
            for i, (lab, v, col) in enumerate(data):
                h = (v - 20) / 10 * 3.0
                x = -4.3 + i * 1.9
                bar = Rectangle(width=1.1, height=h, fill_color=col, fill_opacity=0.85, stroke_width=0)
                bar.move_to([x, base_y + h / 2, 0])
                val = zh(f"{v:.2f}", 22, col).next_to(bar, UP, 0.1)
                name = zh(lab, 18, theme.FG).next_to(bar, DOWN, 0.15)
                bars.add(VGroup(bar, val, name))
            cap = zh("Kimi K2：同一份维基文本，SimpleQA 准确率（y 轴从 20 起）", 20, theme.MUTED)
            cap.move_to([-2.4, 2.45, 0])
            self.play(FadeIn(cap), run_time=self.fit(0.6))
            self.play(LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in bars], lag_ratio=0.4),
                      LaggedStart(*[FadeIn(VGroup(b[1], b[2])) for b in bars], lag_ratio=0.4),
                      run_time=self.fit(3))
            side = VGroup(card([("Nemotron-CC", 24, theme.FG), ("6.3T 里 1.9T 是改写/合成", 20, theme.MUTED)],
                               4.0, 1.1, theme.ATTN),
                          card([("Phi-4", 24, theme.FG), ("合成 40% + 网页改写 15%", 20, theme.MUTED)],
                               4.0, 1.1, theme.ATTN),
                          card([("风险", 24, theme.GRAD), ("幻觉；改写模型的许可证", 20, theme.MUTED)],
                               4.0, 1.1, theme.GRAD)).arrange(DOWN, buff=0.3).move_to([3.9, 0.35, 0])
            self.wait(self.remaining() * 0.15)
            self.play(LaggedStart(*[FadeIn(s) for s in side[:2]], lag_ratio=0.5), run_time=self.fit(1.5))
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(side[2]), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.fade_all(bars, cap, side)

        # ── S11 配比与代理实验 ───────────────────────────────────────────
        with self.shot("S11"):
            self.play(*self.set_heading("配比：靠对照实验，不靠公式"), run_time=self.fit(0.8))
            mix = [("英文网页", 0.45, theme.INPUT), ("中文网页", 0.30, theme.GRAD),
                   ("代码", 0.15, theme.OUTPUT), ("数学", 0.10, theme.PARAM)]
            from manim import AnnularSector  # noqa: PLC0415

            start = math.pi / 2
            pie, labs = VGroup(), VGroup()
            for name, w, col in mix:
                ang = -2 * math.pi * w
                sec = AnnularSector(inner_radius=0.0, outer_radius=1.55, angle=ang, start_angle=start,
                                    fill_color=col, fill_opacity=0.85, stroke_width=2,
                                    stroke_color=theme.BG).move_arc_center_to([-4.0, 0.1, 0])
                mid = start + ang / 2
                u = [math.cos(mid), math.sin(mid), 0]
                edge = [-4.0 + 1.65 * u[0], 0.1 + 1.65 * u[1], 0]
                lab = zh(f"{name} {w:.2f}", 22, col).next_to(edge, u, buff=0.1)
                pie.add(sec)
                labs.add(lab)
                start += ang
            pl = zh("主线暂定配比", 22, theme.MUTED).move_to([-4.0, -2.35, 0])
            steps = VGroup(zh("Puro-2B 的代理实验", 24, theme.HIGHLIGHT),
                           zh("① 同一个 Qwen3-0.6B checkpoint", 22),
                           zh("② 每个候选数据续训 84 亿 token", 22),
                           zh("③ 候选占比 0 → 80% 线性升上去", 22),
                           zh("④ 15 项基准 → 能力向量 → 定配比", 22)).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
            steps.move_to([2.9, 0.3, 0])
            self.wait(self.remaining() * 0.1)
            self.play(LaggedStart(*[FadeIn(s) for s in steps], lag_ratio=0.4), run_time=self.fit(3.5))
            self.wait(self.remaining() * 0.3)
            self.play(LaggedStart(*[FadeIn(p) for p in pie], lag_ratio=0.2), FadeIn(labs), FadeIn(pl),
                      run_time=self.fit(2))
            self.wait(self.remaining() - 0.6)
            self.fade_all(pie, labs, pl, steps)

        # ── S12 我们的两个消融 ───────────────────────────────────────────
        with self.shot("S12"):
            self.play(*self.set_heading("同样的算力，只换数据（配对比较 bpb）"), run_time=self.fit(0.8))
            badge = self.show_badge()
            b = ABL1["bpb"]
            lo, hi = 2.9, 3.6
            groups = VGroup()
            for gi, lang in enumerate(["en", "zh"]):
                for si in range(2):
                    for ci, (name, col) in enumerate([("原样脏网页", theme.GRAD), ("过滤后", theme.OUTPUT)]):
                        v = b[name][lang][si]
                        h = (v - lo) / (hi - lo) * 2.5
                        x = -5.6 + gi * 2.9 + si * 1.3 + ci * 0.5
                        bar = Rectangle(width=0.45, height=h, fill_color=col, fill_opacity=0.85,
                                        stroke_width=0).move_to([x, -1.2 + h / 2, 0])
                        val = mono(f"{v:.3f}", 14, col).rotate(math.pi / 2).next_to(bar, UP, 0.08)
                        groups.add(VGroup(bar, val))
            glabels = VGroup(zh("英文 种子0 / 1", 18, theme.MUTED).move_to([-4.8, -1.5, 0]),
                             zh("中文 种子0 / 1", 18, theme.MUTED).move_to([-1.9, -1.5, 0]))
            legend = VGroup(zh("原样脏网页", 18, theme.GRAD), zh("过滤后", 18, theme.OUTPUT)).arrange(
                RIGHT, buff=0.4).move_to([-3.35, 2.45, 0])
            axis_note = zh(f"验证集 bpb（y 轴从 {lo} 起，越低越好）", 16, theme.MUTED).move_to([-3.35, 2.05, 0])
            self.play(FadeIn(legend), FadeIn(axis_note), run_time=self.fit(0.6))
            self.play(LaggedStart(*[GrowFromEdge(g[0], DOWN) for g in groups], lag_ratio=0.1),
                      FadeIn(VGroup(*[g[1] for g in groups])), FadeIn(glabels), run_time=self.fit(2.5))
            d_en = [x - y for x, y in zip(b["原样脏网页"]["en"], b["过滤后"]["en"])]
            d_zh = [x - y for x, y in zip(b["原样脏网页"]["zh"], b["过滤后"]["zh"])]
            diff = zh(f"配对差值：英 {d_en[0]:.3f} / {d_en[1]:.3f}  中 {d_zh[0]:.3f} / {d_zh[1]:.3f}", 18,
                      theme.HIGHLIGHT).move_to([-3.35, -2.05, 0])
            self.wait(self.remaining() * 0.15)
            self.play(FadeIn(diff), run_time=self.fit(0.8))
            # 右边：配比
            m = ABL2["bpb"]
            rows = [["配比（英/中/码）", "英文", "中文", "代码"]]
            mixes = {"均衡": "0.45/0.45/0.10", "英文为主": "0.80/0.10/0.10", "代码为主": "0.25/0.25/0.50"}
            best = {k: min(m, key=lambda n: m[n][k]) for k in ["en", "zh", "code"]}
            for name in m:
                rows.append([f"{name} {mixes.get(name, '')}", *[f"{m[name][k]:.3f}" for k in ["en", "zh", "code"]]])
            tb = table(rows, [2.9, 1.0, 1.0, 1.0], size=18).move_to([3.6, 0.4, 0])
            for j, k in enumerate(["en", "zh", "code"]):
                i = list(m).index(best[k]) + 1
                tb[i][j + 1].set_color(theme.OUTPUT)
            cap = zh("三种配比（绿色 = 该领域最低）", 18, theme.MUTED).next_to(tb, UP, 0.2)
            self.wait(self.remaining() * 0.25)
            self.play(FadeIn(cap), FadeIn(tb), run_time=self.fit(1.2))
            self.wait(self.remaining() - 0.6)
            self.fade_all(groups, glabels, legend, axis_note, diff, tb, cap, badge)

        # ── S13 去污染 ───────────────────────────────────────────────────
        with self.shot("S13"):
            self.play(*self.set_heading("去污染：训练数据不能见过考题"), run_time=self.fit(0.8))
            badge = self.show_badge()
            words = D["leak_example"]["question"].split()
            qline1 = mono(" ".join(words[:15]), 17, theme.FG)
            qline2 = mono(" ".join(words[15:30]), 17, theme.FG)
            q = VGroup(zh("考题", 20, theme.ATTN), qline1, qline2).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            if q.width > 12.8:
                q.scale_to_fit_width(12.8)
            q.move_to([0, 2.0, 0])
            self.play(FadeIn(q), run_time=self.fit(1))
            gram = mono("13-gram：" + " ".join(words[:13]), 17, theme.HIGHLIGHT).move_to([0, 0.95, 0])
            if gram.width > 12.6:
                gram.scale_to_fit_width(12.6)
            box = SurroundingRectangle(gram, color=theme.HIGHLIGHT, buff=0.08)
            self.wait(self.remaining() * 0.12)
            self.play(FadeIn(gram), Create(box), run_time=self.fit(1))
            v = D["decontam"]["variants"]
            rows = [["n", "原样", "改大小写标点", "改写", "其它命中"]]
            for n, _, caught, other in D["decontam"]["rows"]:
                rows.append([str(n), f"{caught['verbatim']}/{v['verbatim']}",
                             f"{caught['case_punct']}/{v['case_punct']}",
                             f"{caught['paraphrase']}/{v['paraphrase']}", str(other)])
            tb = table(rows, [1.0, 1.4, 2.2, 1.2, 1.6], size=20).move_to([0, -1.1, 0])
            self.wait(self.remaining() * 0.15)
            self.play(FadeIn(tb), run_time=self.fit(1.2))
            hl = SurroundingRectangle(tb[3], color=theme.HIGHLIGHT, buff=0.05)
            self.play(Create(hl), run_time=self.fit(0.6))
            note = zh("n=13 的“其它命中”：同一首宋词在语料里出现了两次 → 真泄漏", 18, theme.GRAD)
            note.next_to(tb, DOWN, 0.2)
            self.wait(self.remaining() * 0.55)
            self.play(FadeIn(note), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.fade_all(q, gram, box, tb, hl, note, badge)

        # ── S14 主线分词器 ───────────────────────────────────────────────
        with self.shot("S14"):
            self.play(*self.set_heading("词表：比“读同样多的文字要花多少算力”"), run_time=self.fit(0.8))
            zero_rows = [r for r in VOCAB if r["label"].startswith("zero")]
            vs = [r["V"] for r in zero_rows]
            lx0, lx1 = math.log2(min(vs)) - 0.2, math.log2(max(vs)) + 0.2
            ys = [r["bpt"][k] for r in zero_rows for k in ("en", "zh", "code")]
            ymin, ymax = math.floor(min(ys) * 2) / 2, math.ceil(max(ys) * 2) / 2
            ax = Axes(x_range=[lx0, lx1, 1], y_range=[ymin, ymax, 0.5], x_length=5.4, y_length=3.6,
                      axis_config={"color": theme.MUTED}, y_axis_config={"include_numbers": True,
                                                                        "font_size": 18},
                      tips=False).move_to([-3.5, 0.2, 0])
            xt = VGroup(*[zh(f"{v // 1024}K" if v % 1024 == 0 else f"{v / 1024:.0f}K", 16, theme.MUTED)
                          .next_to(ax.c2p(math.log2(v), ymin), DOWN, 0.12)
                          for v in vs if v in (16384, 32768, 65536, 131072)])
            yl = zh("字节 / token", 18, theme.MUTED).next_to(ax, UP, 0.1).align_to(ax, LEFT)
            self.play(Create(ax), FadeIn(xt), FadeIn(yl), run_time=self.fit(1))
            lines = VGroup()
            for k, col, name in [("en", theme.INPUT, "英文"), ("zh", theme.GRAD, "中文"), ("code", theme.OUTPUT, "代码")]:
                pts = [(math.log2(r["V"]), r["bpt"][k]) for r in zero_rows]
                ln = polyline_in_axes(ax, pts, color=col, stroke_width=3)
                lab = zh(name, 18, col).next_to(ax.c2p(*pts[-1]), RIGHT, 0.1)
                lines.add(VGroup(ln, lab))
            self.play(LaggedStart(*[Create(g[0]) for g in lines], lag_ratio=0.3),
                      FadeIn(VGroup(*[g[1] for g in lines])), run_time=self.fit(2))
            rows = [["词表", "总参数", "FLOPs/字节"]]
            pick = [r for r in zero_rows if r["V"] in (32768, 65536, 98304, 131072, 151936)]
            for r in pick:
                tag = f"{r['V'] // 1024}K" if r["V"] % 1024 == 0 else f"{r['V']:,}"
                rows.append([tag, f"{r['total']:.0f}M", f"{r['fpb_rel']:.3f}"])
            tb = table(rows, [1.5, 1.6, 1.8], size=20).move_to([3.3, 0.9, 0])
            for i, r in enumerate(pick, start=1):
                if r["total"] > 800:
                    tb[i][1].set_color(theme.GRAD)
            limit = zh("总参数上限 0.8B（红色 = 超线）", 18, theme.GRAD).next_to(tb, DOWN, 0.2)
            self.wait(self.remaining() * 0.1)
            self.play(FadeIn(tb), FadeIn(limit), run_time=self.fit(1.2))
            chosen = VOCAB_CHOICE
            idx = next((i for i, r in enumerate(pick, start=1) if r["V"] == chosen), None)
            hl = SurroundingRectangle(tb[idx] if idx else tb[0], color=theme.HIGHLIGHT, buff=0.05)
            self.wait(self.remaining() * 0.25)
            self.play(Create(hl), run_time=self.fit(0.6))
            rx = VGroup(zh("预切分正则 → qwen3.5：", 20, theme.HIGHLIGHT),
                        zh("印地语 36 → 14 块，泰语 16 → 1 块，中英代码完全相同", 20)).arrange(RIGHT, buff=0.2)
            rx.move_to([0.8, -2.25, 0])
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(rx), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.fade_all(ax, xt, yl, lines, tb, limit, hl, rx)

        # ── S15 主线进度：生产级流水线 ───────────────────────────────────
        with self.shot("S15"):
            self.play(*self.set_heading("主线进度：一个配置，十个阶段，一份清单"), run_time=self.fit(0.8))
            badge = self.show_badge()
            cfg = card([("configs/tiny/data.toml", 22, theme.FG)], 3.8, 0.7, theme.PARAM).move_to([-4.4, 2.0, 0])
            man = card([("manifest.json", 22, theme.FG)], 2.8, 0.7, theme.OUTPUT).move_to([4.6, 2.0, 0])
            mid = zh("zero.data.pipeline", 22, theme.MUTED).move_to([0.1, 2.0, 0])
            a1 = Arrow(cfg.get_right(), mid.get_left(), buff=0.1, color=theme.MUTED)
            a2 = Arrow(mid.get_right(), man.get_left(), buff=0.1, color=theme.MUTED)
            self.play(FadeIn(cfg), Create(a1), FadeIn(mid), Create(a2), FadeIn(man), run_time=self.fit(1.5))
            f = MANIFEST["funnel"]
            stages = [s["stage"] for s in next(iter(f.values()))]
            stage_zh = {"raw": "读取", "clean": "清洗", "langid": "语言", "heuristics": "规则", "score": "打分",
                        "dedup": "去重", "decontam": "去污染"}
            rows = [["来源", *[stage_zh.get(s, s) for s in stages], "训练 token"]]
            for src, st in f.items():
                rows.append([src, *[str(s["docs"]) for s in st],
                             f"{MANIFEST['shards'][src]['train']['tokens']:,}"])
            tb = table(rows, [2.3] + [0.95] * len(stages) + [1.7], size=19).move_to([0, 0.2, 0])
            self.wait(self.remaining() * 0.1)
            self.play(FadeIn(tb), run_time=self.fit(1.2))
            extra = zh(f"分词器 {MANIFEST['tokenizer']['vocab_size']} 词表 · 用时 {MANIFEST['seconds']:.0f} 秒 · "
                       "许可证与出处写进清单", 20, theme.MUTED).move_to([0, -1.35, 0])
            self.play(FadeIn(extra), run_time=self.fit(0.8))
            todo = zh("真实数据的下载、过滤与配比实验：待 GPU 训练后补充", 24, theme.HIGHLIGHT).move_to([0, -2.1, 0])
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(todo), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.fade_all(cfg, man, mid, a1, a2, tb, extra, todo, badge)

        # ── S16 小结 ─────────────────────────────────────────────────────
        with self.shot("S16"):
            self.play(*self.set_heading("小结"), run_time=self.fit(0.8))
            items = VGroup(
                zh("规则抓格式", 30, theme.INPUT), zh("去重抓副本", 30, theme.PARAM),
                zh("分类器抓不像话的文字", 30, theme.ATTN), zh("去污染守住考题", 30, theme.GRAD),
                zh("配比靠对照实验，词表看算力账", 30, theme.OUTPUT),
            ).arrange(DOWN, buff=0.35).move_to([0, 0.45, 0])
            self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.2) for i in items], lag_ratio=0.5),
                      run_time=self.fit(5))
            nxt = zh("下一章：预训练工程", 28, theme.HIGHLIGHT).move_to([0, -2.2, 0])
            self.wait(self.remaining() * 0.5)
            self.play(FadeIn(nxt), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(items), FadeOut(nxt), *self.set_heading(None), run_time=self.fit(0.8))

