"""Chapter 7 video: language modeling and tokenization.

From "guess the next character" to byte-level BPE.

The code in ../code/ calculates all numbers in the frames (see the fact list in script.md).
The results of the slow calculations (BPE training, the vocabulary-size sweep, the bigram
evaluation) are cached in video/out/cache.json.
Render: bash chapters/07-tokenization-language-model/video/build.sh
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
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
    CurvedArrow,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    LaggedStart,
    Line,
    ManimColor,
    MathTex,
    Rectangle,
    RoundedRectangle,
    SurroundingRectangle,
    Text,
    Transform,
    VGroup,
    Write,
    interpolate_color,
)

from video_kit import theme
from video_kit.scene import NarratedScene, code_block, polyline_in_axes, zh

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
REPO = HERE.parents[2]
CACHE = HERE / "out" / "cache.json"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bpe_mod = _load("ch07_bpe", "02_bpe.py")
bigram = _load("ch07_bigram", "03_bigram.py")

MERGE_TEXT = "不亦说乎，"
VOCAB_SWEEP = [272, 512, 1024, 2048, 4096, 8192, 16384, 32768]
HEAT_CHARS = [" ", "e", "t", "o", "a", "h", "s", "n", "r", "i"]


def merge_steps(bpe, text: str) -> list[list[list[int]]]:
    """Record the state after each merge when text is encoded.

    State = list of tokens. Each token is its list of bytes.
    """
    chunks = [list(c.encode("utf-8")) for c in bpe_mod.pretokenize(text)]
    ids = [list(c) for c in chunks]  # id sequence of each chunk
    states = []

    def snapshot():
        states.append([list(bpe.vocab[i]) for c in ids for i in c])

    snapshot()
    for ci in range(len(ids)):
        while len(ids[ci]) >= 2:
            pairs = list(zip(ids[ci], ids[ci][1:]))
            pair = min(pairs, key=lambda p: bpe.merges.get(p, float("inf")))
            if pair not in bpe.merges:
                break
            ids[ci] = bpe_mod.merge(ids[ci], pair, bpe.merges[pair])
            snapshot()
    return states


def compute_data() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text("utf-8"))
    sys.path.insert(0, str(REPO))
    from zero.tokenizer import train_bpe  # noqa: PLC0415

    train, val = bpe_mod.load_splits()
    bpe = bpe_mod.BPE().train("\n".join(train.values()), vocab_size=1024)
    data: dict = {}
    data["history"] = [(i, list(b), c) for i, b, c in bpe.history[:18]]
    data["merge_states"] = merge_steps(bpe, MERGE_TEXT)
    data["ratio_mine"] = {k: len(v.encode()) / len(bpe.encode(v)) for k, v in val.items()}
    prod = train_bpe(["\n".join(train.values())], vocab_size=1024 + 16)
    data["ratio_zero"] = {k: len(v.encode()) / len(prod.encode(v)) for k, v in val.items()}

    big = {k: (bpe_mod.CORPUS / f).read_text("utf-8")[:-40_000] for k, f in bpe_mod.FILES.items()}
    sweep = []
    for V in VOCAB_SWEEP:
        tok = train_bpe(list(big.values()), vocab_size=V)
        sweep.append((tok.vocab_size, sum(len(tok.encode(v)) for v in val.values())))
    data["sweep"] = sweep

    # bigram: evaluate three tokenizations on three corpora, and sample
    evals, samples = {}, {}
    for k in ["en", "zh", "code"]:
        tr, va = bigram.splits(k)
        char = bigram.CharTok(tr)
        for name, tok in [("字节", bigram.ByteTok()), ("字符", char), ("BPE", bigram.BPETok(bpe))]:
            if name == "字符":
                nb = [len(c.encode()) for c in va]
            else:
                nb = [len(tok.token_bytes(i)) for i in tok.encode(va)]
            r = bigram.run(tok, tr, va, lambda ids, nb=nb: nb)
            evals[f"{k}|{name}"] = {x: float(r[x]) for x in ("nats", "bits", "ppl", "bpb")}
            if (k, name) in [("en", "BPE"), ("zh", "字符")]:
                samples[k] = bigram.sample(r["counts"], tok, tok.encode("\n")[0],
                                           30 if name == "BPE" else 50, seed=1)
    data["evals"], data["samples"] = evals, samples

    # Character bigram counts for English (heat map)
    tr, _ = bigram.splits("en")
    char = bigram.CharTok(tr)
    counts = bigram.count_bigrams(char.encode(tr), char.V)
    idx = [char.stoi[c] for c in HEAT_CHARS]
    data["heat"] = counts[np.ix_(idx, idx)].astype(int).tolist()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    return data


D = compute_data()

MONO = "DejaVu Sans Mono"


def mono(text: str, size: float = 24, color: str = theme.FG) -> Text:
    return Text(text, font=MONO, font_size=size, color=color)


def show_bytes(b: list[int]) -> str:
    """Show the text of a token if its bytes decode as UTF-8. Otherwise show hex."""
    try:
        return bytes(b).decode("utf-8").replace(" ", "␣").replace("\n", "↵")
    except UnicodeDecodeError:
        return " ".join(f"{x:02x}" for x in b)


def token_box(b: list[int], unit: float = 0.62, color=theme.INPUT, size: float = 22) -> VGroup:
    """Box for one token: the width is proportional to the number of bytes."""
    w = unit * len(b) - 0.08
    rect = RoundedRectangle(width=w, height=0.62, corner_radius=0.08, color=color,
                            fill_color=color, fill_opacity=0.18, stroke_width=2)
    text = show_bytes(b)
    label = zh(text, size) if not all(c in "0123456789abcdef " for c in text) else mono(text, size - 4)
    if label.width > w - 0.08:
        label.scale_to_fit_width(w - 0.08)
    label.move_to(rect)
    return VGroup(rect, label)


def token_row(tokens: list[list[int]], y: float, unit: float = 0.62, **kw) -> VGroup:
    boxes = [token_box(t, unit, **kw) for t in tokens]
    total = sum(unit * len(t) for t in tokens)
    x = -total / 2
    for box, t in zip(boxes, tokens):
        box.move_to([x + unit * len(t) / 2, y, 0])
        x += unit * len(t)
    return VGroup(*boxes)


class ChapterScene(NarratedScene):
    chapter_label = "第 7 章"
    chapter_title = "语言建模与分词"

    def construct(self) -> None:
        # ── S01 Opening ──────────────────────────────────────────────────
        with self.shot("S01"):
            card = self.chapter_card()
            sub = zh("从“猜下一个字”到 byte-level BPE", 32, theme.HIGHLIGHT).next_to(card, DOWN, 0.6)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(card), FadeOut(sub), run_time=self.fit(0.8))

        # ── S02 A language model: guess the next token ───────────────────
        with self.shot("S02"):
            self.play(*self.set_heading("语言模型：已知前文，猜下一个 token"), run_time=self.fit(0.8))
            chars = "学而时习之"
            boxes = VGroup(*[
                VGroup(RoundedRectangle(width=0.9, height=0.9, corner_radius=0.1, color=theme.INPUT,
                                        fill_color=theme.INPUT, fill_opacity=0.15),
                       zh(c, 40)) for c in chars
            ]).arrange(RIGHT, buff=1.55).move_to([0, 1.7, 0])
            self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in boxes], lag_ratio=0.3),
                      run_time=self.fit(2))
            conds = VGroup()
            for i in range(1, len(chars)):
                prev = chars[:i] if i <= 2 else "…" + chars[i - 1]
                lab = zh(f"p({chars[i]} | {prev})", 22, theme.OUTPUT).next_to(boxes[i], DOWN, 0.3)
                conds.add(lab)
            first = zh(f"p({chars[0]})", 22, theme.OUTPUT).next_to(boxes[0], DOWN, 0.3)
            conds.add_to_back(first)
            self.play(LaggedStart(*[FadeIn(c) for c in conds], lag_ratio=0.35), run_time=self.fit(3))
            chain = MathTex(r"p(x_1,\dots,x_T)=\prod_{t=1}^{T} p(x_t \mid x_{<t})", font_size=46)
            chain.move_to([0, -0.3, 0])
            self.play(Write(chain), run_time=self.fit(2))
            loss = MathTex(r"L=-\frac{1}{T}\sum_{t} \log p(x_t \mid x_{<t})", font_size=40,
                           color=theme.GRAD).move_to([-1.6, -1.75, 0])
            note = zh("= 第 5 章的交叉熵\n   类别 = 词表", 24, theme.MUTED).next_to(loss, RIGHT, 0.4)
            self.wait(self.remaining() * 0.25)
            self.play(Write(loss), run_time=self.fit(1.5))
            self.play(FadeIn(note), run_time=self.fit(0.8))
            q = zh("token 到底是什么？", 30, theme.HIGHLIGHT).move_to([0, 2.75, 0])
            self.wait(self.remaining() - 2.2)
            self.play(FadeIn(q), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [boxes, conds, chain, loss, note, q]],
                      run_time=self.fit(0.6))

        # ── S03 Split by characters ──────────────────────────────────────
        with self.shot("S03"):
            self.play(*self.set_heading("按字符切：每个不同的字符一个编号"), run_time=self.fit(0.8))
            en = VGroup(*[token_box(list(c.encode()), 0.7, size=26) for c in "To be"]).arrange(RIGHT, buff=0.08)
            zhrow = VGroup(*[
                VGroup(RoundedRectangle(width=0.62, height=0.62, corner_radius=0.08, color=theme.INPUT,
                                        fill_color=theme.INPUT, fill_opacity=0.18, stroke_width=2),
                       zh(c, 26)) for c in "学而时习之"
            ]).arrange(RIGHT, buff=0.08)
            en.move_to([-3.5, 1.5, 0])
            zhrow.move_to([3.5, 1.5, 0])
            self.play(FadeIn(en), FadeIn(zhrow), run_time=self.fit(1))
            en_stat = VGroup(zh("英文（莎士比亚）", 26, theme.MUTED),
                             zh("65 种字符", 40, theme.OUTPUT)).arrange(DOWN, buff=0.25)
            zh_stat = VGroup(zh("中文（诗词，约 43 万字）", 26, theme.MUTED),
                             zh("5,297 种字", 40, theme.GRAD)).arrange(DOWN, buff=0.25)
            en_stat.move_to([-3.5, 0.0, 0])
            zh_stat.move_to([3.5, 0.0, 0])
            self.wait(self.remaining() * 0.12)
            self.play(FadeIn(en_stat), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.12)
            self.play(FadeIn(zh_stat), run_time=self.fit(0.8))
            unk = VGroup(RoundedRectangle(width=1.6, height=0.7, corner_radius=0.1, color=theme.GRAD,
                                          fill_color=theme.GRAD, fill_opacity=0.25),
                         mono("<unk>", 26, theme.GRAD))
            unk_txt = zh("后 10% 里有 198 个字训练时没见过", 26, theme.FG)
            row = VGroup(unk_txt, unk).arrange(RIGHT, buff=0.4).move_to([0, -1.7, 0])
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(unk_txt), run_time=self.fit(0.8))
            self.play(FadeIn(unk, scale=1.3), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [en, zhrow, en_stat, zh_stat, row]], run_time=self.fit(0.6))

        # ── S04 Split by bytes ───────────────────────────────────────────
        with self.shot("S04"):
            self.play(*self.set_heading("按字节切：UTF-8，词表固定 256"), run_time=self.fit(0.8))
            xue = VGroup(RoundedRectangle(width=1.1, height=1.1, corner_radius=0.1, color=theme.INPUT,
                                          fill_color=theme.INPUT, fill_opacity=0.15),
                         zh("学", 52)).move_to([-4.2, 1.5, 0])
            bs = list("学".encode())
            bx = VGroup(*[
                VGroup(RoundedRectangle(width=1.0, height=0.8, corner_radius=0.08, color=theme.PARAM,
                                        fill_color=theme.PARAM, fill_opacity=0.15),
                       mono(str(b), 28)) for b in bs
            ]).arrange(RIGHT, buff=0.12).move_to([0.6, 1.5, 0])
            arr = Arrow(xue.get_right(), bx.get_left(), buff=0.2, color=theme.MUTED)
            utf = zh("UTF-8", 22, theme.MUTED).next_to(arr, UP, 0.08)
            self.play(FadeIn(xue), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.2)
            self.play(Create(arr), FadeIn(utf), LaggedStart(*[FadeIn(b) for b in bx], lag_ratio=0.3),
                      run_time=self.fit(1.5))
            rule = zh("英文字母 1 字节 · 常用汉字 3 字节", 26, theme.FG).move_to([0, 0.45, 0])
            self.play(FadeIn(rule), run_time=self.fit(0.8))
            s = "学而时习之"
            five = VGroup(*[
                VGroup(RoundedRectangle(width=1.86, height=0.55, corner_radius=0.08, color=theme.INPUT,
                                        fill_color=theme.INPUT, fill_opacity=0.15), zh(c, 26))
                for c in s
            ]).arrange(RIGHT, buff=0.08).move_to([0, -0.6, 0])
            fifteen = VGroup(*[
                VGroup(RoundedRectangle(width=0.58, height=0.55, corner_radius=0.06, color=theme.PARAM,
                                        fill_color=theme.PARAM, fill_opacity=0.15), mono(str(b), 16))
                for b in s.encode()
            ]).arrange(RIGHT, buff=0.06).move_to([0, -1.5, 0])
            l5 = zh("5 个字", 22, theme.MUTED).next_to(five, LEFT, 0.2)
            l15 = zh("15 个字节", 22, theme.MUTED).next_to(fifteen, LEFT, 0.2)
            if l15.get_left()[0] < -6.9:
                VGroup(five, fifteen).shift(RIGHT * 0.5)
                l5.next_to(five, LEFT, 0.2)
                l15.next_to(fifteen, LEFT, 0.2)
            self.wait(self.remaining() * 0.15)
            self.play(FadeIn(five), FadeIn(l5), run_time=self.fit(0.8))
            self.play(LaggedStart(*[FadeIn(b) for b in fifteen], lag_ratio=0.08), FadeIn(l15),
                      run_time=self.fit(1.8))
            cost = zh("代价：中文序列变长 2.8 倍", 28, theme.GRAD).move_to([0, -2.35, 0])
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(cost), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [xue, bx, arr, utf, rule, five, fifteen, l5, l15, cost]],
                      run_time=self.fit(0.6))

        # ── S05 Animation of the BPE merges ──────────────────────────────
        with self.shot("S05"):
            self.play(*self.set_heading("BPE：反复把最常见的相邻一对合并成新 token"), run_time=self.fit(0.8))
            algo = VGroup(
                zh("1. 从 256 个字节出发", 24, theme.FG),
                zh("2. 数所有相邻对，最常见的一对 → 新 token", 24, theme.FG),
                zh("3. 重复，直到词表够大", 24, theme.FG),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([0, 1.85, 0])
            self.play(FadeIn(algo), run_time=self.fit(1))
            self.wait(self.remaining() * 0.25)
            states = D["merge_states"]
            src = zh(f"编码「{MERGE_TEXT}」：按学到的顺序重放合并", 24, theme.MUTED).move_to([0, 0.45, 0])
            row = token_row(states[0], -0.5, color=theme.PARAM, size=20)
            count = zh(f"{len(states[0])} 个 token", 28, theme.HIGHLIGHT).move_to([0, -1.7, 0])
            self.play(FadeIn(src), FadeIn(row), FadeIn(count), run_time=self.fit(1))
            n_steps = len(states) - 1
            per = (self.remaining() - 3.5) / max(n_steps, 1)
            for k in range(1, len(states)):
                prev, cur = states[k - 1], states[k]
                new_row = token_row(cur, -0.5, color=theme.PARAM, size=20)
                # Find the merge position j: prev[j] + prev[j+1] == cur[j]
                j = next(i for i in range(len(cur)) if cur[i] != prev[i])
                anims = []
                for i in range(len(cur)):
                    if i < j:
                        anims.append(Transform(row[i], new_row[i]))
                    elif i == j:
                        anims.append(Transform(VGroup(row[j], row[j + 1]), new_row[j]))
                    else:
                        anims.append(Transform(row[i + 1], new_row[i]))
                new_count = zh(f"{len(cur)} 个 token", 28, theme.HIGHLIGHT).move_to(count)
                self.play(*anims, Transform(count, new_count), run_time=self.fit(min(per, 1.2)))
                self.remove(*row)
                row = new_row
                self.add(row)
                self.wait(max(0.05, per - 1.2))
            fallback = zh("「说」没学到整字 → 退回字节片段，不丢信息", 26, theme.OUTPUT).move_to([0, -2.35, 0])
            self.play(FadeIn(fallback), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in list(self.mobjects) if m is not self._heading],
                      run_time=self.fit(0.6))

        # ── S06 The first merges that BPE learns ─────────────────────────
        with self.shot("S06"):
            self.play(*self.set_heading("最先学到的合并（英文 + 中文 + 代码各 6 万字符）"),
                      run_time=self.fit(0.8))
            hist = D["history"]
            rows = VGroup()
            for new_id, b, c in hist:
                r = VGroup(mono(str(new_id), 24, theme.MUTED), token_box(b, 0.62, color=theme.PARAM, size=22),
                           zh(f"{c:,} 次", 22, theme.FG))
                r[1].move_to([0, 0, 0])
                r[0].next_to(r[1], LEFT, 0.3)
                r[2].next_to(r[1], RIGHT, 0.3)
                rows.add(r)
            cols = VGroup(*[VGroup(*rows[k:k + 6]).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
                            for k in (0, 6, 12)]).arrange(RIGHT, buff=0.6, aligned_edge=UP)
            if cols.width > 13.2:
                cols.scale_to_fit_width(13.2)
            cols.move_to([0, 0.75, 0])
            self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rows], lag_ratio=0.25),
                      run_time=self.fit(4))
            tags = [
                (0, "全角标点的前两个字节", theme.HIGHLIGHT),
                (5, "汉字的“半个字”", theme.OUTPUT),
                (1, "代码缩进", theme.INPUT),
                (15, "英文字母对", theme.ATTN),
            ]
            marks = VGroup()
            legend_x = [-4.4, -1.0, 1.7, 4.3]
            for (i, txt, col), lx in zip(tags, legend_x):
                box = SurroundingRectangle(rows[i][1], color=col, buff=0.06)
                lab = VGroup(Rectangle(width=0.3, height=0.22, color=col), zh(txt, 20, col)).arrange(
                    RIGHT, buff=0.15).move_to([lx, -1.9, 0])
                marks.add(VGroup(box, lab))
            for m in marks:
                self.wait(self.remaining() * 0.12)
                self.play(Create(m[0]), FadeIn(m[1]), run_time=self.fit(0.8))
            longtok = zh("后面学到的长 token：␣return  ␣import  ␣Citizen  MENENIUS", 22, theme.MUTED)
            longtok.move_to([0, -2.35, 0])
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(longtok), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(rows), FadeOut(marks), FadeOut(longtok), run_time=self.fit(0.6))

        # ── S07 The trade-off of the vocabulary size ─────────────────────
        with self.shot("S07"):
            self.play(*self.set_heading("词表大小：序列长度 vs embedding 参数"), run_time=self.fit(0.8))
            sweep = D["sweep"]
            ax = Axes(x_range=[8, 15.5, 1], y_range=[0, 100, 20], x_length=5.6, y_length=3.6,
                      axis_config={"color": theme.MUTED, "font_size": 18}, tips=False,
                      y_axis_config={"include_numbers": True}).move_to([-3.4, 0.0, 0])
            xt = VGroup(*[mono(f"2^{k}", 16, theme.MUTED).next_to(ax.c2p(k, 0), DOWN, 0.12)
                          for k in range(8, 16)])
            xl = zh("词表大小 V（对数）", 20, theme.MUTED).next_to(ax, DOWN, 0.45)
            yl = zh("验证集 token 数（千）", 20, theme.MUTED).next_to(ax, UP, 0.1)
            pts = [(math.log2(v), n / 1000) for v, n in sweep]
            curve = polyline_in_axes(ax, pts, color=theme.INPUT, stroke_width=4)
            dots = VGroup(*[
                RoundedRectangle(width=0.1, height=0.1, corner_radius=0.05, color=theme.INPUT,
                                 fill_opacity=1).move_to(ax.c2p(x, y)) for x, y in pts
            ])
            self.play(Create(ax), FadeIn(xt), FadeIn(xl), FadeIn(yl), run_time=self.fit(1))
            self.play(Create(curve), FadeIn(dots), run_time=self.fit(2.5))
            a_lab = zh(f"{sweep[0][1]:,}", 18, theme.FG).next_to(ax.c2p(*pts[0]), RIGHT, 0.15)
            b_lab = zh(f"{sweep[-1][1]:,}", 18, theme.FG).next_to(ax.c2p(*pts[-1]), UP, 0.2)
            drop = (sweep[-2][1] - sweep[-1][1]) / sweep[-2][1]
            c_lab = zh(f"16K→32K 只少 {drop:.0%}", 20, theme.HIGHLIGHT).move_to(ax.c2p(13.2, 62))
            self.play(FadeIn(a_lab), FadeIn(b_lab), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.12)
            self.play(FadeIn(c_lab), run_time=self.fit(0.8))
            # Right: bar chart of the embedding parameters
            models = [("GPT-2", 50_257, 768), ("本课主线（暂定）", 65_536, 1280),
                      ("Qwen3-0.6B", 151_936, 1024), ("Qwen3.5-0.8B", 248_320, 1024)]
            bar_title = zh("embedding 参数 = V × d", 24, theme.PARAM).move_to([3.6, 1.95, 0])
            self.wait(self.remaining() * 0.15)
            self.play(FadeIn(bar_title), run_time=self.fit(0.6))
            bars = VGroup()
            scale = 2.4 / 260
            for i, (name, V, d) in enumerate(models):
                val = V * d / 1e6
                y = 1.25 - i * 0.85
                lab = zh(name, 20, theme.FG).move_to([3.6, y + 0.27, 0])
                bar = Rectangle(width=max(val * scale, 0.05), height=0.28, color=theme.PARAM,
                                fill_color=theme.PARAM, fill_opacity=0.7, stroke_width=0)
                bar.move_to([2.2 + val * scale / 2, y - 0.08, 0])
                num = zh(f"{val:.1f}M", 20, theme.HIGHLIGHT if "3.5" in name else theme.FG)
                num.next_to(bar, RIGHT, 0.12)
                lab.align_to(bar, LEFT)
                bars.add(VGroup(lab, bar, num))
            per = self.fit(0.7, reserve=2)
            for b in bars:
                self.play(FadeIn(b[0]), GrowFromEdge(b[1], LEFT), FadeIn(b[2]), run_time=per)
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [ax, xt, xl, yl, curve, dots, a_lab, b_lab, c_lab,
                                              bar_title, bars]], run_time=self.fit(0.6))

        # ── S08 The bigram count table ───────────────────────────────────
        with self.shot("S08"):
            self.play(*self.set_heading("bigram：只看前 1 个 token，数一数"), run_time=self.fit(0.8))
            heat = np.array(D["heat"], dtype=float)
            logc = np.log1p(heat) / np.log1p(heat.max())
            n = len(HEAT_CHARS)
            cell = 0.36
            origin = np.array([-5.1, 1.75, 0])
            grid = VGroup()
            for i in range(n):
                for j in range(n):
                    col = interpolate_color(ManimColor(theme.BG), ManimColor(theme.PARAM), float(logc[i, j]))
                    sq = Rectangle(width=cell, height=cell, stroke_width=0.5, stroke_color=theme.BG,
                                   fill_color=col, fill_opacity=1)
                    sq.move_to(origin + np.array([j * cell, -i * cell, 0]))
                    grid.add(sq)
            lbl = lambda c: "␣" if c == " " else c  # noqa: E731
            rlab = VGroup(*[mono(lbl(c), 18, theme.INPUT).next_to(grid[i * n], LEFT, 0.12)
                            for i, c in enumerate(HEAT_CHARS)])
            clab = VGroup(*[mono(lbl(c), 18, theme.OUTPUT).next_to(grid[j], UP, 0.1)
                            for j, c in enumerate(HEAT_CHARS)])
            rt = zh("当前", 18, theme.INPUT).next_to(rlab, LEFT, 0.12)
            ct = zh("下一个", 18, theme.OUTPUT).next_to(clab, UP, 0.08)
            self.play(FadeIn(rt), FadeIn(ct), FadeIn(rlab), FadeIn(clab), run_time=self.fit(0.8))
            self.play(LaggedStart(*[FadeIn(s) for s in grid], lag_ratio=0.01), run_time=self.fit(2))
            ti, hj = HEAT_CHARS.index("t"), HEAT_CHARS.index("h")
            hl = SurroundingRectangle(grid[ti * n + hj], color=theme.HIGHLIGHT, buff=0.02)
            th = zh(f"t→h：{int(heat[ti, hj]):,} 次", 22, theme.HIGHLIGHT).next_to(grid, DOWN, 0.3)
            self.wait(self.remaining() * 0.12)
            self.play(Create(hl), FadeIn(th), run_time=self.fit(0.8))
            formula = MathTex(r"p(b\mid a)=\frac{\mathrm{count}(a,b)}{\sum_j \mathrm{count}(a,j)}",
                              font_size=42).move_to([3.3, 1.0, 0])
            self.play(Write(formula), run_time=self.fit(1.5))
            mle = zh("数频率 = 最大似然解（第 5 章）", 24, theme.FG).next_to(formula, DOWN, 0.4)
            self.play(FadeIn(mle), run_time=self.fit(0.8))
            smooth = MathTex(r"\frac{\mathrm{count}(a,b)+\alpha}{\sum_j \mathrm{count}(a,j)+\alpha V}",
                             font_size=38, color=theme.GRAD).next_to(mle, DOWN, 0.45)
            sl = zh("平滑：没见过的组合概率不为 0", 22, theme.GRAD).next_to(smooth, DOWN, 0.25)
            self.wait(self.remaining() * 0.3)
            self.play(Write(smooth), FadeIn(sl), run_time=self.fit(1.2))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [grid, rlab, clab, rt, ct, hl, th, formula, mle, smooth, sl]],
                      run_time=self.fit(0.6))

        # ── S09 Sampling ────────────────────────────────────────────────
        with self.shot("S09"):
            self.play(*self.set_heading("从 bigram 抽样：只看前 1 个 token"), run_time=self.fit(0.8))
            en_lines = [x for x in D["samples"]["en"].strip().split("\n") if x]
            zh_lines = [x for x in D["samples"]["zh"].strip().split("\n")][:7]
            en_t = VGroup(zh("英文 · BPE", 24, theme.MUTED),
                          *[mono(x, 24) for x in en_lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
            en_t.move_to([-3.4, 0, 0]).align_to([0, 2.4, 0], UP)
            zh_t = VGroup(zh("中文 · 字符", 24, theme.MUTED), *[zh(x, 24) for x in zh_lines]).arrange(
                DOWN, aligned_edge=LEFT, buff=0.12).move_to([3.3, 0, 0]).align_to([0, 2.4, 0], UP)
            for g in (en_t, zh_t):
                if g.height > 4.3:
                    g.scale_to_fit_height(4.3)
            self.play(FadeIn(en_t[0]), run_time=self.fit(0.5))
            self.play(LaggedStart(*[FadeIn(x) for x in en_t[1:]], lag_ratio=0.4), run_time=self.fit(2.5))
            self.play(FadeIn(zh_t[0]), run_time=self.fit(0.5))
            self.play(LaggedStart(*[FadeIn(x) for x in zh_t[1:]], lag_ratio=0.4), run_time=self.fit(2.5))
            verdict = zh("格式像模像样，内容毫无意义", 28, theme.HIGHLIGHT).move_to([0, -2.35, 0])
            self.wait(self.remaining() * 0.15)
            self.play(FadeIn(verdict), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(en_t), FadeOut(zh_t), FadeOut(verdict), run_time=self.fit(0.6))

        # ── S10 Evaluation: nats → bits → perplexity → bpb ───────────────
        with self.shot("S10"):
            self.play(*self.set_heading("评估：从 nats 到 bits-per-byte"), run_time=self.fit(0.8))
            nats = VGroup(zh("nats / token", 26, theme.GRAD), MathTex(r"-\tfrac{1}{T}\sum \ln p", font_size=34))
            bits = VGroup(zh("bits / token", 26, theme.GRAD), MathTex(r"\text{nats}/\ln 2", font_size=34))
            ppl = VGroup(zh("困惑度", 26, theme.GRAD), MathTex(r"e^{\text{nats}}=2^{\text{bits}}", font_size=34))
            chain = VGroup(nats, bits, ppl)
            for g in chain:
                g.arrange(DOWN, buff=0.2)
            chain.arrange(RIGHT, buff=1.4).move_to([0, 1.6, 0])
            arrows = VGroup(*[Arrow(chain[i].get_right(), chain[i + 1].get_left(), buff=0.15,
                                    color=theme.MUTED) for i in range(2)])
            self.play(FadeIn(nats), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.1)
            self.play(Create(arrows[0]), FadeIn(bits), run_time=self.fit(0.8))
            self.play(Create(arrows[1]), FadeIn(ppl), run_time=self.fit(0.8))
            per_tok = Rectangle(width=chain.width + 0.5, height=chain.height + 0.4, color=theme.MUTED)
            per_tok.move_to(chain)
            pt = zh("都是“每个 token”的：不同分词器不可比", 24, theme.MUTED).next_to(per_tok, DOWN, 0.15)
            self.wait(self.remaining() * 0.12)
            self.play(Create(per_tok), FadeIn(pt), run_time=self.fit(1))
            bpb = MathTex(r"\text{bpb}=\frac{\sum_t -\ln p(x_t\mid\cdot)}{\ln 2\times \text{total bytes}}",
                          r"=\frac{\text{bits/token}}{\text{bytes/token}}", font_size=40)
            bpb[0][0:3].set_color(theme.HIGHLIGHT)
            bpb.move_to([0, -1.05, 0])
            self.wait(self.remaining() * 0.35)
            self.play(Write(bpb), run_time=self.fit(2))
            same = zh("同一段文本的字节数固定 → 与分词器无关", 26, theme.OUTPUT).move_to([0, -2.35, 0])
            self.play(FadeIn(same), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [chain, arrows, per_tok, pt, bpb, same]], run_time=self.fit(0.6))

        # ── S11 Real numbers: perplexity can mislead ─────────────────────
        with self.shot("S11"):
            self.play(*self.set_heading("英文验证集上的 bigram：困惑度 vs bpb"), run_time=self.fit(0.8))
            ev = D["evals"]
            names = ["字节", "字符", "BPE"]
            colors = [theme.PARAM, theme.INPUT, theme.OUTPUT]

            def panel(key: str, title: str, vmax: float, cx: float, fmt: str) -> VGroup:
                g = VGroup()
                base_y = -1.3
                g.add(zh(title, 26, theme.FG).move_to([cx, 2.25, 0]))
                g.add(Line([cx - 2.2, base_y, 0], [cx + 2.2, base_y, 0], color=theme.MUTED))
                for i, (nm, col) in enumerate(zip(names, colors)):
                    v = ev[f"en|{nm}"][key]
                    h = 3.0 * v / vmax
                    x = cx - 1.4 + i * 1.4
                    bar = Rectangle(width=0.8, height=h, fill_color=col, fill_opacity=0.75,
                                    stroke_width=0).move_to([x, base_y + h / 2, 0])
                    g.add(VGroup(bar, zh(format(v, fmt), 22, theme.FG).next_to(bar, UP, 0.08),
                                 zh(nm, 22, col).next_to([x, base_y, 0], DOWN, 0.15)))
                return g

            left = panel("ppl", "困惑度（每 token，越低越好）", 40, -3.5, ".1f")
            right = panel("bpb", "bits-per-byte（越低越好）", 4.0, 3.5, ".3f")
            self.play(FadeIn(left[:2]), run_time=self.fit(0.6))
            self.play(*[GrowFromEdge(b[0], DOWN) for b in left[2:]], *[FadeIn(b[1:]) for b in left[2:]],
                      run_time=self.fit(1.5))
            self.wait(self.remaining() * 0.18)
            self.play(FadeIn(right[:2]), run_time=self.fit(0.6))
            self.play(*[GrowFromEdge(b[0], DOWN) for b in right[2:]], *[FadeIn(b[1:]) for b in right[2:]],
                      run_time=self.fit(1.5))
            gain = 1 - ev["en|BPE"]["bpb"] / ev["en|字节"]["bpb"]
            g_txt = zh(f"BPE 反而好 {gain:.0%}", 26, theme.OUTPUT).move_to([3.5, -2.3, 0])
            self.play(FadeIn(g_txt), Create(SurroundingRectangle(right[4], color=theme.OUTPUT, buff=0.08)),
                      run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.3)
            other = zh(f"中文：字符 {ev['zh|字符']['bpb']:.3f} < BPE {ev['zh|BPE']['bpb']:.3f}"
                       f"　代码：BPE {ev['code|BPE']['bpb']:.3f} 最低", 22, theme.MUTED).move_to([-3.2, -2.3, 0])
            if other.width > 6.4:
                other.scale_to_fit_width(6.4)
            self.play(FadeIn(other), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in list(self.mobjects) if m is not self._heading],
                      run_time=self.fit(0.6))

        # ── S12 From minimal code to production code ─────────────────────
        with self.shot("S12"):
            self.play(*self.set_heading("从极简到生产级：zero/tokenizer.py"), run_time=self.fit(0.8))
            src = '''tok = HFTokenizer(models.BPE())
tok.normalizer = normalizers.NFC()
tok.pre_tokenizer = Sequence([
    Split(Regex(PRETOKENIZE_REGEX)),
    ByteLevel(use_regex=False)])
trainer = BpeTrainer(vocab_size=...,
    special_tokens=DEFAULT_SPECIAL_TOKENS)'''
            code = code_block(src, 17)
            # With code_block aligned left, leading spaces have no width, and the indentation
            # is lost. Add the indentation back by hand with the width of a monospace character.
            cw = Text("M" * 20, font="Noto Sans Mono", font_size=17).width / 20
            for line_m, line in zip(code, src.splitlines()):
                line_m.shift(RIGHT * cw * (len(line) - len(line.lstrip(" "))))
            code.move_to([-3.2, 1.25, 0]).align_to([-6.6, 0, 0], LEFT)
            self.play(FadeIn(code), run_time=self.fit(1))
            notes = [(1, "Unicode 规范化"), (3, "Qwen 同款正则，数字逐个切"), (6, "16 个特殊 token")]
            tags = VGroup()
            for idx, t in notes:
                tags.add(zh(t, 18, theme.HIGHLIGHT).next_to(code[idx], RIGHT, 0.25))
            for tg in tags:
                self.wait(self.remaining() * 0.06)
                self.play(FadeIn(tg), run_time=self.fit(0.6))
            rm, rz = D["ratio_mine"], D["ratio_zero"]
            head = VGroup(*[zh(x, 20, theme.MUTED) for x in ["字节/token", "英文", "中文", "代码"]])
            r1 = VGroup(zh("手写 BPE", 20), *[zh(f"{rm[k]:.2f}", 20) for k in ["en", "zh", "code"]])
            r2 = VGroup(zh("zero", 20, theme.OUTPUT), *[zh(f"{rz[k]:.2f}", 20, theme.OUTPUT)
                                                        for k in ["en", "zh", "code"]])
            table = VGroup(*head, *r1, *r2).arrange_in_grid(rows=3, cols=4, buff=(0.45, 0.2))
            table.move_to([3.6, -1.4, 0])
            ttl = zh("同样 768 次合并，验证集上的压缩率", 20, theme.FG).next_to(table, UP, 0.15)
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(ttl), FadeIn(table), run_time=self.fit(1))
            ok = zh("tests/test_tokenizer.py：12 项通过 · 分片 uint32 + <|endoftext|>", 20, theme.OUTPUT)
            ok.move_to([0, -2.4, 0])
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(ok), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(*[FadeOut(m) for m in [code, tags, ttl, table, ok]], run_time=self.fit(0.6))

        # ── S13 The next chapter ─────────────────────────────────────────
        with self.shot("S13"):
            self.play(*self.set_heading("bigram 只看前 1 个 token"), run_time=self.fit(0.8))
            s = "学而时习之，不亦说乎"
            boxes = VGroup(*[
                VGroup(RoundedRectangle(width=0.8, height=0.8, corner_radius=0.08, color=theme.INPUT,
                                        fill_color=theme.INPUT, fill_opacity=0.15), zh(c, 32))
                for c in s
            ]).arrange(RIGHT, buff=0.18).move_to([0, 0.2, 0])
            self.play(FadeIn(boxes), run_time=self.fit(1))
            last = boxes[-1]
            one = CurvedArrow(last.get_top(), boxes[-2].get_top(), angle=1.2, color=theme.PARAM)
            one_l = zh("bigram", 22, theme.PARAM).next_to(one, UP, 0.1)
            self.play(Create(one), FadeIn(one_l), run_time=self.fit(1))
            summary = VGroup(
                zh("语言模型 = 猜下一个 token", 24, theme.FG),
                zh("byte-level BPE：从字节出发合并常见片段", 24, theme.FG),
                zh("跨分词器比较看 bits-per-byte", 24, theme.FG),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([0, 2.15, 0])
            self.play(FadeIn(summary), run_time=self.fit(1))
            self.wait(self.remaining() * 0.35)
            many = VGroup(*[
                CurvedArrow(last.get_bottom() + DOWN * 0.05, boxes[i].get_bottom() + DOWN * 0.05,
                            angle=-1.2, color=theme.ATTN, stroke_width=2, tip_length=0.15)
                for i in range(len(s) - 1)
            ])
            self.play(FadeOut(one), FadeOut(one_l), *self.set_heading("下一章：注意力 —— 看前面所有 token"),
                      LaggedStart(*[Create(a) for a in many], lag_ratio=0.1), run_time=self.fit(2))
            nxt = zh("下一章：注意力", 34, theme.HIGHLIGHT).move_to([0, -2.3, 0])
            nbox = Rectangle(width=nxt.width + 0.6, height=nxt.height + 0.3, color=theme.HIGHLIGHT).move_to(nxt)
            self.play(FadeIn(nxt), Create(nbox), run_time=self.fit(1))
            self.wait(self.remaining() - 1.0)
            self.play(*[FadeOut(m) for m in [boxes, many, summary, nxt, nbox]], *self.set_heading(None),
                      run_time=self.fit(1.0))
