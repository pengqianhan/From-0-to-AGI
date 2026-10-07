"""Video for Chapter 26: the state of open models.

Put the architectures of the whole course into one tree.

The code in ../code/ calculates all values on screen (see the fact list in script.md):
01_panorama.py (layer composition, KV cache, adoption matrix, the "next version" of the main-line model),
02_evolution_tree.py (evolution tree), 03_meta_params.py (count the parameters from the config).
The cache video/out/cache.json keeps the results. Delete it to calculate them again.
Render: bash chapters/26-open-model-panorama/video/build.sh
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
    Create,
    DashedVMobject,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    LaggedStart,
    Line,
    Rectangle,
    RoundedRectangle,
    VGroup,
    Write,
)

from video_kit import theme
from video_kit.scene import NarratedScene, zh

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
CACHE = HERE / "out" / "cache.json"
GiB = 2**30
TEAL = "#4FC1C9"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _num(s: str) -> float:
    """Convert "2.8T", "104B", "~23B" from a model card → number of parameters."""
    s = s.strip().lstrip("~")
    return float(s[:-1]) * {"T": 1e12, "B": 1e9, "M": 1e6}[s[-1]]


def compute() -> dict:
    """Calculate all numbers for the video with the real code in ../code."""
    pan = _load("panorama", "01_panorama.py")
    tree = _load("evolution_tree", "02_evolution_tree.py").build_tree()
    meta = _load("meta_params", "03_meta_params.py").run()
    ms = pan.load_models()
    d: dict = {"tree": tree}
    d["cards"] = [
        dict(name=ms[m]["name"], total=ms[m]["card"]["total_params"],
             active=ms[m]["card"]["active_params"], dissect=m in pan.DISSECT)
        for m in pan.FLAGSHIPS
    ]
    strip_ids = ["gpt-oss-120b", "qwen3.8-2.4t-a95b", "kimi-k3", "deepseek-v4-pro", "glm-5.3",
                 "minimax-m3", "qwen3.5-0.8b", "main"]
    d["strips"] = [dict(name=ms[m]["name"], kinds=pan.layer_kinds(ms[m]),
                        comp=pan.composition(pan.layer_kinds(ms[m]))) for m in strip_ids]
    # Parameters: for flagship models, use the model card.
    # For small dense models, use the count from the config.
    param_ids = ["main", "qwen3.5-0.8b", "gpt-oss-120b", "minimax-m3", "glm-5.3", "deepseek-v4-pro",
                 "qwen3.8-2.4t-a95b", "kimi-k3"]
    d["params"] = []
    for m in param_ids:
        card = ms[m].get("card", {})
        if card.get("total_params"):
            tot, act = _num(card["total_params"]), _num(card["active_params"])
            label = f"{card['total_params']} / {card['active_params']}"
        else:
            tot = act = meta[m]["total"]
            label = f"{tot / 1e9:.2f}B（稠密）"
        d["params"].append(dict(name=ms[m]["name"], total=tot, active=act, label=label,
                                meta_total=meta[m]["total"]))
    kv_ids = ["deepseek-v4-pro", "qwen3.5-0.8b", "kimi-k3", "gpt-oss-120b", "glm-5.3",
              "qwen3.8-2.4t-a95b", "minimax-m3"]
    d["kv"] = [dict(id=m, name=ms[m]["name"], gib=pan.kv_numbers(ms[m])["kv_128k"] / GiB) for m in kv_ids]
    d["kv_main_32k"] = pan.kv_numbers(ms["main"])["kv_32k"] / GiB
    feats = {m: pan.features(ms[m]) for m in pan.FLAGSHIPS}
    d["counts"] = {k: sum(k in feats[m] for m in pan.FLAGSHIPS)
                   for k in ("RMSNorm 前置", "SwiGLU/GLU", "MoE", "RoPE", "共享专家", "MTP",
                             "无辅助损失均衡", "QK-Norm", "混合线性注意力", "稀疏注意力", "MLA", "滑动窗口")}
    d["next"] = [dict(name=n, gib=pan.kv_cache_bytes(lay, 32768) / GiB)
                 for n, lay in pan.next_version_layouts().items()]
    return d


def get_data() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    d = compute()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


D = get_data()
NODES = {n["id"]: n for n in D["tree"]["nodes"]}

# Layout of the evolution tree (schematic): column 0 is the dense block.
# The 3 columns at the right are the branches.
C1, C2, C3 = -1.25, 1.75, 4.8
POS = {
    "moe": (C1, 2.35), "auxfree": (C2, 2.65), "latentmoe": (C2, 2.05),
    "gqa": (C1, 1.2), "mla": (C2, 1.2), "csa": (C3, 1.2),
    "swa": (C1, 0.25), "sparse": (C2, 0.25),
    "hybrid": (C1, -0.7), "kda": (C2, -0.7),
    "mtp": (C1, -1.6),
    "mhc": (C1, -2.35),
}
LABEL = {
    "moe": "MoE 细粒度+共享", "auxfree": "无辅助损失均衡", "latentmoe": "Latent MoE",
    "gqa": "GQA", "mla": "MLA 潜向量", "csa": "CSA/HCA 压缩",
    "swa": "滑动窗口", "sparse": "稀疏注意力", "hybrid": "混合线性 3:1", "kda": "KDA",
    "mtp": "MTP", "mhc": "mHC / AttnRes",
}
KIND_COLOR = {"GQA": theme.INPUT, "MHA": theme.INPUT, "SWA": TEAL, "GDN": theme.OUTPUT,
              "KDA": theme.OUTPUT, "MLA": theme.ATTN, "MLA+DSA": theme.ATTN, "CSA": theme.PARAM,
              "HCA": theme.GRAD, "GQA+稀疏": theme.HIGHLIGHT}


def hbar(width: float, color: str, height: float = 0.36, opacity: float = 0.85) -> Rectangle:
    return Rectangle(width=max(width, 0.02), height=height, fill_color=color,
                     fill_opacity=opacity, stroke_width=0)


def node_box(nid: str) -> VGroup:
    """Draw one node of the evolution tree.

    Green = consensus, yellow = new, gray dashed = frontier note.
    """
    n = NODES[nid]
    status = n["status"]
    color = {"consensus": theme.OUTPUT, "new": theme.HIGHLIGHT, "frontier": theme.MUTED}.get(status, theme.FG)
    t = zh(LABEL[nid], 19, theme.FG if status != "frontier" else theme.MUTED)
    ch = zh(n["chapter"].split("、")[0], 14, theme.MUTED)
    body = VGroup(t, ch).arrange(DOWN, buff=0.04)
    box = RoundedRectangle(corner_radius=0.08, width=2.45, height=0.52, stroke_color=color,
                           stroke_width=2.2, fill_color=color, fill_opacity=0.12 if status != "frontier" else 0)
    if status == "frontier":
        box = VGroup(DashedVMobject(box, num_dashes=28))
    g = VGroup(box, body)
    g.move_to([*POS[nid], 0])
    body.move_to(box.get_center())
    return g


def edge(a: VGroup, b: VGroup) -> Line:
    return Line(a.get_right(), b.get_left(), stroke_width=2, color=theme.MUTED)


class ChapterScene(NarratedScene):
    chapter_label = "第 26 章"
    chapter_title = "当前最先进开源模型全景"

    def construct(self) -> None:
        for i in range(1, 15):
            getattr(self, f"s{i:02d}")()

    # ── S01 Opening ──────────────────────────────────────────────────────
    def s01(self) -> None:
        with self.shot("S01"):
            card = self.chapter_card()
            sub = zh("把整门课的架构放进一棵树", 30, theme.HIGHLIGHT).next_to(card, DOWN, 0.6)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(card), FadeOut(sub), run_time=self.fit(0.8))

    # ── S02 The newest flagship models ───────────────────────────────────
    def s02(self) -> None:
        with self.shot("S02"):
            self.play(*self.set_heading("2026-09 各家最新旗舰（读自 Hugging Face）"), run_time=self.fit(0.8))
            cards = VGroup()
            for c in D["cards"]:
                color = theme.HIGHLIGHT if c["dissect"] else theme.MUTED
                name = zh(c["name"], 24)
                par = zh(f"总 {c['total']} · 激活 {c['active']}", 20, theme.PARAM)
                tag = zh("本章拆解" if c["dissect"] else "同代对照", 16, color)
                body = VGroup(name, par, tag).arrange(DOWN, buff=0.14)
                box = RoundedRectangle(corner_radius=0.12, width=4.1, height=1.55, stroke_color=color,
                                       stroke_width=2)
                body.move_to(box.get_center())
                cards.add(VGroup(box, body))
            cards.arrange_in_grid(rows=2, cols=3, buff=(0.35, 0.45)).move_to([0, 0.35, 0])
            self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in cards], lag_ratio=0.35),
                      run_time=self.fit(4, reserve=1))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(cards), run_time=self.fit(0.6))

    # ── S03 Root and trunk of the tree ───────────────────────────────────
    def s03(self) -> None:
        with self.shot("S03"):
            self.play(*self.set_heading("一棵树：整门课的架构演化"), run_time=self.fit(0.8))
            g2 = VGroup(RoundedRectangle(corner_radius=0.08, width=1.15, height=0.9, stroke_color=theme.FG,
                                         stroke_width=2),
                        VGroup(zh("GPT-2", 20), zh("2019", 14, theme.MUTED)).arrange(DOWN, buff=0.05))
            g2[1].move_to(g2[0].get_center())
            g2.move_to([-6.4, 0.2, 0])
            items = ["现代稠密块", "Pre-Norm RMSNorm", "RoPE", "SwiGLU", "GQA", "QK-Norm", "共享 embedding"]
            lines = VGroup(*[zh(t, 22 if i == 0 else 18, theme.HIGHLIGHT if i == 0 else theme.FG)
                             for i, t in enumerate(items)]).arrange(DOWN, buff=0.2)
            block = RoundedRectangle(corner_radius=0.12, width=2.55, height=lines.height + 0.5,
                                     stroke_color=theme.INPUT, stroke_width=2.5, fill_color=theme.INPUT,
                                     fill_opacity=0.1)
            lines.move_to(block.get_center())
            self.block = VGroup(block, lines).move_to([-4.3, 0.2, 0])
            e0 = Line(g2.get_right(), self.block.get_left(), stroke_width=2, color=theme.MUTED)
            self.play(FadeIn(g2), run_time=self.fit(1))
            self.wait(self.remaining() * 0.15)
            self.play(Create(e0), FadeIn(self.block[0]), FadeIn(lines[0]), run_time=self.fit(1))
            self.play(LaggedStart(*[FadeIn(x, shift=RIGHT * 0.15) for x in lines[1:]], lag_ratio=0.4),
                      run_time=self.fit(3.5, reserve=2.5))
            ours = zh("主线模型 = 这根主干", 20, theme.OUTPUT).next_to(self.block, DOWN, 0.2)
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(ours), run_time=self.fit(0.8))
            self.tree = VGroup(g2, e0, self.block, ours)
            self.boxes: dict[str, VGroup] = {}

    def grow(self, ids: list[str], run: float = 2.0) -> None:
        """Grow a group of new nodes from their parent.

        The parent is the dense block or a node that is already on screen.
        """
        anims = []
        for nid in ids:
            b = node_box(nid)
            parent = NODES[nid]["parent"]
            src = self.boxes.get(parent, self.block)
            e = edge(src, b)
            self.boxes[nid] = b
            self.tree.add(e, b)
            anims.append(LaggedStart(Create(e), FadeIn(b, shift=RIGHT * 0.15), lag_ratio=0.5))
        self.play(LaggedStart(*anims, lag_ratio=0.6), run_time=self.fit(run))

    # ── S04 MoE ──────────────────────────────────────────────────────────
    def s04(self) -> None:
        with self.shot("S04"):
            self.play(*self.set_heading("分枝一：前馈层变稀疏（第 24 章）"), run_time=self.fit(0.8))
            self.grow(["moe"], 1.5)
            n = D["counts"]
            note = zh(f"6 个旗舰里：MoE {n['MoE']} 家 · 共享专家 {n['共享专家']} 家 · 无辅助损失 {n['无辅助损失均衡']} 家",
                      18, theme.MUTED).move_to([3.1, -2.35, 0])
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(note), run_time=self.fit(0.6))
            self.grow(["auxfree"], 1.2)
            self.wait(self.remaining() * 0.35)
            self.grow(["latentmoe"], 1.2)
            self.wait(self.remaining() - 0.5)
            self.play(FadeOut(note), run_time=self.fit(0.5))

    # ── S05 KV compression ───────────────────────────────────────────────
    def s05(self) -> None:
        with self.shot("S05"):
            self.play(*self.set_heading("分枝二：每个位置少存点（第 10、21 章）"), run_time=self.fit(0.8))
            self.grow(["gqa"], 1.0)
            self.wait(self.remaining() * 0.15)
            self.grow(["mla"], 1.2)
            self.wait(self.remaining() * 0.3)
            self.grow(["csa"], 1.2)
            ratio = next(k["gib"] for k in D["kv"] if k["id"] == "deepseek-v4-pro")
            note = zh(f"DeepSeek-V4-Pro：128K 只要 {ratio * 1024:.0f} MiB", 17, theme.PARAM)
            note.next_to(self.boxes["csa"], DOWN, 0.15).align_to(self.boxes["csa"], RIGHT)
            self.play(FadeIn(note), run_time=self.fit(0.6))
            self.wait(self.remaining() - 0.5)
            self.play(FadeOut(note), run_time=self.fit(0.5))

    # ── S06 Look at only a part / do not store ───────────────────────────
    def s06(self) -> None:
        with self.shot("S06"):
            self.play(*self.set_heading("分枝三、四：只看一部分，或者干脆不存（第 22、23 章）"),
                      run_time=self.fit(0.8))
            self.grow(["swa"], 1.0)
            self.grow(["sparse"], 1.2)
            users = NODES["sparse"]["users"]
            note = VGroup(zh("在用：" + "、".join(u.split("-")[0] for u in users), 16, theme.HIGHLIGHT),
                          zh("2026 年过门槛，做法各异", 16, theme.HIGHLIGHT)).arrange(DOWN, buff=0.06)
            note.next_to(self.boxes["sparse"], RIGHT, 0.25)
            self.play(FadeIn(note), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.35)
            self.grow(["hybrid"], 1.0)
            self.grow(["kda"], 1.0)
            self.wait(self.remaining() - 0.5)
            self.play(FadeOut(note), run_time=self.fit(0.5))

    # ── S07 MTP and the residual stream ──────────────────────────────────
    def s07(self) -> None:
        with self.shot("S07"):
            self.play(*self.set_heading("分枝五、六：MTP 与残差流改造"), run_time=self.fit(0.8))
            self.grow(["mtp"], 1.0)
            self.wait(self.remaining() * 0.3)
            self.grow(["mhc"], 1.0)
            leg = VGroup(*[VGroup(Rectangle(width=0.35, height=0.22, stroke_color=c, stroke_width=2,
                                            fill_color=c, fill_opacity=0.15), zh(t, 17)).arrange(RIGHT, buff=0.1)
                           for c, t in ((theme.OUTPUT, "共识"), (theme.HIGHLIGHT, "新晋共识"),
                                        (theme.MUTED, "前沿观察（虚线）"))]).arrange(RIGHT, buff=0.4)
            leg.move_to([3.2, -1.85, 0])
            self.play(FadeIn(leg), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(self.tree), FadeOut(leg), run_time=self.fit(0.8))

    # ── S08 Colors of the layers ─────────────────────────────────────────
    def s08(self) -> None:
        with self.shot("S08"):
            self.play(*self.set_heading("逐层拆开：每一格是一层"), run_time=self.fit(0.8))
            legend_items = [("全注意力", theme.INPUT), ("滑动窗口", TEAL), ("线性 GDN/KDA", theme.OUTPUT),
                            ("MLA", theme.ATTN), ("CSA", theme.PARAM), ("HCA", theme.GRAD),
                            ("块稀疏", theme.HIGHLIGHT)]
            leg = VGroup(*[VGroup(hbar(0.28, c, 0.2), zh(t, 16)).arrange(RIGHT, buff=0.08)
                           for t, c in legend_items])
            leg.arrange_in_grid(rows=2, cols=4, buff=(0.3, 0.12), flow_order="rd")
            leg.move_to([3.4, 3.15, 0])
            x0, width = -3.9, 10.8
            cell = width / 93
            rows = VGroup()
            for i, s in enumerate(D["strips"]):
                y = 2.35 - i * 0.64
                cells = VGroup(*[Rectangle(width=cell * 0.82, height=0.34, stroke_width=0,
                                           fill_color=KIND_COLOR[k], fill_opacity=0.9)
                                 .move_to([x0 + (j + 0.5) * cell, y, 0]) for j, k in enumerate(s["kinds"])])
                name = zh(s["name"], 17, theme.OUTPUT if s["name"].startswith("主线") else theme.FG)
                name.move_to([x0 - 0.15, y, 0], aligned_edge=RIGHT)
                comp = zh(s["comp"], 13, theme.MUTED).next_to(cells, RIGHT, 0.12)
                if comp.get_right()[0] > 7.0:
                    comp.next_to(cells, DOWN, 0.03).align_to(cells, LEFT)
                rows.add(VGroup(name, cells, comp))
            self.play(FadeIn(leg), run_time=self.fit(0.8))
            self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rows[:-1]], lag_ratio=0.35),
                      run_time=self.fit(5, reserve=4))
            self.wait(self.remaining() * 0.45)
            self.play(FadeIn(rows[-1], shift=RIGHT * 0.2), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(rows, leg)), run_time=self.fit(0.6))

    # ── S09 Parameters ───────────────────────────────────────────────────
    def s09(self) -> None:
        with self.shot("S09"):
            self.play(*self.set_heading("总参数（暗色长条）与激活参数（亮色），对数刻度"), run_time=self.fit(0.8))
            x0, per_dec, lo = -1.9, 1.62, 8  # 1e8 is at x0; each factor of 10 is 1.62 units

            def xpos(v: float) -> float:
                return x0 + (math.log10(v) - lo) * per_dec

            axis = Line([x0, -2.2, 0], [xpos(1e13), -2.2, 0], stroke_width=1.5, color=theme.MUTED)
            ticks = VGroup()
            for e, t in ((8, "1亿"), (9, "10亿"), (10, "100亿"), (11, "1000亿"), (12, "1万亿"), (13, "10万亿")):
                ticks.add(zh(t, 14, theme.MUTED).move_to([xpos(10**e), -2.45, 0]))
            rows = VGroup()
            for i, p in enumerate(D["params"]):
                y = 2.35 - i * 0.6
                ours = p["name"].startswith("主线")
                tb = hbar(xpos(p["total"]) - x0, theme.PARAM, 0.36, 0.3).move_to([x0, y, 0], aligned_edge=LEFT)
                ab = hbar(xpos(p["active"]) - x0, theme.OUTPUT if ours else theme.PARAM, 0.36, 0.95)
                ab.move_to([x0, y, 0], aligned_edge=LEFT)
                name = zh(p["name"], 17, theme.OUTPUT if ours else theme.FG).move_to([x0 - 0.15, y, 0],
                                                                                   aligned_edge=RIGHT)
                val = zh(p["label"], 15, theme.MUTED).next_to(tb, RIGHT, 0.12)
                rows.add(VGroup(name, tb, ab, val))
            self.play(Create(axis), FadeIn(ticks), run_time=self.fit(0.8))
            self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rows], lag_ratio=0.3),
                      run_time=self.fit(4.5, reserve=3))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(rows, axis, ticks)), run_time=self.fit(0.6))

    # ── S10 KV cache ─────────────────────────────────────────────────────
    def s10(self) -> None:
        with self.shot("S10"):
            self.play(*self.set_heading("128K 上下文的 KV cache（BF16）"), run_time=self.fit(0.8))
            color = {"deepseek-v4-pro": theme.PARAM, "qwen3.5-0.8b": theme.OUTPUT, "kimi-k3": theme.OUTPUT,
                     "gpt-oss-120b": TEAL, "glm-5.3": theme.ATTN, "qwen3.8-2.4t-a95b": theme.OUTPUT,
                     "minimax-m3": theme.HIGHLIGHT}
            rows_data = sorted(D["kv"], key=lambda r: r["gib"])
            x0 = -1.6
            scale = 7.0 / max(r["gib"] for r in rows_data)
            rows = VGroup()
            for i, r in enumerate(rows_data):
                y = 2.35 - i * 0.58
                b = hbar(r["gib"] * scale, color[r["id"]], 0.36).move_to([x0, y, 0], aligned_edge=LEFT)
                n = zh(r["name"], 18).move_to([x0 - 0.2, y, 0], aligned_edge=RIGHT)
                txt = f"{r['gib'] * 1024:.0f} MiB" if r["gib"] < 1 else f"{r['gib']:.2f} GiB"
                v = zh(txt, 18, color[r["id"]]).next_to(b, RIGHT, 0.15)
                rows.add(VGroup(b, n, v))
            self.play(LaggedStart(*[FadeIn(g, shift=RIGHT * 0.2) for g in rows], lag_ratio=0.3),
                      run_time=self.fit(4, reserve=3.5))
            y = 2.35 - len(rows_data) * 0.58 - 0.12
            mb = hbar(D["kv_main_32k"] * scale, theme.INPUT, 0.36).move_to([x0, y, 0], aligned_edge=LEFT)
            mn = zh("主线模型（只到 32K）", 18, theme.OUTPUT).move_to([x0 - 0.2, y, 0], aligned_edge=RIGHT)
            mv = zh(f"{D['kv_main_32k']:.2f} GiB", 18, theme.INPUT).next_to(mb, RIGHT, 0.15)
            self.wait(self.remaining() * 0.25)
            self.play(GrowFromEdge(mb, LEFT), FadeIn(mn), FadeIn(mv), run_time=self.fit(1))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(rows, mb, mn, mv)), run_time=self.fit(0.6))

    # ── S11 Consensus and divergence ─────────────────────────────────────
    def s11(self) -> None:
        with self.shot("S11"):
            self.play(*self.set_heading("按“至少 3 家”判定（6 个最新旗舰 + 前几章）"), run_time=self.fit(0.8))
            n = D["counts"]
            left_items = [("RMSNorm 前置", "RMSNorm 前置"), ("门控 FFN（SwiGLU 等）", "SwiGLU/GLU"),
                          ("MoE", "MoE"), ("RoPE", "RoPE"), ("共享专家", "共享专家"), ("MTP", "MTP"),
                          ("无辅助损失均衡", "无辅助损失均衡"), ("QK-Norm", "QK-Norm")]
            lt = zh("已成共识的主干", 24, theme.OUTPUT)
            lrows = VGroup(*[zh(f"{a}　{n[k]}/6", 19) for a, k in left_items]).arrange(DOWN, buff=0.16,
                                                                                         aligned_edge=LEFT)
            left = VGroup(lt, lrows).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to([-3.6, 0.35, 0])
            rt = zh("长上下文：三派，都过了门槛", 24, theme.HIGHLIGHT)
            rrows = VGroup(
                zh(f"混合线性注意力　{n['混合线性注意力']}/6 + Nemotron", 19),
                zh(f"稀疏注意力　{n['稀疏注意力']}/6（2026 新晋）", 19),
                zh(f"MLA　{n['MLA']}/6 + Mistral", 19),
                zh(f"滑动窗口　{n['滑动窗口']}/6 + Gemma、OLMo", 19),
            ).arrange(DOWN, buff=0.16, aligned_edge=LEFT)
            right = VGroup(rt, rrows).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to([2.9, 1.0, 0])
            fr = zh("前沿（1–2 家）：压缩注意力 · mHC / AttnRes · Latent MoE · NoPE", 18, theme.MUTED)
            fr.move_to([2.2, -1.6, 0])
            self.play(FadeIn(lt), run_time=self.fit(0.6))
            self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.15) for r in lrows], lag_ratio=0.3),
                      run_time=self.fit(3, reserve=5))
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(rt), LaggedStart(*[FadeIn(r, shift=RIGHT * 0.15) for r in rrows], lag_ratio=0.3),
                      run_time=self.fit(3, reserve=2.5))
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(fr), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(left, right, fr)), run_time=self.fit(0.6))

    # ── S12 Why the main-line model uses only the trunk ──────────────────
    def s12(self) -> None:
        with self.shot("S12"):
            self.play(*self.set_heading("主线模型：只用稠密共识块（GOAL 3.3）"), run_time=self.fit(0.8))
            main = next(s for s in D["strips"] if s["name"].startswith("主线"))
            cell = 0.26
            strip = VGroup(*[Rectangle(width=cell * 0.82, height=0.4, stroke_width=0, fill_color=theme.INPUT,
                                       fill_opacity=0.9).move_to([-3.6 + j * cell, 2.3, 0])
                             for j in range(len(main["kinds"]))])
            lab = zh("689.5M · 28 层 GQA · 32K", 18, theme.OUTPUT).next_to(strip, RIGHT, 0.3)
            reasons = [("MoE", "用显存换算力；0.7B 切专家，矩阵太小"),
                       ("MLA", "只在大 MoE 里；和 QK-Norm 不兼容"),
                       ("稀疏 / 压缩注意力", "省的是 10 万 token 以上的算力"),
                       ("混合线性注意力", "最值得试，但记性差一点：留给下一版")]
            rows = VGroup()
            for i, (a, b) in enumerate(reasons):
                y = 1.25 - i * 0.72
                t = zh(a, 21, theme.PARAM).move_to([-2.2, y, 0], aligned_edge=RIGHT)
                x = zh("现在不用", 17, theme.GRAD).next_to(t, RIGHT, 0.3)
                r = zh(b, 19).next_to(x, RIGHT, 0.3)
                rows.add(VGroup(t, x, r))
            bottom = zh("在这个规模上：架构的差异 ≪ 数据与后训练的差异", 22, theme.HIGHLIGHT).move_to([0, -2.25, 0])
            self.play(FadeIn(strip), FadeIn(lab), run_time=self.fit(1))
            self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.15) for r in rows], lag_ratio=0.5),
                      run_time=self.fit(5, reserve=3))
            self.wait(self.remaining() * 0.4)
            self.play(Write(bottom), run_time=self.fit(1.2))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(strip, lab, rows, bottom)), run_time=self.fit(0.6))

    # ── S13 The next version ─────────────────────────────────────────────
    def s13(self) -> None:
        with self.shot("S13"):
            self.play(*self.set_heading("主线“下一版”：32K 上下文的 KV cache"), run_time=self.fit(0.8))
            x0 = -1.0
            scale = 6.0 / max(r["gib"] for r in D["next"])
            rows = VGroup()
            cols = [theme.INPUT, TEAL, theme.OUTPUT]
            for i, r in enumerate(D["next"]):
                y = 1.9 - i * 0.8
                b = hbar(r["gib"] * scale, cols[i], 0.45).move_to([x0, y, 0], aligned_edge=LEFT)
                n = zh(r["name"], 19).move_to([x0 - 0.2, y, 0], aligned_edge=RIGHT)
                txt = f"{r['gib'] * 1024:.0f} MiB" if r["gib"] < 1 else f"{r['gib']:.2f} GiB"
                v = zh(txt, 20, cols[i]).next_to(b, RIGHT, 0.15)
                rows.add(VGroup(b, n, v))
            plan = VGroup(zh("① 先加 1 层 MTP：代价小，推理时当草稿", 21, theme.HIGHLIGHT),
                          zh("② 再试 3:1 混合线性：先做同算力对比，专测精确复制", 21, theme.HIGHLIGHT)
                          ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).move_to([0, -1.2, 0])
            self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rows], lag_ratio=0.4),
                      run_time=self.fit(3, reserve=4))
            self.wait(self.remaining() * 0.25)
            self.play(FadeIn(plan[0]), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(plan[1]), run_time=self.fit(0.8))
            self.wait(self.remaining() - 0.6)
            self.play(FadeOut(VGroup(rows, plan)), run_time=self.fit(0.6))

    # ── S14 Back to the start ────────────────────────────────────────────
    def s14(self) -> None:
        with self.shot("S14"):
            self.play(*self.set_heading("从 y = ax + b 到最先进的开源模型"), run_time=self.fit(0.8))
            steps = ["y = ax + b", "神经网络", "反向传播", "注意力", "现代 Transformer",
                     "预训练", "后训练", "架构演进", "今天的旗舰"]
            boxes = VGroup()
            for i, s in enumerate(steps):
                t = zh(s, 19, theme.FG)
                b = RoundedRectangle(corner_radius=0.1, width=t.width + 0.4, height=0.55,
                                     stroke_color=theme.INPUT if i < len(steps) - 1 else theme.HIGHLIGHT,
                                     stroke_width=2)
                t.move_to(b.get_center())
                boxes.add(VGroup(b, t))
            row1 = VGroup(*boxes[:5]).arrange(RIGHT, buff=0.45).move_to([0, 1.7, 0])
            row2 = VGroup(*boxes[5:]).arrange(RIGHT, buff=0.45).move_to([0, 0.5, 0])
            arrows = VGroup()
            for a, b in zip(boxes[:-1], boxes[1:]):
                if a in row1 and b in row2:
                    continue
                arrows.add(Line(a.get_right(), b.get_left(), stroke_width=2, color=theme.MUTED))
            loop = zh("每一步都是：算损失 → 求梯度 → 走一小步", 22, theme.OUTPUT).move_to([0, -0.55, 0])
            nxt = zh("第二步：在 GPU 上训练主线模型", 28, theme.HIGHLIGHT).move_to([0, -1.6, 0])
            self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.15) for b in boxes], lag_ratio=0.3),
                      Create(arrows), run_time=self.fit(4, reserve=4))
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(loop), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.45)
            self.play(Write(nxt), run_time=self.fit(1.2))
            self.wait(self.remaining() - 0.8)
            self.play(FadeOut(VGroup(row1, row2, arrows, loop, nxt)), *self.set_heading(None),
                      run_time=self.fit(0.8))
