"""第 4 章视频：反向传播与自动微分 —— 让计算机替你求导

画面里的数值由 ../code/ 中的代码真实计算（见 script.md 事实清单）；
唯一例外是 S11 的计时数字（F8），它是 04_pytorch_compare.py 的一次实测，每次运行会不同。
渲染：bash chapters/04-backprop-autograd/video/build.sh
"""

from __future__ import annotations

import importlib.util
import math
import random
from pathlib import Path

import numpy as np
import torch
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Axes,
    ChangeDecimalToValue,
    Circle,
    Create,
    CurvedArrow,
    DecimalNumber,
    Dot,
    FadeIn,
    FadeOut,
    GrowArrow,
    Indicate,
    LaggedStart,
    MathTex,
    Rectangle,
    RoundedRectangle,
    SurroundingRectangle,
    Text,
    Transform,
    VGroup,
    VMobject,
    Write,
)

from video_kit import theme
from video_kit.scene import NarratedScene, zh

CODE = Path(__file__).resolve().parent.parent / "code"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tm = _load("train_mlp", "03_train_mlp.py")
eng = tm.engine  # 和 03_train_mlp.py 共用同一个 Value 类
gc = _load("grad_check", "02_grad_check.py")
cmp = _load("pytorch_compare", "04_pytorch_compare.py")
V = eng.Value

# ── 所有画面数字：在这里用代码算出来 ──────────────────────────────────────────
# S03 链式法则：u = 3x，y = u²，x = 2
_x = V(2.0)
_u = _x * 3
_y = _u**2
_y.backward()
CHAIN = dict(x=_x.data, u=_u.data, y=_y.data, du_dx=3.0, dy_du=_u.grad, dy_dx=_x.grad)

# S04/S05 计算图：L = (a·b + c)²
A, B, C = V(2.0), V(-3.0), V(10.0)
D = A * B
E = D + C
L = E**2
L.backward()
G = {k: v for k, v in dict(a=A, b=B, c=C, d=D, e=E, L=L).items()}

# S07 分叉：y = x·x + x，x = 3
FX = V(3.0)
FM = FX * FX
FY = FM + FX
FY.backward()

# S09 梯度检验
_f, _leaves = gc.expression_case(eng)
_, EXPR_REL, EXPR_AUTO, EXPR_NUM = gc.check(_f, _leaves)
_f, _params = gc.mlp_case(eng)
_, MLP_REL, _, _ = gc.check(_f, _params)
_buggy = gc.buggy_engine()
_, BUG_EXPR_REL, _, _ = gc.check(*gc.expression_case(_buggy))
_, BUG_MLP_REL, _, _ = gc.check(*gc.mlp_case(_buggy))
N_PARAMS = len(_params)

# S10 训练
XS, YS = tm.make_data()
SNAP_STEPS = [0, 10, 50, 100, 200, 300, 500]
NET, LOSSES, SNAPS = tm.train(steps=500, lr=0.1, snapshot_at=SNAP_STEPS)
random.seed(0)
_probe = eng.MLP(1, [8, 8, 1])
N_NODES = tm.count_nodes(tm.mse(_probe, XS, YS))

# S12 与 PyTorch 对拍
random.seed(0)
_net = tm.MLP(1, [8, 8, 1])
_model = cmp.to_torch(_net)
_xt, _yt = torch.tensor(XS).unsqueeze(1), torch.tensor(YS).unsqueeze(1)
_lv = tm.mse(_net, XS, YS)
_net.zero_grad()
_lv.backward()
_lt = ((_model(_xt) - _yt) ** 2).mean()
_model.zero_grad()
_lt.backward()
PARAM_ROWS = []
for (_name, _p), _g in zip(_model.named_parameters(), cmp.value_grads_as_tensors(_net),
                           strict=True):
    PARAM_ROWS.append((_name, tuple(_p.shape), bool(torch.allclose(_p.grad, _g, rtol=1e-9,
                                                                     atol=1e-12))))
MAX_DIFF = max((p.grad - g).abs().max().item()
               for p, g in zip(_model.parameters(), cmp.value_grads_as_tensors(_net), strict=True))
_opt = torch.optim.SGD(_model.parameters(), lr=0.1)
for _ in range(500):
    _l = ((_model(_xt) - _yt) ** 2).mean()
    _opt.zero_grad()
    _l.backward()
    _opt.step()
TORCH_FINAL = ((_model(_xt) - _yt) ** 2).mean().item()

# S11 计时：04_pytorch_compare.py 的一次实测（毫秒），每次运行都会不同，见 README 第 9 节
TIMING = [(20, 43.7, 0.355), (200, 775.3, 0.414)]

MONO = "Noto Sans Mono"


def mono(text: str, size: float = 20, color: str = theme.FG) -> Text:
    return Text(text, font=MONO, font_size=size, color=color)


def code_block(lines: list[str], size: float = 20, buff: float = 0.14) -> VGroup:
    """多行代码：Text 不渲染行首空格，这里按缩进宽度手动右移，保留 Python 缩进。"""
    char_w = mono("x" * 20, size).width / 20
    ms = [mono(line.lstrip(), size) for line in lines]
    block = VGroup(*ms).arrange(DOWN, aligned_edge=LEFT, buff=buff)
    for m, line in zip(ms, lines, strict=True):
        m.shift(RIGHT * char_w * (len(line) - len(line.lstrip())))
    return block


def fmt(v: float) -> str:
    """-6.0 → '-6'；2.5 → '2.5'。"""
    return f"{v:g}"


def vbox(tex: str, color: str = theme.FG, width: float = 1.45) -> VGroup:
    """值节点：圆角框 + 里面的 MathTex。"""
    label = MathTex(tex, font_size=30, color=color)
    box = RoundedRectangle(width=max(width, label.width + 0.3), height=0.62, corner_radius=0.12,
                           color=color, stroke_width=2.5)
    return VGroup(box, label.move_to(box))


def opnode(tex: str) -> VGroup:
    c = Circle(radius=0.3, color=theme.MUTED, stroke_width=2.5)
    return VGroup(c, MathTex(tex, font_size=30, color=theme.MUTED).move_to(c))


def edge(m1, m2, color=theme.MUTED) -> Arrow:
    return Arrow(m1.get_right(), m2.get_left(), buff=0.06, color=color, stroke_width=3,
                 max_tip_length_to_length_ratio=0.18, tip_length=0.18)


def set_label(node: VGroup, tex: str, color: str) -> MathTex:
    return MathTex(tex, font_size=30, color=color).move_to(node[0])


class ChapterScene(NarratedScene):
    chapter_label = "第 4 章"
    chapter_title = "反向传播与自动微分"

    def build_graph(self):
        """L = (a·b + c)² 的计算图（S04、S05 共用）。"""
        nodes = {
            "a": vbox(r"a", theme.PARAM).move_to([-5.7, 1.55, 0]),
            "b": vbox(r"b", theme.PARAM).move_to([-5.7, 0.05, 0]),
            "c": vbox(r"c", theme.PARAM).move_to([-5.7, -1.45, 0]),
            "mul": opnode(r"\times").move_to([-3.9, 0.8, 0]),
            "d": vbox(r"d").move_to([-2.3, 0.8, 0]),
            "add": opnode(r"+").move_to([-0.7, -0.3, 0]),
            "e": vbox(r"e").move_to([0.9, -0.3, 0]),
            "pow": opnode(r"(\,)^2").move_to([2.5, -0.3, 0]),
            "L": vbox(r"L", theme.OUTPUT).move_to([4.1, -0.3, 0]),
        }
        nodes["pow"][0].scale(1.3)
        nodes["pow"][1].scale(0.9)
        edges = {
            ("a", "mul"): edge(nodes["a"], nodes["mul"]),
            ("b", "mul"): edge(nodes["b"], nodes["mul"]),
            ("mul", "d"): edge(nodes["mul"], nodes["d"]),
            ("d", "add"): edge(nodes["d"], nodes["add"]),
            ("c", "add"): edge(nodes["c"], nodes["add"]),
            ("add", "e"): edge(nodes["add"], nodes["e"]),
            ("e", "pow"): edge(nodes["e"], nodes["pow"]),
            ("pow", "L"): edge(nodes["pow"], nodes["L"]),
        }
        return nodes, edges

    def pulse(self, arrow: Arrow, color: str, reverse: bool = False, run_time: float = 0.6):
        """一个小圆点沿边移动：前向从左到右，反向从右到左。"""
        s, e = arrow.get_start(), arrow.get_end()
        if reverse:
            s, e = e, s
        dot = Dot(s, radius=0.08, color=color)
        self.add(dot)
        self.play(dot.animate.move_to(e), run_time=run_time)
        self.remove(dot)

    def construct(self) -> None:
        # ── S01 片头 ─────────────────────────────────────────────────────
        with self.shot("S01"):
            card = self.chapter_card()
            sub = zh("让计算机替你求导", 32, theme.HIGHLIGHT).next_to(card, DOWN, 0.6)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.wait(max(0.05, self.remaining() - 0.8))
            self.play(FadeOut(card), FadeOut(sub), run_time=self.fit(0.8))

        # ── S02 手推梯度的痛 ─────────────────────────────────────────────
        with self.shot("S02"):
            self.play(*self.set_heading("手推梯度：模型每变一次，就要重推一次"),
                      run_time=self.fit(0.8))
            r1_t = zh("第 1 章：2 个参数", 26, theme.MUTED)
            r1_f = MathTex(r"\frac{\partial L}{\partial a}=\frac{2}{N}\sum(\hat y_i-y_i)x_i",
                           r"\quad",
                           r"\frac{\partial L}{\partial b}=\frac{2}{N}\sum(\hat y_i-y_i)",
                           font_size=32)
            row1 = VGroup(r1_t, r1_f).arrange(DOWN, buff=0.2).move_to([0, 1.75, 0])
            r2_t = zh("第 3 章：两层网络，第一层权重的梯度", 26, theme.MUTED)
            r2_f = MathTex(r"\frac{\partial L}{\partial W_1}=X^\top\Big[\big(\tfrac{2}{N}(\hat Y-Y)"
                           r"\,W_2^\top\big)\odot\mathbf{1}(XW_1+b_1>0)\Big]", font_size=32)
            row2 = VGroup(r2_t, r2_f).arrange(DOWN, buff=0.2).move_to([0, -0.1, 0])
            r3 = zh("再加一层？换个激活函数？换个损失？—— 全部重推", 28, theme.GRAD)
            r3.move_to([0, -1.75, 0])
            self.play(FadeIn(row1), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.15)
            self.play(FadeIn(r2_t), Write(r2_f), run_time=self.fit(2.5))
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(r3, shift=UP * 0.2), run_time=self.fit(1))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(FadeOut(row1), FadeOut(row2), FadeOut(r3), run_time=self.fit(0.6))

        # ── S03 链式法则 ─────────────────────────────────────────────────
        with self.shot("S03"):
            self.play(*self.set_heading("链式法则：沿路的局部导数相乘"), run_time=self.fit(0.8))
            bx = vbox(rf"x={fmt(CHAIN['x'])}", theme.INPUT, 1.8).move_to([-4.5, 1.2, 0])
            bu = vbox(rf"u=3x={fmt(CHAIN['u'])}", theme.FG, 2.4).move_to([0, 1.2, 0])
            by = vbox(rf"y=u^2={fmt(CHAIN['y'])}", theme.OUTPUT, 2.4).move_to([4.5, 1.2, 0])
            e1, e2 = edge(bx, bu), edge(bu, by)
            l1 = MathTex(rf"\times {fmt(CHAIN['du_dx'])}", font_size=34,
                         color=theme.HIGHLIGHT).next_to(e1, UP, 0.12)
            l2 = MathTex(rf"\times {fmt(CHAIN['dy_du'])}", font_size=34,
                         color=theme.HIGHLIGHT).next_to(e2, UP, 0.12)
            n1 = zh("du/dx", 20, theme.MUTED).next_to(e1, DOWN, 0.12)
            n2 = zh("dy/du = 2u", 20, theme.MUTED).next_to(e2, DOWN, 0.12)
            self.play(FadeIn(bx), FadeIn(bu), FadeIn(by), GrowArrow(e1), GrowArrow(e2),
                      run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.12)
            self.play(FadeIn(l1), FadeIn(n1), run_time=self.fit(0.8))
            self.pulse(e1, theme.HIGHLIGHT, run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.12)
            self.play(FadeIn(l2), FadeIn(n2), run_time=self.fit(0.8))
            self.pulse(e2, theme.HIGHLIGHT, run_time=self.fit(0.8))
            # 小变化被逐段放大：Δx = 0.01 → Δu = 0.03 → Δy ≈ 0.36
            dx = 0.01
            du = CHAIN["du_dx"] * dx
            dy_true = (CHAIN["u"] + du) ** 2 - CHAIN["y"]
            deltas = VGroup(
                MathTex(rf"\Delta x={dx:g}", font_size=30, color=theme.INPUT).next_to(bx, DOWN, 0.9),
                MathTex(rf"\Delta u={du:.2f}", font_size=30).next_to(bu, DOWN, 0.9),
                MathTex(rf"\Delta y\approx{dy_true:.2f}", font_size=30,
                        color=theme.OUTPUT).next_to(by, DOWN, 0.9),
            )
            self.play(LaggedStart(*[FadeIn(d) for d in deltas], lag_ratio=0.5),
                      run_time=self.fit(1.8))
            rule = MathTex(r"\frac{dy}{dx}", "=", r"\frac{dy}{du}", r"\cdot", r"\frac{du}{dx}", "=",
                           rf"{fmt(CHAIN['dy_du'])}\times{fmt(CHAIN['du_dx'])}={fmt(CHAIN['dy_dx'])}",
                           font_size=40).move_to([0, -1.7, 0])
            rule[6].set_color(theme.HIGHLIGHT)
            self.wait(self.remaining() * 0.2)
            self.play(Write(rule), run_time=self.fit(1.5))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(*[FadeOut(m) for m in [bx, bu, by, e1, e2, l1, l2, n1, n2, deltas, rule]],
                      run_time=self.fit(0.6))

        # ── S04 计算图：前向 ─────────────────────────────────────────────
        nodes, edges = self.build_graph()
        formula = MathTex(r"L=(a\cdot b+c)^2", font_size=40).move_to([4.6, 2.3, 0])
        with self.shot("S04"):
            self.play(*self.set_heading("计算图：前向时把每一步运算记下来"), Write(formula),
                      run_time=self.fit(1))
            self.play(LaggedStart(*[FadeIn(n) for n in nodes.values()], lag_ratio=0.1),
                      LaggedStart(*[GrowArrow(e) for e in edges.values()], lag_ratio=0.1),
                      run_time=self.fit(2))
            self.wait(self.remaining() * 0.08)
            # 叶子节点先有值
            leaf_vals = [Transform(nodes[k][1], set_label(nodes[k], rf"{k}={fmt(G[k].data)}",
                                                           theme.PARAM)) for k in "abc"]
            self.play(*leaf_vals, run_time=self.fit(0.8))
            order = [(("a", "mul"), ("b", "mul"), ("mul", "d"), "d", theme.FG),
                     (("d", "add"), ("c", "add"), ("add", "e"), "e", theme.FG),
                     (("e", "pow"), None, ("pow", "L"), "L", theme.OUTPUT)]
            per = (self.remaining() - 2.5) / 3
            for e_in1, e_in2, e_out, key, color in order:
                ins = [edges[e_in1]] + ([edges[e_in2]] if e_in2 else [])
                dots = [Dot(a.get_start(), radius=0.08, color=theme.OUTPUT) for a in ins]
                self.add(*dots)
                self.play(*[d.animate.move_to(a.get_end()) for d, a in zip(dots, ins, strict=True)],
                          run_time=self.fit(per * 0.3))
                self.remove(*dots)
                self.pulse(edges[e_out], theme.OUTPUT, run_time=self.fit(per * 0.25))
                self.play(Transform(nodes[key][1], set_label(nodes[key],
                                                              rf"{key}={fmt(G[key].data)}", color)),
                          run_time=self.fit(per * 0.3))
            note = zh("每个节点记住：值、由谁算出、用什么运算", 24, theme.HIGHLIGHT)
            note.move_to([2.6, -1.9, 0])
            self.play(FadeIn(note), run_time=self.fit(0.8))

        # ── S05 反向传播 ─────────────────────────────────────────────────
        with self.shot("S05"):
            self.play(*self.set_heading("反向：上游梯度 × 局部导数"), FadeOut(note),
                      run_time=self.fit(0.8))

            def grad_lbl(key: str) -> MathTex:
                lbl = MathTex(rf"\bar{{{key}}}={fmt(G[key].grad)}", font_size=30, color=theme.GRAD)
                return lbl.next_to(nodes[key], DOWN, 0.12)

            legend = MathTex(r"\bar{v}", r"=\partial L/\partial v", font_size=30,
                             color=theme.GRAD).move_to([4.6, 1.5, 0])
            self.play(FadeIn(legend), run_time=self.fit(0.6))
            gL = grad_lbl("L")
            self.play(FadeIn(gL, shift=DOWN * 0.1), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.05)
            locs = {
                "pow": MathTex(r"\times 2e", font_size=26, color=theme.HIGHLIGHT),
                "add": MathTex(r"\times 1", font_size=26, color=theme.HIGHLIGHT),
                "mul_a": MathTex(r"\times b", font_size=26, color=theme.HIGHLIGHT),
                "mul_b": MathTex(r"\times a", font_size=26, color=theme.HIGHLIGHT),
            }
            locs["pow"].next_to(nodes["pow"], UP, 0.15)
            locs["add"].next_to(nodes["add"], UP, 0.15)
            locs["mul_a"].next_to(edges[("a", "mul")].get_center(), UP, 0.12)
            locs["mul_b"].next_to(edges[("b", "mul")].get_center(), DOWN, 0.12)
            per = (self.remaining() - 3) / 4
            # ² 节点
            self.play(FadeIn(locs["pow"]), run_time=self.fit(per * 0.3))
            self.pulse(edges[("pow", "L")], theme.GRAD, reverse=True, run_time=self.fit(per * 0.2))
            self.pulse(edges[("e", "pow")], theme.GRAD, reverse=True, run_time=self.fit(per * 0.2))
            ge = grad_lbl("e")
            self.play(FadeIn(ge), run_time=self.fit(per * 0.3))
            # + 节点：原样分给 d 和 c
            self.play(FadeIn(locs["add"]), run_time=self.fit(per * 0.3))
            self.pulse(edges[("add", "e")], theme.GRAD, reverse=True, run_time=self.fit(per * 0.2))
            dots = [Dot(edges[k].get_end(), radius=0.08, color=theme.GRAD)
                    for k in [("d", "add"), ("c", "add")]]
            self.add(*dots)
            self.play(dots[0].animate.move_to(edges[("d", "add")].get_start()),
                      dots[1].animate.move_to(edges[("c", "add")].get_start()),
                      run_time=self.fit(per * 0.3))
            self.remove(*dots)
            gd, gc_ = grad_lbl("d"), grad_lbl("c")
            self.play(FadeIn(gd), FadeIn(gc_), run_time=self.fit(per * 0.3))
            self.wait(max(0.1, per * 0.3))
            # × 节点：交换相乘
            self.play(FadeIn(locs["mul_a"]), FadeIn(locs["mul_b"]), run_time=self.fit(per * 0.3))
            self.pulse(edges[("mul", "d")], theme.GRAD, reverse=True, run_time=self.fit(per * 0.2))
            dots = [Dot(edges[k].get_end(), radius=0.08, color=theme.GRAD)
                    for k in [("a", "mul"), ("b", "mul")]]
            self.add(*dots)
            self.play(dots[0].animate.move_to(edges[("a", "mul")].get_start()),
                      dots[1].animate.move_to(edges[("b", "mul")].get_start()),
                      run_time=self.fit(per * 0.3))
            self.remove(*dots)
            ga, gb = grad_lbl("a"), grad_lbl("b")
            self.play(FadeIn(ga), FadeIn(gb), run_time=self.fit(per * 0.3))
            calc = MathTex(rf"\bar a=\bar d\cdot b={fmt(D.grad)}\times({fmt(B.data)})={fmt(A.grad)}",
                           font_size=30, color=theme.FG).move_to([3.0, -1.9, 0])
            self.play(Write(calc), run_time=self.fit(1.2))
            self.wait(max(0.05, self.remaining() - 0.7))
            graph_all = VGroup(*nodes.values(), *edges.values(), *locs.values(), gL, ge, gd, gc_,
                               ga, gb, calc, legend, formula)
            self.play(FadeOut(graph_all), run_time=self.fit(0.7))

        # ── S06 每种运算只写一次 ─────────────────────────────────────────
        with self.shot("S06"):
            self.play(*self.set_heading("每种运算：前向一行 + 局部导数一行"), run_time=self.fit(0.8))
            rows = [
                (r"c=a+b", r"\partial c/\partial a=1,\ \partial c/\partial b=1"),
                (r"c=a\cdot b", r"\partial c/\partial a=b,\ \partial c/\partial b=a"),
                (r"c=a^k", r"\partial c/\partial a=k\,a^{k-1}"),
                (r"c=\tanh a", r"\partial c/\partial a=1-c^2"),
                (r"c=e^a", r"\partial c/\partial a=c"),
                (r"c=\ln a", r"\partial c/\partial a=1/a"),
            ]
            table = VGroup()
            for lhs, rhs in rows:
                table.add(MathTex(lhs, font_size=30), MathTex(rhs, font_size=30,
                                                                 color=theme.HIGHLIGHT))
            table.arrange_in_grid(rows=len(rows), cols=2, col_alignments="ll", buff=(0.5, 0.28))
            table.move_to([-3.4, 0.1, 0])
            code_lines = [
                "def __mul__(self, other):",
                "    out = Value(self.data * other.data)",
                "    def _backward():",
                "        self.grad += other.data * out.grad",
                "        other.grad += self.data * out.grad",
                "    out._backward = _backward",
                "    return out",
            ]
            code = code_block(code_lines, 19, 0.16).move_to([3.55, 0.3, 0])
            for i in range(len(table) // 2):
                self.play(FadeIn(table[2 * i]), FadeIn(table[2 * i + 1]),
                          run_time=self.fit(0.5, reserve=5))
            self.play(FadeIn(code), run_time=self.fit(1))
            hl = SurroundingRectangle(VGroup(code[3], code[4]), color=theme.GRAD, buff=0.08)
            self.wait(self.remaining() * 0.2)
            self.play(Create(hl), run_time=self.fit(0.8))
            tag = zh("局部导数 × 上游梯度，传给输入", 22, theme.GRAD).next_to(code, DOWN, 0.3)
            self.play(FadeIn(tag), run_time=self.fit(0.6))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(FadeOut(table), FadeOut(code), FadeOut(hl), FadeOut(tag),
                      run_time=self.fit(0.6))

        # ── S07 分叉：梯度要相加 ─────────────────────────────────────────
        with self.shot("S07"):
            self.play(*self.set_heading("分叉：一个节点被用了几次，梯度就加几次"),
                      run_time=self.fit(0.8))
            nx = vbox(rf"x={fmt(FX.data)}", theme.PARAM).move_to([-4.6, 0.2, 0])
            nmul = opnode(r"\times").move_to([-1.9, 1.6, 0])
            nm = vbox(rf"m=x\cdot x={fmt(FM.data)}", theme.FG, 2.2).move_to([0.2, 1.6, 0])
            nadd = opnode("+").move_to([2.3, 0.2, 0])
            ny = vbox(rf"y={fmt(FY.data)}", theme.OUTPUT).move_to([4.0, 0.2, 0])
            c1 = CurvedArrow(nx.get_top() + RIGHT * 0.2, nmul.get_left() + UP * 0.1, angle=-0.5,
                             color=theme.MUTED, stroke_width=3, tip_length=0.18)
            c2 = CurvedArrow(nx.get_right() + UP * 0.1, nmul.get_bottom(), angle=0.4,
                             color=theme.MUTED, stroke_width=3, tip_length=0.18)
            s3 = Arrow(nx.get_right() + DOWN * 0.1, nadd.get_left(), buff=0.06, color=theme.MUTED,
                       stroke_width=3, tip_length=0.18)
            e_mm = edge(nmul, nm)
            e_ma = Arrow(nm.get_bottom() + RIGHT * 0.4, nadd.get_top(), buff=0.06,
                         color=theme.MUTED, stroke_width=3, tip_length=0.18)
            e_ay = edge(nadd, ny)
            title = MathTex(r"y=x\cdot x+x", font_size=38).move_to([4.6, 2.3, 0])
            self.play(FadeIn(VGroup(nx, nmul, nm, nadd, ny)), Create(c1), Create(c2),
                      GrowArrow(s3), GrowArrow(e_mm), GrowArrow(e_ma), GrowArrow(e_ay),
                      Write(title), run_time=self.fit(1.5))
            self.wait(self.remaining() * 0.12)
            # 三条路各带回一份梯度：乘法给两个 x 各 x·1 = 3，加法给 x 1
            g_mul = FX.data  # ∂m/∂x（每个乘数位置）× ∂y/∂m(=1)
            g1 = MathTex(fmt(g_mul), font_size=34, color=theme.GRAD).move_to(c1.point_from_proportion(0.5) + UP * 0.3 + LEFT * 0.2)
            g2 = MathTex(fmt(g_mul), font_size=34, color=theme.GRAD).move_to(c2.point_from_proportion(0.5) + DOWN * 0.3 + RIGHT * 0.25)
            g3 = MathTex("1", font_size=34, color=theme.GRAD).next_to(s3, DOWN, 0.12)
            self.play(Indicate(c1, color=theme.GRAD), FadeIn(g1), run_time=self.fit(1))
            self.play(Indicate(c2, color=theme.GRAD), FadeIn(g2), run_time=self.fit(1))
            self.play(Indicate(s3, color=theme.GRAD), FadeIn(g3), run_time=self.fit(1))
            total = MathTex(rf"\bar x={fmt(g_mul)}+{fmt(g_mul)}+1={fmt(FX.grad)}", r"\ (=2x+1)",
                            font_size=34, color=theme.GRAD)
            total[1].set_color(theme.MUTED)
            total.next_to(nx, DOWN, 0.75).shift(RIGHT * 0.9)
            self.play(Write(total), run_time=self.fit(1.2))
            code = mono("self.grad += other.data * out.grad", 22).move_to([-1.2, -2.2, 0])
            box = SurroundingRectangle(code[9:11], color=theme.HIGHLIGHT, buff=0.06)
            self.wait(self.remaining() * 0.1)
            self.play(FadeIn(code), Create(box), run_time=self.fit(0.8))
            bug = VGroup(zh(f"改成 = ：相对误差 {BUG_EXPR_REL:.2f}", 22, theme.GRAD),
                         zh("PyTorch 也累加 → 每步先 zero_grad()", 22, theme.MUTED)
                         ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([4.4, -1.55, 0])
            self.wait(self.remaining() * 0.15)
            self.play(FadeIn(bug[0]), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(bug[1]), run_time=self.fit(0.6))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(*[FadeOut(m) for m in [nx, nmul, nm, nadd, ny, c1, c2, s3, e_mm, e_ma, e_ay,
                                             title, g1, g2, g3, total, code, box, bug]],
                      run_time=self.fit(0.6))

        # ── S08 拓扑排序 ─────────────────────────────────────────────────
        with self.shot("S08"):
            self.play(*self.set_heading("拓扑排序：先收齐梯度，再往上游传"), run_time=self.fit(0.8))
            order = ["a", "b", "d", "c", "e", "L"]
            parents = {"d": ["a", "b"], "e": ["d", "c"], "L": ["e"]}
            row = VGroup(*[vbox(k, theme.PARAM if k in "abc" else
                                (theme.OUTPUT if k == "L" else theme.FG), 1.0) for k in order])
            row.arrange(RIGHT, buff=0.95).move_to([0, 0.3, 0])
            nums = VGroup(*[zh(str(i + 1), 22, theme.MUTED).next_to(row[i], DOWN, 0.15)
                            for i in range(len(order))])
            links = VGroup()
            for child, ps in parents.items():
                ci = order.index(child)
                for p in ps:
                    pi = order.index(p)
                    links.add(CurvedArrow(row[pi].get_top(), row[ci].get_top(),
                                          angle=-0.9 if ci - pi > 1 else -0.6, color=theme.MUTED,
                                          stroke_width=2.5, tip_length=0.15))
            self.play(LaggedStart(*[FadeIn(r) for r in row], lag_ratio=0.15), FadeIn(nums),
                      run_time=self.fit(1.5))
            self.play(Create(links), run_time=self.fit(1.2))
            topo_note = zh("每个节点都排在用到它的节点前面", 24, theme.MUTED).move_to([0, 2.3, 0])
            self.play(FadeIn(topo_note), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.12)
            back = zh("反向：6 → 5 → … → 1", 24, theme.GRAD).move_to([0, 2.3, 0])
            self.play(Transform(topo_note, back), run_time=self.fit(0.6))
            code = code_block([
                "topo = 拓扑排序(self)",
                "self.grad = 1.0",
                "for v in reversed(topo):",
                "    v._backward()",
            ]).move_to([0, -1.75, 0])
            code[0][5:9].set_color(theme.HIGHLIGHT)
            per = (self.remaining() - 3) / len(order)
            grads = VGroup()
            for k in reversed(order):
                i = order.index(k)
                g = MathTex(fmt(G[k].grad), font_size=28, color=theme.GRAD)
                g.next_to(nums[i], DOWN, 0.15)
                grads.add(g)
                self.play(row[i][0].animate.set_stroke(theme.GRAD, width=5), FadeIn(g),
                          run_time=self.fit(min(0.8, per * 0.6)))
                self.wait(max(0.05, per * 0.4))
            self.play(FadeIn(code), run_time=self.fit(0.8))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(*[FadeOut(m) for m in [row, nums, links, topo_note, grads, code]],
                      run_time=self.fit(0.6))

        # ── S09 梯度检验 ─────────────────────────────────────────────────
        with self.shot("S09"):
            self.play(*self.set_heading("梯度检验：autograd 对不对？"), run_time=self.fit(0.8))
            numf = MathTex(r"\frac{\partial L}{\partial p}\approx\frac{L(p+\varepsilon)-L(p-\varepsilon)}{2\varepsilon}",
                           font_size=36).move_to([0, 1.95, 0])
            self.play(Write(numf), run_time=self.fit(1.5))
            hdr = VGroup(zh("参数", 22, theme.MUTED), zh("autograd", 22, theme.GRAD),
                         zh("数值梯度", 22, theme.INPUT))
            cells = [hdr[0], hdr[1], hdr[2]]
            for name, a, n in zip(["x", "y", "z"], EXPR_AUTO, EXPR_NUM, strict=True):
                cells += [MathTex(name, font_size=28), MathTex(f"{a:+.6f}", font_size=28),
                          MathTex(f"{n:+.6f}", font_size=28)]
            tbl = VGroup(*cells).arrange_in_grid(rows=4, cols=3, buff=(0.8, 0.18))
            tbl.move_to([-3.2, -0.3, 0])
            cap = zh("一个用到全部运算的表达式", 20, theme.MUTED).next_to(tbl, UP, 0.2)
            self.wait(self.remaining() * 0.1)
            self.play(FadeIn(cap), FadeIn(tbl), run_time=self.fit(1.2))
            res = VGroup(
                zh(f"表达式：相对误差 {EXPR_REL:.1e}", 24, theme.OUTPUT),
                zh(f"MLP（{N_PARAMS} 个参数）：{MLP_REL:.1e}", 24, theme.OUTPUT),
                zh(f"+= 改成 = ：{BUG_EXPR_REL:.2f} / {BUG_MLP_REL:.2f}", 24, theme.GRAD),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.4, -0.3, 0])
            self.wait(self.remaining() * 0.15)
            self.play(FadeIn(res[0]), run_time=self.fit(0.6))
            self.play(FadeIn(res[1]), run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(res[2]), run_time=self.fit(0.6))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(*[FadeOut(m) for m in [numf, cap, tbl, res]], run_time=self.fit(0.6))

        # ── S10 训练 MLP ─────────────────────────────────────────────────
        with self.shot("S10"):
            axes = Axes(x_range=[-3.2, 3.2, 1], y_range=[-1.5, 1.5, 0.5], x_length=6.2,
                        y_length=4.2, tips=False,
                        axis_config={"color": theme.MUTED, "include_numbers": False},
                        ).move_to([-3.4, 0.25, 0])
            dots = VGroup(*[Dot(axes.c2p(x, y), radius=0.06, color=theme.INPUT)
                            for x, y in zip(XS, YS, strict=True)])
            sin_pts = [axes.c2p(x, math.sin(x)) for x in np.linspace(-3, 3, 80)]
            sin_curve = VMobject(color=theme.MUTED, stroke_width=2).set_points_smoothly(sin_pts)
            sin_lbl = MathTex(r"\sin x", font_size=28, color=theme.MUTED).next_to(
                axes.c2p(1.6, 1.0), UP, 0.1)

            def pred_curve(step: int) -> VMobject:
                pts = [axes.c2p(x, max(-1.5, min(1.5, y)))
                       for x, y in zip(tm.GRID, SNAPS[step], strict=True)]
                return VMobject(color=theme.OUTPUT, stroke_width=4).set_points_as_corners(pts)

            self.play(*self.set_heading(f"用自己的引擎训练 MLP(1, [8, 8, 1])：{N_PARAMS} 个参数"),
                      Create(axes), run_time=self.fit(1))
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.05),
                      Create(sin_curve), FadeIn(sin_lbl), run_time=self.fit(1.5))
            code = code_block([
                "loss = mse(net, xs, ys)",
                "net.zero_grad()",
                "loss.backward()",
                "for p in net.parameters():",
                "    p.data -= lr * p.grad",
            ]).move_to([3.5, 1.1, 0])
            self.wait(self.remaining() * 0.1)
            self.play(FadeIn(code), run_time=self.fit(1))
            table = VGroup(zh("步数", 22, theme.MUTED),
                           DecimalNumber(0, num_decimal_places=0, font_size=30),
                           zh("损失", 22, theme.GRAD),
                           DecimalNumber(LOSSES[0], num_decimal_places=4, font_size=30),
                           ).arrange_in_grid(rows=2, cols=2, flow_order="dr", buff=(0.9, 0.15))
            table.move_to([3.5, -1.0, 0])
            curve = pred_curve(0)
            self.play(Create(curve), FadeIn(table), run_time=self.fit(1))
            self.wait(self.remaining() * 0.12)
            per = (self.remaining() - 1.5) / (len(SNAP_STEPS) - 1)
            for s in SNAP_STEPS[1:]:
                self.play(Transform(curve, pred_curve(s)), ChangeDecimalToValue(table[1], s),
                          ChangeDecimalToValue(table[3], LOSSES[s]), run_time=self.fit(per * 0.8))
                self.wait(max(0.05, per * 0.2))
            nohand = zh("没有一行手推梯度", 26, theme.HIGHLIGHT).move_to([3.5, -2.1, 0])
            self.play(FadeIn(nohand), run_time=self.fit(0.8))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(*[FadeOut(m) for m in [axes, dots, sin_curve, sin_lbl, code, table, curve,
                                             nohand]], run_time=self.fit(0.6))

        # ── S11 代价：标量图很慢 ─────────────────────────────────────────
        with self.shot("S11"):
            self.play(*self.set_heading("代价：标量计算图很慢"), run_time=self.fit(0.8))
            cols = 62
            grid = VGroup()
            for i in range(N_NODES):
                r, c = divmod(i, cols)
                grid.add(Dot([c * 0.075, -r * 0.055, 0], radius=0.018, color=theme.INPUT))
            grid.move_to([-4.0, 0.6, 0])
            cnt = VGroup(zh("一次前向（20 个样本）", 22, theme.FG),
                         zh(f"= {N_NODES} 个 Value 节点", 22, theme.FG)).arrange(DOWN, buff=0.1)
            cnt.next_to(grid, DOWN, 0.2)
            self.play(FadeIn(grid, lag_ratio=0.002), run_time=self.fit(2))
            self.play(FadeIn(cnt), run_time=self.fit(0.6))
            hdr = [zh("样本数", 22, theme.MUTED), zh("Value", 22, theme.PARAM),
                   zh("PyTorch", 22, theme.OUTPUT)]
            cells = list(hdr)
            for n, tv, tt in TIMING:
                cells += [MathTex(str(n), font_size=30), MathTex(rf"{tv:.0f}\,\mathrm{{ms}}",
                                                                 font_size=30),
                          MathTex(rf"{tt:.2f}\,\mathrm{{ms}}", font_size=30)]
            tbl = VGroup(*cells).arrange_in_grid(rows=3, cols=3, buff=(0.6, 0.25))
            tbl.move_to([3.2, 1.4, 0])
            tcap = zh("一次前向 + 反向（一次实测，因机器而异）", 18, theme.MUTED).next_to(tbl, UP, 0.2)
            self.wait(self.remaining() * 0.1)
            self.play(FadeIn(tcap), FadeIn(tbl), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.35)
            vjp = VGroup(
                zh("张量图：一次矩阵乘法 = 一个节点", 22, theme.HIGHLIGHT),
                MathTex(r"Y=XW", r"\ \Rightarrow\ ", r"\frac{\partial L}{\partial X}=G\,W^\top",
                        font_size=36),
                zh("G = ∂L/∂Y：向量-雅可比积（VJP）", 20, theme.MUTED),
            ).arrange(DOWN, buff=0.22).move_to([3.2, -1.35, 0])
            vjp[1][2].set_color(theme.GRAD)
            self.play(FadeIn(vjp[0]), run_time=self.fit(0.8))
            self.play(Write(vjp[1]), FadeIn(vjp[2]), run_time=self.fit(1.5))
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(*[FadeOut(m) for m in [grid, cnt, tcap, tbl, vjp]], run_time=self.fit(0.6))

        # ── S12 从极简到生产级 ───────────────────────────────────────────
        with self.shot("S12"):
            self.play(*self.set_heading("从极简到生产级：和 torch.autograd 对拍"),
                      run_time=self.fit(0.8))
            lt = zh("Value 引擎（本章）", 26, theme.PARAM).move_to([-3.6, 2.2, 0])
            rt = zh("PyTorch nn.Sequential", 26, theme.OUTPUT).move_to([3.6, 2.2, 0])
            self.play(FadeIn(lt), FadeIn(rt), run_time=self.fit(0.8))
            cells = []
            for name, shape, ok in PARAM_ROWS:
                cells += [mono(name, 20), mono(str(shape), 20),
                          zh("allclose ✓" if ok else "✗", 22, theme.OUTPUT if ok else theme.GRAD)]
            rows = VGroup(*cells).arrange_in_grid(rows=len(PARAM_ROWS), cols=3,
                                                  col_alignments="llc", buff=(0.5, 0.14))
            rows.move_to([0, 0.55, 0])
            self.play(LaggedStart(*[FadeIn(VGroup(*cells[3 * i:3 * i + 3]))
                                    for i in range(len(PARAM_ROWS))], lag_ratio=0.3),
                      run_time=self.fit(2.2))
            summary = VGroup(
                zh(f"梯度最大差 {MAX_DIFF:.1e}（双精度舍入误差）", 22, theme.FG),
                zh(f"SGD 500 步后损失：{LOSSES[-1]:.10f}  vs  {TORCH_FINAL:.10f}", 22, theme.FG),
                zh("torch.autograd.gradcheck：用数值梯度检验，同一个思路", 22, theme.MUTED),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([0, -1.75, 0])
            self.wait(self.remaining() * 0.1)
            for s in summary:
                self.play(FadeIn(s), run_time=self.fit(0.6))
                self.wait(self.remaining() * 0.3)
            self.wait(max(0.05, self.remaining() - 0.6))
            self.play(*[FadeOut(m) for m in [lt, rt, rows, summary]], run_time=self.fit(0.6))

        # ── S13 小结与下一章 ─────────────────────────────────────────────
        with self.shot("S13"):
            steps_ = [("前向：记录计算图", theme.OUTPUT),
                      ("反向：按拓扑序的逆序", theme.GRAD),
                      ("每个节点：上游梯度 × 局部导数", theme.GRAD),
                      ("分叉处：梯度相加（+=）", theme.HIGHLIGHT)]
            items = VGroup(*[zh(f"{i + 1}. {t}", 30, c) for i, (t, c) in enumerate(steps_)])
            items.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([0, 0.7, 0])
            self.play(*self.set_heading("小结"), run_time=self.fit(0.6))
            per = (self.remaining() * 0.6) / 4
            for it in items:
                self.play(FadeIn(it, shift=RIGHT * 0.2), run_time=self.fit(0.6))
                self.wait(max(0.05, per - 0.6))
            nxt = zh("下一章：分类与概率（softmax、交叉熵）", 30, theme.HIGHLIGHT).move_to([0, -1.9, 0])
            nbox = Rectangle(width=nxt.width + 0.6, height=nxt.height + 0.3, color=theme.HIGHLIGHT)
            nbox.move_to(nxt)
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(nxt), Create(nbox), run_time=self.fit(1))
            self.wait(max(0.05, self.remaining() - 1.0))
            self.play(FadeOut(items), FadeOut(nxt), FadeOut(nbox), *self.set_heading(None),
                      run_time=self.fit(1.0))
