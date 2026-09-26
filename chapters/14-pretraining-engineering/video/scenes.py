"""第 14 章视频：预训练工程 —— 混合精度、FlashAttention、数据并行与断点续训

画面里的所有数值都由 ../code/ 与 zero/tools/ 真实计算（见 script.md 事实清单）。
结果缓存在 video/out/cache.json；删掉它会重新计算。tiny 预训练日志读自 out/ch14/
（先按 README"主线进度"的命令跑出来；没有日志时相关镜头显示"未找到日志"）。
渲染：bash chapters/14-pretraining-engineering/video/build.sh
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Circle,
    Create,
    CurvedArrow,
    DashedLine,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    LaggedStart,
    Line,
    MathTex,
    Rectangle,
    RoundedRectangle,
    Square,
    Text,
    Transform,
    VGroup,
    Write,
)

from video_kit import theme
from video_kit.scene import NarratedScene, polyline_in_axes, zh

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
ROOT = HERE.parents[2]
CACHE = HERE / "out" / "cache.json"
MONO = "Noto Sans Mono"
GIB = 2**30


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compute() -> dict:
    """从 ../code 与 zero/tools 真实计算视频要用的全部数字。"""
    import torch

    sys.path.insert(0, str(ROOT))
    torch.set_num_threads(1)
    d: dict = {}

    # ① 算账（01_step_cost.py）
    sc = _load("step_cost", "01_step_cost.py")
    cfg = sc.shape(ROOT / "configs/main/pretrain.toml")
    fpt, n_matmul, n_total = sc.flops_per_token(cfg["model"], cfg["seq_len"])
    tr = cfg["train"]
    tok_step = tr["micro_batch_size"] * tr["grad_accum_steps"] * 8 * cfg["seq_len"]
    d["step"] = dict(fpt=fpt, n_total=n_total, tok_step=tok_step, total=fpt * 500e9)
    d["mfu_rows"] = []
    for mfu in (0.3, 0.4, 0.5):
        tps = 8 * sc.H100_BF16_DENSE * mfu / fpt
        hours = 500e9 / tps / 3600
        d["mfu_rows"].append(dict(mfu=mfu, days=hours / 24, cost=hours * 8 * sc.PRICE))

    # ③ CPU 上的 MFU（极小配置演示）
    tiny = sc.shape(ROOT / "configs/tiny/pretrain.toml")
    tfpt, _, tn = sc.flops_per_token(tiny["model"], tiny["seq_len"])
    log = _read_log(ROOT / "out/ch14/pretrain/log.jsonl")
    peak = sc.cpu_peak_flops()
    if log:
        tps = sorted(r["tok_per_s"] for r in log if r["step"] > 1)
        med = tps[len(tps) // 2]
        d["cpu_mfu"] = dict(peak=peak, tps=med, fpt=tfpt, mfu=med * tfpt / peak)

    # ② 显存（05_memory.py + zero/tools/memory_calc.py）
    mem = _load("memory_demo", "05_memory.py")
    from torch import nn

    from zero.config import load_model_config
    from zero.model import Transformer
    from zero.tools.memory_calc import STRATEGIES, estimate_memory

    torch.manual_seed(0)
    m = Transformer(mem.TINY)
    toks = torch.randint(0, mem.TINY.vocab_size, (mem.B, mem.T))
    fp32 = mem.saved_activation_bytes(m, toks, False)
    bf16 = mem.saved_activation_bytes(m, toks, True)
    ck = Transformer(mem.TINY)
    ck.load_state_dict(m.state_dict())
    ck.layers = nn.ModuleList([mem.CheckpointedBlock(b) for b in ck.layers])
    ckb = mem.saved_activation_bytes(ck, toks, True)
    d["act_ratio"] = [1.0, bf16 / fp32, ckb / fp32]
    main_cfg = load_model_config(ROOT / "configs/main/pretrain.toml")
    d["mem"] = {}
    for mb in (4, 8):
        e = estimate_memory(main_cfg, mb, 4096, num_gpus=8, strategy="ddp")
        d["mem"][str(mb)] = dict(params=e.params, grads=e.grads, optim=e.optimizer,
                                 buckets=e.ddp_buckets, copies=e.weight_copies,
                                 act=e.activations, logits=e.logits_transient, total=e.total)
    d["static"] = {s: estimate_memory(main_cfg, 1, 4096, num_gpus=8, strategy=s).static
                   for s in STRATEGIES}
    d["p16"] = 16 * n_total

    # 精度（02_precision.py）
    pr = _load("precision", "02_precision.py")
    d["bits"] = {k: pr.bits(3.14159, v[0]) for k, v in pr.FORMATS.items()}
    d["pi"] = {k: float(torch.tensor(3.14159, dtype=v[0])) for k, v in pr.FORMATS.items()}
    d["sums"] = {k: pr.repeat_add(0.01, 10_000, v[0]) for k, v in pr.FORMATS.items()}
    w = {}
    for k, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        x = torch.tensor(1.0, dtype=dt)
        for _ in range(1000):
            x = x - torch.tensor(1e-3, dtype=dt)
        w[k] = float(x)
    d["wupd"] = w
    d["fp16_70000"] = str(float(torch.tensor(70000.0, dtype=torch.float16)))

    # online softmax（03）与分块注意力（04）
    osm = _load("online_softmax", "03_online_softmax.py")
    trace, mfin, lfin = osm.online_softmax_stats(osm.EXAMPLE)
    d["osm"] = dict(x=osm.EXAMPLE, trace=trace,
                    probs=[math.exp(v - mfin) / lfin for v in osm.EXAMPLE])
    ta = _load("tiled_attention", "04_tiled_attention.py")
    q, k, v = (torch.randn(512, 64, dtype=torch.float64) for _ in range(3))
    ref, _ = ta.naive_attention(q, k, v)
    out, _, _ = ta.tiled_attention(q, k, v, 64, 64)
    d["tiled_diff"] = float((out - ref).abs().max())
    d["sp_gib"] = 2 * 4096 * 4096 * 16 * 8 * 2 / GIB

    # 数据并行（06）：多进程部分用子进程跑脚本，解析它的表格
    outp = subprocess.run([sys.executable, str(CODE / "06_ddp_by_hand.py")], capture_output=True,
                          text=True, check=True, cwd=ROOT).stdout
    rows = re.findall(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", outp, re.M)
    d["ddp_rows"] = [[int(r[0]), float(r[1]), float(r[2]), float(r[3])] for r in rows]
    d["ddp_pdiff"] = re.search(r"2 进程 ([\d.e+-]+)", outp).group(1)
    d["ring_send"] = re.search(r"每张卡发出 ([\d,]+)", outp).group(1)

    # 断点续训（07）
    rs = _load("resume_demo", "07_resume.py")
    ref_run = rs.Run()
    ref_l = [ref_run.train_step() for _ in range(rs.STEPS)]
    run = rs.Run()
    import io

    ckpt = None
    for _ in range(rs.CRASH_AT):
        run.train_step()
        if run.step == rs.SAVE_AT:
            buf = io.BytesIO()
            torch.save(run.state_dict(), buf)
            ckpt = buf.getvalue()
    res_l = rs.resume(ckpt)
    d["resume"] = dict(ref=ref_l, res=res_l, save=rs.SAVE_AT, crash=rs.CRASH_AT,
                       same=all(a == b for a, b in zip(ref_l[rs.SAVE_AT:], res_l)))
    d["resume_skip"] = {key: max(abs(a - b) for a, b in zip(ref_l[rs.SAVE_AT:], rs.resume(ckpt, key)))
                        for key in ("optim", "sched", "data_rng", "torch_rng")}

    # tiny 预训练日志（极小配置演示）
    d["tiny"] = [[r["step"], r["loss"], r.get("val_loss")] for r in log]
    d["tiny_resume"] = [[r["step"], r["loss"], r.get("val_loss")]
                        for r in _read_log(ROOT / "out/ch14/resume/log.jsonl")]
    ddp_log = ROOT / "out/ch14/ddp2.log"
    m2 = re.search(r"step\s+200/200 \| loss ([\d.]+).*val ([\d.]+)",
                   ddp_log.read_text()) if ddp_log.exists() else None
    d["ddp2"] = [float(m2.group(1)), float(m2.group(2))] if m2 else None
    return d


def get_data() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    d = compute()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


D = get_data()


def mono(text: str, size: float = 24, color: str = theme.FG) -> Text:
    return Text(text, font=MONO, font_size=size, color=color)


def hbar(value: float, vmax: float, width: float, color: str, height: float = 0.42) -> Rectangle:
    w = max(0.03, value / vmax * width)
    return Rectangle(width=w, height=height, stroke_width=0, fill_color=color, fill_opacity=0.9)


def bit_row(label: str, exp: int, man: int, cell: float, y: float) -> VGroup:
    cells = VGroup()
    colors = [theme.MUTED] + [theme.INPUT] * exp + [theme.PARAM] * man
    for c in colors:
        cells.add(Rectangle(width=cell, height=0.42, stroke_color=theme.BG, stroke_width=1.5,
                            fill_color=c, fill_opacity=0.85))
    cells.arrange(RIGHT, buff=0)
    lab = mono(label, 26, theme.FG)
    lab.move_to([-6.3, y, 0])
    cells.next_to(lab, RIGHT, 0.35)
    cells.align_to([-5.6, 0, 0], LEFT).set_y(y)
    return VGroup(lab, cells)


class ChapterScene(NarratedScene):
    chapter_label = "第 14 章"
    chapter_title = "预训练工程"

    def construct(self) -> None:
        for i in range(1, 18):
            getattr(self, f"s{i:02d}")()

    def clear_all(self, *mobs, t: float = 0.6) -> None:
        self.wait(max(0.1, self.remaining() - t))
        self.play(*[FadeOut(m) for m in mobs], run_time=self.fit(t))

    # ── S01 片头 ─────────────────────────────────────────────────────────
    def s01(self):
        with self.shot("S01"):
            card = self.chapter_card()
            sub = zh("混合精度 · FlashAttention · 数据并行 · 断点续训", 30,
                     theme.HIGHLIGHT).next_to(card, DOWN, 0.6)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=self.fit(1.5))
            self.play(Write(sub), run_time=self.fit(1.5))
            self.clear_all(card, sub, t=0.8)

    # ── S02 先算账 ───────────────────────────────────────────────────────
    def s02(self):
        with self.shot("S02"):
            self.play(*self.set_heading("先算账：一步预训练要多少算力"), run_time=self.fit(0.8))
            st = D["step"]
            f = MathTex(r"\text{FLOPs/token}=6N+12\,L\,q_{\dim}\,T", font_size=38)
            f.move_to([-3.4, 1.9, 0])
            lines = VGroup(
                zh(f"主线模型：{st['n_total'] / 1e6:.1f}M 参数，T = 4096", 24, theme.FG),
                zh(f"每 token：{st['fpt'] / 1e9:.2f} × 10⁹ 次运算", 24, theme.PARAM),
                zh(f"每步 {st['tok_step']:,} token", 24, theme.FG),
                zh(f"500B token 共 {st['total'] / 1e21:.2f} × 10²¹ 次", 24, theme.PARAM),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(f, DOWN, 0.5).align_to(f, LEFT)
            self.play(Write(f), run_time=self.fit(1.5))
            self.play(LaggedStart(*[FadeIn(x) for x in lines], lag_ratio=0.4), run_time=self.fit(3))
            title = zh("8×H100 上要多久（$2.5/卡时）", 24, theme.MUTED).move_to([3.6, 2.2, 0])
            self.play(FadeIn(title), run_time=self.fit(0.6))
            rows = VGroup()
            vmax = max(r["days"] for r in D["mfu_rows"])
            for i, r in enumerate(D["mfu_rows"]):
                y = 1.3 - i * 1.0
                lab = mono(f"MFU {r['mfu']:.1f}", 24, theme.FG).move_to([1.5, y, 0])
                color = theme.HIGHLIGHT if r["mfu"] == 0.4 else theme.INPUT
                b = hbar(r["days"], vmax, 3.4, color).next_to(lab, RIGHT, 0.25)
                b.align_to([2.3, 0, 0], LEFT)
                val = zh(f"{r['days']:.1f} 天  ${r['cost']:,.0f}", 22, theme.FG).next_to(b, DOWN, 0.08)
                val.align_to(b, LEFT)
                rows.add(VGroup(lab, b, val))
            self.play(LaggedStart(*[GrowFromEdge(r[1], LEFT) for r in rows], lag_ratio=0.3),
                      *[FadeIn(r[0]) for r in rows], run_time=self.fit(2))
            self.play(*[FadeIn(r[2]) for r in rows], run_time=self.fit(1))
            self.clear_all(f, lines, title, rows)

    # ── S03 显存里有什么 ─────────────────────────────────────────────────
    def s03(self):
        with self.shot("S03"):
            self.play(*self.set_heading("显存里装了什么（每张卡，DDP）"), run_time=self.fit(0.8))
            scale = 4.6 / 120  # GiB → 画面高度
            base = -2.4
            cap = DashedLine([-1.0, base + 80 * scale, 0], [6.6, base + 80 * scale, 0],
                             color=theme.GRAD, stroke_width=3)
            cap_l = zh("80 GB", 22, theme.GRAD).next_to(cap, RIGHT, 0.1).shift(LEFT * 0.9 + UP * 0.2)
            legend = VGroup()
            parts = [("参数 FP32", "params", theme.PARAM), ("梯度", "grads", theme.GRAD),
                     ("AdamW m、v", "optim", theme.ATTN), ("桶 + 副本", "misc", theme.MUTED),
                     ("激活 + logits", "act", theme.INPUT)]
            stacks = VGroup()
            for j, mb in enumerate(("4", "8")):
                mm = D["mem"][mb]
                vals = dict(params=mm["params"], grads=mm["grads"], optim=mm["optim"],
                            misc=mm["buckets"] + mm["copies"], act=mm["act"] + mm["logits"])
                x = 1.4 + j * 2.6
                y = base
                col = VGroup()
                for _name, key, color in parts:
                    h = vals[key] / GIB * scale
                    r = Rectangle(width=1.3, height=max(h, 0.02), stroke_color=theme.BG,
                                  stroke_width=1, fill_color=color, fill_opacity=0.9)
                    r.move_to([x, y + h / 2, 0])
                    y += h
                    col.add(r)
                tot = zh(f"{mm['total'] / GIB:.1f} GiB", 24,
                         theme.GRAD if mm["total"] > 80 * GIB else theme.OUTPUT).move_to([x, y + 0.25, 0])
                lab = zh(f"micro batch {mb}", 22, theme.FG).move_to([x, base - 0.25, 0])
                stacks.add(VGroup(col, tot, lab))
            for name, _k, color in parts:
                sq = Square(0.28, stroke_width=0, fill_color=color, fill_opacity=0.9)
                t = zh(name, 22, theme.FG).next_to(sq, RIGHT, 0.15)
                legend.add(VGroup(sq, t))
            legend.arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([-4.6, 0.6, 0])
            p16 = VGroup(zh("参数 + 梯度 + m + v = 16 字节/参数", 20, theme.HIGHLIGHT),
                         zh(f"→ {D['p16'] / GIB:.1f} GiB", 20, theme.HIGHLIGHT)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            p16.move_to([0, -1.6, 0]).align_to([-6.6, 0, 0], LEFT)
            self.play(FadeIn(legend), run_time=self.fit(1))
            self.play(Create(cap), FadeIn(cap_l), run_time=self.fit(0.8))
            s4, s8 = stacks
            self.play(LaggedStart(*[GrowFromEdge(r, DOWN) for r in s4[0][:4]], lag_ratio=0.3),
                      FadeIn(s4[2]), run_time=self.fit(2))
            self.play(FadeIn(p16), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.25)
            self.play(GrowFromEdge(s4[0][4], DOWN), FadeIn(s4[1]), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.3)
            self.play(LaggedStart(*[GrowFromEdge(r, DOWN) for r in s8[0]], lag_ratio=0.15),
                      FadeIn(s8[1]), FadeIn(s8[2]), run_time=self.fit(1.8))
            self.clear_all(legend, cap, cap_l, stacks, p16)

    # ── S04 激活检查点与梯度累积 ─────────────────────────────────────────
    def s04(self):
        with self.shot("S04"):
            self.play(*self.set_heading("两招：激活检查点、梯度累积"), run_time=self.fit(0.8))
            names = ["FP32", "BF16 autocast", "BF16 + 激活检查点"]
            colors = [theme.MUTED, theme.INPUT, theme.OUTPUT]
            rows = VGroup()
            for i, (n, r, c) in enumerate(zip(names, D["act_ratio"], colors)):
                y = 2.0 - i * 0.65
                lab = zh(n, 22, theme.FG).move_to([-4.6, y, 0])
                b = hbar(r, 1.0, 6.0, c, 0.4).move_to([0, y, 0]).align_to([-2.6, 0, 0], LEFT)
                val = mono(f"{r:.0%}", 22, theme.FG).next_to(b, RIGHT, 0.15)
                rows.add(VGroup(lab, b, val))
            cap = zh("保存的激活（tiny 模型实测）", 20, theme.MUTED).move_to([2.2, 2.55, 0])
            self.play(FadeIn(cap), *[FadeIn(r[0]) for r in rows], run_time=self.fit(0.8))
            self.play(LaggedStart(*[GrowFromEdge(r[1], LEFT) for r in rows], lag_ratio=0.4),
                      run_time=self.fit(2))
            self.play(*[FadeIn(r[2]) for r in rows], run_time=self.fit(0.6))
            self.wait(self.remaining() * 0.35)
            # 梯度累积：4 个小梯度块 → 1 个大块
            y = -1.2
            smalls = VGroup(*[Square(0.55, stroke_width=0, fill_color=theme.GRAD, fill_opacity=0.35 + 0.15 * i)
                              for i in range(4)]).arrange(RIGHT, buff=0.55).move_to([-3.2, y, 0])
            labs = VGroup(*[zh(f"micro {i + 1}", 18, theme.MUTED).next_to(s, DOWN, 0.12)
                            for i, s in enumerate(smalls)])
            big = Square(1.1, stroke_width=0, fill_color=theme.GRAD, fill_opacity=0.9).move_to([2.2, y, 0])
            arr = Arrow([-0.5, y, 0], [1.4, y, 0], color=theme.MUTED, buff=0.05)
            txt = zh("loss / k，梯度自动累加 → 一次 step", 22, theme.FG).move_to([0.2, -2.35, 0])
            self.play(LaggedStart(*[FadeIn(s) for s in smalls], lag_ratio=0.3), FadeIn(labs),
                      run_time=self.fit(1.5))
            self.play(Create(arr), Transform(smalls.copy(), big), FadeIn(big), run_time=self.fit(1.2))
            self.play(FadeIn(txt), run_time=self.fit(0.6))
            self.clear_all(cap, rows, smalls, labs, big, arr, txt)
            for m in list(self.mobjects):
                if m is not getattr(self, "_heading", None):
                    self.remove(m)

    # ── S05 三种浮点格式 ─────────────────────────────────────────────────
    def s05(self):
        with self.shot("S05"):
            self.play(*self.set_heading("三种浮点格式：指数管范围，尾数管精度"), run_time=self.fit(0.8))
            cell = 0.36
            specs = [("FP32", 8, 23, "fp32", 2.0), ("BF16", 8, 7, "bf16", 0.4), ("FP16", 5, 10, "fp16", -1.2)]
            rows = VGroup()
            maxes = {"fp32": "最大 3.4×10³⁸", "bf16": "最大 3.4×10³⁸", "fp16": "最大 65,504"}
            for name, e, mbits, key, y in specs:
                r = bit_row(name, e, mbits, cell, y)
                info = zh(f"1 + {e} + {mbits}    π → {D['pi'][key]:.7f}    {maxes[key]}", 20,
                          theme.FG).next_to(r[1], DOWN, 0.12).align_to(r[1], LEFT)
                rows.add(VGroup(r, info))
            leg = VGroup(
                VGroup(Square(0.25, stroke_width=0, fill_color=theme.MUTED, fill_opacity=0.85),
                       zh("符号", 20, theme.FG)).arrange(RIGHT, buff=0.1),
                VGroup(Square(0.25, stroke_width=0, fill_color=theme.INPUT, fill_opacity=0.85),
                       zh("指数（范围）", 20, theme.FG)).arrange(RIGHT, buff=0.1),
                VGroup(Square(0.25, stroke_width=0, fill_color=theme.PARAM, fill_opacity=0.85),
                       zh("尾数（精度）", 20, theme.FG)).arrange(RIGHT, buff=0.1),
            ).arrange(RIGHT, buff=0.5).move_to([0, -2.35, 0])
            self.play(FadeIn(leg), run_time=self.fit(0.6))
            for r in rows:
                self.play(FadeIn(r[0][0]), LaggedStart(*[FadeIn(c) for c in r[0][1]], lag_ratio=0.02),
                          run_time=self.fit(1.4))
                self.play(FadeIn(r[1]), run_time=self.fit(0.5))
                self.wait(self.remaining() * 0.2)
            self.clear_all(rows, leg)

    # ── S06 精度不够会怎样 ───────────────────────────────────────────────
    def s06(self):
        with self.shot("S06"):
            self.play(*self.set_heading("精度不够：BF16 算，FP32 存"), run_time=self.fit(0.8))
            t1 = zh("0.01 连加一万次（答案 100）", 24, theme.MUTED).move_to([-3.5, 2.2, 0])
            counters = VGroup()
            for i, key in enumerate(("fp32", "bf16", "fp16")):
                v = D["sums"][key]
                c = theme.OUTPUT if abs(v - 100) < 1 else theme.GRAD
                row = VGroup(mono(key.upper(), 26, theme.FG), mono(f"{v:.3f}", 30, c)).arrange(RIGHT, buff=0.6)
                row.move_to([-3.5, 1.3 - i * 0.8, 0])
                counters.add(row)
            t2 = zh("权重 1.0，每步减 10⁻³，走 1000 步", 24, theme.MUTED).move_to([3.4, 2.2, 0])
            wrows = VGroup(
                VGroup(zh("BF16 权重", 24, theme.FG), mono(f"{D['wupd']['bf16']:.4f}", 30, theme.GRAD)).arrange(RIGHT, buff=0.5),
                VGroup(zh("FP32 权重", 24, theme.FG), mono(f"{D['wupd']['fp32']:.4f}", 30, theme.OUTPUT)).arrange(RIGHT, buff=0.5),
            ).arrange(DOWN, buff=0.4).move_to([3.4, 1.0, 0])
            rule = zh("矩阵乘用 BF16 算；主权重、梯度累加、优化器状态用 FP32 存", 24,
                      theme.HIGHLIGHT).move_to([0, -1.0, 0])
            of = zh(f"FP16：70000 → {D['fp16_70000']}，小梯度 → 0（需要 loss scaling）", 22,
                    theme.GRAD).move_to([0, -1.9, 0])
            self.play(FadeIn(t1), run_time=self.fit(0.6))
            self.play(LaggedStart(*[FadeIn(r) for r in counters], lag_ratio=0.4), run_time=self.fit(2))
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(t2), FadeIn(wrows), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.25)
            self.play(FadeIn(rule), run_time=self.fit(0.8))
            self.wait(self.remaining() * 0.35)
            self.play(FadeIn(of), run_time=self.fit(0.8))
            self.clear_all(t1, counters, t2, wrows, rule, of)

    # ── S07 注意力慢在读写 ───────────────────────────────────────────────
    def s07(self):
        with self.shot("S07"):
            self.play(*self.set_heading("注意力慢在读写，不在计算"), run_time=self.fit(0.8))
            hbm = RoundedRectangle(width=4.2, height=3.4, corner_radius=0.15, stroke_color=theme.INPUT,
                                   fill_color=theme.INPUT, fill_opacity=0.1).move_to([-3.8, 0.2, 0])
            hl = zh("显存 HBM：80 GB，慢", 24, theme.INPUT).next_to(hbm, UP, 0.1)
            sram = RoundedRectangle(width=1.8, height=1.2, corner_radius=0.1, stroke_color=theme.OUTPUT,
                                    fill_color=theme.OUTPUT, fill_opacity=0.15).move_to([3.6, 0.7, 0])
            sl = zh("片上 SRAM：小，快", 24, theme.OUTPUT).next_to(sram, UP, 0.1)
            S = Square(1.3, stroke_color=theme.ATTN, fill_color=theme.ATTN, fill_opacity=0.3).move_to([-4.6, 0.2, 0])
            P = Square(1.3, stroke_color=theme.ATTN, fill_color=theme.ATTN, fill_opacity=0.6).move_to([-3.0, 0.2, 0])
            Sl = mono("S=QKᵀ", 20, theme.FG).move_to(S)
            Pl = mono("P", 22, theme.FG).move_to(P)
            tt = zh("T × T", 20, theme.MUTED).next_to(VGroup(S, P), DOWN, 0.15)
            self.play(FadeIn(hbm), FadeIn(hl), FadeIn(sram), FadeIn(sl), run_time=self.fit(1.2))
            self.play(FadeIn(S), FadeIn(Sl), FadeIn(P), FadeIn(Pl), FadeIn(tt), run_time=self.fit(1))
            arrows = VGroup()
            for k in range(3):
                y = 0.9 - k * 0.45
                a1 = Arrow([-1.6, y, 0], [2.6, y + 0.1, 0], color=theme.MUTED, buff=0, stroke_width=3)
                a2 = Arrow([2.6, y - 0.15, 0], [-1.6, y - 0.25, 0], color=theme.GRAD, buff=0, stroke_width=3)
                arrows.add(a1, a2)
            steps = zh("写 S → 读 S 算 softmax → 写 P → 读 P 乘 V", 22, theme.FG).move_to([2.4, -0.9, 0])
            self.play(LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.2), run_time=self.fit(2.5))
            self.play(FadeIn(steps), run_time=self.fit(0.6))
            big = zh(f"主线一层的 S + P：{D['sp_gib']:.1f} GiB", 28, theme.HIGHLIGHT).move_to([0.5, -2.3, 0])
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(big), run_time=self.fit(0.8))
            self.clear_all(hbm, hl, sram, sl, S, P, Sl, Pl, tt, arrows, steps, big)

    # ── S08 online softmax ───────────────────────────────────────────────
    def s08(self):
        with self.shot("S08"):
            self.play(*self.set_heading("online softmax：一遍扫描，边走边改正"), run_time=self.fit(0.8))
            o = D["osm"]
            f = MathTex(r"m_j=\max(m_{j-1},x_j),\quad l_j=l_{j-1}\,e^{m_{j-1}-m_j}+e^{x_j-m_j}",
                        font_size=34).move_to([0, 2.3, 0])
            self.play(Write(f), run_time=self.fit(1.5))
            xs = VGroup()
            for i, xv in enumerate(o["x"]):
                box = Square(0.9, stroke_color=theme.INPUT, fill_color=theme.INPUT, fill_opacity=0.1)
                box.move_to([-3.6 + i * 1.2, 1.0, 0])
                xs.add(VGroup(box, mono(f"{xv:g}", 30, theme.FG).move_to(box)))
            xl = zh("x", 26, theme.MUTED).next_to(xs, LEFT, 0.3)
            ml = zh("m（最大值）", 24, theme.HIGHLIGHT).move_to([-4.8, -0.3, 0])
            ll = zh("l（指数和）", 24, theme.OUTPUT).move_to([-4.8, -1.1, 0])
            mv = mono("−∞", 30, theme.HIGHLIGHT).move_to([-2.6, -0.3, 0])
            lv = mono("0", 30, theme.OUTPUT).move_to([-2.6, -1.1, 0])
            self.play(FadeIn(xs), FadeIn(xl), FadeIn(ml), FadeIn(ll), FadeIn(mv), FadeIn(lv),
                      run_time=self.fit(1))
            per = max(0.4, (self.remaining() - 3.0) / len(o["x"]))
            prev_m = None
            note = None
            for i, (mj, lj) in enumerate(o["trace"]):
                hl = xs[i][0].copy().set_stroke(theme.HIGHLIGHT, 5)
                anims = [FadeIn(hl),
                         Transform(mv, mono(f"{mj:g}", 30, theme.HIGHLIGHT).move_to(mv)),
                         Transform(lv, mono(f"{lj:.4f}", 30, theme.OUTPUT).move_to(lv).align_to(lv, LEFT))]
                if note is not None:
                    anims.append(FadeOut(note))
                    note = None
                if prev_m is not None and mj > prev_m:
                    note = zh(f"新最大值：旧的 l × e^({prev_m:g}−{mj:g}) = {math.exp(prev_m - mj):.4f}",
                              22, theme.HIGHLIGHT).move_to([1.6, -0.7, 0])
                    anims.append(FadeIn(note))
                self.play(*anims, run_time=self.fit(per * 0.6))
                self.wait(per * 0.3)
                self.play(FadeOut(hl), run_time=self.fit(per * 0.1))
                prev_m = mj
            if note is not None:
                self.play(FadeOut(note), run_time=self.fit(0.3))
            res = zh("softmax = " + "、".join(f"{p:.4f}" for p in o["probs"]) + "   与三遍写法最大差 0",
                     22, theme.FG).move_to([0, -2.2, 0])
            self.play(FadeIn(res), run_time=self.fit(0.8))
            self.clear_all(f, xs, xl, ml, ll, mv, lv, res)

    # ── S09 分块注意力 ───────────────────────────────────────────────────
    def s09(self):
        with self.shot("S09"):
            self.play(*self.set_heading("FlashAttention：分块 + online softmax"), run_time=self.fit(0.8))
            n, c = 6, 0.62
            x0, y0 = -5.6, 2.1
            grid = VGroup()
            for i in range(n):
                for j in range(n):
                    sq = Square(c, stroke_color=theme.MUTED, stroke_width=1.5,
                                fill_color=theme.ATTN, fill_opacity=0.12)
                    sq.move_to([x0 + j * c + c / 2, y0 - i * c - c / 2, 0])
                    grid.add(sq)
            ql = zh("Q 块 →", 20, theme.INPUT).rotate(math.pi / 2).next_to(grid, LEFT, 0.15)
            kl = zh("K、V 块 →", 20, theme.OUTPUT).next_to(grid, UP, 0.1)
            self.play(Create(grid), FadeIn(ql), FadeIn(kl), run_time=self.fit(1.5))
            skip = VGroup(*[grid[i * n + j] for i in range(n) for j in range(n) if j > i])
            self.play(skip.animate.set_fill(theme.BG, 1).set_stroke(theme.MUTED, 0.8), run_time=self.fit(1))
            skl = zh("右上：因果 mask，整块跳过", 20, theme.MUTED).move_to([x0 + n * c / 2, y0 - n * c - 0.3, 0])
            self.play(FadeIn(skl), run_time=self.fit(0.5))
            state = VGroup(
                zh("第 4 块 Q 的每一行：", 22, theme.FG),
                zh("m ← max(m, 块内最大)", 22, theme.HIGHLIGHT),
                zh("l ← l·e^(m旧−m新) + Σ e^(s−m新)", 22, theme.OUTPUT),
                zh("O ← O·e^(m旧−m新) + P_块 · V_块", 22, theme.ATTN),
                zh("最后 O / l；只存 lse = m + log l", 22, theme.FG),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([2.9, 0.8, 0])
            self.play(FadeIn(state[0]), run_time=self.fit(0.5))
            row = 3
            cur = None
            for j in range(row + 1):
                sq = grid[row * n + j]
                hl = sq.copy().set_fill(theme.HIGHLIGHT, 0.7).set_stroke(theme.HIGHLIGHT, 3)
                anims = [FadeIn(hl)]
                if cur is not None:
                    anims.append(cur.animate.set_fill(theme.ATTN, 0.5).set_stroke(theme.MUTED, 1.5))
                if j + 1 < len(state) - 1:
                    anims.append(FadeIn(state[j + 1]))
                self.play(*anims, run_time=self.fit(0.9))
                cur = hl
            self.play(FadeIn(state[4]), run_time=self.fit(0.6))
            diff = VGroup(zh(f"CPU 上对拍：与朴素注意力最大差 {D['tiled_diff']:.1e}", 20, theme.OUTPUT),
                          zh("精确注意力，不是近似", 20, theme.OUTPUT)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            diff.move_to([2.9, -1.3, 0]).align_to(state, LEFT)
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(diff), run_time=self.fit(0.8))
            self.clear_all(grid, ql, kl, skl, state, diff)
            for m in list(self.mobjects):
                if m is not getattr(self, "_heading", None):
                    self.remove(m)

    # ── S10 数据并行 ─────────────────────────────────────────────────────
    def s10(self):
        with self.shot("S10"):
            self.play(*self.set_heading("数据并行：梯度求平均"), run_time=self.fit(0.8))
            cards = VGroup()
            for i, x in enumerate((-4.8, -1.6)):
                box = RoundedRectangle(width=2.4, height=1.5, corner_radius=0.12, stroke_color=theme.PARAM,
                                       fill_color=theme.PARAM, fill_opacity=0.1).move_to([x, 1.4, 0])
                t = zh(f"卡 {i}：完整模型", 22, theme.PARAM).move_to(box).shift(UP * 0.35)
                dt = zh(f"数据的第 {i + 1} 半", 20, theme.INPUT).move_to(box).shift(DOWN * 0.3)
                cards.add(VGroup(box, t, dt))
            grads = VGroup(*[Square(0.45, stroke_width=0, fill_color=theme.GRAD, fill_opacity=0.8)
                             .next_to(cards[i], DOWN, 0.25) for i in range(2)])
            avg = zh("all-reduce：求和 ÷ 2", 22, theme.GRAD).move_to([-3.2, -0.6, 0])
            self.play(FadeIn(cards), run_time=self.fit(1))
            self.play(FadeIn(grads), run_time=self.fit(0.8))
            self.play(grads[0].animate.move_to([-3.5, -0.05, 0]), grads[1].animate.move_to([-2.9, -0.05, 0]),
                      FadeIn(avg), run_time=self.fit(1.2))
            code = VGroup(
                mono("dist.all_reduce(p.grad, SUM)", 22, theme.FG),
                mono("p.grad /= world", 22, theme.FG),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([-3.2, -1.6, 0])
            self.play(FadeIn(code), run_time=self.fit(0.8))
            hdr = VGroup(*[zh(t, 20, theme.MUTED) for t in ("步", "一次 2B 条", "梯度累积", "2 进程")])
            table = VGroup()
            xs = [0.9, 2.3, 3.95, 5.6]
            for k, h in enumerate(hdr):
                h.move_to([xs[k], 2.0, 0])
            for r_i, r in enumerate(D["ddp_rows"]):
                y = 1.35 - r_i * 0.55
                cells = [mono(str(r[0]), 20, theme.FG)] + [mono(f"{v:.6f}", 18, theme.OUTPUT) for v in r[1:]]
                for k, cc in enumerate(cells):
                    cc.move_to([xs[k], y, 0])
                table.add(VGroup(*cells))
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(hdr), LaggedStart(*[FadeIn(r) for r in table], lag_ratio=0.3),
                      run_time=self.fit(2))
            same = zh(f"逐位相同；最终参数最大差 {D['ddp_pdiff']}", 22, theme.HIGHLIGHT).move_to([3.4, -1.3, 0])
            self.play(FadeIn(same), run_time=self.fit(0.8))
            self.clear_all(cards, grads, avg, code, hdr, table, same)

    # ── S11 环形 all-reduce ──────────────────────────────────────────────
    def s11(self):
        with self.shot("S11"):
            self.play(*self.set_heading("环形 all-reduce"), run_time=self.fit(0.8))
            n = 4
            center, rad = [-3.2, 0.1, 0], 1.9
            cols = [theme.INPUT, theme.PARAM, theme.ATTN, theme.OUTPUT]
            nodes = VGroup()
            chunks = []
            for i in range(n):
                ang = math.pi / 2 - i * 2 * math.pi / n
                pos = [center[0] + rad * math.cos(ang), center[1] + rad * math.sin(ang), 0]
                circ = Circle(0.72, stroke_color=theme.FG, stroke_width=2).move_to(pos)
                lab = zh(f"卡{i}", 18, theme.MUTED).next_to(circ, UP if i == 0 else DOWN, 0.08)
                blocks = VGroup(*[Rectangle(width=0.22, height=0.22 + 0.0, stroke_width=0,
                                            fill_color=cols[k], fill_opacity=0.35) for k in range(n)])
                blocks.arrange(RIGHT, buff=0.06).move_to(pos)
                chunks.append(blocks)
                nodes.add(VGroup(circ, lab, blocks))
            ring = VGroup()
            for i in range(n):
                a = nodes[i][0].get_center()
                b = nodes[(i + 1) % n][0].get_center()
                ring.add(CurvedArrow(a + (b - a) * 0.28, a + (b - a) * 0.72, angle=-0.4,
                                     color=theme.MUTED, stroke_width=3))
            self.play(FadeIn(nodes), Create(ring), run_time=self.fit(1.5))
            phase = zh("① reduce-scatter：N−1 轮，每轮传一块、累加", 18, theme.FG).move_to([3.6, 1.8, 0]).align_to([0.0, 0, 0], LEFT)
            self.play(FadeIn(phase), run_time=self.fit(0.6))
            per = max(0.5, (self.remaining() - 6) / 6)
            # reduce-scatter：第 r 轮，卡 i 把第 (i−r) 块发给右邻居，右邻居这一块变深
            for r in range(n - 1):
                anims = []
                for i in range(n):
                    c = (i - r) % n
                    dst = (i + 1) % n
                    anims.append(chunks[dst][c].animate.set_fill(opacity=min(1.0, 0.35 + 0.22 * (r + 1))))
                self.play(*[a.animate.set_color(theme.HIGHLIGHT) for a in ring], *anims, run_time=self.fit(per * 0.7))
                self.play(*[a.animate.set_color(theme.MUTED) for a in ring], run_time=self.fit(per * 0.3))
            phase2 = zh("② all-gather：再 N−1 轮，把完整的块传一圈", 18, theme.FG).move_to([3.6, 1.1, 0]).align_to([0.0, 0, 0], LEFT)
            self.play(FadeIn(phase2), run_time=self.fit(0.6))
            for r in range(n - 1):
                self.play(*[chunks[i][k].animate.set_fill(opacity=1.0) for i in range(n) for k in range(n)
                            if (k - i) % n <= r + 1 or r == n - 2],
                          *[a.animate.set_color(theme.HIGHLIGHT) for a in ring], run_time=self.fit(per * 0.7))
                self.play(*[a.animate.set_color(theme.MUTED) for a in ring], run_time=self.fit(per * 0.3))
            res = VGroup(
                zh("每卡发送 = 2(N−1)/N × 自己的梯度量", 22, theme.HIGHLIGHT),
                zh(f"4 卡模拟：每卡 1,000,000 个 → 发出 {D['ring_send']} 个", 18, theme.FG),
                zh("几乎与卡数无关", 20, theme.FG),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.6, -0.6, 0]).align_to([0.0, 0, 0], LEFT)
            self.play(FadeIn(res), run_time=self.fit(0.8))
            self.clear_all(nodes, ring, phase, phase2, res)

    # ── S12 FSDP 切分 ────────────────────────────────────────────────────
    def s12(self):
        with self.shot("S12"):
            self.play(*self.set_heading("ZeRO / FSDP：把状态切开"), run_time=self.fit(0.8))
            layers = [("参数", theme.PARAM), ("梯度", theme.GRAD), ("优化器", theme.ATTN)]

            def gpus(x0, sharded):
                g = VGroup()
                for i in range(4):
                    col = VGroup()
                    for _n, color in layers:
                        row = VGroup(*[Rectangle(width=0.32, height=0.3, stroke_color=theme.BG, stroke_width=1,
                                                 fill_color=color,
                                                 fill_opacity=0.9 if (not sharded or s == i) else 0.08)
                                       for s in range(4)]).arrange(RIGHT, buff=0)
                        col.add(row)
                    col.arrange(DOWN, buff=0.08)
                    box = VGroup(col, zh(f"卡{i}", 18, theme.MUTED).next_to(col, DOWN, 0.08))
                    g.add(box)
                g.arrange(RIGHT, buff=0.3).move_to([x0, 1.3, 0])
                return g

            left = gpus(-3.6, False)
            right = gpus(3.4, True)
            ll = zh("DDP：每卡一份全量", 22, theme.FG).next_to(left, UP, 0.2)
            rl = zh("FSDP：每卡只存 1/N", 22, theme.FG).next_to(right, UP, 0.2)
            leg = VGroup(*[VGroup(Square(0.22, stroke_width=0, fill_color=c, fill_opacity=0.9), zh(n, 18, theme.FG))
                           .arrange(RIGHT, buff=0.08) for n, c in layers]).arrange(RIGHT, buff=0.35)
            leg.move_to([0, 0.05, 0])
            self.play(FadeIn(left), FadeIn(ll), FadeIn(leg), run_time=self.fit(1.2))
            self.wait(self.remaining() * 0.15)
            self.play(FadeIn(right), FadeIn(rl), run_time=self.fit(1.2))
            names = [("ddp", "DDP"), ("zero1", "ZeRO-1"), ("zero2", "ZeRO-2"), ("fsdp", "FSDP")]
            vmax = max(D["static"].values())
            bars = VGroup()
            for i, (k, n) in enumerate(names):
                y = -0.55 - i * 0.5
                lab = mono(n, 20, theme.FG).move_to([-4.3, y, 0])
                b = hbar(D["static"][k], vmax, 5.2, theme.INPUT, 0.34).move_to([0, y, 0]).align_to([-3.4, 0, 0], LEFT)
                v = mono(f"{D['static'][k] / GIB:.2f} GiB", 20, theme.FG).next_to(b, RIGHT, 0.15)
                bars.add(VGroup(lab, b, v))
            cap = zh("主线每卡静态显存（8 卡）", 20, theme.MUTED).move_to([4.9, -1.7, 0])
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(cap), LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.3), run_time=self.fit(2))
            self.clear_all(left, right, ll, rl, leg, bars, cap)

    # ── S13 MFU ──────────────────────────────────────────────────────────
    def s13(self):
        with self.shot("S13"):
            self.play(*self.set_heading("MFU：离硬件峰值还有多远"), run_time=self.fit(0.8))
            f = MathTex(r"\mathrm{MFU}=\frac{\text{tokens/s}\times\text{FLOPs/token}}{\text{peak FLOPs/s}}",
                        font_size=40).move_to([-3.2, 1.6, 0])
            self.play(Write(f), run_time=self.fit(1.5))
            reps = [("PaLM 540B", 46.2, 46.2), ("Llama 3 405B", 38, 43), ("Nemotron-4 340B", 41.0, 42.4)]
            rows = VGroup()
            for i, (n, lo, hi) in enumerate(reps):
                y = 2.0 - i * 0.8
                lab = zh(n, 22, theme.FG).move_to([2.3, y, 0])
                b = hbar(hi, 60, 3.0, theme.OUTPUT, 0.36).move_to([0, y, 0]).align_to([3.7, 0, 0], LEFT)
                val = mono(f"{lo:g}%" if lo == hi else f"{lo:g}–{hi:g}%", 20, theme.FG).next_to(b, DOWN, 0.06)
                rows.add(VGroup(lab, b, val))
            src = zh("公开报告", 18, theme.MUTED).next_to(rows, UP, 0.15)
            self.play(FadeIn(src), LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.3), run_time=self.fit(2))
            self.wait(self.remaining() * 0.3)
            badge = self.show_badge()
            cm = D.get("cpu_mfu")
            if cm:
                txt = VGroup(
                    zh(f"tiny 模型在本机单线程 CPU：{cm['tps']:,.0f} tok/s × {cm['fpt'] / 1e6:.2f}M FLOPs/token", 22, theme.FG),
                    zh(f"÷ 实测峰值 {cm['peak'] / 1e9:.1f} GFLOPS ≈ {cm['mfu']:.1%}（只演示算法，不能和 GPU 比）", 22,
                       theme.HIGHLIGHT),
                ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([0, -1.5, 0])
            else:
                txt = zh("未找到 tiny 预训练日志", 22, theme.GRAD).move_to([0, -1.5, 0])
            self.play(FadeIn(txt), run_time=self.fit(1))
            self.clear_all(f, rows, src, txt, badge)

    # ── S14 loss spike ───────────────────────────────────────────────────
    def s14(self):
        with self.shot("S14"):
            self.play(*self.set_heading("loss spike：训练中途突然爆炸"), run_time=self.fit(0.8))
            from manim import Axes

            ax = Axes(x_range=[0, 100, 20], y_range=[2, 6, 1], x_length=5.6, y_length=3.6,
                      axis_config={"color": theme.MUTED, "include_ticks": False}).move_to([-3.4, 0.2, 0])
            xl = zh("训练步", 20, theme.MUTED).next_to(ax, DOWN, 0.1)
            yl = zh("loss", 20, theme.MUTED).next_to(ax, LEFT, 0.1)
            # 示意曲线（不是真实数据）：平滑下降，第 60 步附近一个尖峰
            pts = []
            for i in range(0, 101):
                base = 2.6 + 2.8 * math.exp(-i / 22)
                spike = 2.0 * math.exp(-((i - 60) / 1.6) ** 2)
                pts.append((i, base + spike))
            curve = polyline_in_axes(ax, pts, color=theme.INPUT, stroke_width=4)
            note = zh("示意图", 18, theme.MUTED).next_to(ax, UP, 0.05).align_to(ax, RIGHT)
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(note), run_time=self.fit(1))
            self.play(Create(curve), run_time=self.fit(2.5))
            cause = zh("尖峰 = 特定数据 × 特定参数状态（PaLM）", 22, theme.HIGHLIGHT).move_to([-3.2, -2.3, 0])
            self.play(FadeIn(cause), run_time=self.fit(0.8))
            items = VGroup(*[zh(t, 22, c) for t, c in [
                ("梯度裁剪（范数 1.0）", theme.FG), ("warmup + 合适的学习率", theme.FG),
                ("QK-Norm：注意力分数不失控", theme.FG), ("过滤长串重复的 n-gram", theme.FG),
                ("回滚到尖峰前，跳过那几批", theme.GRAD)]]).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
            items.move_to([3.6, 0.3, 0])
            self.wait(self.remaining() * 0.15)
            self.play(LaggedStart(*[FadeIn(it, shift=RIGHT * 0.2) for it in items], lag_ratio=0.5),
                      run_time=self.fit(4))
            self.clear_all(ax, xl, yl, note, curve, cause, items)

    # ── S15 断点续训 ─────────────────────────────────────────────────────
    def s15(self):
        with self.shot("S15"):
            self.play(*self.set_heading("断点续训：恢复全部状态"), run_time=self.fit(0.8))
            r = D["resume"]
            n = len(r["ref"])
            x0, x1, y = -6.0, 1.0, 1.9
            tl = Line([x0, y, 0], [x1, y, 0], color=theme.MUTED, stroke_width=4)

            def xp(step):
                return x0 + (x1 - x0) * step / n

            flag = VGroup(Line([xp(r["save"]), y, 0], [xp(r["save"]), y + 0.5, 0], color=theme.OUTPUT, stroke_width=4),
                          zh(f"第 {r['save']} 步存档", 18, theme.OUTPUT).move_to([xp(r["save"]) - 0.2, y + 0.75, 0]))
            crash = VGroup(mono("✕", 34, theme.GRAD).move_to([xp(r["crash"]), y, 0]),
                           zh(f"第 {r['crash']} 步崩溃 → 回到第 {r['save']} 步存档续训", 18, theme.GRAD)
                           .next_to([xp(r["crash"]), y - 0.3, 0], DOWN, 0.05).align_to([xp(r["save"]) - 0.3, 0, 0], LEFT))
            self.play(Create(tl), run_time=self.fit(0.8))
            self.play(FadeIn(flag), run_time=self.fit(0.6))
            self.play(FadeIn(crash), run_time=self.fit(0.6))
            from manim import Axes

            lo, hi = min(r["ref"]) - 0.05, max(r["ref"]) + 0.05
            ax = Axes(x_range=[0, n, 5], y_range=[lo, hi, (hi - lo) / 4], x_length=x1 - x0, y_length=2.3,
                      axis_config={"color": theme.MUTED, "include_ticks": False}).move_to([(x0 + x1) / 2, -0.6, 0])
            ref_c = polyline_in_axes(ax, [(i + 1, v) for i, v in enumerate(r["ref"])], color=theme.INPUT, stroke_width=4)
            res_c = polyline_in_axes(ax, [(r["save"] + 1 + i, v) for i, v in enumerate(r["res"])],
                                     color=theme.HIGHLIGHT, stroke_width=3)
            lg = VGroup(zh("不中断", 18, theme.INPUT), zh("续训", 18, theme.HIGHLIGHT)).arrange(RIGHT, buff=0.4)
            lg.next_to(ax, DOWN, 0.1)
            self.play(Create(ax), Create(ref_c), FadeIn(lg), run_time=self.fit(1.5))
            self.play(Create(res_c), run_time=self.fit(1.2))
            same = zh("续训 15 步逐位相同" if r["same"] else "续训结果不一致", 22, theme.OUTPUT).move_to([4.4, -1.7, 0])
            self.play(FadeIn(same), run_time=self.fit(0.6))
            labels = [("optim", "不恢复优化器"), ("sched", "不恢复调度器"), ("data_rng", "不恢复数据位置"),
                      ("torch_rng", "不恢复随机数")]
            vmax = max(D["resume_skip"].values())
            bars = VGroup()
            for i, (k, t) in enumerate(labels):
                yy = 1.7 - i * 0.75
                lab = zh(t, 20, theme.FG).move_to([3.3, yy, 0])
                b = hbar(D["resume_skip"][k], vmax, 1.8, theme.GRAD, 0.3).move_to([0, yy, 0]).align_to([4.6, 0, 0], LEFT)
                v = mono(f"{D['resume_skip'][k]:.1e}", 18, theme.FG).next_to(b, DOWN, 0.05).align_to(b, LEFT)
                bars.add(VGroup(lab, b, v))
            cap = zh("少恢复一项：loss 最大偏差", 20, theme.MUTED).move_to([4.4, 2.5, 0])
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(cap), LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.3), run_time=self.fit(2))
            self.clear_all(tl, flag, crash, ax, ref_c, res_c, lg, same, cap, bars)

    # ── S16 极小配置演示 ─────────────────────────────────────────────────
    def s16(self):
        with self.shot("S16"):
            self.play(*self.set_heading("zero 生产级代码：CPU 上真跑一遍"), run_time=self.fit(0.8))
            badge = self.show_badge()
            tiny, tres = D["tiny"], D["tiny_resume"]
            from manim import Axes

            if not tiny:
                msg = zh("未找到 out/ch14 的 tiny 预训练日志", 24, theme.GRAD)
                self.play(FadeIn(msg), run_time=self.fit(0.6))
                self.clear_all(msg, badge)
                return
            ax = Axes(x_range=[0, 200, 50], y_range=[5, 8, 1], x_length=6.2, y_length=3.6,
                      axis_config={"color": theme.MUTED, "include_numbers": True, "font_size": 20}).move_to([-3.0, 0.1, 0])
            xl = zh("步", 20, theme.MUTED).next_to(ax, DOWN, 0.1)
            yl = zh("训练 loss", 20, theme.MUTED).next_to(ax, UP, 0.05).align_to(ax, LEFT)
            c1 = polyline_in_axes(ax, [(s, l) for s, l, _ in tiny], color=theme.INPUT, stroke_width=4)
            dots = VGroup(*[Circle(0.06, color=theme.INPUT, fill_opacity=1).move_to(ax.c2p(s, l)) for s, l, _ in tiny])
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=self.fit(1))
            self.play(Create(c1), FadeIn(dots), run_time=self.fit(2))
            first, last = tiny[0], tiny[-1]
            info = VGroup(
                zh("1.31M 参数，200 步，单线程 CPU", 22, theme.FG),
                zh(f"loss {first[1]:.2f} → {last[1]:.2f}，val {last[2]:.2f}", 22, theme.FG),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.7, 1.6, 0]).align_to([0.7, 0, 0], LEFT)
            self.play(FadeIn(info), run_time=self.fit(0.8))
            ck = DashedLine(ax.c2p(100, 5), ax.c2p(100, 8), color=theme.OUTPUT)
            ckl = zh("第 100 步存档", 18, theme.OUTPUT).next_to(ck, UP, 0.05)
            self.wait(self.remaining() * 0.15)
            self.play(Create(ck), FadeIn(ckl), run_time=self.fit(0.6))
            if tres:
                c2 = polyline_in_axes(ax, [(100, [x for x in tiny if x[0] == 100][0][1])] + [(s, l) for s, l, _ in tres],
                                      color=theme.HIGHLIGHT, stroke_width=3)
                same = all(any(a[0] == b[0] and a[1] == b[1] for a in tiny) for b in tres)
                rtxt = VGroup(zh("删掉第 200 步存档，同一条命令重跑：", 20, theme.HIGHLIGHT),
                              zh("第 125–200 步 loss 逐位相同" if same else "结果不一致", 20, theme.HIGHLIGHT)
                              ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([3.3, 0.2, 0]).align_to([0.7, 0, 0], LEFT)
                self.play(Create(c2), FadeIn(rtxt), run_time=self.fit(1.5))
            else:
                rtxt = zh("未找到续训日志", 22, theme.GRAD).move_to([3.3, 0.3, 0])
                self.play(FadeIn(rtxt), run_time=self.fit(0.5))
            lines = [f"torchrun 2 进程 DDP：跑通，val {D['ddp2'][1]:.2f}" if D.get("ddp2") else "torchrun：未找到日志",
                     "pytest：续训、DDP、显存计算器 11 项通过"]
            more = VGroup(*[zh(t, 22, theme.FG) for t in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
            more.move_to([3.4, -1.2, 0]).align_to([0.7, 0, 0], LEFT)
            self.wait(self.remaining() * 0.2)
            self.play(FadeIn(more), run_time=self.fit(0.8))
            self.clear_all(ax, xl, yl, c1, dots, info, ck, ckl, rtxt, more, badge)
            for m in list(self.mobjects):
                if m is not getattr(self, "_heading", None):
                    self.remove(m)

    # ── S17 小结与下一章 ─────────────────────────────────────────────────
    def s17(self):
        with self.shot("S17"):
            self.play(*self.set_heading("小结"), run_time=self.fit(0.8))
            items = ["算账\n算力·显存·时间", "BF16 算\nFP32 存", "FlashAttention\n分块 + online softmax",
                     "DDP 求平均\nFSDP 切状态", "续训\n恢复全部状态"]
            boxes = VGroup()
            for t in items:
                txt = zh(t, 20, theme.FG)
                if txt.width > 2.35:
                    txt.scale_to_fit_width(2.35)
                box = RoundedRectangle(width=2.6, height=1.3, corner_radius=0.12, stroke_color=theme.INPUT,
                                       fill_color=theme.INPUT, fill_opacity=0.1)
                txt.move_to(box)
                boxes.add(VGroup(box, txt))
            boxes.arrange(RIGHT, buff=0.15).move_to([0, 1.2, 0])
            self.play(LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.5), run_time=self.fit(4))
            plan = zh("第二步：先花不到 50 美元在真 GPU 上逐项验证、实测 MFU，再定预训练预算", 22,
                      theme.HIGHLIGHT).move_to([0, -0.4, 0])
            self.wait(self.remaining() * 0.3)
            self.play(FadeIn(plan), run_time=self.fit(0.8))
            nxt = zh("下一章：中期训练与长上下文", 30, theme.OUTPUT).move_to([0, -1.6, 0])
            self.wait(self.remaining() * 0.4)
            self.play(FadeIn(nxt), run_time=self.fit(0.8))
            self.clear_all(boxes, plan, nxt)
            self.play(*self.set_heading(None), run_time=0.3)
