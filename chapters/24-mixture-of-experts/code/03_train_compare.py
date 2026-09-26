"""第 24 章 · 极简代码 3：同一个小语言模型，FFN 换成 MoE，和稠密比

模型：第 10 章的字符级莎士比亚小模型（4 层、宽 128、4 个头，Pre-Norm RMSNorm + RoPE），只换 FFN：

  稠密-384      ：SwiGLU 宽 384                       —— 激活参数（≈ 每 token 算力）的基准
  稠密-1536     ：SwiGLU 宽 1536                      —— 总参数和 MoE 差不多
  MoE-无均衡    ：8 专家 × 宽 192，选 2（激活宽 384）   —— 不做任何负载均衡
  MoE-辅助损失  ：同上 + L_aux，α = 0.01
  MoE-无辅助损失：同上 + 偏置均衡，γ = 0.01
  细粒度+共享   ：16 专家 × 宽 96，选 3，+ 1 个宽 96 的共享专家（激活宽 384），偏置均衡

所有 MoE 都用 sigmoid 打分 + top-k 归一化（DeepSeek-V3 的写法；它的对照实验也是在这个设定下比较两种均衡）。
同样的数据顺序、步数、AdamW + warmup + cosine。每种跑 2 个随机种子。训练中每 50 步记一次各层专家负载。

注意：这是百万参数、几分钟训练的极小实验，只能说明"代码对、现象大致如何"，
不能推出大模型上的结论（DeepSeek-V3 报告表 5、Kimi K2 报告图 5 做的才是认真的对比）。
第一次运行要训练 12 个小模型（单线程约 30–40 分钟），结果缓存在 code/out/moe_runs/。
可以按方案名分几个进程并行训练：python 03_train_compare.py 稠密-384 MoE-无均衡 ……，再不带参数运行汇总。
运行：uv run python chapters/24-mixture-of-experts/code/03_train_compare.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CH10 = HERE.parents[1] / "10-inference" / "code" / "01_tiny_model.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tiny = _load("tiny_ch10", CH10)          # 数据、注意力、RMSNorm、RoPE 都用第 10 章的
moe_mod = _load("moe_ch24", HERE / "02_moe_layer.py")

# 名称 → FFN 构造参数
VARIANTS = {
    "稠密-384": dict(kind="dense", hidden=384),
    "稠密-1536": dict(kind="dense", hidden=1536),
    "MoE-无均衡": dict(kind="moe", E=8, K=2, hidden=192, balance="none"),
    "MoE-辅助损失": dict(kind="moe", E=8, K=2, hidden=192, balance="aux"),
    "MoE-无辅助损失": dict(kind="moe", E=8, K=2, hidden=192, balance="free"),
    "细粒度+共享": dict(kind="moe", E=16, K=3, hidden=96, shared=1, balance="free"),
}
SEEDS = (0, 1)
STEPS = 800
LOG_EVERY = 50
SHOW_LAYER = 1  # 打印和视频里展示的层


def make_ffn(v: dict, dim: int) -> nn.Module:
    if v["kind"] == "dense":
        return moe_mod.Expert(dim, v["hidden"])
    return moe_mod.MoE(dim, v["E"], v["K"], v["hidden"], n_shared=v.get("shared", 0),
                       score="sigmoid", balance=v["balance"], aux_coef=0.01, bias_speed=0.01)


class Block(nn.Module):
    def __init__(self, c, ffn: nn.Module) -> None:
        super().__init__()
        self.n1, self.attn = tiny.RMSNorm(c.dim), tiny.Attention(c)
        self.n2, self.ffn = tiny.RMSNorm(c.dim), ffn

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin, None, 0)
        return x + self.ffn(self.n2(x))


class LM(nn.Module):
    def __init__(self, v: dict) -> None:
        super().__init__()
        c = self.c = tiny.Config(vocab_size=tiny.CharData().vocab_size)
        self.emb = nn.Embedding(c.vocab_size, c.dim)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(c, make_ffn(v, c.dim)) for _ in range(c.n_layers))
        self.norm = tiny.RMSNorm(c.dim)
        cos, sin = tiny.rope_tables(c.head_dim, c.max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, ids):
        T = ids.shape[1]
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x, self.cos[:T], self.sin[:T])
        return self.norm(x) @ self.emb.weight.T

    def moes(self):
        return [b.ffn for b in self.blocks if isinstance(b.ffn, moe_mod.MoE)]


def ffn_params(model: LM) -> tuple[int, int]:
    """(FFN 总参数, FFN 激活参数)，4 层合计。"""
    tot = act = 0
    for b in model.blocks:
        f = b.ffn
        n = moe_mod.n_params(f)
        tot += n
        if isinstance(f, moe_mod.MoE):
            n -= (f.E - f.K) * moe_mod.n_params(f.experts[0])
        act += n
    return tot, act


@torch.no_grad()
def val_loss(model: LM, n_batches: int = 20, seq: int = 64) -> float:
    data = tiny.CharData()
    g = torch.Generator().manual_seed(1234)
    model.eval()
    tot = 0.0
    for _ in range(n_batches):
        x, y = data.batch("val", 32, seq, g)
        tot += F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    model.train()
    return tot / n_batches


def train(name: str, seed: int, steps: int = STEPS, bsz: int = 16, seq: int = 64,
          lr: float = 3e-3) -> dict:
    v = VARIANTS[name]
    data = tiny.CharData()
    torch.manual_seed(seed)
    model = LM(v)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    g = torch.Generator().manual_seed(seed)  # 同一个种子 → 所有变体看到同样的数据顺序
    loads, curve = [], []
    t0 = time.time()
    for step in range(steps + 1):
        for pg in opt.param_groups:  # warmup + cosine（第 6 章）
            pg["lr"] = lr * min(1, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / steps))
        x, y = data.batch("train", bsz, seq, g)
        ce = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        aux = sum((m.aux for m in model.moes()), torch.zeros(()))
        opt.zero_grad()
        (ce + aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for m in model.moes():
            if m.balance == "free":
                m.update_bias()          # 无辅助损失均衡：每步之后按这一步的负载更新偏置
        if step % LOG_EVERY == 0 and model.moes():
            loads.append([(m.load / m.load.sum()).tolist() for m in model.moes()])
        if step % 100 == 0:
            curve.append((step, ce.item()))
    tot, act = ffn_params(model)
    return dict(name=name, seed=seed, val=val_loss(model), loads=loads, curve=curve,
                ffn_total=tot, ffn_active=act, total=moe_mod.n_params(model),
                secs=time.time() - t0)


def run(only: list[str] | None = None) -> list[dict]:
    """每个 (方案, 种子) 训练一次，结果存成 code/out/moe_runs/*.pt（*.pt 被 .gitignore 忽略），
    之后直接读缓存。only 给出方案名时只训练这些（可以开几个进程并行跑，最后再不带参数汇总）。"""
    runs = OUT / "moe_runs"
    runs.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        for i, name in enumerate(VARIANTS):
            path = runs / f"v{i}_s{seed}.pt"
            if path.exists():
                rows.append(torch.load(path, weights_only=False))
                continue
            if only is not None and name not in only:
                continue
            r = train(name, seed)
            print(f"  训练 {name} 种子 {seed}：{r['secs']:.0f}s，验证 loss {r['val']:.3f}", flush=True)
            torch.save(r, path)
            rows.append(r)
    return rows


def imbalance(frac: list[float]) -> float:
    """最忙专家的负载 ÷ 平均负载（1 = 完全均衡；E 个专家时最大为 E/K）。"""
    return max(frac) * len(frac)


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for name in VARIANTS:
        rs = [r for r in rows if r["name"] == name]
        vals = [r["val"] for r in rs]
        s = dict(name=name, vals=vals, mean=sum(vals) / len(vals), ffn_total=rs[0]["ffn_total"],
                 ffn_active=rs[0]["ffn_active"], total=rs[0]["total"])
        if rs[0]["loads"]:
            # 最后一次记录的负载：每层的"最大/平均"，取 4 层和 2 个种子的平均与最大
            ims = [imbalance(layer) for r in rs for layer in r["loads"][-1]]
            s["imb_mean"], s["imb_max"] = sum(ims) / len(ims), max(ims)
            dead = [sum(v < 0.01 for v in layer) for r in rs for layer in r["loads"][-1]]
            s["dead"] = max(dead)  # 负载 < 1% 的专家数（单层最多几个）
        out.append(s)
    return out


if __name__ == "__main__":
    only = sys.argv[1:] or None
    rows = run(only)
    if only is not None:
        sys.exit(0)
    print(f"4 层、宽 128 的字符级语言模型，训练 {STEPS} 步 × {len(SEEDS)} 个种子；FFN 参数为 4 层合计")
    print(f"  {'方案':14}{'FFN 总参数':>11}{'FFN 激活':>10}{'全模型':>10}   验证 loss 均值 [各种子]"
          f"      负载 最大/平均（均值, 最坏）  近乎闲置的专家")
    for s in summarize(rows):
        vals = " ".join(f"{v:.3f}" for v in s["vals"])
        load = (f"{s['imb_mean']:.2f}, {s['imb_max']:.2f}" if "imb_mean" in s else "  —")
        dead = f"{s['dead']}" if "dead" in s else "—"
        print(f"  {s['name']:14}{s['ffn_total']:11,d}{s['ffn_active']:10,d}{s['total']:10,d}"
              f"   {s['mean']:.3f} [{vals}]      {load:>12}              {dead}")
    spread = max(max(s["vals"]) - min(s["vals"]) for s in summarize(rows))
    print(f"  同一方案换种子，验证 loss 最多相差 {spread:.3f}；差距小于这个量级时不能当真")
    print(f"\n  各层负载的「最大/平均」：训练开始（step 0）→ 结束（step {STEPS}），种子 0")
    for name in ("MoE-无均衡", "MoE-辅助损失", "MoE-无辅助损失"):
        r = next(r for r in rows if r["name"] == name and r["seed"] == 0)
        pairs = "  ".join(f"第{i}层 {imbalance(a):.2f}→{imbalance(b):.2f}"
                          for i, (a, b) in enumerate(zip(r["loads"][0], r["loads"][-1])))
        print(f"  {name:14}{pairs}")
    print(f"\n  第 {SHOW_LAYER} 层（从 0 数）的专家负载占比随训练变化（种子 0；均匀是 0.125）：")
    for name in ("MoE-无均衡", "MoE-辅助损失", "MoE-无辅助损失"):
        r = next(r for r in rows if r["name"] == name and r["seed"] == 0)
        print(f"  {name}")
        for i in (0, 2, 6, len(r["loads"]) - 1):
            frac = " ".join(f"{v:.2f}" for v in r["loads"][i][SHOW_LAYER])
            print(f"    step {i * LOG_EVERY:4d}: [{frac}]  最大/平均 "
                  f"{imbalance(r['loads'][i][SHOW_LAYER]):.2f}")
