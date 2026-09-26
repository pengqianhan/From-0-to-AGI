"""第 12 章 · 极简代码 3：阶梯实验的第一步 —— 先给每个尺寸找到合适的学习率

"小规模实验可以预测大规模结果"有个前提：每个小模型都调好了超参。没调好的小模型会把
scaling law 扭歪（Lourie et al. 2026, arXiv:2608.11859）。所以阶梯实验先做学习率扫描：

  1. 4 个尺寸（非 embedding 参数约 1 万 → 20 万）× 5 个学习率，每个跑同样的 token 数；
  2. 每个尺寸取验证 loss 最低的学习率 η*；
  3. 拟合 η*(N) = c · N^(-k)（对数坐标下的一条直线），外推给更大的模型用（第 4 个脚本）。

模型就是第 9 章的 TinyTransformer（字节级，词表 256），数据是 assets/tiny_corpus/shakespeare.txt。
学习率调度用 WSD（第 6 章）：warmup → 恒定 → 最后 20% 线性降到 0。

运行：uv run python chapters/12-scaling-laws/code/03_lr_sweep.py
      （单线程，共享 CPU 上约 4–6 分钟；结果缓存到 out/ch12/lr_sweep.json，再跑直接读缓存，加 --fresh 重跑）
这是"极小配置演示"：几万参数、十几万 token，结论只说明方法，不代表主线模型。
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "out" / "ch12"
_spec = importlib.util.spec_from_file_location(
    "tiny_tf", ROOT / "chapters/09-modern-transformer/code/02_tiny_transformer.py"
)
tt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tt)

SEQ_LEN = 64
BATCH = 8  # 小 batch：同样的 token 数走更多步，这个尺度上学得更快（实测）
TOKENS_PER_STEP = SEQ_LEN * BATCH  # 512 个字节 / 步
WARMUP = 20

# 阶梯：(名字, dim, 层数)；head_dim 固定 16，SwiGLU 中间维度 ≈ 8/3·dim 取 16 的倍数
LADDER = [("s1", 16, 2), ("s2", 32, 2), ("s3", 48, 3), ("s4", 64, 4)]
HELD_OUT = ("s5", 96, 4)  # 比阶梯最大的再大 2.3 倍，只用来检验外推
SWEEP_LRS = [2.5e-3, 5e-3, 1e-2, 2e-2, 4e-2]
SWEEP_TOKENS = 131_072  # 256 步


def make_config(dim: int, n_layers: int):
    ffn = 16 * round(8 / 3 * dim / 16)
    return tt.Config(dim=dim, n_layers=n_layers, n_heads=dim // 16, ffn_dim=ffn, seq_len=SEQ_LEN)


def make_model(dim: int, n_layers: int):
    """第 9 章的模型，只改一处：输入 embedding 和 lm_head 不共享。
    这个尺度上共享会让模型在"只会猜高频字节"的平台期卡很久（实测），曲线噪声大到拟合不出规律；
    zero 的 tiny 配置也因为同样的原因不共享（configs/tiny/pretrain.toml 的注释）。"""
    model = tt.TinyTransformer(make_config(dim, n_layers))
    model.lm_head.weight = nn.Parameter(torch.randn(model.cfg.vocab_size, dim) * 0.02)
    return model


def non_embedding_params(dim: int, n_layers: int) -> int:
    """scaling law 里的 N：参与矩阵乘的参数（除了输入 embedding 查表以外的全部，含 lm_head）。
    这样 6N 就是每 token 的训练 FLOPs（不算注意力项），和 01_flops.py 的口径一致。"""
    model = make_model(dim, n_layers)
    return sum(p.numel() for p in model.parameters()) - model.tok_emb.weight.numel()


def load_bytes():
    raw = (ROOT / "assets/tiny_corpus/shakespeare.txt").read_bytes()
    data = torch.tensor(list(raw), dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:]


def batch(data, g):
    ix = torch.randint(len(data) - SEQ_LEN - 1, (BATCH,), generator=g)
    x = torch.stack([data[i : i + SEQ_LEN] for i in ix])
    y = torch.stack([data[i + 1 : i + SEQ_LEN + 1] for i in ix])
    return x, y


@torch.no_grad()
def val_bpb(model, val, n_batches=24) -> float:
    """验证集 bits-per-byte（固定的 24 个 batch，所有运行用同一份）。"""
    g = torch.Generator().manual_seed(123)
    model.eval()
    losses = [F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
              for x, y in (batch(val, g) for _ in range(n_batches))]
    model.train()
    return float(np.mean(losses) / math.log(2))


def train_step(model, opt, x, y, lr):
    for group in opt.param_groups:
        group["lr"] = lr
    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


def train_wsd_branches(dim, n_layers, lr, budgets, seed=0, data=None):
    """一次训练得到多个 token 预算的结果（WSD 分叉）。

    主干：warmup 后保持峰值学习率一直训到 0.8 × 最大预算；
    每个预算 D_k：在主干的 0.8·D_k 处复制一份模型 + 优化器，接一段 0.2·D_k 的线性衰减，
    衰减完测验证 loss。这样 3 个预算只花约 1.35 倍最大预算的算力，而不是 1.75 倍。
    """
    train, val = data or load_bytes()
    torch.manual_seed(seed)
    model = make_model(dim, n_layers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    g = torch.Generator().manual_seed(seed)
    steps = {D: D // TOKENS_PER_STEP for D in budgets}
    branch_at = {int(0.8 * s): D for D, s in steps.items()}
    trunk_end = max(branch_at)
    results = {}
    for step in range(trunk_end + 1):
        if step in branch_at:  # 分叉：复制当前状态，接一段衰减
            D = branch_at[step]
            m2 = copy.deepcopy(model)
            o2 = torch.optim.AdamW(m2.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
            o2.load_state_dict(copy.deepcopy(opt.state_dict()))
            g2 = torch.Generator().manual_seed(seed * 1000 + D % 997)
            n_decay = steps[D] - step
            for i in range(n_decay):
                train_step(m2, o2, *batch(train, g2), lr * (1 - (i + 1) / n_decay))
            results[D] = val_bpb(m2, val)
        if step == trunk_end:
            break
        warm = min(1.0, (step + 1) / WARMUP)
        train_step(model, opt, *batch(train, g), lr * warm)
    return results


def fit_power_law(x, y):
    """log y = log c − k·log x 的最小二乘，返回 (c, k)。"""
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    return float(np.exp(intercept)), float(-slope)


def run_sweep(fresh=False):
    path = OUT / "lr_sweep.json"
    if path.exists() and not fresh:
        return json.loads(path.read_text())
    data = load_bytes()
    rows, t0 = [], time.time()
    for name, dim, L in LADDER:
        n = non_embedding_params(dim, L)
        for lr in SWEEP_LRS:
            bpb = train_wsd_branches(dim, L, lr, [SWEEP_TOKENS], data=data)[SWEEP_TOKENS]
            rows.append({"size": name, "dim": dim, "layers": L, "N": n, "lr": lr, "val_bpb": bpb})
            print(f"  {name} N={n:>7,}  lr={lr:<7g} val {bpb:.4f} bit/字节  ({time.time() - t0:4.0f}s)")
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"tokens": SWEEP_TOKENS, "rows": rows}
    path.write_text(json.dumps(result, indent=1))
    return result


def best_lrs(sweep):
    best = {}
    for r in sweep["rows"]:
        if r["size"] not in best or r["val_bpb"] < best[r["size"]]["val_bpb"]:
            best[r["size"]] = r
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="忽略缓存，重新训练")
    args = ap.parse_args()
    torch.set_num_threads(1)  # 共享 CPU 上单线程最快；本机空闲时可删掉
    print(f"学习率扫描：每个运行 {SWEEP_TOKENS:,} 个字节（token），batch {BATCH}×{SEQ_LEN}")
    sweep = run_sweep(args.fresh)
    table = {}
    for r in sweep["rows"]:
        table.setdefault((r["size"], r["N"]), {})[r["lr"]] = r["val_bpb"]
    print("\n验证 loss（bit/字节），每行一个尺寸，* 为该尺寸最优：")
    print(f"{'尺寸':>4} {'N':>8} | " + " ".join(f"{lr:>8g}" for lr in SWEEP_LRS))
    best = best_lrs(sweep)
    for (name, n), row in table.items():
        cells = [f"{row[lr]:>7.4f}{'*' if lr == best[name]['lr'] else ' '}" for lr in SWEEP_LRS]
        print(f"{name:>4} {n:>8,} | " + " ".join(cells))
    ns = [best[s]["N"] for s, _, _ in LADDER]
    lrs = [best[s]["lr"] for s, _, _ in LADDER]
    c, k = fit_power_law(ns, lrs)
    n5 = non_embedding_params(*HELD_OUT[1:])
    print(f"\n拟合 η*(N) = {c:.3g} · N^(-{k:.3f})")
    print(f"外推到留出尺寸 {HELD_OUT[0]}（N={n5:,}）：η* ≈ {c * n5 ** -k:.4g}")
    fixed = SWEEP_LRS[-1]
    print(f"\n如果所有尺寸都用同一个学习率 {fixed:g}（不调参）：")
    for (name, n), row in table.items():
        print(f"  {name} N={n:>7,}  {row[fixed]:.4f}  比调好的差 {row[fixed] - best[name]['val_bpb']:+.4f}")


if __name__ == "__main__":
    main()
