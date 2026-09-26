"""第 23 章 · 极简代码 5：联想回忆（associative recall）——纯线性掉队，混合找回来

任务（Zoology 的 MQAR 的缩小版）：序列前半段是 N 个随机的"键 值"对，后半段反复给出其中某个键，
模型要在下一个位置答出它对应的值：

    k₇ v₃₁  k₂ v₉  k₄₀ v₁₂ … │ k₂ v₉  k₇ v₃₁  k₂ v₉ …
    └──────── N 个键值对 ────────┘ └─ 查询：只在"键"后面的位置计分 ─┘

答对需要把 N 个键值对**原样**记住——这正是固定大小状态的软肋（第 5 节的容量表）。
比较 4 种 2 层小模型（字母含义同 04_hybrid_lm.py）：AA（纯注意力）、LL（纯朴素线性）、
GG（纯 Gated DeltaNet）、GA（1 层 Gated DeltaNet + 1 层全注意力）。
线性层的 head_dim 故意取得很小（16），让状态容量明显不够用，差别才看得清。

训练：每步随机取 N ∈ [4, 24]，同样的数据顺序；权重缓存在 code/out/*.pt。
运行：uv run python chapters/23-linear-attention-hybrid/code/05_associative_recall.py
"""

from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
_spec = importlib.util.spec_from_file_location("hybrid_lm", HERE / "04_hybrid_lm.py")
lm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lm)

N_KEYS, N_VALS = 64, 64  # 键：0..63，值：64..127
PAD = N_KEYS + N_VALS  # 128
VOCAB = PAD + 1
L = 64  # 模型输入长度
TRAIN_N = (4, 24)
EVAL_N = [4, 8, 12, 16, 20, 24]
PATTERNS = ["AA", "LL", "GG", "GA"]
DIM, HEADS = 64, 4  # head_dim = 16


def make_batch(bsz: int, N: int, g: torch.Generator):
    """返回 (x, y, mask)：x、y 形状 (bsz, L)，mask 标出要计分的位置（输入是查询键、目标是它的值）。"""
    keys = torch.argsort(torch.rand(bsz, N_KEYS, generator=g), dim=1)[:, :N]  # 每条序列 N 个不同的键
    vals = torch.randint(N_VALS, (bsz, N), generator=g) + N_KEYS
    Q = (L - 2 * N) // 2
    qi = torch.randint(N, (bsz, Q), generator=g)
    qk, qv = keys.gather(1, qi), vals.gather(1, qi)
    pairs = torch.stack([keys, vals], -1).flatten(1)  # k v k v …
    queries = torch.stack([qk, qv], -1).flatten(1)
    seq = torch.full((bsz, L + 1), PAD)
    seq[:, : 2 * N] = pairs
    seq[:, 2 * N : 2 * N + 2 * Q] = queries
    mask = torch.zeros(bsz, L, dtype=torch.bool)
    mask[:, 2 * N : 2 * N + 2 * Q : 2] = True
    return seq[:, :-1], seq[:, 1:], mask


def train(pattern: str, steps: int = 1500, bsz: int = 64, lr: float = 3e-3, seed: int = 0,
          verbose: bool = True):
    path = OUT / f"recall_{pattern}_s{steps}_seed{seed}.pt"
    torch.manual_seed(seed)
    model = lm.TinyLM(VOCAB, pattern, dim=DIM, n_heads=HEADS, ffn=2 * DIM)
    if path.exists():
        model.load_state_dict(torch.load(path, weights_only=True))
        return model.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for step in range(steps + 1):
        for pg in opt.param_groups:
            pg["lr"] = lr * min(1, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / steps))
        N = int(torch.randint(TRAIN_N[0], TRAIN_N[1] + 1, (1,), generator=g))
        x, y, m = make_batch(bsz, N, g)
        loss = F.cross_entropy(model(x)[m], y[m])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if verbose and step % 250 == 0:
            print(f"  [{pattern}] step {step:4d}  loss {loss.item():.3f}  ({time.time() - t0:.0f}s)")
    OUT.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path)
    return model.eval()


@torch.no_grad()
def accuracy(model, N: int, n: int = 256) -> float:
    x, y, m = make_batch(n, N, torch.Generator().manual_seed(1000 + N))
    return (model(x).argmax(-1)[m] == y[m]).float().mean().item()


def state_numbers(pattern: str) -> str:
    """每种结构推理时要存的东西（L=64 的输入）：A 层存 2·L·DIM 个 K/V 数，线性层存 H·dh·dh。"""
    dh = DIM // HEADS
    parts = []
    for c in pattern:
        parts.append(f"KV {2 * L * DIM}" if c == "A" else f"状态 {HEADS * dh * dh}")
    return " + ".join(parts)


def results(verbose: bool = False) -> dict:
    acc = {}
    for p in PATTERNS:
        model = train(p, verbose=verbose)
        acc[p] = [accuracy(model, N) for N in EVAL_N]
    return {"patterns": PATTERNS, "N": EVAL_N, "acc": acc}


def main() -> None:
    r = results(verbose=True)
    print(f"\n== 联想回忆准确率（2 层、宽 64、线性层 head_dim 16；每个 N 测 256 条序列；随机猜 = 1/{N_VALS}）==")
    print(f"{'结构':>4} | " + " | ".join(f"N={n:>2}" for n in EVAL_N) + " | 推理时每层要存的数")
    for p in PATTERNS:
        print(f"{p:>4} | " + " | ".join(f"{a:>4.0%}" for a in r["acc"][p]) + f" | {state_numbers(p)}")


if __name__ == "__main__":
    main()
