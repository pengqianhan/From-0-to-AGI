"""第 23 章 · 极简代码 4：同一个小语言模型，四种"混合方式"

每层的 token mixer 用一个字母表示：
  A = softmax 注意力（因果、RoPE，推理时要存 KV cache）
  L = 朴素线性注意力（S ← S + v kᵀ，只累加）
  G = Gated DeltaNet（S ← α S (I − β k kᵀ) + β v kᵀ）
比较四种 4 层结构：AAAA（纯注意力）、LLLL、GGGG（纯线性）、GGGA（3:1 混合，Qwen3.5 的排法）。

  - 训练：assets/tiny_corpus/shakespeare.txt，字符级，每种 800 步（CPU 单线程每种约 1–3 分钟）；
  - 报告：验证集 loss，以及推理缓存（KV cache + 线性层状态）随上下文长度的增长。
权重缓存在 code/out/*.pt（已被 .gitignore 忽略），05_associative_recall.py 复用这里的模型定义。

运行：uv run python chapters/23-linear-attention-hybrid/code/04_hybrid_lm.py
"""

from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "out"

_spec = importlib.util.spec_from_file_location("delta_rule", HERE / "03_delta_rule.py")
delta = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(delta)


# ── 两种线性算子的分块形式（批量，形状 (B, H, T, d)）──────────────────────────────
def linear_chunked(q, k, v, C: int = 32):
    """朴素线性注意力的分块形式：块内 (QKᵀ ⊙ M) V，块间传状态 S ← S + Vᵀ K。"""
    T = q.shape[-2]
    S = q.new_zeros(*q.shape[:-2], v.shape[-1], q.shape[-1])
    mask = torch.ones(C, C).tril()
    out = []
    for s in range(0, T, C):
        qc, kc, vc = q[..., s : s + C, :], k[..., s : s + C, :], v[..., s : s + C, :]
        out.append(qc @ S.mT + ((qc @ kc.mT) * mask) @ vc)
        S = S + vc.mT @ kc
    return torch.cat(out, -2)


def rope(x, base: float = 10000.0):
    """x: (B, H, T, d)。标准 RoPE（第 9 章）。"""
    T, d = x.shape[-2], x.shape[-1]
    f = 1.0 / base ** (torch.arange(0, d, 2).float() / d)
    ang = torch.outer(torch.arange(T).float(), f)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class SoftmaxAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.h, self.dh = n_heads, dim // n_heads
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.wqkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)
        return self.wo(o.transpose(1, 2).reshape(B, T, D))


class LinearMixer(nn.Module):
    """kind='linear'：朴素线性注意力；kind='gdn'：Gated DeltaNet。
    两者共用：q/k/v 投影 → 因果短卷积(核长 4) + SiLU → q、k 做 L2 归一化 → 核心算子 → 每头 RMSNorm → 输出投影。"""

    def __init__(self, dim: int, n_heads: int, kind: str = "gdn", conv: int = 4) -> None:
        super().__init__()
        self.h, self.dh, self.kind, self.conv_k = n_heads, dim // n_heads, kind, conv
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.conv = nn.Conv1d(3 * dim, 3 * dim, conv, groups=3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.out_norm = nn.RMSNorm(self.dh)
        if kind == "gdn":
            self.wb = nn.Linear(dim, n_heads)  # β_t = sigmoid(·)：写入强度
            self.wa = nn.Linear(dim, n_heads)  # α_t = exp(−e^{A} softplus(·))：衰减门
            self.A_log = nn.Parameter(torch.zeros(n_heads))

    def forward(self, x):
        B, T, D = x.shape
        h = self.wqkv(x).transpose(1, 2)
        h = F.silu(self.conv(F.pad(h, (self.conv_k - 1, 0)))).transpose(1, 2)  # 因果：只看过去
        q, k, v = h.view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q = F.normalize(q, dim=-1) / math.sqrt(self.dh)
        k = F.normalize(k, dim=-1)
        if self.kind == "linear":
            o = linear_chunked(q, k, v)
        else:
            beta = torch.sigmoid(self.wb(x)).transpose(1, 2)  # (B, H, T)
            g = (-self.A_log.exp() * F.softplus(self.wa(x))).transpose(1, 2)  # 对数衰减 ≤ 0
            o, _ = delta.gated_delta_chunked(q, k, v, g, beta, C=32)
        o = self.out_norm(o)
        return self.wo(o.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, kind: str, ffn: int) -> None:
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(dim), nn.RMSNorm(dim)
        self.mix = (
            SoftmaxAttention(dim, n_heads) if kind == "A"
            else LinearMixer(dim, n_heads, "linear" if kind == "L" else "gdn")
        )
        self.w_gate = nn.Linear(dim, ffn, bias=False)
        self.w_up = nn.Linear(dim, ffn, bias=False)
        self.w_down = nn.Linear(ffn, dim, bias=False)

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        h = self.n2(x)
        return x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))


class TinyLM(nn.Module):
    """pattern 例如 "GGGA"：每个字母一层。"""

    def __init__(self, vocab: int, pattern: str, dim: int = 128, n_heads: int = 4,
                 ffn: int = 384) -> None:
        super().__init__()
        self.pattern = pattern
        self.emb = nn.Embedding(vocab, dim)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(dim, n_heads, c, ffn) for c in pattern)
        self.norm = nn.RMSNorm(dim)

    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x)
        return self.norm(x) @ self.emb.weight.T


# ── 数据与训练 ──────────────────────────────────────────────────────────────────
class CharData:
    def __init__(self) -> None:
        text = (ROOT / "assets" / "tiny_corpus" / "shakespeare.txt").read_text("utf-8")
        self.chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(self.chars)}
        ids = torch.tensor([stoi[c] for c in text])
        n = int(len(ids) * 0.9)
        self.train, self.val = ids[:n], ids[n:]

    def batch(self, split, bsz, seq, g):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - seq - 1, (bsz,), generator=g)
        x = torch.stack([data[i : i + seq] for i in ix])
        y = torch.stack([data[i + 1 : i + seq + 1] for i in ix])
        return x, y


def train_lm(pattern: str, steps: int = 800, bsz: int = 16, seq: int = 128, lr: float = 3e-3,
             seed: int = 0):
    data = CharData()
    path = OUT / f"lm_{pattern}_s{steps}_seed{seed}.pt"
    torch.manual_seed(seed)
    model = TinyLM(len(data.chars), pattern)
    if path.exists():
        model.load_state_dict(torch.load(path, weights_only=True))
        return model.eval(), None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    g = torch.Generator().manual_seed(seed)  # 同一个种子 → 四种结构看到完全相同的数据
    t0 = time.time()
    for step in range(steps + 1):
        for pg in opt.param_groups:  # warmup + cosine（第 6 章）
            pg["lr"] = lr * min(1, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / steps))
        x, y = data.batch("train", bsz, seq, g)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            print(f"  [{pattern}] step {step:4d}  train loss {loss.item():.3f}  ({time.time() - t0:.0f}s)")
    OUT.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path)
    return model.eval(), time.time() - t0


@torch.no_grad()
def val_loss(model, seq: int = 128, n_batches: int = 20) -> float:
    data = CharData()
    g = torch.Generator().manual_seed(1234)
    tot = 0.0
    for _ in range(n_batches):
        x, y = data.batch("val", 16, seq, g)
        tot += F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    return tot / n_batches


def cache_bytes(pattern: str, T: int, dim: int = 128, n_heads: int = 4, conv: int = 4) -> int:
    """推理时一条序列的缓存：A 层存 K、V（2·T·dim 个数，BF16）；L/G 层存状态（H·dh·dh，FP32）+ 卷积尾巴。"""
    dh = dim // n_heads
    total = 0
    for c in pattern:
        if c == "A":
            total += 2 * T * dim * 2
        else:
            total += n_heads * dh * dh * 4 + 3 * dim * (conv - 1) * 2
    return total


PATTERNS = ["AAAA", "LLLL", "GGGG", "GGGA"]


def main() -> None:
    rows = []
    for p in PATTERNS:
        model, secs = train_lm(p)
        n_params = sum(x.numel() for x in model.parameters())
        rows.append((p, n_params, val_loss(model), secs))
    print("\n== 验证集 loss（字符级，nats/字符；800 步，同一数据顺序，种子 0）==")
    print(f"{'结构':>6} | {'参数量':>9} | {'val loss':>8} | 训练用时")
    for p, n, vl, secs in rows:
        print(f"{p:>6} | {n:>9,} | {vl:>8.3f} | {'(已缓存)' if secs is None else f'{secs:.0f}s'}")

    print("\n== 一条序列的推理缓存（KB）：KV cache 随长度增长，线性层的状态不增长 ==")
    Ts = (128, 1024, 8192, 65536)
    print(f"{'结构':>6} | " + " | ".join(f"T={T:>6}" for T in Ts))
    for p in PATTERNS:
        print(f"{p:>6} | " + " | ".join(f"{cache_bytes(p, T) / 1024:>8.0f}" for T in Ts))


if __name__ == "__main__":
    main()
