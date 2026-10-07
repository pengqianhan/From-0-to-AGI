"""Chapter 23 · Minimal code 4: one small language model, four "hybrid layouts"

One letter gives the token mixer of each layer:
  A = softmax attention (causal, RoPE; stores a KV cache at inference)
  L = naive linear attention (S ← S + v kᵀ, only adds)
  G = Gated DeltaNet (S ← α S (I − β k kᵀ) + β v kᵀ)
We compare four 4-layer layouts: AAAA (pure attention), LLLL, GGGG (pure linear),
and GGGA (3:1 hybrid, the layer order of Qwen3.5).

  - Training: assets/tiny_corpus/shakespeare.txt, character level, 800 steps for each layout
    (a few minutes each on one CPU thread; 10 to 30 min each when other jobs share the CPU of the build machine).
  - Report: the validation loss, and how the inference cache (KV cache + linear-layer state)
    grows with the context length.
The weights are cached in code/out/*.pt (.gitignore ignores them).
05_associative_recall.py uses the model definitions of this file.

Run: uv run python chapters/23-linear-attention-hybrid/code/04_hybrid_lm.py
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


# ── Chunkwise forms of the two linear operators (batched, shape (B, H, T, d)) ─────────
def linear_chunked(q, k, v, C: int = 32):
    """Chunkwise form of naive linear attention: (QKᵀ ⊙ M) V in a chunk;
    pass the state S ← S + Vᵀ K between chunks."""
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
    """x: (B, H, T, d). Standard RoPE (Chapter 9)."""
    T, d = x.shape[-2], x.shape[-1]
    f = 1.0 / base ** (torch.arange(0, d, 2).float() / d)
    ang = torch.outer(torch.arange(T).float(), f)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class SoftmaxAttention(nn.Module):
    """If conv > 0, q/k/v also go through the same short causal convolution as in LinearMixer.
    The recall experiment in 05 uses this. Then all layouts have the same "local mixing" ability,
    and the differences come only from the global token mixer."""

    def __init__(self, dim: int, n_heads: int, conv: int = 0) -> None:
        super().__init__()
        self.h, self.dh, self.conv_k = n_heads, dim // n_heads, conv
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        if conv > 0:
            self.conv = nn.Conv1d(3 * dim, 3 * dim, conv, groups=3 * dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        h = self.wqkv(x)
        if self.conv_k > 0:
            h = F.silu(self.conv(F.pad(h.transpose(1, 2), (self.conv_k - 1, 0)))).transpose(1, 2)
        q, k, v = h.view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)
        return self.wo(o.transpose(1, 2).reshape(B, T, D))


class LinearMixer(nn.Module):
    """kind='linear': naive linear attention; kind='gdn': Gated DeltaNet.
    Both use: q/k/v projection → short causal convolution (kernel size 4) + SiLU
    → L2 normalization of q and k → core operator → RMSNorm per head → output projection."""

    def __init__(self, dim: int, n_heads: int, kind: str = "gdn", conv: int = 4) -> None:
        super().__init__()
        self.h, self.dh, self.kind, self.conv_k = n_heads, dim // n_heads, kind, conv
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.conv = nn.Conv1d(3 * dim, 3 * dim, conv, groups=3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.out_norm = nn.RMSNorm(self.dh)
        if kind == "gdn":
            self.wb = nn.Linear(dim, n_heads)  # β_t = sigmoid(·): write strength
            self.wa = nn.Linear(dim, n_heads)  # α_t = exp(−e^{A} softplus(· + dt_bias)): decay gate
            self.A_log = nn.Parameter(torch.zeros(n_heads))
            # Mamba-2 style initialization: softplus(dt_bias) of the heads is geometric from 0.001 to 0.1.
            # Then the initial α ≈ 0.9 to 0.999. Without it, α starts at about 0.5,
            # and the layer forgets everything after 10 to 20 steps.
            dt = torch.exp(torch.linspace(math.log(1e-3), math.log(1e-1), n_heads))
            self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x):
        B, T, D = x.shape
        h = self.wqkv(x).transpose(1, 2)
        h = F.silu(self.conv(F.pad(h, (self.conv_k - 1, 0)))).transpose(1, 2)  # causal: sees only the past
        q, k, v = h.view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q = F.normalize(q, dim=-1) / math.sqrt(self.dh)
        k = F.normalize(k, dim=-1)
        if self.kind == "linear":
            o = linear_chunked(q, k, v)
        else:
            beta = torch.sigmoid(self.wb(x)).transpose(1, 2)  # (B, H, T)
            g = -self.A_log.exp() * F.softplus(self.wa(x) + self.dt_bias)  # log decay ≤ 0
            g = g.transpose(1, 2)
            o, _ = delta.gated_delta_chunked(q, k, v, g, beta, C=32)
        o = self.out_norm(o)
        return self.wo(o.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, kind: str, ffn: int, attn_conv: int = 0) -> None:
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(dim), nn.RMSNorm(dim)
        self.mix = (
            SoftmaxAttention(dim, n_heads, attn_conv) if kind == "A"
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
    """pattern, for example "GGGA": one letter for each layer.
    attn_conv: if the attention layers also have the short convolution (default: no)."""

    def __init__(self, vocab: int, pattern: str, dim: int = 128, n_heads: int = 4,
                 ffn: int = 384, attn_conv: int = 0) -> None:
        super().__init__()
        self.pattern = pattern
        self.emb = nn.Embedding(vocab, dim)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(dim, n_heads, c, ffn, attn_conv) for c in pattern)
        self.norm = nn.RMSNorm(dim)

    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x)
        return self.norm(x) @ self.emb.weight.T


# ── Data and training ─────────────────────────────────────────────────────────────
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
    g = torch.Generator().manual_seed(seed)  # same seed → the four layouts see exactly the same data
    t0 = time.time()
    for step in range(steps + 1):
        for pg in opt.param_groups:  # warmup + cosine (Chapter 6)
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
    """Inference cache of one sequence: an A layer stores K and V (2·T·dim numbers, BF16);
    an L/G layer stores the state (H·dh·dh, FP32) + the convolution tail."""
    dh = dim // n_heads
    total = 0
    for c in pattern:
        if c == "A":
            total += 2 * T * dim * 2
        else:
            total += n_heads * dh * dh * 4 + 3 * dim * (conv - 1) * 2
    return total


PATTERNS = ["AAAA", "LLLL", "GGGG", "GGGA"]

# Real configuration of Qwen3.5-0.8B (config.json of huggingface.co/Qwen/Qwen3.5-0.8B, text_config, read 2026-09)
QWEN35_08B = {
    "layer_types": ["linear_attention"] * 3 + ["full_attention"],  # × 6, full_attention_interval = 4
    "repeat": 6,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "linear_num_value_heads": 16,
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
}


def qwen35_cache_mib(T: int, all_full: bool = False) -> tuple[float, float]:
    """Cache of one sequence of Qwen3.5-0.8B (MiB): KV in BF16, recurrent state in FP32
    (mamba_ssm_dtype in the config).
    all_full=True: a baseline that changes all 24 layers to full-attention layers with the same configuration.
    Returns (KV, linear-layer state)."""
    c = QWEN35_08B
    types = c["layer_types"] * c["repeat"]
    if all_full:
        types = ["full_attention"] * len(types)
    n_full = types.count("full_attention")
    n_lin = len(types) - n_full
    kv = n_full * 2 * c["num_key_value_heads"] * c["head_dim"] * T * 2
    conv_dim = 2 * c["linear_num_key_heads"] * c["linear_key_head_dim"] + (
        c["linear_num_value_heads"] * c["linear_value_head_dim"]
    )
    state = n_lin * (
        c["linear_num_value_heads"] * c["linear_key_head_dim"] * c["linear_value_head_dim"] * 4
        + conv_dim * (c["linear_conv_kernel_dim"] - 1) * 2
    )
    return kv / 2**20, state / 2**20


def main() -> None:
    rows = []
    for p in PATTERNS:
        model, secs = train_lm(p)
        n_params = sum(x.numel() for x in model.parameters())
        rows.append((p, n_params, val_loss(model), secs))
    print("\n== Validation loss (character level, nats/char; 800 steps, same data order, seed 0) ==")
    print(f"{'Arch':>6} | {'Params':>9} | {'val loss':>8} | train time")
    for p, n, vl, secs in rows:
        print(f"{p:>6} | {n:>9,} | {vl:>8.3f} | {'(cached)' if secs is None else f'{secs:.0f}s'}")

    print("\n== Inference cache of one sequence (KB): the KV cache grows with the length; the linear-layer state does not ==")
    Ts = (128, 1024, 8192, 65536)
    print(f"{'Arch':>6} | " + " | ".join(f"T={T:>6}" for T in Ts))
    for p in PATTERNS:
        print(f"{p:>6} | " + " | ".join(f"{cache_bytes(p, T) / 1024:>8.0f}" for T in Ts))

    print("\n== Real model: Qwen3.5-0.8B (24 layers = 6 × [3 Gated DeltaNet + 1 full attention]), cache of one sequence ==")
    print(f"{'Context':>8} | {'KV(6 attn)':>9} | {'State(18 GDN)':>14} | {'Total':>8} | {'If 24 full attn':>16}")
    for T in (4096, 32768, 262144):
        kv, st = qwen35_cache_mib(T)
        full = sum(qwen35_cache_mib(T, all_full=True))
        print(f"{T:>8,} | {kv:>6.0f} MiB | {st:>10.1f} MiB | {kv + st:>4.0f} MiB | {full:>12,.0f} MiB")


if __name__ == "__main__":
    main()
