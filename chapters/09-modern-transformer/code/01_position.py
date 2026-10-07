"""Chapter 9 · Minimal code 1: attention cannot see word order → RoPE writes the position into q, k with a "rotation"

1. Without position information, change the order of the words before the last position.
   The attention output at the last position does not change;
2. RoPE: put the same pair (q, k) at different positions. The dot product depends only on
   the distance between them (m − n), not on the absolute positions;
3. The rotation does not change the length of a vector. Different dimension pairs rotate
   at different speeds (frequency ω_i = θ^(-2i/d)).
Run: uv run python chapters/09-modern-transformer/code/01_position.py
"""

import importlib.util
import math
from pathlib import Path

import torch

torch.set_num_threads(1)  # many jobs share the CPU of the build machine; on your computer you can remove this line

_spec = importlib.util.spec_from_file_location(
    "tiny", Path(__file__).with_name("02_tiny_transformer.py"))
tiny = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tiny)

torch.manual_seed(0)
D = 8  # the dimension of one attention head (head_dim)


def causal_attention(x, wq, wk, wv, use_rope=False):
    """Single-head causal attention (Chapter 8), with optional RoPE on q and k. x: (T, D)."""
    T = x.shape[0]
    q, k, v = x @ wq, x @ wk, x @ wv
    if use_rope:
        cos, sin = tiny.rope_cos_sin(D, T)
        q, k = tiny.apply_rope(q, cos, sin), tiny.apply_rope(k, cos, sin)
    scores = (q @ k.T / math.sqrt(D)).masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), -1e9)
    return scores.softmax(-1) @ v


# ── 1. Change the order of the words before the last position. Does the output change? ──
# Example sentences: "狗咬人了" = "the dog bit the man", "人咬狗了" = "the man bit the dog".
# "了" is the last word in both sentences. It marks a completed action.
words = ["狗", "咬", "人", "了"]
emb = {w: torch.randn(D) for w in words}
wq, wk, wv = (torch.randn(D, D) / math.sqrt(D) for _ in range(3))
a = torch.stack([emb[w] for w in ["狗", "咬", "人", "了"]])
b = torch.stack([emb[w] for w in ["人", "咬", "狗", "了"]])
print("1) Attention output of the last word “了”, when the words before it have a different order:")
for rope in (False, True):
    oa, ob = causal_attention(a, wq, wk, wv, rope)[-1], causal_attention(b, wq, wk, wv, rope)[-1]
    tag = "with RoPE  " if rope else "no position"
    print(f"   {tag}: “狗咬人了” vs “人咬狗了” max difference = {(oa - ob).abs().max().item():.2e}")

# ── 2. RoPE: the dot product depends only on the relative position ──────────────
q0, k0 = torch.randn(D), torch.randn(D)
cos, sin = tiny.rope_cos_sin(D, 64)


def rope_at(x, pos):
    return tiny.apply_rope(x, cos[pos], sin[pos])


print("\n2) The same pair q, k at different positions (m, n). Dot product after the rotation, q_m·k_n:")
print(f"   without rotation, q·k = {q0 @ k0:.4f}")
for m, n in [(3, 1), (10, 8), (50, 48), (5, 1), (40, 36), (1, 3)]:
    dot = rope_at(q0, m) @ rope_at(k0, n)
    print(f"   m={m:2d}, n={n:2d}, m−n={m - n:+d}  →  q_m·k_n = {dot:.4f}")

# ── 3. The length does not change; the rotation speed of each dimension pair ────
print(f"\n3) The rotation does not change the length: |q| = {q0.norm():.4f}; at position 37, |q| = {rope_at(q0, 37).norm():.4f}")
inv_freq = 10000.0 ** (-torch.arange(0, D, 2).float() / D)
for i, w in enumerate(inv_freq):
    print(f"   dimension pair {i}: {w:.4f} rad per position, {2 * math.pi / w:8.1f} positions for one full turn")
