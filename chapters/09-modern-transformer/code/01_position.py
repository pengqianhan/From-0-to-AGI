"""第 9 章 · 极简代码 1：注意力分不清顺序 → RoPE 用"旋转"把位置写进 q、k

1. 没有位置信息时，把前文打乱顺序，最后一个位置的注意力输出一模一样；
2. RoPE：同一对 (q, k) 放在不同位置，点积只取决于"相隔多远"（m − n），不取决于绝对位置；
3. 旋转不改变向量长度；不同维度对转得快慢不同（频率 ω_i = θ^(-2i/d)）。
运行：uv run python chapters/09-modern-transformer/code/01_position.py
"""

import importlib.util
import math
from pathlib import Path

import torch

torch.set_num_threads(1)  # 构建环境多任务共享 CPU；本机可以删掉这行

_spec = importlib.util.spec_from_file_location(
    "tiny", Path(__file__).with_name("02_tiny_transformer.py"))
tiny = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tiny)

torch.manual_seed(0)
D = 8  # 一个注意力头的维度 head_dim


def causal_attention(x, wq, wk, wv, use_rope=False):
    """单头因果注意力（第 8 章），可选在 q、k 上加 RoPE。x: (T, D)。"""
    T = x.shape[0]
    q, k, v = x @ wq, x @ wk, x @ wv
    if use_rope:
        cos, sin = tiny.rope_cos_sin(D, T)
        q, k = tiny.apply_rope(q, cos, sin), tiny.apply_rope(k, cos, sin)
    scores = (q @ k.T / math.sqrt(D)).masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), -1e9)
    return scores.softmax(-1) @ v


# ── 1. 打乱前文，最后一个位置的输出变不变？ ─────────────────────────────────
words = ["狗", "咬", "人", "了"]
emb = {w: torch.randn(D) for w in words}
wq, wk, wv = (torch.randn(D, D) / math.sqrt(D) for _ in range(3))
a = torch.stack([emb[w] for w in ["狗", "咬", "人", "了"]])
b = torch.stack([emb[w] for w in ["人", "咬", "狗", "了"]])
print("1) 最后一个词“了”的注意力输出，前文顺序不同：")
for rope in (False, True):
    oa, ob = causal_attention(a, wq, wk, wv, rope)[-1], causal_attention(b, wq, wk, wv, rope)[-1]
    tag = "加 RoPE  " if rope else "无位置信息"
    print(f"   {tag}：“狗咬人了” vs “人咬狗了” 最大差 = {(oa - ob).abs().max().item():.2e}")

# ── 2. RoPE：点积只取决于相对位置 ───────────────────────────────────────────
q0, k0 = torch.randn(D), torch.randn(D)
cos, sin = tiny.rope_cos_sin(D, 64)


def rope_at(x, pos):
    return tiny.apply_rope(x, cos[pos], sin[pos])


print("\n2) 同一对 q、k 放在不同位置 (m, n)，旋转后的点积 q_m·k_n：")
print(f"   不旋转时 q·k = {q0 @ k0:.4f}")
for m, n in [(3, 1), (10, 8), (50, 48), (5, 1), (40, 36), (1, 3)]:
    dot = rope_at(q0, m) @ rope_at(k0, n)
    print(f"   m={m:2d}, n={n:2d}, m−n={m - n:+d}  →  q_m·k_n = {dot:.4f}")

# ── 3. 长度不变；各维度对的转速 ─────────────────────────────────────────────
print(f"\n3) 旋转不改变长度：|q| = {q0.norm():.4f}，转到位置 37 后 |q| = {rope_at(q0, 37).norm():.4f}")
inv_freq = 10000.0 ** (-torch.arange(0, D, 2).float() / D)
for i, w in enumerate(inv_freq):
    print(f"   第 {i} 对维度：每前进一个位置转 {w:.4f} 弧度，转一圈要 {2 * math.pi / w:8.1f} 个位置")
