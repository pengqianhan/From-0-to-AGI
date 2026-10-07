"""Chapter 8 · Minimal code 2: (multi-head) causal self-attention from scratch,
with a parity check against PyTorch

    Q = X Wq,  K = X Wk,  V = X Wv                 (three linear projections, y = XW from Chapter 2)
    Attention(Q, K, V) = softmax(Q Kᵀ / √d + mask) V  (scaled dot product + causal mask)
    Multi-head: split C into H parts of d = C / H dimensions, run attention on each part,
    concatenate the results, and multiply by Wo

Uses only PyTorch. It finishes in less than one second on the CPU.
Run: uv run python chapters/08-attention/code/02_attention_from_scratch.py
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

# One thread is fastest for small tensors. Many jobs share the CPU of the build machine.
# You can remove this line on your computer.
torch.set_num_threads(1)

VERBOSE = False  # MultiHeadAttention.forward prints the shape after each step


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True):
    """Scaled dot-product attention. q, k, v: (..., T, d) → output (..., T, d), weights (..., T, T)."""
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)  # QKᵀ / √d
    if causal:
        T = q.shape[-2]
        future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))  # future positions → −∞
    weights = torch.softmax(scores, dim=-1)  # each row sums to 1
    return weights @ v, weights  # weighted average of V


def show(name: str, t: torch.Tensor, dims: str) -> None:
    if VERBOSE:
        print(f"  {name:<28} {str(tuple(t.shape)):<16} {dims}")


class MultiHeadAttention(nn.Module):
    def __init__(self, C: int, n_heads: int) -> None:
        super().__init__()
        assert C % n_heads == 0
        self.H, self.d = n_heads, C // n_heads
        self.wq = nn.Linear(C, C, bias=False)
        self.wk = nn.Linear(C, C, bias=False)
        self.wv = nn.Linear(C, C, bias=False)
        self.wo = nn.Linear(C, C, bias=False)  # output projection: mixes the concatenated heads once more

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        B, T, C = x.shape
        show("x", x, "(B, T, C)")
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        show("q = x @ Wq", q, "(B, T, C)")
        # Split into heads: C = H × d. Then move the head dimension to the front,
        # so that it acts as part of the batch
        q = q.view(B, T, self.H, self.d).transpose(1, 2)
        k = k.view(B, T, self.H, self.d).transpose(1, 2)
        v = v.view(B, T, self.H, self.d).transpose(1, 2)
        show("q after split into heads", q, "(B, H, T, d)")
        out, w = attention(q, k, v, causal=True)
        show("weights softmax(QKᵀ/√d)", w, "(B, H, T, T)")
        show("output of each head w @ v", out, "(B, H, T, d)")
        out = out.transpose(1, 2).reshape(B, T, C)  # concatenate the heads again
        show("concatenate heads", out, "(B, T, C)")
        out = self.wo(out)
        show("output = concat @ Wo", out, "(B, T, C)")
        return (out, w) if return_weights else out


def mha_with_sdpa(m: MultiHeadAttention, x: torch.Tensor) -> torch.Tensor:
    """The same weights, but the attention step uses PyTorch's scaled_dot_product_attention."""
    B, T, C = x.shape
    q = m.wq(x).view(B, T, m.H, m.d).transpose(1, 2)
    k = m.wk(x).view(B, T, m.H, m.d).transpose(1, 2)
    v = m.wv(x).view(B, T, m.H, m.d).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return m.wo(out.transpose(1, 2).reshape(B, T, C))


def main() -> None:
    global VERBOSE
    torch.manual_seed(0)
    B, T, C, H = 2, 8, 32, 4
    x = torch.randn(B, T, C)
    m = MultiHeadAttention(C, H)

    print(f"B={B} (batch size)  T={T} (sequence length)  C={C} (channels)  H={H} (heads)  d=C/H={C // H}")
    print("Shape after each step:")
    VERBOSE = True
    with torch.no_grad():
        ours, w = m(x, return_weights=True)
    VERBOSE = False

    print("\nAttention weights of sample 1, head 1 (the upper triangle is all 0):")
    torch.set_printoptions(precision=2, sci_mode=False)
    print(w[0, 0])
    print("Sum of each row:", w[0, 0].sum(-1))

    with torch.no_grad():
        ref = mha_with_sdpa(m, x)
    print(
        f"\nMax difference from F.scaled_dot_product_attention(is_causal=True): {(ours - ref).abs().max():.1e}"
    )

    # Causality test: change the last 3 tokens.
    # The outputs at the first 5 positions must not change at all
    x2 = x.clone()
    x2[:, 5:] = torch.randn(B, 3, C)
    with torch.no_grad():
        out2 = m(x2)
    print(
        f"New inputs at positions 5–7: max output change at positions 0–4: {(out2[:, :5] - ours[:, :5]).abs().max():.1e}"
        f"; at positions 5–7: {(out2[:, 5:] - ours[:, 5:]).abs().max():.2f}"
    )

    n_params = sum(p.numel() for p in m.parameters())
    print(f"Parameters: {n_params} = 4 × C² = 4 × {C}² (Wq, Wk, Wv, Wo; does not depend on the number of heads)")


if __name__ == "__main__":
    main()
