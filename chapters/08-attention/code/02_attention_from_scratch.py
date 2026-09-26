"""第 8 章 · 极简代码 2：从零实现（多头）因果自注意力，并和 PyTorch 官方实现对拍

    Q = X Wq,  K = X Wk,  V = X Wv                 （三个线性投影，第 2 章的 y = XW）
    Attention(Q, K, V) = softmax(Q Kᵀ / √d + mask) V  （缩放点积 + 因果 mask）
    多头：把 C 维切成 H 份，每份 d = C / H 维各算一次，拼回来再乘 Wo

只用 PyTorch，CPU 上一秒内跑完。
运行：uv run python chapters/08-attention/code/02_attention_from_scratch.py
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

VERBOSE = False  # MultiHeadAttention.forward 里打印每一步的形状


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True):
    """缩放点积注意力。q, k, v: (..., T, d) → 输出 (..., T, d)，权重 (..., T, T)。"""
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)            # QKᵀ / √d
    if causal:
        T = q.shape[-2]
        future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))      # 未来位置 → −∞
    weights = torch.softmax(scores, dim=-1)                     # 每行和为 1
    return weights @ v, weights                                 # 加权平均 V


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
        self.wo = nn.Linear(C, C, bias=False)  # 输出投影：把各头拼起来的结果再混合一次

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        B, T, C = x.shape
        show("x", x, "(B, T, C)")
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        show("q = x @ Wq", q, "(B, T, C)")
        # 切头：C = H × d，再把头的维度换到前面，当成"批"的一部分
        q = q.view(B, T, self.H, self.d).transpose(1, 2)
        k = k.view(B, T, self.H, self.d).transpose(1, 2)
        v = v.view(B, T, self.H, self.d).transpose(1, 2)
        show("q 切头后", q, "(B, H, T, d)")
        out, w = attention(q, k, v, causal=True)
        show("权重 softmax(QKᵀ/√d)", w, "(B, H, T, T)")
        show("每个头的输出 w @ v", out, "(B, H, T, d)")
        out = out.transpose(1, 2).reshape(B, T, C)               # 拼回去
        show("拼接各头", out, "(B, T, C)")
        out = self.wo(out)
        show("输出 = 拼接 @ Wo", out, "(B, T, C)")
        return (out, w) if return_weights else out


def mha_with_sdpa(m: MultiHeadAttention, x: torch.Tensor) -> torch.Tensor:
    """同一组权重，注意力那一步换成 PyTorch 官方的 scaled_dot_product_attention。"""
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

    print(f"B={B}（批大小）  T={T}（序列长度）  C={C}（通道数）  H={H}（头数）  d=C/H={C // H}")
    print("每一步的形状：")
    VERBOSE = True
    with torch.no_grad():
        ours, w = m(x, return_weights=True)
    VERBOSE = False

    print("\n第 1 个样本、第 1 个头的注意力权重（上三角全是 0）：")
    torch.set_printoptions(precision=2, sci_mode=False)
    print(w[0, 0])
    print("每行的和：", w[0, 0].sum(-1))

    with torch.no_grad():
        ref = mha_with_sdpa(m, x)
    print(f"\n和 F.scaled_dot_product_attention(is_causal=True) 的最大差：{(ours - ref).abs().max():.1e}")

    # 因果性检验：改掉最后 3 个 token，前 5 个位置的输出必须一字不变
    x2 = x.clone()
    x2[:, 5:] = torch.randn(B, 3, C)
    with torch.no_grad():
        out2 = m(x2)
    print(f"改掉位置 5–7 的输入后，位置 0–4 输出的最大变化：{(out2[:, :5] - ours[:, :5]).abs().max():.1e}"
          f"；位置 5–7 的最大变化：{(out2[:, 5:] - ours[:, 5:]).abs().max():.2f}")

    n_params = sum(p.numel() for p in m.parameters())
    print(f"参数量：{n_params} = 4 × C² = 4 × {C}²（Wq、Wk、Wv、Wo，和头数无关）")


if __name__ == "__main__":
    main()
