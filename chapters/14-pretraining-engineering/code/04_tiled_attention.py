"""第 14 章 · 极简代码 4：分块（tiling）注意力 —— FlashAttention 前向的骨架

朴素注意力要把完整的 T×T 分数矩阵 S = QKᵀ/√d 和概率矩阵 P = softmax(S) 写进显存，再读回来乘 V。
T = 4096 时每个头每条序列就是 1600 多万个数。GPU 上注意力慢，主要不是算得慢，而是这些读写慢
（memory-bound）。

FlashAttention（Dao et al., 2022）的做法：
  - 把 Q 切成 Br 行一块，K、V 切成 Bc 行一块；每次只把一块 Q 和一块 K、V 放进片上高速缓存（SRAM）；
  - 用上一个脚本的 online softmax，逐块更新每一行的 (m, l, O)，S 和 P 从不完整写出；
  - 反向时也不存 P，只存每行的 logsumexp（lse = m + log l），需要时按块重算 P = exp(S − lse)。

这里用 PyTorch 在 CPU 上写出这个循环（只为看清算法，不追求速度），验证它和朴素注意力、
PyTorch 的 scaled_dot_product_attention 结果一致，并统计中间张量的大小。

运行：uv run python chapters/14-pretraining-engineering/code/04_tiled_attention.py   （几秒）
"""

import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)


def naive_attention(q, k, v, causal=True):
    """q, k, v: (T, d)。完整地算出 T×T 的 S 和 P。"""
    T, d = q.shape
    s = q @ k.T / math.sqrt(d)                       # (T, T) 分数矩阵，整个写出来
    if causal:
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
    p = torch.softmax(s, dim=-1)                     # (T, T) 概率矩阵，也整个写出来
    return p @ v, s.numel() + p.numel()


def tiled_attention(q, k, v, br=64, bc=64, causal=True):
    """分块 + online softmax。返回输出、每行的 logsumexp、以及最大的中间块大小。"""
    T, d = q.shape
    scale = 1 / math.sqrt(d)
    out = torch.empty_like(q)
    lse = torch.empty(T, dtype=q.dtype)
    biggest = 0
    for i0 in range(0, T, br):                       # 外层：一块 Q（Br 行）
        qi = q[i0:i0 + br]
        m = torch.full((qi.shape[0],), -math.inf, dtype=q.dtype)   # 每行的当前最大值
        l = torch.zeros(qi.shape[0], dtype=q.dtype)                 # 每行的当前指数和
        o = torch.zeros(qi.shape[0], d, dtype=q.dtype)              # 每行的未归一化输出
        for j0 in range(0, T, bc):                   # 内层：一块 K、V（Bc 行）
            if causal and j0 > i0 + qi.shape[0] - 1:
                break                                # 整块都在对角线右上方：因果 mask 全挡掉，跳过
            s = qi @ k[j0:j0 + bc].T * scale         # (Br, Bc) —— 只有这么大
            if causal:
                rows = torch.arange(i0, i0 + qi.shape[0])[:, None]
                cols = torch.arange(j0, j0 + s.shape[1])[None, :]
                s = s.masked_fill(cols > rows, float("-inf"))
            biggest = max(biggest, s.numel())
            m_new = torch.maximum(m, s.max(dim=1).values)
            alpha = torch.exp(m - m_new)             # 旧状态的改正系数 exp(m_旧 − m_新)
            p = torch.exp(s - m_new[:, None])        # 这一块的未归一化概率
            l = l * alpha + p.sum(dim=1)
            o = o * alpha[:, None] + p @ v[j0:j0 + bc]
            m = m_new
        out[i0:i0 + br] = o / l[:, None]             # 最后才除以 l
        lse[i0:i0 + br] = m + torch.log(l)           # 反向传播只需要存这个
    return out, lse, biggest


def main():
    torch.set_num_threads(1)
    T, d = 512, 64
    q, k, v = (torch.randn(T, d, dtype=torch.float64) for _ in range(3))

    print(f"① 正确性（T = {T}，d = {d}，float64）")
    ref, naive_elems = naive_attention(q, k, v)
    sdpa = F.scaled_dot_product_attention(q[None, None], k[None, None], v[None, None], is_causal=True)[0, 0]
    print(f"  {'块大小 Br×Bc':<12} {'与朴素的最大差':>14} {'与 SDPA 的最大差':>16} {'最大中间块':>10}")
    for br, bc in [(64, 64), (128, 32), (32, 128), (100, 70)]:
        out, _, biggest = tiled_attention(q, k, v, br, bc)
        print(f"  {f'{br}×{bc}':<12} {(out - ref).abs().max().item():>14.1e} "
              f"{(out - sdpa).abs().max().item():>16.1e} {biggest:>10,}")
    print(f"  朴素注意力的中间张量：S 和 P 共 {naive_elems:,} 个数（2·T²）")

    print("\n② 反向传播不存 P：用 logsumexp 按块重算")
    out, lse, _ = tiled_attention(q, k, v)
    s = q @ k.T / math.sqrt(d)
    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
    p_recomputed = torch.exp(s[:64] - lse[:64, None])     # 只重算第一块 Q 对应的行
    print(f"  前 64 行重算的 P 与 softmax(S) 的最大差：{(p_recomputed - torch.softmax(s[:64], -1)).abs().max().item():.1e}")
    print(f"  每行只存 1 个数（lse），而不是 T = {T} 个概率")

    print("\n③ 放到主线模型的尺度（T = 4096，16 个头，micro batch 8，BF16 2 字节）")
    Tm, heads, bsz = 4096, 16, 8
    per_layer = 2 * Tm * Tm * heads * bsz * 2          # S 和 P
    print(f"  朴素注意力一层的 S + P：{per_layer / 2**30:.1f} GiB；28 层要为反向保留 P：{per_layer / 2 * 28 / 2**30:.0f} GiB")
    print(f"  分块注意力：每行存 lse（fp32）一层 {Tm * heads * bsz * 4 / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
