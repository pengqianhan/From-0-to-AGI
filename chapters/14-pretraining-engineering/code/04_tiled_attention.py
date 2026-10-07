"""Chapter 14 · Minimal code 4: tiled attention. This is the skeleton of the FlashAttention forward pass.

Naive attention writes the full T×T score matrix S = QKᵀ/√d and the probability matrix P = softmax(S)
to GPU memory. Then it reads them back to multiply by V. At T = 4096, each head of each sequence
has more than 16 million numbers. Attention on a GPU is slow mainly because of these reads and writes,
not because of the calculation (memory-bound).

The method of FlashAttention (Dao et al., 2022):
  - Cut Q into blocks of Br rows and K, V into blocks of Bc rows. Put only one block of Q and one
    block of K, V into the fast on-chip cache (SRAM) at a time.
  - Use the online softmax of the last script to update (m, l, O) of each row block by block.
    S and P are never written out in full.
  - The backward pass also does not store P. It stores only the logsumexp of each row
    (lse = m + log l) and calculates P = exp(S − lse) again block by block when it needs P.

This script writes the loop in PyTorch on the CPU (to show the algorithm, not for speed).
It checks that the result is the same as naive attention and as PyTorch's
scaled_dot_product_attention, and it counts the size of the intermediate tensors.

Run: uv run python chapters/14-pretraining-engineering/code/04_tiled_attention.py   (a few seconds)
"""

import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)


def naive_attention(q, k, v, causal=True):
    """q, k, v: (T, d). Calculate the full T×T matrices S and P."""
    T, d = q.shape
    s = q @ k.T / math.sqrt(d)                       # (T, T) score matrix, written out in full
    if causal:
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
    p = torch.softmax(s, dim=-1)                     # (T, T) probability matrix, also written out in full
    return p @ v, s.numel() + p.numel()


def tiled_attention(q, k, v, br=64, bc=64, causal=True):
    """Tiles + online softmax. Return the output, the logsumexp of each row, and the size of the largest intermediate block."""
    T, d = q.shape
    scale = 1 / math.sqrt(d)
    out = torch.empty_like(q)
    lse = torch.empty(T, dtype=q.dtype)
    biggest = 0
    for i0 in range(0, T, br):                       # outer loop: one block of Q (Br rows)
        qi = q[i0:i0 + br]
        m = torch.full((qi.shape[0],), -math.inf, dtype=q.dtype)   # current maximum of each row
        l = torch.zeros(qi.shape[0], dtype=q.dtype)                 # current sum of exponentials of each row
        o = torch.zeros(qi.shape[0], d, dtype=q.dtype)              # output of each row, not normalized yet
        for j0 in range(0, T, bc):                   # inner loop: one block of K, V (Bc rows)
            if causal and j0 > i0 + qi.shape[0] - 1:
                break                                # the full block is above the diagonal: the causal mask hides all of it, so skip it
            s = qi @ k[j0:j0 + bc].T * scale         # (Br, Bc): only this size
            if causal:
                rows = torch.arange(i0, i0 + qi.shape[0])[:, None]
                cols = torch.arange(j0, j0 + s.shape[1])[None, :]
                s = s.masked_fill(cols > rows, float("-inf"))
            biggest = max(biggest, s.numel())
            m_new = torch.maximum(m, s.max(dim=1).values)
            alpha = torch.exp(m - m_new)             # correction factor for the old state, exp(m_old − m_new)
            p = torch.exp(s - m_new[:, None])        # probabilities of this block, not normalized yet
            l = l * alpha + p.sum(dim=1)
            o = o * alpha[:, None] + p @ v[j0:j0 + bc]
            m = m_new
        out[i0:i0 + br] = o / l[:, None]             # divide by l only at the end
        lse[i0:i0 + br] = m + torch.log(l)           # the backward pass needs to store only this
    return out, lse, biggest


def main():
    torch.set_num_threads(1)
    T, d = 512, 64
    q, k, v = (torch.randn(T, d, dtype=torch.float64) for _ in range(3))

    print(f"① Correctness (T = {T}, d = {d}, float64)")
    ref, naive_elems = naive_attention(q, k, v)
    sdpa = F.scaled_dot_product_attention(q[None, None], k[None, None], v[None, None], is_causal=True)[0, 0]
    print(f"  {'block Br×Bc':<12} {'max diff naive':>14} {'max diff SDPA':>16} {'max block':>10}")
    for br, bc in [(64, 64), (128, 32), (32, 128), (100, 70)]:
        out, _, biggest = tiled_attention(q, k, v, br, bc)
        print(f"  {f'{br}×{bc}':<12} {(out - ref).abs().max().item():>14.1e} "
              f"{(out - sdpa).abs().max().item():>16.1e} {biggest:>10,}")
    print(f"  Intermediate tensors of naive attention: S and P have {naive_elems:,} numbers in total (2·T²)")

    print("\n② The backward pass does not store P: it uses logsumexp to calculate P again block by block")
    out, lse, _ = tiled_attention(q, k, v)
    s = q @ k.T / math.sqrt(d)
    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
    p_recomputed = torch.exp(s[:64] - lse[:64, None])     # calculate again only the rows of the first block of Q
    print(f"  Max difference between P calculated again for the first 64 rows and softmax(S): {(p_recomputed - torch.softmax(s[:64], -1)).abs().max().item():.1e}")
    print(f"  Each row stores only 1 number (lse), not T = {T} probabilities")

    print("\n③ At the scale of the main-line model (T = 4096, 16 heads, micro batch 8, BF16 2 bytes)")
    Tm, heads, bsz = 4096, 16, 8
    per_layer = 2 * Tm * Tm * heads * bsz * 2          # S and P
    print(f"  S + P of naive attention in one layer: {per_layer / 2**30:.1f} GiB; to keep P for the backward pass in 28 layers: {per_layer / 2 * 28 / 2**30:.0f} GiB")
    print(f"  Tiled attention: lse (fp32) of each row, per layer {Tm * heads * bsz * 4 / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
