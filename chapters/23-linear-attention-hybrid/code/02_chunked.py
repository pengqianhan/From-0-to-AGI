"""Chapter 23 · Minimal code 2: decay gate + chunkwise parallel form

The recurrent form is good for inference (O(1) per step). But training processes a full sequence at once.
A token-by-token for loop cannot run in parallel, so the GPU waits.
The fully parallel form calculates a T×T matrix. The compromise is the **chunkwise** form:
  - In a chunk (length C): use the parallel form, one matrix multiplication.
  - Between chunks: pass on the state S, which compresses all earlier chunks.
The script also adds a **decay gate** (the common skeleton of RetNet / GLA / Mamba-2):
    S_t = α_t S_{t-1} + v_t k_tᵀ,   α_t = exp(g_t) ∈ (0, 1]
A smaller α_t forgets old information faster.
The script makes sure that the three algorithms give the same numbers. Then it compares their speed.

Run: uv run python chapters/23-linear-attention-hybrid/code/02_chunked.py
"""

from __future__ import annotations

import time

import torch

torch.set_num_threads(1)


def recurrent(q, k, v, g):
    """q, k: (T, d_k); v: (T, d_v); g: (T,) log decay ≤ 0. State S: (d_v, d_k)."""
    S = torch.zeros(v.shape[1], q.shape[1])
    out = torch.empty_like(v)
    for t in range(q.shape[0]):
        S = g[t].exp() * S + torch.outer(v[t], k[t])  # S_t = α_t S_{t-1} + v_t k_tᵀ
        out[t] = S @ q[t]  # o_t = S_t q_t
    return out


def parallel(q, k, v, g):
    """Fully parallel: O = (Q Kᵀ ⊙ D) V, D_ij = α_{j+1}···α_i = exp(b_i − b_j) (j ≤ i), b = cumsum(g)."""
    b = g.cumsum(0)
    D = (b[:, None] - b[None, :]).masked_fill(~torch.ones(len(g), len(g)).tril().bool(), -torch.inf)
    return ((q @ k.T) * D.exp()) @ v


def chunked(q, k, v, g, C: int = 64):
    """Chunkwise: parallel in a chunk, recurrent between chunks.
    T must be a multiple of C (the production version pads with zeros)."""
    T = q.shape[0]
    S = torch.zeros(v.shape[1], q.shape[1])
    out = torch.empty_like(v)
    mask = torch.ones(C, C).tril().bool()
    for s in range(0, T, C):
        qc, kc, vc = q[s : s + C], k[s : s + C], v[s : s + C]
        b = g[s : s + C].cumsum(0)  # cumulative log decay in the chunk
        D = (b[:, None] - b[None, :]).masked_fill(~mask, -torch.inf).exp()
        inter = (qc * b.exp()[:, None]) @ S.T  # read the state from the chunk start (decayed to position i)
        intra = ((qc @ kc.T) * D) @ vc  # parallel attention in the chunk
        out[s : s + C] = inter + intra
        S = b[-1].exp() * S + vc.T @ (kc * (b[-1] - b).exp()[:, None])  # compress the full chunk into the state
    return out


def timed(fn, *args, reps: int = 3) -> tuple[torch.Tensor, float]:
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(*args)
    return out, (time.perf_counter() - t0) / reps * 1e3


def main() -> None:
    torch.manual_seed(0)
    d = 64
    print("== The three algorithms give the same numbers (T=512, d=64, q/k L2-normalized) ==")
    T = 512
    q = torch.nn.functional.normalize(torch.randn(T, d), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(T, d), dim=-1)
    v = torch.randn(T, d)
    g = -torch.rand(T) * 0.1  # α_t ∈ (0.905, 1]
    r, p, c = recurrent(q, k, v, g), parallel(q, k, v, g), chunked(q, k, v, g)
    print(f"max|recurrent − parallel| = {(r - p).abs().max():.1e}   max|recurrent − chunked| = {(r - c).abs().max():.1e}")

    print("\n== Effect of the decay gate: read strength of one written token as the distance grows ==")
    qq = torch.zeros(T, d)
    qq[:, 0] = 1.0  # each step reads with the same query
    kk, vv = torch.zeros(T, d), torch.zeros(T, 1)
    kk[0, 0], vv[0, 0] = 1.0, 1.0  # write only once, at step 0: key = e₀, value = 1
    for alpha in (1.0, 0.99, 0.9):
        o = recurrent(qq, kk, vv, torch.full((T,), float(torch.tensor(alpha).log())))
        print(f"α = {alpha:<5}: read at distance 10 / 100 / 500 = {o[10, 0]:.3f} / {o[100, 0]:.3f} / "
              f"{o[500, 0]:.3f} (theory: α^distance)")

    print("\n== Speed (one head d=64, one CPU thread, ms; the parallel form calculates a T×T matrix) ==")
    print(f"{'T':>6} | {'Recurrent':>14} | {'Chunk C=64':>10} | {'Parallel':>8}")
    for T in (256, 1024, 4096):
        q = torch.nn.functional.normalize(torch.randn(T, d), dim=-1)
        k = torch.nn.functional.normalize(torch.randn(T, d), dim=-1)
        v, g = torch.randn(T, d), -torch.rand(T) * 0.1
        _, tr = timed(recurrent, q, k, v, g, reps=1)
        _, tc = timed(chunked, q, k, v, g)
        _, tp = timed(parallel, q, k, v, g)
        print(f"{T:>6} | {tr:>14.1f} | {tc:>10.1f} | {tp:>8.1f}")


if __name__ == "__main__":
    main()
