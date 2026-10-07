"""Chapter 23 · Minimal code 1: remove the softmax, and attention becomes an RNN

Three parts:
  1. Softmax attention: to generate token t, it must read all t K and V vectors.
     Thus the KV cache grows linearly with the length.
  2. Without the softmax (we use a feature map φ instead), we can change the order of the matrix products:
         (φ(Q) φ(K)ᵀ ⊙ M) V   ==   token-by-token recurrence  S_t = S_{t-1} + v_t φ(k_t)ᵀ,  o_t = S_t φ(q_t)
     The left side is a T×T "attention matrix". The right side has only one d_v×d_k state matrix S.
     The script makes sure that the two give the same numbers.
  3. Cost of one decode step: softmax attention becomes slower as the context grows.
     Linear attention has a constant cost for each step.

Run: uv run python chapters/23-linear-attention-hybrid/code/01_linear_attention.py
"""

from __future__ import annotations

import time

import torch

torch.set_num_threads(1)  # the build machine shares its CPU; on your computer you can remove this line


def phi(x: torch.Tensor) -> torch.Tensor:
    """Feature map: makes q and k non-negative vectors (Katharopoulos et al. 2020 use elu(x) + 1)."""
    return torch.nn.functional.elu(x) + 1


def softmax_attention(q, k, v):
    """Standard causal attention: o_t = Σ_{j≤t} softmax_j(q_t·k_j / √d) v_j. q, k: (T, d); v: (T, d_v)."""
    T, d = q.shape
    scores = q @ k.T / d**0.5  # (T, T)
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    return scores.masked_fill(~mask, float("-inf")).softmax(-1) @ v


def linear_attention_parallel(q, k, v):
    """Parallel form: O = (φ(Q) φ(K)ᵀ ⊙ M) V, where M is the lower-triangular causal mask. It calculates a T×T matrix."""
    T = q.shape[0]
    A = phi(q) @ phi(k).T  # (T, T): the same "who reads whom" matrix as softmax attention, but without the softmax
    return (A * torch.ones(T, T).tril()) @ v


def linear_attention_recurrent(q, k, v):
    """Recurrent form: keeps only one (d_v, d_k) state S. This is an RNN."""
    S = torch.zeros(v.shape[1], q.shape[1])
    out = []
    for t in range(q.shape[0]):
        S = S + torch.outer(v[t], phi(k[t]))  # S_t = S_{t-1} + v_t φ(k_t)ᵀ   (write)
        out.append(S @ phi(q[t]))  # o_t = S_t φ(q_t)                (read)
    return torch.stack(out)


def kv_cache_vs_state(d: int = 64, bytes_per: int = 2) -> list[tuple[int, int, int]]:
    """One layer, one head: KV cache = 2·T·d numbers; linear-attention state = d·d numbers, independent of T."""
    rows = []
    for T in (128, 1024, 8192, 65536, 262144):
        rows.append((T, 2 * T * d * bytes_per, d * d * bytes_per))
    return rows


def _median_ms(fn, reps: int) -> float:
    for _ in range(5):  # warmup
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sorted(times)[reps // 2] * 1e3


@torch.no_grad()
def decode_step_time(T: int, d: int = 64, reps: int = 60) -> tuple[float, float]:
    """Time of the attention part to generate the next token after T earlier tokens (ms, median)."""
    g = torch.Generator().manual_seed(0)
    K, V = torch.randn(T, d, generator=g), torch.randn(T, d, generator=g)
    S = torch.randn(d, d, generator=g)
    q, k, v = (torch.randn(d, generator=g) for _ in range(3))

    def softmax_step():  # score all T keys, then take the weighted sum of all T values
        return (K @ q / d**0.5).softmax(0) @ V

    def linear_step():  # update the d×d state once and read it once
        return (S + torch.outer(v, phi(k))) @ phi(q)

    return _median_ms(softmax_step, reps), _median_ms(linear_step, reps)


def main() -> None:
    torch.manual_seed(0)
    T, d = 256, 16
    q, k, v = torch.randn(T, d), torch.randn(T, d), torch.randn(T, d)

    print("== 1. Associativity: parallel form == recurrent form ==")
    par = linear_attention_parallel(q, k, v)
    rec = linear_attention_recurrent(q, k, v)
    rel = ((par - rec).abs().max() / par.abs().max()).item()
    print(f"T={T}, d={d}: max relative error between the two outputs {rel:.1e} (float32 rounding level)")
    print(f"Parallel form stores the intermediate matrix φ(Q)φ(K)ᵀ: {T}×{T} = {T * T} numbers")
    print(f"Recurrent form stores the state S: {d}×{d} = {d * d} numbers (independent of T)")
    soft = softmax_attention(q, k, v)
    print(f"Compare: the softmax attention output also has the shape {tuple(soft.shape)}. But the softmax "
          "normalizes all T scores of a row together. We cannot split it, so it has no recurrent form")

    print("\n== 2. Memory at inference: KV cache vs fixed-size state (one layer, one head, d=64, BF16) ==")
    print(f"{'Context T':>12} | {'KV cache':>12} | {'Linear state':>14}")
    for T_, kv, st in kv_cache_vs_state():
        print(f"{T_:>12,} | {kv / 1024:>9,.0f} KB | {st / 1024:>11,.0f} KB")

    print("\n== 3. Attention cost to generate the next token (one head d=64, one CPU thread, ms) ==")
    print(f"{'Context':>10} | {'Softmax attn':>14} | {'Linear':>10}")
    for T_ in (1024, 8192, 65536, 262144):
        ts, tl = decode_step_time(T_)
        print(f"{T_:>10,} | {ts:>14.3f} | {tl:>10.3f}")


if __name__ == "__main__":
    main()
