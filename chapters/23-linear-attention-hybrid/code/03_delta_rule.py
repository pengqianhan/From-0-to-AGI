"""Chapter 23 · Minimal code 3: the delta rule — from "only add" to "overwrite", then add a decay gate

Think of the state S (d_v × d_k) as a "key → value" table. Write (k, v), then read with k.
We want the read to give S k ≈ v.
  - Linear attention can only add: S ← S + v kᵀ. Write the same key two times,
    and the read gives the sum of the two values.
    When we write more items, different keys interfere with each other. The read becomes less accurate.
  - The delta rule (DeltaNet) first reads the old answer S k. Then it writes back only the "difference":
        S ← S + β (v − S k) kᵀ  =  S (I − β k kᵀ) + β v kᵀ
    When β = 1 and ‖k‖ = 1, S k is exactly v after the write. The rule overwrites; it does not add.
  - Gated DeltaNet also multiplies by a decay gate: S ← α S (I − β k kᵀ) + β v kᵀ.

Four small experiments: overwrite, capacity, forgetting in a stream of writes,
chunkwise form == recurrent form.
Run: uv run python chapters/23-linear-attention-hybrid/code/03_delta_rule.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

torch.set_num_threads(1)


# ── Core: batched recurrent and chunkwise forms (shape (..., T, d), state S: (..., d_v, d_k)) ──
def gated_delta_recurrent(q, k, v, g, beta, S=None):
    """Token by token: S ← α_t S (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ, o_t = S q_t. g=None gives DeltaNet."""
    *lead, T, dk = q.shape
    S = q.new_zeros(*lead, v.shape[-1], dk) if S is None else S
    out = []
    for t in range(T):
        if g is not None:
            S = S * g[..., t, None, None].exp()  # decay: S ← α_t S
        kt = k[..., t, :]
        old = (S @ kt[..., None])[..., 0]  # old answer S k_t
        u = beta[..., t, None] * (v[..., t, :] - old)  # write only the difference
        S = S + u[..., :, None] * kt[..., None, :]  # S ← S + u_t k_tᵀ
        out.append((S @ q[..., t, :, None])[..., 0])
    return torch.stack(out, -2), S


def gated_delta_chunked(q, k, v, g, beta, C: int = 32):
    """Chunkwise form (T must be a multiple of C).
    The value u_i that position i of a chunk really writes depends on the earlier u_j:
        u_i = β_i ( v_i − e^{b_i} S₀ k_i − Σ_{j<i} e^{b_i−b_j} (k_i·k_j) u_j )
    This is a lower-triangular system (I + A) U = ….
    One triangular solve replaces the token-by-token loop (UT transform)."""
    *lead, T, dk = q.shape
    g = torch.zeros(*lead, T) if g is None else g
    S = q.new_zeros(*lead, v.shape[-1], dk)
    mask = torch.ones(C, C, dtype=torch.bool).tril()
    eye = torch.eye(C)
    out = []
    for s in range(0, T, C):
        qc, kc, vc = q[..., s : s + C, :], k[..., s : s + C, :], v[..., s : s + C, :]
        bt, b = beta[..., s : s + C], g[..., s : s + C].cumsum(-1)
        D = (b[..., :, None] - b[..., None, :]).masked_fill(~mask, -torch.inf).exp()
        A = (bt[..., :, None] * (kc @ kc.mT) * D).tril(-1)  # A_ij = β_i e^{b_i−b_j} k_i·k_j, j<i
        rhs = bt[..., None] * (vc - (kc * b.exp()[..., None]) @ S.mT)  # β_i (v_i − e^{b_i} S₀ k_i)
        u = torch.linalg.solve_triangular(eye + A, rhs, upper=False)
        out.append((qc * b.exp()[..., None]) @ S.mT + ((qc @ kc.mT) * D) @ u)
        S = b[..., -1, None, None].exp() * S + u.mT @ (kc * (b[..., -1:] - b).exp()[..., None])
    return torch.cat(out, -2), S


# ── Writes of one stream, for the demos ───────────────────────────────────────────
def write_stream(keys, values, rule: str, alpha=1.0, beta: float = 1.0):
    """Write N (k, v) pairs one after the other into a d_v×d_k state. Return S. rule: 'linear' or 'delta'.
    alpha is one number (fixed decay) or a sequence of length N (one per step: a data-dependent gate)."""
    S = torch.zeros(values.shape[1], keys.shape[1])
    alphas = [alpha] * len(keys) if isinstance(alpha, float) else alpha
    for k, v, alpha in zip(keys, values, alphas):
        if rule == "linear":
            S = alpha * S + torch.outer(v, k)
        else:
            S = alpha * S
            S = S + beta * torch.outer(v - S @ k, k)
    return S


def read_error(S, keys, values) -> float:
    """Read with each key. Return the mean relative error of the read value, ‖S k − v‖ / ‖v‖."""
    return ((keys @ S.T - values).norm(dim=1) / values.norm(dim=1)).mean().item()


def attention_read_error(keys, values, scale: float = 30.0) -> float:
    """Baseline: softmax attention stores all (k, v) pairs unchanged (KV cache). It "looks up" the table with the key."""
    w = (keys @ keys.T * scale).softmax(-1)
    return ((w @ values - values).norm(dim=1) / values.norm(dim=1)).mean().item()


def main() -> None:
    torch.manual_seed(0)
    d = 64

    print("== 1. Write the same key two times: first v1, then v2 (β = 1) ==")
    k = F.normalize(torch.randn(d), dim=0)
    v1, v2 = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    for rule in ("linear", "delta"):
        S = write_stream(torch.stack([k, k]), torch.stack([v1, v2]), rule)
        print(f"{rule:>6}: read with k gives {[round(x, 3) for x in (S @ k).tolist()]}")
    print("(Linear attention reads v1 + v2 = [1, 1]. The delta rule reads v2 = [0, 1]: it overwrote the old value)")

    print(f"\n== 2. Capacity: write N random (k, v) pairs into a {d}×{d} state, then read each one back ==")
    print(f"{'N':>5} | {'Linear':>10} | {'Delta rule':>10} | {'Softmax (stores all KV)':>24}")
    for N in (16, 32, 64, 128, 256):
        keys = F.normalize(torch.randn(N, d), dim=1)
        values = torch.randn(N, d)
        e_lin = read_error(write_stream(keys, values, "linear"), keys, values)
        e_del = read_error(write_stream(keys, values, "delta"), keys, values)
        e_att = attention_read_error(keys, values)
        print(f"{N:>5} | {e_lin:>10.3f} | {e_del:>10.3f} | {e_att:>24.3f}")
    print("(Relative error: 0 = perfect read; 1 ≈ the read is noise as large as the true value)")

    print("\n== 3. Write a stream of 1024 (k, v) pairs; read only the 32 newest ==")
    keys = F.normalize(torch.randn(1024, d), dim=1)
    values = torch.randn(1024, d)
    recent = slice(1024 - 32, 1024)
    reset = [1.0] * 1024
    # Data-dependent gate: the topic changes where the 32 newest items start.
    # The gate closes and clears the old content.
    reset[1024 - 32] = 0.0
    rows = [("linear attention", "linear", 1.0), ("linear + fixed α=0.95", "linear", 0.95),
            ("delta rule", "delta", 1.0), ("delta + fixed α=0.95", "delta", 0.95),
            ("linear + topic reset α=0", "linear", reset), ("delta + topic reset α=0", "delta", reset)]
    for name, rule, alpha in rows:
        S = write_stream(keys, values, rule, alpha=alpha)
        err = read_error(S, keys[recent], values[recent])
        print(f"{name:<24} read error of the 32 newest {err:.3f}")

    print("\n== 4. Chunkwise form == recurrent form (Gated DeltaNet, B=2, H=3, T=128, d=16, chunk length 32) ==")
    B, H, T, dk = 2, 3, 128, 16
    q = F.normalize(torch.randn(B, H, T, dk), dim=-1)
    k = F.normalize(torch.randn(B, H, T, dk), dim=-1)
    v = torch.randn(B, H, T, dk)
    g, beta = -torch.rand(B, H, T) * 0.2, torch.rand(B, H, T)
    o1, S1 = gated_delta_recurrent(q, k, v, g, beta)
    o2, S2 = gated_delta_chunked(q, k, v, g, beta, C=32)
    print(f"max|output diff| = {(o1 - o2).abs().max():.1e}   max|final state diff| = {(S1 - S2).abs().max():.1e}")


if __name__ == "__main__":
    main()
