"""第 23 章 · 极简代码 3：delta 规则 —— 从"只会累加"到"会覆盖"，再加上衰减门

把状态 S（d_v × d_k）看成一张"键 → 值"的表：写入 (k, v)，之后用 k 去读，希望读出 S k ≈ v。
  - 线性注意力只会累加：S ← S + v kᵀ。同一个 key 写两次，读出来是两个值的和；
    写的东西多了，不同 key 之间互相串扰，读出来越来越糊。
  - delta 规则（DeltaNet）先读出旧答案 S k，只把"差值"写回去：
        S ← S + β (v − S k) kᵀ  =  S (I − β k kᵀ) + β v kᵀ
    β = 1 且 ‖k‖ = 1 时，写完以后 S k 恰好等于 v —— 覆盖，而不是累加。
  - Gated DeltaNet 再乘一个衰减门：S ← α S (I − β k kᵀ) + β v kᵀ。

四个小实验：覆盖、容量、流式写入时的遗忘、分块形式 == 递推形式。
运行：uv run python chapters/23-linear-attention-hybrid/code/03_delta_rule.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

torch.set_num_threads(1)


# ── 核心：批量版的递推与分块（形状 (..., T, d)，状态 S: (..., d_v, d_k)）─────────────
def gated_delta_recurrent(q, k, v, g, beta, S=None):
    """逐 token：S ← α_t S (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ，o_t = S q_t。g=None 即 DeltaNet。"""
    *lead, T, dk = q.shape
    S = q.new_zeros(*lead, v.shape[-1], dk) if S is None else S
    out = []
    for t in range(T):
        if g is not None:
            S = S * g[..., t, None, None].exp()  # 衰减：S ← α_t S
        kt = k[..., t, :]
        old = (S @ kt[..., None])[..., 0]  # 旧答案 S k_t
        u = beta[..., t, None] * (v[..., t, :] - old)  # 只写入差值
        S = S + u[..., :, None] * kt[..., None, :]  # S ← S + u_t k_tᵀ
        out.append((S @ q[..., t, :, None])[..., 0])
    return torch.stack(out, -2), S


def gated_delta_chunked(q, k, v, g, beta, C: int = 32):
    """分块形式（T 需是 C 的整数倍）。块内第 i 个位置真正写入的 u_i 依赖前面的 u_j：
        u_i = β_i ( v_i − e^{b_i} S₀ k_i − Σ_{j<i} e^{b_i−b_j} (k_i·k_j) u_j )
    是一个下三角方程组 (I + A) U = …，用一次三角求解代替逐 token 循环（UT 变换）。"""
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


# ── 演示用的单条流写入 ──────────────────────────────────────────────────────────
def write_stream(keys, values, rule: str, alpha=1.0, beta: float = 1.0):
    """把 N 个 (k, v) 依次写进一个 d_v×d_k 的状态，返回 S。rule: 'linear' 或 'delta'。
    alpha 可以是一个数（固定衰减），也可以是长度 N 的序列（每步一个，数据相关的门）。"""
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
    """用 key 去读，读出值与真实值的平均相对误差 ‖S k − v‖ / ‖v‖。"""
    return ((keys @ S.T - values).norm(dim=1) / values.norm(dim=1)).mean().item()


def attention_read_error(keys, values, scale: float = 30.0) -> float:
    """对照：softmax 注意力把全部 (k, v) 原样存下来（KV cache），用 key 去"查表"。"""
    w = (keys @ keys.T * scale).softmax(-1)
    return ((w @ values - values).norm(dim=1) / values.norm(dim=1)).mean().item()


def main() -> None:
    torch.manual_seed(0)
    d = 64

    print("== 1. 同一个 key 写两次：先写 v1，再写 v2（β = 1）==")
    k = F.normalize(torch.randn(d), dim=0)
    v1, v2 = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    for rule in ("linear", "delta"):
        S = write_stream(torch.stack([k, k]), torch.stack([v1, v2]), rule)
        print(f"{rule:>6}：用 k 读出 {[round(x, 3) for x in (S @ k).tolist()]}")
    print("（线性注意力读出 v1 + v2 = [1, 1]；delta 规则读出 v2 = [0, 1]：覆盖掉了旧值）")

    print(f"\n== 2. 容量：往 {d}×{d} 的状态里写 N 个随机 (k, v)，再逐个读回来 ==")
    print(f"{'N':>5} | {'线性注意力':>10} | {'delta 规则':>10} | {'softmax 注意力(存全部 KV)':>24}")
    for N in (16, 32, 64, 128, 256):
        keys = F.normalize(torch.randn(N, d), dim=1)
        values = torch.randn(N, d)
        e_lin = read_error(write_stream(keys, values, "linear"), keys, values)
        e_del = read_error(write_stream(keys, values, "delta"), keys, values)
        e_att = attention_read_error(keys, values)
        print(f"{N:>5} | {e_lin:>10.3f} | {e_del:>10.3f} | {e_att:>24.3f}")
    print("（相对误差：0 = 完美读回，1 ≈ 读出的东西和真值一样大的噪声）")

    print("\n== 3. 流式写入 1024 个 (k, v)，只读最近写入的 32 个 ==")
    keys = F.normalize(torch.randn(1024, d), dim=1)
    values = torch.randn(1024, d)
    recent = slice(1024 - 32, 1024)
    reset = [1.0] * 1024
    reset[1024 - 32] = 0.0  # 数据相关的门：最近 32 个开始时"换了话题"，门关上，旧内容清空
    rows = [("线性注意力", "linear", 1.0), ("线性 + 固定衰减 α=0.95", "linear", 0.95),
            ("delta 规则", "delta", 1.0), ("delta + 固定衰减 α=0.95", "delta", 0.95),
            ("线性 + 换话题时 α=0", "linear", reset), ("delta + 换话题时 α=0", "delta", reset)]
    for name, rule, alpha in rows:
        S = write_stream(keys, values, rule, alpha=alpha)
        err = read_error(S, keys[recent], values[recent])
        print(f"{name:<24} 最近 32 个的读回误差 {err:.3f}")

    print("\n== 4. 分块形式 == 递推形式（Gated DeltaNet，B=2, H=3, T=128, d=16, 块长 32）==")
    B, H, T, dk = 2, 3, 128, 16
    q = F.normalize(torch.randn(B, H, T, dk), dim=-1)
    k = F.normalize(torch.randn(B, H, T, dk), dim=-1)
    v = torch.randn(B, H, T, dk)
    g, beta = -torch.rand(B, H, T) * 0.2, torch.rand(B, H, T)
    o1, S1 = gated_delta_recurrent(q, k, v, g, beta)
    o2, S2 = gated_delta_chunked(q, k, v, g, beta, C=32)
    print(f"max|输出差| = {(o1 - o2).abs().max():.1e}   max|最终状态差| = {(S1 - S2).abs().max():.1e}")


if __name__ == "__main__":
    main()
