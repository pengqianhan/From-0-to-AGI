"""第 23 章 · 极简代码 2：衰减门 + 分块并行（chunkwise）形式

递推形式适合推理（每步 O(1)），但训练时要一次处理整段序列：逐 token 的 for 循环没法并行，
GPU 会闲着。完全并行的形式又要算 T×T 的矩阵。折中办法是**分块**：
  - 块内（长度 C）：用并行形式，一次矩阵乘法；
  - 块间：把前面所有块压缩成的状态 S 传下去。
这里同时加上**衰减门**（RetNet / GLA / Mamba-2 一族的共同骨架）：
    S_t = α_t S_{t-1} + v_t k_tᵀ,   α_t = exp(g_t) ∈ (0, 1]
α_t 越小，旧信息忘得越快。验证三种算法数值相等，再比一比速度。

运行：uv run python chapters/23-linear-attention-hybrid/code/02_chunked.py
"""

from __future__ import annotations

import time

import torch

torch.set_num_threads(1)


def recurrent(q, k, v, g):
    """q, k: (T, d_k)；v: (T, d_v)；g: (T,) 对数衰减 ≤ 0。状态 S: (d_v, d_k)。"""
    S = torch.zeros(v.shape[1], q.shape[1])
    out = torch.empty_like(v)
    for t in range(q.shape[0]):
        S = g[t].exp() * S + torch.outer(v[t], k[t])  # S_t = α_t S_{t-1} + v_t k_tᵀ
        out[t] = S @ q[t]  # o_t = S_t q_t
    return out


def parallel(q, k, v, g):
    """完全并行：O = (Q Kᵀ ⊙ D) V，D_ij = α_{j+1}···α_i = exp(b_i − b_j)（j ≤ i），b = cumsum(g)。"""
    b = g.cumsum(0)
    D = (b[:, None] - b[None, :]).masked_fill(~torch.ones(len(g), len(g)).tril().bool(), -torch.inf)
    return ((q @ k.T) * D.exp()) @ v


def chunked(q, k, v, g, C: int = 64):
    """分块：块内并行、块间递推。T 需是 C 的整数倍（生产版会补零）。"""
    T = q.shape[0]
    S = torch.zeros(v.shape[1], q.shape[1])
    out = torch.empty_like(v)
    mask = torch.ones(C, C).tril().bool()
    for s in range(0, T, C):
        qc, kc, vc = q[s : s + C], k[s : s + C], v[s : s + C]
        b = g[s : s + C].cumsum(0)  # 块内累计对数衰减
        D = (b[:, None] - b[None, :]).masked_fill(~mask, -torch.inf).exp()
        inter = (qc * b.exp()[:, None]) @ S.T  # 读块开始时的状态（衰减到位置 i）
        intra = ((qc @ kc.T) * D) @ vc  # 块内的并行注意力
        out[s : s + C] = inter + intra
        S = b[-1].exp() * S + vc.T @ (kc * (b[-1] - b).exp()[:, None])  # 把整块压进状态
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
    print("== 三种算法数值一致（T=512, d=64，q/k 做了 L2 归一化）==")
    T = 512
    q = torch.nn.functional.normalize(torch.randn(T, d), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(T, d), dim=-1)
    v = torch.randn(T, d)
    g = -torch.rand(T) * 0.1  # α_t ∈ (0.905, 1]
    r, p, c = recurrent(q, k, v, g), parallel(q, k, v, g), chunked(q, k, v, g)
    print(f"max|递推 − 并行| = {(r - p).abs().max():.1e}   max|递推 − 分块| = {(r - c).abs().max():.1e}")

    print("\n== 衰减门的效果：一个 token 写入后，被读出的强度随距离的变化 ==")
    qq = torch.zeros(T, d)
    qq[:, 0] = 1.0  # 每一步都用同一个 query 去读
    kk, vv = torch.zeros(T, d), torch.zeros(T, 1)
    kk[0, 0], vv[0, 0] = 1.0, 1.0  # 只在第 0 步写入一次：key = e₀，value = 1
    for alpha in (1.0, 0.99, 0.9):
        o = recurrent(qq, kk, vv, torch.full((T,), float(torch.tensor(alpha).log())))
        print(f"α = {alpha:<5}：距离 10 / 100 / 500 处读出 {o[10, 0]:.3f} / {o[100, 0]:.3f} / "
              f"{o[500, 0]:.3f}（理论值 α^距离）")

    print("\n== 速度（单头 d=64，CPU 单线程，毫秒；并行形式要算 T×T 矩阵）==")
    print(f"{'T':>6} | {'递推(逐 token)':>14} | {'分块 C=64':>10} | {'完全并行':>8}")
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
