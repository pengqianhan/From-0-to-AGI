"""第 23 章 · 极简代码 1：去掉 softmax，注意力就变成了一个 RNN

三件事：
  1. softmax 注意力：生成第 t 个 token 要看全部 t 个 K、V —— KV cache 随长度线性增长；
  2. 去掉 softmax（换成特征映射 φ）以后，矩阵乘法可以换个顺序算：
         (φ(Q) φ(K)ᵀ ⊙ M) V   ==   逐 token 递推  S_t = S_{t-1} + v_t φ(k_t)ᵀ,  o_t = S_t φ(q_t)
     左边是 T×T 的"注意力矩阵"，右边只有一个 d_v×d_k 的状态矩阵 S —— 验证两者数值相等；
  3. 解码一步的开销：softmax 注意力随上下文变长越来越慢，线性注意力每步是常数。

运行：uv run python chapters/23-linear-attention-hybrid/code/01_linear_attention.py
"""

from __future__ import annotations

import time

import torch

torch.set_num_threads(1)  # 构建环境共享 CPU；读者本机可删掉


def phi(x: torch.Tensor) -> torch.Tensor:
    """特征映射：把 q、k 变成非负向量（Katharopoulos et al. 2020 用 elu(x) + 1）。"""
    return torch.nn.functional.elu(x) + 1


def softmax_attention(q, k, v):
    """标准因果注意力：o_t = Σ_{j≤t} softmax_j(q_t·k_j / √d) v_j。q, k: (T, d)；v: (T, d_v)。"""
    T, d = q.shape
    scores = q @ k.T / d**0.5  # (T, T)
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    return scores.masked_fill(~mask, float("-inf")).softmax(-1) @ v


def linear_attention_parallel(q, k, v):
    """并行形式：O = (φ(Q) φ(K)ᵀ ⊙ M) V，M 是下三角的因果掩码。要算一个 T×T 的矩阵。"""
    T = q.shape[0]
    A = phi(q) @ phi(k).T  # (T, T)：和 softmax 注意力一样的"谁看谁"矩阵，只是没有 softmax
    return (A * torch.ones(T, T).tril()) @ v


def linear_attention_recurrent(q, k, v):
    """递推形式：只维护一个 (d_v, d_k) 的状态 S —— 这就是一个 RNN。"""
    S = torch.zeros(v.shape[1], q.shape[1])
    out = []
    for t in range(q.shape[0]):
        S = S + torch.outer(v[t], phi(k[t]))  # S_t = S_{t-1} + v_t φ(k_t)ᵀ   （写入）
        out.append(S @ phi(q[t]))  # o_t = S_t φ(q_t)                （读出）
    return torch.stack(out)


def kv_cache_vs_state(d: int = 64, bytes_per: int = 2) -> list[tuple[int, int, int]]:
    """单层单头：KV cache = 2·T·d 个数；线性注意力的状态 = d·d 个数，与 T 无关。"""
    rows = []
    for T in (128, 1024, 8192, 65536, 262144):
        rows.append((T, 2 * T * d * bytes_per, d * d * bytes_per))
    return rows


def _median_ms(fn, reps: int) -> float:
    for _ in range(5):  # 预热
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sorted(times)[reps // 2] * 1e3


@torch.no_grad()
def decode_step_time(T: int, d: int = 64, reps: int = 60) -> tuple[float, float]:
    """在已经有 T 个历史 token 时，生成下一个 token 的注意力部分要多久（毫秒，取中位数）。"""
    g = torch.Generator().manual_seed(0)
    K, V = torch.randn(T, d, generator=g), torch.randn(T, d, generator=g)
    S = torch.randn(d, d, generator=g)
    q, k, v = (torch.randn(d, generator=g) for _ in range(3))

    def softmax_step():  # 和全部 T 个 key 算分数、再加权全部 T 个 value
        return (K @ q / d**0.5).softmax(0) @ V

    def linear_step():  # 更新一次 d×d 的状态、读一次
        return (S + torch.outer(v, phi(k))) @ phi(q)

    return _median_ms(softmax_step, reps), _median_ms(linear_step, reps)


def main() -> None:
    torch.manual_seed(0)
    T, d = 256, 16
    q, k, v = torch.randn(T, d), torch.randn(T, d), torch.randn(T, d)

    print("== 1. 结合律：并行形式 == 递推形式 ==")
    par = linear_attention_parallel(q, k, v)
    rec = linear_attention_recurrent(q, k, v)
    rel = ((par - rec).abs().max() / par.abs().max()).item()
    print(f"T={T}, d={d}：两种算法输出的最大相对误差 {rel:.1e}（float32 舍入误差量级）")
    print(f"并行形式要存的中间矩阵 φ(Q)φ(K)ᵀ：{T}×{T} = {T * T} 个数")
    print(f"递推形式要存的状态 S：{d}×{d} = {d * d} 个数（与 T 无关）")
    soft = softmax_attention(q, k, v)
    print(f"对照：softmax 注意力输出的形状也是 {tuple(soft.shape)}，但 softmax 要对每一行的 T 个分数"
          "整体归一化，拆不开，所以没有这种递推形式")

    print("\n== 2. 推理时要存多少：KV cache vs 固定大小的状态（单层单头，d=64，BF16）==")
    print(f"{'上下文长度 T':>12} | {'KV cache':>12} | {'线性注意力状态':>14}")
    for T_, kv, st in kv_cache_vs_state():
        print(f"{T_:>12,} | {kv / 1024:>9,.0f} KB | {st / 1024:>11,.0f} KB")

    print("\n== 3. 生成下一个 token 的注意力开销（单头 d=64，CPU 单线程，毫秒）==")
    print(f"{'已有上下文':>10} | {'softmax 注意力':>14} | {'线性注意力':>10}")
    for T_ in (1024, 8192, 65536, 262144):
        ts, tl = decode_step_time(T_)
        print(f"{T_:>10,} | {ts:>14.3f} | {tl:>10.3f}")


if __name__ == "__main__":
    main()
