"""第 12 章 · 极简代码 2：Chinchilla 公式能回答什么 —— 算力怎么在"模型大小"和"数据量"之间分

Chinchilla（Hoffmann et al. 2022）用几百个训练运行拟合了
    L(N, D) = E + A / N^α + B / D^β
E 是"数据本身的不确定性"（再大的模型也降不下去），A/N^α 是模型太小的代价，B/D^β 是数据太少的代价。
给定算力 C = 6ND，把 D = C/(6N) 代进去，对 N 求最小值，就得到"算力最优"的分配。

这里不训练任何东西，只用两组公开的系数做算术（绝对 loss 只对它们的数据和分词器有意义，不能搬到我们的模型上；
能搬过来的是"形状"：最优比例、过训练的代价）：
  - Hoffmann et al. 2022 原文（Approach 3）：E=1.69, A=406.4, B=410.7, α=0.34, β=0.28
  - Besiroglu et al. 2024（Epoch AI 复现，arXiv:2404.10102）：E=1.8172, A=482.01, B=2085.43, α=0.3478, β=0.3658
    复现者指出原文系数因优化器提前停止而有偏，复现版与 Chinchilla 实际采用的约 20 token/参数一致。

运行：uv run python chapters/12-scaling-laws/code/02_chinchilla.py   （不到 1 秒）
"""

import numpy as np

FITS = {
    "Hoffmann 2022": dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28),
    "Epoch 2024 复现": dict(E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658),
}


def loss(N, D, E, A, B, alpha, beta):
    return E + A / N**alpha + B / D**beta


def compute_optimal(C, E, A, B, alpha, beta):
    """解析解：N_opt = G · (C/6)^(β/(α+β))，G = (αA / βB)^(1/(α+β))。"""
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    N = G * (C / 6) ** (beta / (alpha + beta))
    return N, C / (6 * N)


def compute_optimal_grid(C, fit):
    """数值验证：在等算力曲线上扫一遍 N，取 loss 最小处（IsoFLOP 抛物线的谷底）。"""
    Ns = np.logspace(6, 13, 20001)
    Ls = loss(Ns, C / (6 * Ns), **fit)
    i = int(np.argmin(Ls))
    return Ns[i], C / (6 * Ns[i])


def compute_to_reach(target_loss, fit):
    """算力最优前沿上，达到 target_loss 需要多少算力（二分）。"""
    lo, hi = 1e15, 1e30
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        N, D = compute_optimal(mid, **fit)
        lo, hi = (mid, hi) if loss(N, D, **fit) > target_loss else (lo, mid)
    return hi


def main():
    fit = FITS["Epoch 2024 复现"]
    print("① 算力最优分配（Epoch 复现系数；网格搜索与解析解一致）")
    print(f"{'算力 C':>9} | {'N_opt':>9} {'D_opt':>9} {'token/参数':>9} | {'网格 N_opt':>9}")
    for C in [1e19, 1e21, 1.65e21, 1e23, 1e25]:
        N, D = compute_optimal(C, **fit)
        Ng, _ = compute_optimal_grid(C, fit)
        print(f"{C:>9.2e} | {N / 1e9:>8.2f}B {D / 1e9:>8.1f}B {D / N:>9.1f} | {Ng / 1e9:>8.2f}B")
    for name, f in FITS.items():
        N, D = compute_optimal(1e23, **f)
        print(f"  {name}：C=1e23 时最优 {D / N:.0f} token/参数")

    # 主线预训练：689.5M 参数 × 400B token（见 05_plan_budget.py）。Chinchilla 用粗算口径 C = 6ND，
    # 这里也用它（01_flops.py 里含注意力项的精确口径是 2.78e21，两者差在 4096 长序列的注意力上）
    N_ours, D_ours = 689.5e6, 400e9
    C = 6 * N_ours * D_ours
    N_opt, D_opt = compute_optimal(C, **fit)
    L_ours, L_opt = loss(N_ours, D_ours, **fit), loss(N_opt, D_opt, **fit)
    print(f"\n② 我们的预算（C ≈ {C:.3g}，粗算口径 6ND）：")
    print(f"  算力最优：N = {N_opt / 1e9:.2f}B、D = {D_opt / 1e9:.0f}B → L = {L_opt:.4f}")
    print(f"  主线选择：N = 0.69B、D = {D_ours / 1e9:.0f}B（{D_ours / N_ours:.0f} token/参数）→ L = {L_ours:.4f}")
    C_eq = compute_to_reach(L_ours, fit)
    print(f"  过训练的代价：loss 高 {L_ours - L_opt:.4f}（{(L_ours - L_opt) / L_opt:.1%}）；"
          f"同样的 loss，最优分配只要 {C_eq / C:.0%} 的算力")

    print("\n③ 为什么还要过训练：把推理也算进总账（Sardana & Frankle 2023 的思路）")
    print("  目标：达到同一个 loss；总算力 = 训练 6ND + 推理 2N × 一生要生成的 token 数")
    L_target = L_ours
    print(f"  目标 loss = {L_target:.4f}（即上面主线的 loss）")
    print(f"{'推理 token':>10} | {'最省总算力的 N':>12} {'D':>9} {'token/参数':>9} {'总算力':>10}")
    Ns = np.logspace(8, 11, 3001)
    for D_inf in [0, 1e12, 1e13, 1e14]:
        best = None
        for N in Ns:
            gap = L_target - fit["E"] - fit["A"] / N ** fit["alpha"]
            if gap <= 0:
                continue  # 这个 N 太小，数据再多也到不了目标
            D = (fit["B"] / gap) ** (1 / fit["beta"])
            total = 6 * N * D + 2 * N * D_inf
            if best is None or total < best[0]:
                best = (total, N, D)
        total, N, D = best
        print(f"{D_inf:>10.0e} | {N / 1e9:>11.2f}B {D / 1e9:>8.0f}B {D / N:>9.0f} {total:>10.3g}")
    print("  → 要服务的 token 越多，越该用更小的模型、训更久：这就是小模型\"过训练\"的原因。")

    print("\n④ 真实模型的 token/参数（来源见 README）")
    for name, n, d in [
        ("Chinchilla 70B", 70e9, 1.4e12),
        ("Llama 3 8B", 8e9, 15e12),
        ("Puro-2B", 2.0e9, 1.4e12),
        ("MobileLLM-R1-950M", 0.949e9, 4.2e12),
        ("Qwen3-0.6B", 0.6e9, 36e12),
        ("本课主线（计划）", 0.6895e9, 400e9),
    ]:
        print(f"  {name:<18} {d / n:>8,.0f} token/参数（约为 20 的 {d / n / 20:,.0f} 倍）")


if __name__ == "__main__":
    main()
