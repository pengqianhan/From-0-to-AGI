"""第 25 章 · 极简代码 4：加速比公式——接受率 α、草稿长度 k、草稿成本 c

假设每个草稿 token 被接受的概率都是 α（相互独立），一轮最多产出 k+1 个 token：
  第 1 个 token 一定有（纠正或奖励），第 2 个要第 1 个草稿被接受（概率 α），第 3 个要前两个都被接受（α²）……
  E[每轮产出] = 1 + α + α² + … + α^k = (1 − α^{k+1}) / (1 − α)            （Leviathan 等 2023，式 (1)）
每轮的代价 = 目标模型 1 次前向 + 草稿 k 次前向 = (1 + k·c) 个"目标步"，c = 草稿一步 / 目标一步：
  加速比 = (1 − α^{k+1}) / ((1 − α)(1 + k·c))                              （定理 3.8）
前提：目标模型一次验证 k+1 个位置和生成 1 个 token 一样快（decode 受带宽限制时近似成立）。

运行：uv run python chapters/25-mtp-speculative-decoding/code/04_speedup_formula.py
"""

from __future__ import annotations


def expected_tokens(alpha: float, k: int) -> float:
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def speedup(alpha: float, k: int, c: float) -> float:
    return expected_tokens(alpha, k) / (1 + k * c)


def best_k(alpha: float, c: float, k_max: int = 32) -> tuple[int, float]:
    return max(((k, speedup(alpha, k, c)) for k in range(1, k_max + 1)), key=lambda t: t[1])


if __name__ == "__main__":
    alphas = (0.5, 0.6, 0.7, 0.8, 0.9)
    print("每轮期望产出的 token 数 E = (1 − α^{k+1}) / (1 − α)：")
    print("  α \\ k " + "".join(f"{k:>7d}" for k in (1, 2, 3, 4, 6, 8)) + "    k→∞")
    for a in alphas:
        row = "".join(f"{expected_tokens(a, k):7.2f}" for k in (1, 2, 3, 4, 6, 8))
        print(f"  {a:4.2f}  {row}  {1 / (1 - a):6.2f}")

    for c in (0.0, 0.05, 0.2):
        print(f"\n加速比（c = {c}）：")
        print("  α \\ k " + "".join(f"{k:>7d}" for k in (1, 2, 3, 4, 6, 8)) + "   最优 k")
        for a in alphas:
            row = "".join(f"{speedup(a, k, c):7.2f}" for k in (1, 2, 3, 4, 6, 8))
            kb, sb = best_k(a, c)
            best = f"k 越大越好（上限 {1 / (1 - a):.2f}×）" if c == 0 else f"k={kb}（{sb:.2f}×）"
            print(f"  {a:4.2f}  {row}   {best}")

    print("\n对照论文里的数字：")
    print(
        f"  Leviathan 等表 1：α=0.8、k=5、c=0 → {speedup(0.8, 5, 0):.2f}×（论文 3.69×）；"
        f"α=0.9、k=10 → {speedup(0.9, 10, 0):.2f}×（论文 6.86×）"
    )
    for a in (0.85, 0.90):
        print(
            f"  DeepSeek-V3 的 MTP 自推测（k=1）：第二个 token 接受率 {a:.2f} → 每次前向 "
            f"{expected_tokens(a, 1):.2f} 个 token（论文报告 TPS 提到 1.8 倍）"
        )
    print("\n最优 k 的规律：α 越高、c 越小，越值得多猜几个；α 低时猜多了只是白白浪费草稿的算力。")
