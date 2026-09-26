"""第 2 章 · 极简代码 3：Y = X @ W + b —— 线性层就是一次矩阵乘法

场景：4 套房子，每套 3 个特征（面积、卧室数、距市中心距离）。
先用一层 (3 → 2)，再用一层 (2 → 1)，看清每一步的形状；
最后验证：两层线性层叠在一起，等于一层线性层（第 3 章的伏笔）。
运行：uv run python chapters/02-from-scalar-to-matrix/code/03_linear_layer.py
"""

import numpy as np

# 4 套房子，每套 3 个特征。形状 (4, 3)：4 个样本（batch 维），每个样本 3 个特征
HOUSES = np.array([
    [80,  2, 5.0],   # 80㎡，2 室，距市中心 5km
    [120, 3, 2.0],   # 120㎡，3 室，距市中心 2km
    [60,  1, 8.0],   # 60㎡，1 室，距市中心 8km
    [100, 3, 3.5],   # 100㎡，3 室，距市中心 3.5km
])


def linear(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Y = X @ W + b。X: (batch, k)，W: (k, n)，b: (n,) → Y: (batch, n)。"""
    return X @ W + b


def init_params(seed: int = 42):
    """两层的参数。W1: (3, 2)，第 j 列 = 第 j 个输出用的 3 个权重；b1: (2,)，每个输出一个偏置。"""
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((3, 2)) * 0.1
    b1 = np.array([1.0, -1.0])
    W2 = rng.standard_normal((2, 1)) * 0.1
    b2 = np.array([0.5])
    return W1, b1, W2, b2


if __name__ == "__main__":
    X = HOUSES
    print(f"输入 X：形状 {X.shape}（4 套房子 × 3 个特征）")

    W1, b1, W2, b2 = init_params()

    # ── 第一层：3 维 → 2 维 ────────────────────────────────────────────
    H = linear(X, W1, b1)
    print("\n第一层 H = X @ W1 + b1")
    print(f"  {X.shape} @ {W1.shape} → {(X @ W1).shape}，再 + b1 {b1.shape}（广播到每一行）→ {H.shape}")
    print(f"  H =\n{H.round(3)}")

    # ── 第二层：2 维 → 1 维 ────────────────────────────────────────────
    Y = linear(H, W2, b2)
    print("\n第二层 Y = H @ W2 + b2")
    print(f"  {H.shape} @ {W2.shape} → {Y.shape}")
    print(f"  Y = {Y.ravel().round(4)}")

    print("\n数据流：(4, 3) --[@W1 + b1]--> (4, 2) --[@W2 + b2]--> (4, 1)")

    # ── 伏笔：两层线性 = 一层线性 ──────────────────────────────────────
    W = W1 @ W2              # (3, 1)
    b = b1 @ W2 + b2         # (1,)
    Y_one = linear(X, W, b)
    print("\n把两层合并成一层：W = W1 @ W2，b = b1 @ W2 + b2")
    print(f"  合并后的 W 形状 {W.shape}，b 形状 {b.shape}")
    print(f"  两层的输出与一层的输出最大差 = {np.abs(Y - Y_one).max():.1e}")
    print("  结论：线性层叠多少层，都还是一个线性函数。要拟合曲线，得加点“非线性”——这是第 3 章的事。")
