"""第 3 章 · 极简代码 1：直线拟合不了曲线，线性层叠多少层也还是直线

两件事：
1. 用最小二乘找到"最好的直线"去拟合 y = sin(2x)，看看它能好到哪里。
2. 把两层（甚至三层）线性层叠起来：(X·W1 + b1)·W2 + b2 = X·(W1·W2) + (b1·W2 + b2)，
   用数字验证它和"一层线性层"完全等价。
只用 NumPy，CPU 上不到 1 秒跑完。
运行：uv run python chapters/03-neural-network/code/01_linear_is_not_enough.py
"""

import numpy as np


def make_data(n: int = 100):
    """本章的数据：x 在 [−3, 3] 上均匀取 100 个点，y = sin(2x)。形状都是 (N, 1)。"""
    x = np.linspace(-3, 3, n).reshape(-1, 1)
    y = np.sin(2 * x)
    return x, y


def best_line(x: np.ndarray, y: np.ndarray):
    """最小二乘：所有直线里 MSE 最小的那一条（第 1 章的解析解）。"""
    a, b = np.polyfit(x[:, 0], y[:, 0], deg=1)
    mse = float(np.mean((a * x + b - y) ** 2))
    return a, b, mse


def collapse(layers):
    """把一串线性层 [(W1, b1), (W2, b2), ...] 合并成一层 (W, b)。

    (X·W1 + b1)·W2 + b2 = X·(W1·W2) + (b1·W2 + b2)，再往后叠也一样。
    """
    W, b = layers[0]
    for W_next, b_next in layers[1:]:
        W, b = W @ W_next, b @ W_next + b_next
    return W, b


def forward_stack(x: np.ndarray, layers) -> np.ndarray:
    """老老实实一层一层算：h ← h·W + b。"""
    h = x
    for W, b in layers:
        h = h @ W + b
    return h


def random_stacks(seed: int = 0):
    """随机造两个纯线性网络：两层 1→8→1、三层 1→8→8→1（每层 h ← h·W + b，中间不加激活）。"""
    rng = np.random.default_rng(seed)
    two = [(rng.normal(size=(1, 8)), rng.normal(size=8)),      # 第 1 层：1 → 8
           (rng.normal(size=(8, 1)), rng.normal(size=1))]      # 第 2 层：8 → 1
    three = [(rng.normal(size=(1, 8)), rng.normal(size=8)),
             (rng.normal(size=(8, 8)), rng.normal(size=8)),
             (rng.normal(size=(8, 1)), rng.normal(size=1))]
    return {"两层 1→8→1": two, "三层 1→8→8→1": three}


def stack_demo(x: np.ndarray):
    """对每个随机线性网络：参数量、合并后的 (W, b)、逐层算与合并算的最大差。"""
    rows = []
    for name, layers in random_stacks().items():
        W, b = collapse(layers)
        diff = float(np.max(np.abs(forward_stack(x, layers) - (x @ W + b))))
        n_params = sum(Wi.size + bi.size for Wi, bi in layers)
        rows.append((name, n_params, W, b, diff))
    return rows


if __name__ == "__main__":
    x, y = make_data()
    a, b, mse = best_line(x, y)
    print("1) 用直线拟合 y = sin(2x)")
    print(f"   最好的直线：y = {a:.3f}·x + {b:.3f}，MSE = {mse:.4f}")
    print(f"   对照：y 的方差 = {float(np.var(y)):.4f}（直接猜平均值的 MSE）")

    print("\n2) 线性层叠起来，还是一层线性层")
    for name, n_params, W, bb, diff in stack_demo(x):
        print(f"   {name}：{n_params} 个参数，合并后 y = {W.item():.3f}·x {bb.item():+.3f}，"
              f"逐层算 vs 合并算 最大差 = {diff:.1e}")
    print("   → 参数再多，函数图像仍是一条直线，MSE 不可能低于上面那条最好的直线。")
