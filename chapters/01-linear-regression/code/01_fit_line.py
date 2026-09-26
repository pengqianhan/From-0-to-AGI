"""第 1 章 · 极简代码 1：用梯度下降拟合一条直线 y = ax + b

只用 NumPy，CPU 上不到 1 秒跑完。
运行：uv run python chapters/01-linear-regression/code/01_fit_line.py
"""

import numpy as np


def make_data(n: int = 50, a_true: float = 2.0, b_true: float = 1.0, noise: float = 0.5,
              seed: int = 0):
    """造一批数据：真实关系是 y = 2x + 1，再加一点噪声。"""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 5, size=n)
    y = a_true * x + b_true + rng.normal(0, noise, size=n)
    return x, y


def mse_loss(a: float, b: float, x: np.ndarray, y: np.ndarray) -> float:
    """均方误差 L(a, b) = 1/N · Σ (a·x_i + b − y_i)²"""
    y_hat = a * x + b
    return float(np.mean((y_hat - y) ** 2))


def gradients(a: float, b: float, x: np.ndarray, y: np.ndarray):
    """手算梯度：
    ∂L/∂a = 2/N · Σ (ŷ_i − y_i) · x_i
    ∂L/∂b = 2/N · Σ (ŷ_i − y_i)
    """
    err = (a * x + b) - y
    grad_a = 2 * np.mean(err * x)
    grad_b = 2 * np.mean(err)
    return grad_a, grad_b


def gradient_descent(x, y, lr: float = 0.05, steps: int = 200, a0: float = -1.0,
                     b0: float = 4.0):
    """梯度下降：沿着梯度的反方向走一小步，重复很多次。返回每一步的 (a, b, loss)。"""
    a, b = a0, b0
    history = [(a, b, mse_loss(a, b, x, y))]
    for _ in range(steps):
        grad_a, grad_b = gradients(a, b, x, y)
        a = a - lr * grad_a          # a ← a − η · ∂L/∂a
        b = b - lr * grad_b          # b ← b − η · ∂L/∂b
        history.append((a, b, mse_loss(a, b, x, y)))
    return history


if __name__ == "__main__":
    x, y = make_data()
    history = gradient_descent(x, y)

    print("步数     a        b        损失")
    for step in [0, 1, 2, 5, 10, 20, 50, 100, 200]:
        a, b, loss = history[step]
        print(f"{step:>4}  {a:7.3f}  {b:7.3f}  {loss:9.4f}")

    # 对照：最小二乘的解析解（只有线性模型才有这种"一步到位"的公式）
    a_ls, b_ls = np.polyfit(x, y, deg=1)
    print(f"\n解析解（np.polyfit）：a = {a_ls:.3f}, b = {b_ls:.3f}")
    print("真实值：            a = 2.000, b = 1.000（有噪声，所以不会完全一样）")
