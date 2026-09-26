"""第 2 章 · 极简代码 4：多元线性回归 —— 矩阵形式的梯度下降

第 1 章：一个输入 x，两个参数 a、b，ŷ = a·x + b。
这一章：3 个输入（面积、卧室数、距市中心距离），ŷ = X @ W + b。
损失、梯度、更新四步完全不变，只是把标量换成了矩阵：
    ŷ     = X @ W + b                 (N, 3) @ (3, 1) + (1,) → (N, 1)
    L     = 1/N · Σ (ŷ − y)²
    ∂L/∂W = 2/N · Xᵀ @ (ŷ − y)        (3, N) @ (N, 1) → (3, 1)，和 W 同形状
    ∂L/∂b = 2/N · Σ (ŷ − y)
运行：uv run python chapters/02-from-scalar-to-matrix/code/04_multivariate_regression.py
"""

import numpy as np

FEATURES = ["面积(㎡)", "卧室数", "距市中心(km)"]
W_TRUE = np.array([[0.8], [5.0], [-3.0]])   # 真实权重：每平米 0.8 万，每间卧室 5 万，每远 1km 少 3 万
B_TRUE = 20.0


def make_data(n: int = 200, noise: float = 5.0, seed: int = 0):
    """造 n 套房子。X: (n, 3)，y: (n, 1)，价格单位：万元。"""
    rng = np.random.default_rng(seed)
    area = rng.uniform(50, 150, size=n)
    rooms = rng.integers(1, 5, size=n).astype(float)
    dist = rng.uniform(1, 20, size=n)
    X = np.stack([area, rooms, dist], axis=1)
    y = X @ W_TRUE + B_TRUE + rng.normal(0, noise, size=(n, 1))
    return X, y


def standardize(X: np.ndarray):
    """每个特征减均值、除标准差，让三个特征的尺度一样（原因见 critical_lr）。"""
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    return (X - mu) / sigma, mu, sigma


def critical_lr(X: np.ndarray) -> float:
    """第 1 章的临界学习率 2/λ_max，推广到多个特征：Hessian = 2/N · [X 1]ᵀ[X 1]。"""
    A = np.hstack([X, np.ones((len(X), 1))])
    lam_max = np.linalg.eigvalsh(2 / len(X) * A.T @ A).max()
    return 2 / lam_max


def mse(W, b, X, y) -> float:
    return float(np.mean((X @ W + b - y) ** 2))


def gradients(W, b, X, y):
    """矩阵形式的梯度：一行代码算出所有权重的梯度。"""
    n = len(X)
    err = X @ W + b - y                  # (N, 1)：残差 ŷ − y
    grad_W = 2 / n * X.T @ err           # (3, N) @ (N, 1) → (3, 1)
    grad_b = 2 / n * err.sum(axis=0)     # (1,)
    return grad_W, grad_b


def gradient_descent(X, y, lr: float = 0.1, steps: int = 200):
    """返回每一步的 (W, b, loss)。起点：W 全 0，b = 0。"""
    W = np.zeros((X.shape[1], 1))
    b = np.zeros(1)
    history = [(W.copy(), b.copy(), mse(W, b, X, y))]
    for _ in range(steps):
        grad_W, grad_b = gradients(W, b, X, y)
        W = W - lr * grad_W              # W ← W − η · ∂L/∂W
        b = b - lr * grad_b              # b ← b − η · ∂L/∂b
        history.append((W.copy(), b.copy(), mse(W, b, X, y)))
    return history


def to_original_units(W, b, mu, sigma):
    """标准化空间里的 (W, b) 换算回原始单位：ŷ = ((X − μ)/σ) @ W + b。"""
    W_orig = W / sigma[:, None]
    b_orig = b - (mu / sigma) @ W
    return W_orig, b_orig


if __name__ == "__main__":
    X, y = make_data()
    Xs, mu, sigma = standardize(X)
    print(f"X 形状 {X.shape}，y 形状 {y.shape}")
    print(f"三个特征的标准差：{np.round(sigma, 2)}（尺度差很多）")
    print(f"临界学习率 2/λ_max：原始特征 {critical_lr(X):.2e}，标准化后 {critical_lr(Xs):.3f}\n")

    history = gradient_descent(Xs, y, lr=0.1, steps=200)
    print("步数     w_面积    w_卧室    w_距离        b        损失")
    for step in [0, 1, 5, 10, 20, 50, 100, 200]:
        W, b, loss = history[step]
        w = W.ravel()
        print(f"{step:>4}  {w[0]:8.3f}  {w[1]:8.3f}  {w[2]:8.3f}  {b[0]:8.3f}  {loss:10.3f}")

    W, b, _ = history[-1]
    W_o, b_o = to_original_units(W, b, mu, sigma)

    # 对照：最小二乘解析解
    A = np.hstack([X, np.ones((len(X), 1))])
    sol = np.linalg.lstsq(A, y, rcond=None)[0]

    print("\n换算回原始单位（万元）：")
    print(f"{'':14s}{'梯度下降':>10s}{'解析解':>10s}{'真实值':>10s}")
    for i, name in enumerate(FEATURES):
        print(f"w_{name:12s}{W_o[i, 0]:10.3f}{sol[i, 0]:10.3f}{W_TRUE[i, 0]:10.3f}")
    print(f"{'b':14s}{b_o[0]:10.3f}{sol[3, 0]:10.3f}{B_TRUE:10.3f}")

    new = np.array([[100.0, 3.0, 5.0]])
    pred = ((new - mu) / sigma) @ W + b
    print(f"\n新房子 100㎡、3 室、距市中心 5km：预测 {pred[0, 0]:.1f} 万元")
