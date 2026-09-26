"""第 6 章 · 极简代码 4：优化器 —— SGD → 动量 → Adam → AdamW

实验 1：一个"各个方向陡峭程度差 1000 倍"的碗 L(θ) = ½ Σ hᵢ θᵢ²，
        比较 SGD、动量、Adam 每个参数每一步走多远、200 步后还剩多少损失。
实验 2：权重衰减。同样的 λ，"Adam + L2 正则"和 AdamW（解耦权重衰减）
        对梯度大小不同的参数，衰减力度一样吗？
运行：uv run python chapters/06-training-stability/code/04_optimizers.py
"""

import numpy as np

H = np.array([100.0, 10.0, 1.0, 0.1])   # 四个参数方向上的曲率（碗的陡峭程度）


def grad(theta: np.ndarray) -> np.ndarray:
    """L = ½ Σ hᵢ θᵢ² 的梯度：∂L/∂θᵢ = hᵢ θᵢ。"""
    return H * theta


def loss(theta: np.ndarray) -> float:
    return float(0.5 * np.sum(H * theta**2))


def sgd(steps, lr):
    theta, hist = np.ones(4), []
    for _ in range(steps):
        step = -lr * grad(theta)                    # θ ← θ − η·g
        theta = theta + step
        hist.append((theta.copy(), step))
    return hist


def momentum(steps, lr, beta=0.9):
    theta, v, hist = np.ones(4), np.zeros(4), []
    for _ in range(steps):
        v = beta * v + grad(theta)                  # v ← β·v + g   （速度：梯度的累积）
        step = -lr * v                              # θ ← θ − η·v
        theta = theta + step
        hist.append((theta.copy(), step))
    return hist


def adam(steps, lr, b1=0.9, b2=0.95, eps=1e-8):
    theta, m, v, hist = np.ones(4), np.zeros(4), np.zeros(4), []
    for t in range(1, steps + 1):
        g = grad(theta)
        m = b1 * m + (1 - b1) * g                   # 一阶矩：梯度的滑动平均（方向）
        v = b2 * v + (1 - b2) * g * g               # 二阶矩：梯度平方的滑动平均（尺度）
        m_hat, v_hat = m / (1 - b1**t), v / (1 - b2**t)   # 偏差修正
        step = -lr * m_hat / (np.sqrt(v_hat) + eps)       # 每个参数除以自己的梯度尺度
        theta = theta + step
        hist.append((theta.copy(), step))
    return hist


def weight_decay_demo(decoupled: bool, steps=3000, lr=0.01, wd=0.1, n=1000, seed=0):
    """两组参数，各 n 个，初值都是 1。损失本身不提供任何信号，梯度只是噪声：
    A 组噪声标准差 0.01（梯度小），B 组 10（梯度大）。权重衰减本应把两组一样地往 0 拉。
    返回两组参数的平均值。"""
    rng = np.random.default_rng(seed)
    sigma = np.repeat([0.01, 10.0], n)
    w, m, v = np.ones(2 * n), np.zeros(2 * n), np.zeros(2 * n)
    b1, b2 = 0.9, 0.95
    for t in range(1, steps + 1):
        g = rng.normal(0, sigma)
        if not decoupled:
            g = g + wd * w                          # Adam + L2：衰减项混进梯度，一起被除以 √v
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        if decoupled:
            w = w - lr * wd * w                     # AdamW：衰减单独做，不经过 √v
        w = w - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + 1e-8)
    return w[:n].mean(), w[n:].mean()


if __name__ == "__main__":
    lr_sgd = 0.019                                  # SGD 的临界学习率是 2/100 = 0.02
    runs = {
        "SGD": sgd(200, lr_sgd),
        "动量 (β=0.9)": momentum(200, lr_sgd),
        "Adam": adam(200, 0.05),
    }
    print("碗的曲率 h =", H.tolist(), "（最陡和最平的方向差 1000 倍），起点 θ = [1, 1, 1, 1]")
    print(f"SGD 与动量 η = {lr_sgd}（再大一点，最陡的方向就发散），Adam η = 0.05\n")
    print("第 1 步，每个参数走了多远 |Δθᵢ|：")
    for name, hist in runs.items():
        print(f"  {name:<12s}" + "".join(f"{abs(s):>10.4f}" for s in hist[0][1]))
    print("\n200 步之后：")
    for name, hist in runs.items():
        th = hist[-1][0]
        hit = next((i + 1 for i, (t, _) in enumerate(hist) if loss(t) < 1e-3), None)
        print(f"  {name:<12s} θ = [" + ", ".join(f"{x:8.4f}" for x in th) + f"]   损失 = {loss(th):.2e}"
              f"   损失 < 0.001 用了 {hit if hit else '超过 200'} 步")
    print(f"  （起点损失 = {loss(np.ones(4)):.2f}）\n")

    print("权重衰减 λ = 0.1、η = 0.01、3000 步，梯度只有噪声；两组参数的平均值：")
    print("                    A 组（梯度噪声 0.01）   B 组（梯度噪声 10）")
    for name, dec in [("Adam + L2 正则", False), ("AdamW（解耦）", True)]:
        a, b = weight_decay_demo(dec)
        print(f"  {name:<16s}{a:>16.3f}{b:>22.3f}")
    print(f"  只看衰减项，理论值 (1 − ηλ)^3000 = {(1 - 0.01 * 0.1) ** 3000:.3f}")
