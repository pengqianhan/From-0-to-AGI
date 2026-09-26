"""第 3 章 · 极简代码 3：两层 MLP 拟合 y = sin(2x)，手推梯度 + 梯度下降

模型（行向量写法，和第 2 章的 y = XW + b 一致）：
    Z = X·W1 + b1        (N, 1) → (N, H)   第 1 层：线性
    A = ReLU(Z)          (N, H)            激活：把每个数小于 0 的部分砍成 0
    Ŷ = A·W2 + b2        (N, H) → (N, 1)   第 2 层：线性
    L = mean((Ŷ − Y)²)                     损失：和第 1 章一样的均方误差
梯度是用链式法则手推的（第 4 章会系统讲反向传播，这里先用、并用数值梯度验证它没推错）。
只用 NumPy，CPU 上约 25 秒跑完（大部分时间花在第 3 部分的多种子实验）。
运行：uv run python chapters/03-neural-network/code/03_mlp_numpy.py
"""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "linear_is_not_enough", Path(__file__).with_name("01_linear_is_not_enough.py")
)
lin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lin)
make_data = lin.make_data  # x ∈ [−3, 3] 上 100 个点，y = sin(2x)

LR = 0.01
STEPS = 20000


def init_params(hidden: int, seed: int = 0) -> dict:
    """随机初始化。b1 的取法让每个隐藏单元的折点（x = −b1/W1）一开始就散落在 [−3, 3] 里。
    （初始化有很多讲究，第 6 章再系统讲。）"""
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1, size=(1, hidden))
    b1 = rng.uniform(-3, 3, size=hidden) * np.abs(W1[0])
    W2 = rng.normal(0, 1 / np.sqrt(hidden), size=(hidden, 1))
    b2 = np.zeros(1)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def act_fn(z: np.ndarray, act: str) -> np.ndarray:
    return np.maximum(0.0, z) if act == "relu" else z        # "linear"：不用激活


def act_grad(z: np.ndarray, act: str) -> np.ndarray:
    return (z > 0).astype(z.dtype) if act == "relu" else np.ones_like(z)


def forward(p: dict, x: np.ndarray, act: str = "relu"):
    z = x @ p["W1"] + p["b1"]         # Z = X·W1 + b1
    a = act_fn(z, act)                # A = ReLU(Z)
    y_hat = a @ p["W2"] + p["b2"]     # Ŷ = A·W2 + b2
    return y_hat, (z, a)


def mse(p: dict, x, y, act: str = "relu") -> float:
    y_hat, _ = forward(p, x, act)
    return float(np.mean((y_hat - y) ** 2))


def gradients(p: dict, x, y, act: str = "relu"):
    """手推梯度：从损失往回，一层一层用链式法则。返回 (损失, 梯度字典)。"""
    n = len(x)
    y_hat, (z, a) = forward(p, x, act)
    loss = float(np.mean((y_hat - y) ** 2))
    d_yhat = 2 * (y_hat - y) / n          # ∂L/∂Ŷ：和第 1 章的 2/N·(ŷ − y) 一样
    d_W2 = a.T @ d_yhat                   # ∂L/∂W2 = Aᵀ · ∂L/∂Ŷ
    d_b2 = d_yhat.sum(axis=0)             # ∂L/∂b2 = Σ ∂L/∂Ŷ
    d_a = d_yhat @ p["W2"].T              # ∂L/∂A  = ∂L/∂Ŷ · W2ᵀ
    d_z = d_a * act_grad(z, act)          # ∂L/∂Z  = ∂L/∂A ⊙ ReLU′(Z)：没激活的单元梯度为 0
    d_W1 = x.T @ d_z                      # ∂L/∂W1 = Xᵀ · ∂L/∂Z
    d_b1 = d_z.sum(axis=0)                # ∂L/∂b1 = Σ ∂L/∂Z
    return loss, {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}


def numerical_gradients(p: dict, x, y, act: str = "relu", eps: float = 1e-6) -> dict:
    """数值梯度：(L(θ+ε) − L(θ−ε)) / 2ε，逐个参数算。慢，但不会推错，用来检验手推的公式。"""
    grads = {}
    for k, v in p.items():
        g = np.zeros_like(v)
        for idx in np.ndindex(v.shape):
            old = v[idx]
            v[idx] = old + eps
            lp = mse(p, x, y, act)
            v[idx] = old - eps
            lm = mse(p, x, y, act)
            v[idx] = old
            g[idx] = (lp - lm) / (2 * eps)
        grads[k] = g
    return grads


def train(x, y, hidden: int, lr: float = LR, steps: int = STEPS, seed: int = 0,
          act: str = "relu", snapshot_steps=()):
    """梯度下降。返回 (最终参数, 每一步的损失, {步数: 当时的参数副本})。"""
    p = init_params(hidden, seed)
    losses, snaps = [], {}
    for step in range(steps + 1):
        if step in snapshot_steps:
            snaps[step] = {k: v.copy() for k, v in p.items()}
        loss, g = gradients(p, x, y, act)
        losses.append(loss)
        if step == steps:
            break
        for k in p:
            p[k] -= lr * g[k]             # θ ← θ − η · ∂L/∂θ，和第 1 章一模一样
    return p, losses, snaps


def pieces(p: dict, x: np.ndarray):
    """把网络输出拆成 H 个"折线片"：第 j 个隐藏单元贡献 W2[j] · ReLU(W1[j]·x + b1[j])。
    返回 (每片的值 (N, H), 每片的折点位置 (H,))。输出 = 各片之和 + b2。"""
    z = x @ p["W1"] + p["b1"]
    contrib = np.maximum(0.0, z) * p["W2"][:, 0]
    kinks = -p["b1"] / p["W1"][0]
    return contrib, kinks


def n_params(hidden: int) -> int:
    return 3 * hidden + 1                 # W1: H，b1: H，W2: H，b2: 1


if __name__ == "__main__":
    x, y = make_data()

    # 1) 梯度检验：手推的梯度和数值梯度对得上吗？
    p0 = init_params(8, seed=0)
    ga, gn = gradients(p0, x, y)[1], numerical_gradients(p0, x, y)
    err = max(np.max(np.abs(ga[k] - gn[k])) for k in p0)
    print(f"1) 梯度检验（宽度 8）：手推 vs 数值梯度，最大差 = {err:.1e}")

    # 2) 不同宽度，学习率 0.01，梯度下降 20000 步
    print(f"\n2) 拟合 y = sin(2x)，学习率 {LR}，梯度下降 {STEPS} 步（种子 0）")
    _, _, line_mse = lin.best_line(x, y)
    print(f"   {'模型':<18}{'参数量':>6}{'第0步':>9}{'第1000步':>10}{'第5000步':>10}{'第20000步':>10}")
    configs = [("两层线性，宽 8", 8, "linear"), ("ReLU，宽 2", 2, "relu"),
               ("ReLU，宽 8", 8, "relu"), ("ReLU，宽 64", 64, "relu")]
    for name, h, act in configs:
        _, losses, _ = train(x, y, h, act=act)
        cols = "".join(f"{losses[s]:>10.4f}" for s in (0, 1000, 5000, STEPS))
        print(f"   {name:<16}{n_params(h):>6}{cols}")
    print(f"   （对照：最好的直线 MSE = {line_mse:.4f}）")

    # 3) 换 5 个随机种子，看结论稳不稳
    print("\n3) 换 5 个随机种子（0–4），20000 步后的损失")
    for h in (2, 8, 64):
        finals = [train(x, y, h, seed=s)[1][-1] for s in range(5)]
        print(f"   宽 {h:>2}：" + "  ".join(f"{v:.4f}" for v in finals)
              + f"   中位数 {np.median(finals):.4f}")

    # 4) 把宽度 8 的网络拆开：输出 = 8 个折线片之和 + b2
    p8, _, _ = train(x, y, 8)
    contrib, kinks = pieces(p8, x)
    y_hat, _ = forward(p8, x)
    diff = np.max(np.abs(contrib.sum(axis=1, keepdims=True) + p8["b2"] - y_hat))
    order = np.argsort(kinks)
    print("\n4) 宽度 8 的网络：每个隐藏单元是一个折点")
    print("   折点位置 x = −b1/W1：" + "  ".join(f"{kinks[j]:.2f}" for j in order))
    print(f"   单条折线在数据范围内的最大幅度 = {np.max(np.abs(contrib)):.2f}"
          f"（网络输出的幅度只有 {np.max(np.abs(y_hat)):.2f}：各片互相抵消）")
    print(f"   各片之和 + b2 与网络输出的最大差 = {diff:.1e}")
