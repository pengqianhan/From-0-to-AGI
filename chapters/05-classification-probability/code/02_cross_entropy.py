"""第 5 章 · 极简代码 2：最大似然 → 交叉熵，以及它的梯度 p − onehot

1. 最大似然：让模型给"正确答案"的概率尽量大；连乘会下溢，所以取 log。
2. 负对数似然（NLL）= 交叉熵（cross-entropy）。
3. softmax + 交叉熵对 logits 的梯度就是 p − onehot，用数值梯度验证。
4. 为什么分类不用 MSE：模型"自信地错了"时，MSE 的梯度几乎为 0。
运行：uv run python chapters/05-classification-probability/code/02_cross_entropy.py
"""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("softmax_mod", Path(__file__).with_name("01_softmax.py"))
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)
softmax = sm.softmax


def log_softmax(z: np.ndarray) -> np.ndarray:
    """log p_k = z_k − logsumexp(z)，同样先减最大值，避免 exp 溢出、也避免 log(0)。"""
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def cross_entropy(logits: np.ndarray, y: np.ndarray) -> float:
    """交叉熵 = 平均负对数似然：L = −1/N · Σ_i log p_{i, y_i}。logits 形状 (N, K)，y 形状 (N,)。"""
    logits = np.atleast_2d(logits)
    y = np.atleast_1d(y)
    return float(-log_softmax(logits)[np.arange(len(y)), y].mean())


def ce_grad(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """∂L/∂z = (p − onehot(y)) / N —— 整章最重要的一行。"""
    logits = np.atleast_2d(logits)
    y = np.atleast_1d(y)
    p = softmax(logits)
    p[np.arange(len(y)), y] -= 1.0
    return p / len(y)


def mse_on_probs(logits: np.ndarray, y: np.ndarray) -> float:
    """对照组：把 softmax 概率和 onehot 做均方误差，L = 1/N · Σ_i Σ_k (p_ik − t_ik)²。"""
    logits = np.atleast_2d(logits)
    y = np.atleast_1d(y)
    p = softmax(logits)
    t = np.eye(p.shape[1])[y]
    return float(((p - t) ** 2).sum(axis=1).mean())


def mse_grad(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """MSE 对 logits 的梯度：链式法则穿过 softmax 的雅可比 J = diag(p) − p pᵀ。
    ∂L/∂z = J · 2(p − t) = 2 · p ⊙ (g − Σ_k p_k g_k)，其中 g = p − t。
    """
    logits = np.atleast_2d(logits)
    y = np.atleast_1d(y)
    p = softmax(logits)
    g = p - np.eye(p.shape[1])[y]
    return 2 * p * (g - (p * g).sum(axis=1, keepdims=True)) / len(y)


def numerical_grad(f, z: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """中心差分：(f(z + ε) − f(z − ε)) / 2ε，逐个元素算（第 1、4 章的梯度检验）。"""
    g = np.zeros_like(z)
    for idx in np.ndindex(z.shape):
        zp, zm = z.copy(), z.copy()
        zp[idx] += eps
        zm[idx] -= eps
        g[idx] = (f(zp) - f(zm)) / (2 * eps)
    return g


def confident_wrong_table(gaps=(0, 2, 4, 6, 8, 10)):
    """真实类别是 0，模型把错误类别 1 的 logit 抬到 s：s 越大，错得越自信。
    返回每个 s 下：p(正确)、CE 损失、CE 梯度大小、MSE 损失、MSE 梯度大小。
    """
    rows = []
    for s in gaps:
        z = np.array([[0.0, float(s), 0.0]])
        y = np.array([0])
        rows.append((
            s,
            softmax(z)[0, 0],
            cross_entropy(z, y),
            np.linalg.norm(ce_grad(z, y)),
            mse_on_probs(z, y),
            np.linalg.norm(mse_grad(z, y)),
        ))
    return rows


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    logits, classes = sm.LOGITS, sm.CLASSES
    p = softmax(logits)

    print("—— 1. 似然：连乘会下溢 ——")
    for n in [10, 100, 1000, 10000]:
        with np.errstate(under="ignore"):
            prod = np.prod(np.full(n, 0.9))
        print(f"{n:>6} 个样本、每个都给正确答案 0.9 的概率：似然 = 0.9^{n} = {prod:.3e}"
              f"，对数似然 = {n * np.log(0.9):.2f}")

    print("\n—— 2. 交叉熵 = −log p(正确类别) ——")
    print("logits =", logits, " softmax =", p)
    for k, name in enumerate(classes):
        print(f"正确答案是「{name}」：p = {p[k]:.4f}，损失 = −ln p = {cross_entropy(logits, np.array([k])):.4f}")
    print(f"什么都不懂时（三类均匀猜）：−ln(1/3) = {np.log(3):.4f}")

    print("\n—— 3. 梯度 = p − onehot，数值验证 ——")
    y = np.array([0])
    z = logits[None, :].copy()
    g_ana = ce_grad(z, y)
    g_num = numerical_grad(lambda t: cross_entropy(t, y), z)
    print("解析梯度 p − onehot：", g_ana[0])
    print("数值梯度（中心差分）：", g_num[0])
    print(f"最大差：{np.abs(g_ana - g_num).max():.2e}")
    rng = np.random.default_rng(0)
    zb, yb = rng.normal(0, 3, size=(8, 5)), rng.integers(0, 5, size=8)
    diff = np.abs(ce_grad(zb, yb) - numerical_grad(lambda t: cross_entropy(t, yb), zb)).max()
    diff_mse = np.abs(mse_grad(zb, yb) - numerical_grad(lambda t: mse_on_probs(t, yb), zb)).max()
    print(f"随机 8 个样本 × 5 类：CE 解析与数值梯度最大差 {diff:.2e}；MSE 最大差 {diff_mse:.2e}")

    print("\n—— 4. 自信地错了：CE vs MSE 的梯度 ——")
    print("错误类别 logit s   p(正确)     CE 损失   |CE 梯度|   MSE 损失   |MSE 梯度|")
    for s, pt, ce, gce, mse, gmse in confident_wrong_table():
        print(f"{s:>12}      {pt:9.2e}   {ce:7.3f}   {gce:9.4f}   {mse:8.4f}   {gmse:10.2e}")
