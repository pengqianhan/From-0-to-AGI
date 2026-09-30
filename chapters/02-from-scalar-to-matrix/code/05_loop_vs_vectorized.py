"""第 2 章 · 极简代码 5：循环版 vs 向量化版，到底差多少？

两个对比，结果都先核对一致，再比速度：
1. 一次前向 Y = X @ W：Python 三重循环 vs 每行一次 np.dot vs 一次 X @ W。
2. 多元线性回归（5000 套房子）训练 200 步：逐样本、逐特征循环算梯度 vs 矩阵形式 2/N·Xᵀ(ŷ − y)。
计时结果取决于机器，每次运行也会略有不同；看数量级即可。
运行：uv run python chapters/02-from-scalar-to-matrix/code/05_loop_vs_vectorized.py
"""

import importlib.util
import time
from pathlib import Path

import numpy as np


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mm = _load("matrix_multiply", "02_matrix_multiply.py")
reg = _load("multivariate_regression", "04_multivariate_regression.py")


def best_time(fn, repeat: int) -> float:
    """跑 repeat 次，取最快一次（秒），减少系统抖动的影响。"""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# ── 对比 1：前向 Y = X @ W ─────────────────────────────────────────────────

def forward_loop(X_list, W_list):
    return mm.matmul(X_list, W_list)                        # 三重循环，全在 Python 里


def forward_rows(X, W):
    return np.stack([X[i] @ W for i in range(len(X))])      # 只循环样本，每行交给 NumPy


def forward_vec(X, W):
    return X @ W                                            # 一次矩阵乘法


# ── 对比 2：训练 200 步 ────────────────────────────────────────────────────

def gradients_loop(W, b, X, y):
    """和 reg.gradients 算同一个东西，但逐样本、逐特征地累加。"""
    n, k = len(X), len(X[0])
    grad_W = [0.0] * k
    grad_b = 0.0
    for i in range(n):
        y_hat = b
        for j in range(k):
            y_hat += X[i][j] * W[j]                         # ŷ_i = Σ_j x_ij·w_j + b
        err = y_hat - y[i]
        for j in range(k):
            grad_W[j] += 2 / n * err * X[i][j]              # ∂L/∂w_j = 2/N Σ_i (ŷ_i − y_i)·x_ij
        grad_b += 2 / n * err
    return grad_W, grad_b


def train_loop(X_list, y_list, lr=0.1, steps=200):
    k = len(X_list[0])
    W, b = [0.0] * k, 0.0
    for _ in range(steps):
        gW, gb = gradients_loop(W, b, X_list, y_list)
        W = [W[j] - lr * gW[j] for j in range(k)]
        b = b - lr * gb
    return W, b


def train_vec(X, y, lr=0.1, steps=200):
    W, b, _ = reg.gradient_descent(X, y, lr=lr, steps=steps)[-1]
    return W, b


def run_benchmarks():
    rng = np.random.default_rng(0)
    results = {}

    # 对比 1：batch = 1000 个样本，100 个输入特征，10 个输出
    X = rng.standard_normal((1000, 100))
    W = rng.standard_normal((100, 10))
    X_list, W_list = X.tolist(), W.tolist()
    Y_ref = forward_vec(X, W)
    assert np.allclose(np.array(forward_loop(X_list, W_list)), Y_ref)
    assert np.allclose(forward_rows(X, W), Y_ref)
    results["forward"] = {
        "shape": (X.shape, W.shape),
        "loop": best_time(lambda: forward_loop(X_list, W_list), 3),
        "rows": best_time(lambda: forward_rows(X, W), 20),
        "vec": best_time(lambda: forward_vec(X, W), 200),
    }

    # 对比 2：用第 4 个脚本的造数据方法，换成 5000 套房子（3 个特征，标准化后）
    Xh, yh = reg.make_data(n=5000)
    Xs, _, _ = reg.standardize(Xh)
    Xs_list, y_list = Xs.tolist(), yh.ravel().tolist()
    W_l, b_l = train_loop(Xs_list, y_list)
    W_v, b_v = train_vec(Xs, yh)
    diff = max(np.abs(np.array(W_l) - W_v.ravel()).max(), abs(b_l - b_v[0]))
    results["train"] = {
        "shape": Xs.shape,
        "loop": best_time(lambda: train_loop(Xs_list, y_list), 2),
        "vec": best_time(lambda: train_vec(Xs, yh), 10),
        "max_diff": diff,
    }
    return results


if __name__ == "__main__":
    r = run_benchmarks()
    f = r["forward"]
    print(f"对比 1：一次前向 Y = X @ W，X {f['shape'][0]}，W {f['shape'][1]}（三种写法结果一致）")
    print(f"  Python 三重循环      {f['loop'] * 1e3:10.2f} ms")
    print(f"  每行一次 np.dot       {f['rows'] * 1e3:10.2f} ms   比三重循环快 {f['loop'] / f['rows']:6.0f} 倍")
    print(f"  一次 X @ W           {f['vec'] * 1e3:10.3f} ms   比三重循环快 {f['loop'] / f['vec']:6.0f} 倍")

    t = r["train"]
    print(f"\n对比 2：多元线性回归训练 200 步，X {t['shape']}")
    print(f"  循环算梯度           {t['loop'] * 1e3:10.2f} ms")
    print(f"  矩阵形式 2/N·Xᵀ(ŷ−y) {t['vec'] * 1e3:10.2f} ms   快 {t['loop'] / t['vec']:6.0f} 倍")
    print(f"  两种写法学到的参数最大差 = {t['max_diff']:.1e}")
