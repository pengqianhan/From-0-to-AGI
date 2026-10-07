"""Chapter 2 · Minimal code 5: loop version vs vectorized version. How large is the difference?

Two comparisons. For each one, the script first makes sure that the results agree. Then it compares the speed:
1. One forward pass Y = X @ W: three nested Python loops vs one np.dot for each row vs one X @ W.
2. Multivariate linear regression (5000 houses), 200 training steps: a loop over samples and features
   to calculate the gradient vs the matrix form 2/N·Xᵀ(ŷ − y).
The times depend on the machine and change a little from run to run. Look only at the order of magnitude.
Run: uv run python chapters/02-from-scalar-to-matrix/code/05_loop_vs_vectorized.py
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
    """Run repeat times and return the fastest time (seconds). This decreases the effect of system noise."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# ── Comparison 1: forward pass Y = X @ W ───────────────────────────────────

def forward_loop(X_list, W_list):
    return mm.matmul(X_list, W_list)                        # three nested loops, all in Python


def forward_rows(X, W):
    return np.stack([X[i] @ W for i in range(len(X))])      # loop only over samples; NumPy does each row


def forward_vec(X, W):
    return X @ W                                            # one matrix multiplication


# ── Comparison 2: 200 training steps ───────────────────────────────────────

def gradients_loop(W, b, X, y):
    """Calculates the same values as reg.gradients, but adds them one sample and one feature at a time."""
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

    # Comparison 1: batch = 1000 samples, 100 input features, 10 outputs
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

    # Comparison 2: use the data function of script 04, with 5000 houses (3 features, standardized)
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
    print(f"Comparison 1: one forward pass Y = X @ W, X {f['shape'][0]}, W {f['shape'][1]} (the three versions agree)")
    print(f"  Three Python loops   {f['loop'] * 1e3:10.2f} ms")
    print(f"  np.dot for each row  {f['rows'] * 1e3:10.2f} ms   faster than three loops by {f['loop'] / f['rows']:6.0f}×")
    print(f"  One X @ W            {f['vec'] * 1e3:10.3f} ms   faster than three loops by {f['loop'] / f['vec']:6.0f}×")

    t = r["train"]
    print(f"\nComparison 2: multivariate linear regression, 200 training steps, X {t['shape']}")
    print(f"  Gradient with loops  {t['loop'] * 1e3:10.2f} ms")
    print(f"  Matrix 2/N·Xᵀ(ŷ−y)   {t['vec'] * 1e3:10.2f} ms   faster by {t['loop'] / t['vec']:6.0f}×")
    print(f"  Maximum difference between the parameters of the two versions = {t['max_diff']:.1e}")
