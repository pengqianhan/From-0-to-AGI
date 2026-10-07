"""Chapter 1 · Minimal code 2: what occurs when the learning rate is too small, correct, or too large?

In this problem, the loss is a quadratic function of (a, b). On a quadratic function,
gradient descent converges only when
    learning rate η < 2 / λ_max,
where λ_max is the largest eigenvalue of the matrix of second derivatives of the loss (the Hessian).
The script calculates this critical value and tests it.
Run: uv run python chapters/01-linear-regression/code/02_learning_rate.py
"""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "fit_line", Path(__file__).with_name("01_fit_line.py")
)
fit_line = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fit_line)


def critical_lr(x: np.ndarray) -> float:
    """The Hessian of L(a, b) is 2/N · [[Σx², Σx], [Σx, N]]. It does not depend on a or b."""
    n = len(x)
    hessian = 2 / n * np.array([[np.sum(x**2), np.sum(x)], [np.sum(x), n]])
    lam_max = np.linalg.eigvalsh(hessian).max()
    return 2 / lam_max


if __name__ == "__main__":
    x, y = fit_line.make_data()
    lr_c = critical_lr(x)
    print(f"Critical learning rate from theory, 2/λ_max = {lr_c:.4f}\n")

    for lr in [0.005, 0.05, 0.9 * lr_c, 1.05 * lr_c]:
        hist = fit_line.gradient_descent(x, y, lr=lr, steps=100)
        a, b, loss = hist[-1]
        trend = "diverges ↑" if loss > hist[0][2] else "converges ↓"
        print(f"η = {lr:.4f}: after 100 steps a = {a:10.3f}, b = {b:10.3f}, loss = {loss:12.4f}  {trend}")
