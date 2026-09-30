"""第 1 章 · 极简代码 2：学习率太小、刚好、太大会怎样？

对这个问题，损失是 (a, b) 的二次函数。二次函数上梯度下降能收敛的条件是
    学习率 η < 2 / λ_max，
其中 λ_max 是损失的二阶导数矩阵（Hessian）的最大特征值。我们直接算出这个临界值来验证。
运行：uv run python chapters/01-linear-regression/code/02_learning_rate.py
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
    """L(a,b) 的 Hessian = 2/N · [[Σx², Σx], [Σx, N]]，与 a、b 无关。"""
    n = len(x)
    hessian = 2 / n * np.array([[np.sum(x**2), np.sum(x)], [np.sum(x), n]])
    lam_max = np.linalg.eigvalsh(hessian).max()
    return 2 / lam_max


if __name__ == "__main__":
    x, y = fit_line.make_data()
    lr_c = critical_lr(x)
    print(f"理论临界学习率 2/λ_max = {lr_c:.4f}\n")

    for lr in [0.005, 0.05, 0.9 * lr_c, 1.05 * lr_c]:
        hist = fit_line.gradient_descent(x, y, lr=lr, steps=100)
        a, b, loss = hist[-1]
        trend = "发散 ↑" if loss > hist[0][2] else "收敛 ↓"
        print(f"η = {lr:.4f}：100 步后 a = {a:10.3f}, b = {b:10.3f}, 损失 = {loss:12.4f}  {trend}")
