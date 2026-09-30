"""第 4 章 · 极简代码 2：梯度检验——自动微分算得对不对？

第 1 章任务 2 用过这个办法：用中心差分
    ∂L/∂p ≈ (L(p + ε) − L(p − ε)) / (2ε)
近似导数，和解析梯度比一比。这里拿它来检验我们的 autograd：
1. 一个用到全部运算的小表达式；
2. 一个 MLP 的损失对全部 97 个参数的梯度；
3. 故意把 backward 里的 `+=` 改成 `=`，看梯度检验能不能抓住这个错；
4. 和第 3 章手推的 6 行梯度公式对拍（同一个 ReLU MLP、同一组初始参数）。

运行：uv run python chapters/04-backprop-autograd/code/02_grad_check.py
"""

import importlib.util
import random
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("engine", HERE / "01_engine.py")
engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engine)
_spec = importlib.util.spec_from_file_location("train_mlp", HERE / "03_train_mlp.py")
train_mlp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_mlp)

EPS = 1e-6


def numerical_grad(f, leaves, i: int) -> float:
    """只动第 i 个叶子节点，用中心差分估计 ∂f/∂leaf_i。f() 每次重新前向。"""
    old = leaves[i].data
    leaves[i].data = old + EPS
    up = f().data
    leaves[i].data = old - EPS
    down = f().data
    leaves[i].data = old
    return (up - down) / (2 * EPS)


def check(f, leaves):
    """返回 (最大绝对误差, 相对误差, autograd 梯度, 数值梯度)。"""
    for v in leaves:
        v.grad = 0.0
    f().backward()
    auto = [v.grad for v in leaves]
    num = [numerical_grad(f, leaves, i) for i in range(len(leaves))]
    diff = [a - n for a, n in zip(auto, num, strict=True)]
    norm = lambda v: sum(t * t for t in v) ** 0.5  # noqa: E731
    rel = norm(diff) / (norm(auto) + norm(num))    # 相对误差：按整个梯度向量算
    return max(abs(d) for d in diff), rel, auto, num


def expression_case(eng):
    """一个把所有运算都用上的表达式（x 被用了好几次，考验分叉处的累加）。"""
    V = eng.Value
    x, y, z = V(0.7), V(-1.3), V(2.1)

    def f():
        a = x * y + z.exp()
        b = (a.log() - x / z).tanh()
        return b * x + (y**2).relu() + (x - 3.0) ** 3 / 10
    return f, [x, y, z]


def mlp_case(eng):
    """MLP(1, [8, 8, 1]) 在 20 个点上的均方误差，对全部 97 个参数。"""
    V = eng.Value
    random.seed(0)
    net = eng.MLP(1, [8, 8, 1])
    xs, ys = train_mlp.make_data()

    def f():  # 和 03_train_mlp.py 的 mse 一样：L = 1/N · Σ (ŷ_i − y_i)²
        return sum(((net([V(x)]) - y) ** 2 for x, y in zip(xs, ys, strict=True)), V(0.0)) / len(xs)
    return f, net.parameters()


def buggy_engine():
    """故意制造 bug：把所有 `.grad +=` 改成 `.grad =`，梯度不再累加而是被覆盖。"""
    src = (HERE / "01_engine.py").read_text(encoding="utf-8").replace(".grad +=", ".grad =")
    src = src.replace('if __name__ == "__main__":', "if False:")
    mod = types.ModuleType("buggy_engine")
    exec(compile(src, "buggy_engine", "exec"), mod.__dict__)
    return mod


def ch3_case(eng):
    """第 3 章的两层 ReLU MLP（宽 8，拟合 sin(2x) 的 100 个点）：用 Value 重建，
    返回 (autograd 梯度, 第 3 章手推梯度)，都是 {参数名: 嵌套列表}。"""
    path = HERE.parent.parent / "03-neural-network" / "code" / "03_mlp_numpy.py"
    spec = importlib.util.spec_from_file_location("ch3_mlp", path)
    ch3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ch3)
    x, y = ch3.make_data()
    p = ch3.init_params(8, seed=0)
    _, hand = ch3.gradients(p, x, y)                       # 第 3 章手推的梯度
    V = eng.Value
    P = {k: [[V(v) for v in row] for row in np.atleast_2d(p[k])] for k in p}  # b1、b2 变成 1 行
    W1, b1, W2, b2 = P["W1"][0], P["b1"][0], [r[0] for r in P["W2"]], P["b2"][0][0]
    total = V(0.0)
    for xi, yi in zip(x[:, 0], y[:, 0], strict=True):
        a = [(float(xi) * w + b).relu() for w, b in zip(W1, b1, strict=True)]   # A = ReLU(X·W1 + b1)
        y_hat = sum((aj * wj for aj, wj in zip(a, W2, strict=True)), b2)  # Ŷ = A·W2 + b2
        total = total + (y_hat - float(yi)) ** 2
    (total / len(x)).backward()
    auto = {k: np.array([[v.grad for v in row] for row in P[k]]).reshape(p[k].shape) for k in P}
    return auto, hand


def run_checks(eng) -> None:
    e_abs, e_rel, auto, num = check(*expression_case(eng))
    print("  [表达式]  autograd:", ", ".join(f"{g:+.6f}" for g in auto))
    print("            数值梯度:", ", ".join(f"{g:+.6f}" for g in num))
    print(f"            最大绝对误差 {e_abs:.1e}，相对误差 {e_rel:.1e}")
    f, params = mlp_case(eng)
    e_abs, e_rel, _, _ = check(f, params)
    print(f"  [MLP]     {len(params)} 个参数：最大绝对误差 {e_abs:.1e}，相对误差 {e_rel:.1e}")


if __name__ == "__main__":
    print("正确的引擎（backward 里用 +=）：")
    run_checks(engine)
    print("\n有 bug 的引擎（把 backward 里的 += 改成 =）：")
    run_checks(buggy_engine())

    print("\n和第 3 章手推的梯度对拍（ReLU MLP，宽 8，100 个点）：")
    auto, hand = ch3_case(engine)
    for k in auto:
        print(f"  {k:<3} 形状 {str(hand[k].shape):<7} 最大差 {np.max(np.abs(auto[k] - hand[k])):.1e}")
