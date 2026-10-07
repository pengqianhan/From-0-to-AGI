"""Chapter 4 · Minimal code 2: gradient check. Does our automatic differentiation give correct gradients?

Task 2 of Chapter 1 used this method. A central difference
    ∂L/∂p ≈ (L(p + ε) − L(p − ε)) / (2ε)
gives an approximate derivative. We compare it with the analytic gradient. Here, we use it to check our autograd:
1. a small expression that uses all operations;
2. the gradient of the loss of an MLP for all 97 parameters;
3. an engine with an intentional bug (`+=` changed to `=` in backward): does the gradient check find the bug?
4. a parity check against the 6 gradient formulas that we derived by hand in Chapter 3
   (the same ReLU MLP, the same initial parameters).

Run: uv run python chapters/04-backprop-autograd/code/02_grad_check.py
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
    """Move only leaf node i and estimate ∂f/∂leaf_i with a central difference. Each call of f() does a new forward pass."""
    old = leaves[i].data
    leaves[i].data = old + EPS
    up = f().data
    leaves[i].data = old - EPS
    down = f().data
    leaves[i].data = old
    return (up - down) / (2 * EPS)


def check(f, leaves):
    """Return (maximum absolute error, relative error, autograd gradient, numerical gradient)."""
    for v in leaves:
        v.grad = 0.0
    f().backward()
    auto = [v.grad for v in leaves]
    num = [numerical_grad(f, leaves, i) for i in range(len(leaves))]
    diff = [a - n for a, n in zip(auto, num, strict=True)]
    norm = lambda v: sum(t * t for t in v) ** 0.5  # noqa: E731
    rel = norm(diff) / (norm(auto) + norm(num))    # relative error over the full gradient vector
    return max(abs(d) for d in diff), rel, auto, num


def expression_case(eng):
    """An expression that uses all operations. It uses x several times, so it tests the sum at fan-out points."""
    V = eng.Value
    x, y, z = V(0.7), V(-1.3), V(2.1)

    def f():
        a = x * y + z.exp()
        b = (a.log() - x / z).tanh()
        return b * x + (y**2).relu() + (x - 3.0) ** 3 / 10
    return f, [x, y, z]


def mlp_case(eng):
    """The mean squared error of MLP(1, [8, 8, 1]) on 20 points, for all 97 parameters."""
    V = eng.Value
    random.seed(0)
    net = eng.MLP(1, [8, 8, 1])
    xs, ys = train_mlp.make_data()

    def f():  # the same as mse in 03_train_mlp.py: L = 1/N · Σ (ŷ_i − y_i)²
        return sum(((net([V(x)]) - y) ** 2 for x, y in zip(xs, ys, strict=True)), V(0.0)) / len(xs)
    return f, net.parameters()


def buggy_engine():
    """Make a bug on purpose: change each `.grad +=` to `.grad =`. Then a new gradient replaces the old one and does not add to it."""
    src = (HERE / "01_engine.py").read_text(encoding="utf-8").replace(".grad +=", ".grad =")
    src = src.replace('if __name__ == "__main__":', "if False:")
    mod = types.ModuleType("buggy_engine")
    exec(compile(src, "buggy_engine", "exec"), mod.__dict__)
    return mod


def ch3_case(eng):
    """Rebuild the two-layer ReLU MLP of Chapter 3 (width 8, fits 100 points of sin(2x)) with Value.

    Return (autograd gradients, hand-derived gradients of Chapter 3). Both are {parameter name: nested list}.
    """
    path = HERE.parent.parent / "03-neural-network" / "code" / "03_mlp_numpy.py"
    spec = importlib.util.spec_from_file_location("ch3_mlp", path)
    ch3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ch3)
    x, y = ch3.make_data()
    p = ch3.init_params(8, seed=0)
    _, hand = ch3.gradients(p, x, y)                       # the gradients derived by hand in Chapter 3
    V = eng.Value
    P = {k: [[V(v) for v in row] for row in np.atleast_2d(p[k])] for k in p}  # b1 and b2 become 1 row
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
    print("  [expression]  autograd:", ", ".join(f"{g:+.6f}" for g in auto))
    print("                numerical:", ", ".join(f"{g:+.6f}" for g in num))
    print(f"                max absolute error {e_abs:.1e}, relative error {e_rel:.1e}")
    f, params = mlp_case(eng)
    e_abs, e_rel, _, _ = check(f, params)
    print(f"  [MLP]         {len(params)} parameters: max absolute error {e_abs:.1e}, relative error {e_rel:.1e}")


if __name__ == "__main__":
    print("Correct engine (backward uses +=):")
    run_checks(engine)
    print("\nEngine with a bug (+= changed to = in backward):")
    run_checks(buggy_engine())

    print("\nParity check with the hand-derived gradients of Chapter 3 (ReLU MLP, width 8, 100 points):")
    auto, hand = ch3_case(engine)
    for k in auto:
        print(f"  {k:<3} shape {str(hand[k].shape):<7} max difference {np.max(np.abs(auto[k] - hand[k])):.1e}")
