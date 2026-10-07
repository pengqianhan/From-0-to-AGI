"""Chapter 3 · Minimal code 3: fit y = sin(2x) with a two-layer MLP, manual gradients, and gradient descent

The model (row-vector form, the same as y = XW + b in Chapter 2):
    Z = X·W1 + b1        (N, 1) → (N, H)   layer 1: linear
    A = ReLU(Z)          (N, H)            activation: set each negative number to 0
    Ŷ = A·W2 + b2        (N, H) → (N, 1)   layer 2: linear
    L = mean((Ŷ − Y)²)                     loss: the same mean squared error as in Chapter 1
We derived the gradients by hand with the chain rule. Chapter 4 explains backpropagation in full.
Here we use the gradients, and we check them against numerical gradients.
Uses only NumPy. It runs in about 25 seconds on a CPU (most of the time is for the multi-seed test in part 3).
Run: uv run python chapters/03-neural-network/code/03_mlp_numpy.py
"""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "linear_is_not_enough", Path(__file__).with_name("01_linear_is_not_enough.py")
)
lin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lin)
make_data = lin.make_data  # 100 points for x ∈ [−3, 3], y = sin(2x)

LR = 0.01
STEPS = 20000


def init_params(hidden: int, seed: int = 0) -> dict:
    """Random initialization. We choose b1 so that the kink of each hidden unit (x = −b1/W1)
    starts at a random point in [−3, 3]. (Initialization has many details. Chapter 6 explains them.)"""
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1, size=(1, hidden))
    b1 = rng.uniform(-3, 3, size=hidden) * np.abs(W1[0])
    W2 = rng.normal(0, 1 / np.sqrt(hidden), size=(hidden, 1))
    b2 = np.zeros(1)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def act_fn(z: np.ndarray, act: str) -> np.ndarray:
    return np.maximum(0.0, z) if act == "relu" else z        # "linear": no activation


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
    """Manual gradients: go back from the loss, one layer at a time, with the chain rule.
    Returns (loss, dict of gradients)."""
    n = len(x)
    y_hat, (z, a) = forward(p, x, act)
    loss = float(np.mean((y_hat - y) ** 2))
    d_yhat = 2 * (y_hat - y) / n          # ∂L/∂Ŷ: the same as 2/N·(ŷ − y) in Chapter 1
    d_W2 = a.T @ d_yhat                   # ∂L/∂W2 = Aᵀ · ∂L/∂Ŷ
    d_b2 = d_yhat.sum(axis=0)             # ∂L/∂b2 = Σ ∂L/∂Ŷ
    d_a = d_yhat @ p["W2"].T              # ∂L/∂A  = ∂L/∂Ŷ · W2ᵀ
    d_z = d_a * act_grad(z, act)          # ∂L/∂Z  = ∂L/∂A ⊙ ReLU′(Z): an inactive unit gets gradient 0
    d_W1 = x.T @ d_z                      # ∂L/∂W1 = Xᵀ · ∂L/∂Z
    d_b1 = d_z.sum(axis=0)                # ∂L/∂b1 = Σ ∂L/∂Z
    return loss, {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}


def numerical_gradients(p: dict, x, y, act: str = "relu", eps: float = 1e-6) -> dict:
    """Numerical gradients: (L(θ+ε) − L(θ−ε)) / 2ε, one parameter at a time.
    This is slow, but it has no derivation errors. We use it to check the manual formulas."""
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
    """Gradient descent. Returns (final parameters, loss at each step, {step: copy of the parameters at that step})."""
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
            p[k] -= lr * g[k]             # θ ← θ − η · ∂L/∂θ, the same as in Chapter 1
    return p, losses, snaps


def pieces(p: dict, x: np.ndarray):
    """Split the network output into H "hinge pieces": hidden unit j adds W2[j] · ReLU(W1[j]·x + b1[j]).
    Returns (the value of each piece (N, H), the kink position of each piece (H,)). Output = sum of the pieces + b2."""
    z = x @ p["W1"] + p["b1"]
    contrib = np.maximum(0.0, z) * p["W2"][:, 0]
    kinks = -p["b1"] / p["W1"][0]
    return contrib, kinks


def n_params(hidden: int) -> int:
    return 3 * hidden + 1                 # W1: H, b1: H, W2: H, b2: 1


if __name__ == "__main__":
    x, y = make_data()

    # 1) Gradient check: do the manual gradients agree with the numerical gradients?
    p0 = init_params(8, seed=0)
    ga, gn = gradients(p0, x, y)[1], numerical_gradients(p0, x, y)
    err = max(np.max(np.abs(ga[k] - gn[k])) for k in p0)
    print(f"1) Gradient check (width 8): manual vs numerical gradients, max difference = {err:.1e}")

    # 2) Different widths, learning rate 0.01, 20000 steps of gradient descent
    print(f"\n2) Fit y = sin(2x), learning rate {LR}, {STEPS} steps of gradient descent (seed 0)")
    _, _, line_mse = lin.best_line(x, y)
    print(f"   {'Model':<16}{'Params':>6}{'Step 0':>10}{'1000':>10}{'5000':>10}{'20000':>10}")
    configs = [("Linear, width 8", 8, "linear"), ("ReLU, width 2", 2, "relu"),
               ("ReLU, width 8", 8, "relu"), ("ReLU, width 64", 64, "relu")]
    for name, h, act in configs:
        _, losses, _ = train(x, y, h, act=act)
        cols = "".join(f"{losses[s]:>10.4f}" for s in (0, 1000, 5000, STEPS))
        print(f"   {name:<16}{n_params(h):>6}{cols}")
    print(f"   (Reference: MSE of the best straight line = {line_mse:.4f})")

    # 3) Use 5 random seeds to check that the result is stable
    print("\n3) 5 random seeds (0–4), loss after 20000 steps")
    for h in (2, 8, 64):
        finals = [train(x, y, h, seed=s)[1][-1] for s in range(5)]
        print(f"   width {h:>2}: " + "  ".join(f"{v:.4f}" for v in finals)
              + f"   median {np.median(finals):.4f}")

    # 4) Take the width-8 network apart: output = sum of 8 hinge pieces + b2
    p8, _, _ = train(x, y, 8)
    contrib, kinks = pieces(p8, x)
    y_hat, _ = forward(p8, x)
    diff = np.max(np.abs(contrib.sum(axis=1, keepdims=True) + p8["b2"] - y_hat))
    order = np.argsort(kinks)
    print("\n4) The width-8 network: each hidden unit is one kink")
    print("   Kink positions x = −b1/W1: " + "  ".join(f"{kinks[j]:.2f}" for j in order))
    print(f"   Max amplitude of one hinge piece in the data range = {np.max(np.abs(contrib)):.2f}"
          f" (the network output has an amplitude of only {np.max(np.abs(y_hat)):.2f}: the pieces cancel each other)")
    print(f"   Max difference between (sum of pieces + b2) and the network output = {diff:.1e}")
