"""Chapter 3 · Minimal code 1: a straight line cannot fit a curve, and stacked linear layers are still a straight line.

The script shows two things:
1. Use least squares to find the "best straight line" for y = sin(2x). See how good this line can be.
2. Stack two (or three) linear layers: (X·W1 + b1)·W2 + b2 = X·(W1·W2) + (b1·W2 + b2).
   Use numbers to show that the stack is identical to one linear layer.
Uses only NumPy. It runs in less than 1 second on a CPU.
Run: uv run python chapters/03-neural-network/code/01_linear_is_not_enough.py
"""

import numpy as np


def make_data(n: int = 100):
    """The data of this chapter: 100 evenly spaced points for x in [−3, 3], y = sin(2x). Both have the shape (N, 1)."""
    x = np.linspace(-3, 3, n).reshape(-1, 1)
    y = np.sin(2 * x)
    return x, y


def best_line(x: np.ndarray, y: np.ndarray):
    """Least squares: the straight line with the smallest MSE (the closed-form solution from Chapter 1)."""
    a, b = np.polyfit(x[:, 0], y[:, 0], deg=1)
    mse = float(np.mean((a * x + b - y) ** 2))
    return a, b, mse


def collapse(layers):
    """Merge a stack of linear layers [(W1, b1), (W2, b2), ...] into one layer (W, b).

    (X·W1 + b1)·W2 + b2 = X·(W1·W2) + (b1·W2 + b2). The same is true for each layer after that.
    """
    W, b = layers[0]
    for W_next, b_next in layers[1:]:
        W, b = W @ W_next, b @ W_next + b_next
    return W, b


def forward_stack(x: np.ndarray, layers) -> np.ndarray:
    """Calculate layer by layer, with no shortcut: h ← h·W + b."""
    h = x
    for W, b in layers:
        h = h @ W + b
    return h


def random_stacks(seed: int = 0):
    """Make two random linear-only networks: 2 layers 1→8→1 and 3 layers 1→8→8→1.

    Each layer is h ← h·W + b. No activation is between the layers.
    """
    rng = np.random.default_rng(seed)
    two = [(rng.normal(size=(1, 8)), rng.normal(size=8)),      # layer 1: 1 → 8
           (rng.normal(size=(8, 1)), rng.normal(size=1))]      # layer 2: 8 → 1
    three = [(rng.normal(size=(1, 8)), rng.normal(size=8)),
             (rng.normal(size=(8, 8)), rng.normal(size=8)),
             (rng.normal(size=(8, 1)), rng.normal(size=1))]
    return {"2 layers 1→8→1": two, "3 layers 1→8→8→1": three}


def stack_demo(x: np.ndarray):
    """For each random linear network: the number of parameters, the merged (W, b),
    and the maximum difference between the layer-by-layer result and the merged result."""
    rows = []
    for name, layers in random_stacks().items():
        W, b = collapse(layers)
        diff = float(np.max(np.abs(forward_stack(x, layers) - (x @ W + b))))
        n_params = sum(Wi.size + bi.size for Wi, bi in layers)
        rows.append((name, n_params, W, b, diff))
    return rows


if __name__ == "__main__":
    x, y = make_data()
    a, b, mse = best_line(x, y)
    print("1) Fit y = sin(2x) with a straight line")
    print(f"   Best straight line: y = {a:.3f}·x + {b:.3f}, MSE = {mse:.4f}")
    print(f"   Reference: variance of y = {float(np.var(y)):.4f} (the MSE if we always predict the mean)")

    print("\n2) Stacked linear layers are still one linear layer")
    for name, n_params, W, bb, diff in stack_demo(x):
        print(f"   {name}: {n_params} parameters, merged y = {W.item():.3f}·x {bb.item():+.3f}, "
              f"max difference layer-by-layer vs merged = {diff:.1e}")
    print("   → More parameters do not help. The graph of the function is still a straight line,"
          " so the MSE cannot be lower than the best straight line above.")
