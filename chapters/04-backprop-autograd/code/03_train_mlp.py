"""Chapter 4 · Minimal code 3: train an MLP with two hidden layers with our own autograd.

Task: fit the curve y = sin(x), x ∈ [−3, 3], 20 points. The network is MLP(1, [8, 8, 1]) with tanh activation.
Unlike Chapter 3, this file has no gradient that we derived by hand. The forward pass calculates the loss,
and one call of loss.backward() gives the gradients of all parameters.
The full training takes about half a minute on a CPU. Scalar autograd is slow; the end of this chapter discusses
this problem.

Run: uv run python chapters/04-backprop-autograd/code/03_train_mlp.py
"""

import importlib.util
import math
import random
import time
from pathlib import Path

_spec = importlib.util.spec_from_file_location("engine", Path(__file__).with_name("01_engine.py"))
engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engine)
Value, MLP = engine.Value, engine.MLP


def make_data(n: int = 20):
    xs = [-3 + 6 * i / (n - 1) for i in range(n)]
    ys = [math.sin(x) for x in xs]
    return xs, ys


GRID = [-3 + 6 * i / 60 for i in range(61)]


def mse(net, xs, ys) -> Value:
    """L = 1/N · Σ (ŷ_i − y_i)². The full expression is made of Value nodes, so it is a computational graph."""
    preds = [net([Value(x)]) for x in xs]
    return sum(((p - y) ** 2 for p, y in zip(preds, ys, strict=True)), Value(0.0)) * (1 / len(xs))


def count_nodes(root: Value) -> int:
    seen, stack = set(), [root]
    while stack:
        v = stack.pop()
        if v not in seen:
            seen.add(v)
            stack.extend(v._prev)
    return len(seen)


def train(steps: int = 500, lr: float = 0.1, seed: int = 0, snapshot_at=(), verbose=False):
    """Return (net, the loss at each step, {step: the predictions on GRID before the update of that step})."""
    random.seed(seed)
    xs, ys = make_data()
    net = MLP(1, [8, 8, 1], act="tanh")
    losses, snaps = [], {}
    for step in range(steps + 1):
        loss = mse(net, xs, ys)           # 1. forward pass: also records the full computational graph
        losses.append(loss.data)
        if step in snapshot_at:           # keep the prediction curve of this step (for plots only, not for training)
            snaps[step] = [net([Value(x)]).data for x in GRID]
        if step == steps:
            break                         # the last step only evaluates and does not update
        net.zero_grad()                   # 2. set the gradients to zero (backward uses +=)
        loss.backward()                   # 3. backward pass: calculates the gradients of all parameters
        for p in net.parameters():        # 4. update: p ← p − η · ∂L/∂p
            p.data -= lr * p.grad
        if verbose and step in (0, 1, 10, 50, 100, 200, 300, 400):
            print(f"{step:>5}  {loss.data:9.5f}")
    return net, losses, snaps


if __name__ == "__main__":
    xs, ys = make_data()
    random.seed(0)
    probe = MLP(1, [8, 8, 1])
    loss = mse(probe, xs, ys)
    print(f"MLP(1, [8, 8, 1]): {len(probe.parameters())} parameters")
    print(f"One forward pass (20 samples) records a computational graph with {count_nodes(loss)} Value nodes\n")

    print(" step       loss")
    t0 = time.perf_counter()
    net, losses, _ = train(verbose=True)
    dt = time.perf_counter() - t0
    print(f"{500:>5}  {losses[-1]:9.5f}")
    print(f"\n500 steps took {dt:.1f} s, {dt / 500 * 1000:.0f} ms per step on average")

    print("\n  x       sin(x)   prediction")
    for x, y in list(zip(xs, ys, strict=True))[::4]:
        print(f"{x:6.2f}  {y:8.3f}  {net([Value(x)]).data:7.3f}")
