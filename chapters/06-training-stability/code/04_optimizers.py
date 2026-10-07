"""Chapter 6 · Minimal code 4: optimizers — SGD → momentum → Adam → AdamW

Experiment 1: a bowl L(θ) = ½ Σ hᵢ θᵢ² whose steepness differs by 1000× between directions.
              For SGD, momentum, and Adam, we compare how far each parameter moves in each
              step, and the loss that is left after 200 steps.
Experiment 2: weight decay. With the same λ, do "Adam + L2 regularization" and AdamW
              (decoupled weight decay) apply the same decay to parameters with small and
              large gradients?
Run: uv run python chapters/06-training-stability/code/04_optimizers.py
"""

import numpy as np

H = np.array([100.0, 10.0, 1.0, 0.1])   # curvature in each of the 4 directions (how steep the bowl is)


def grad(theta: np.ndarray) -> np.ndarray:
    """The gradient of L = ½ Σ hᵢ θᵢ²: ∂L/∂θᵢ = hᵢ θᵢ."""
    return H * theta


def loss(theta: np.ndarray) -> float:
    return float(0.5 * np.sum(H * theta**2))


def sgd(steps, lr):
    theta, hist = np.ones(4), []
    for _ in range(steps):
        step = -lr * grad(theta)                    # θ ← θ − η·g
        theta = theta + step
        hist.append((theta.copy(), step))
    return hist


def momentum(steps, lr, beta=0.9):
    theta, v, hist = np.ones(4), np.zeros(4), []
    for _ in range(steps):
        v = beta * v + grad(theta)                  # v ← β·v + g   (velocity: the sum of past gradients)
        step = -lr * v                              # θ ← θ − η·v
        theta = theta + step
        hist.append((theta.copy(), step))
    return hist


def adam(steps, lr, b1=0.9, b2=0.95, eps=1e-8):
    theta, m, v, hist = np.ones(4), np.zeros(4), np.zeros(4), []
    for t in range(1, steps + 1):
        g = grad(theta)
        m = b1 * m + (1 - b1) * g                   # first moment: moving average of g (direction)
        v = b2 * v + (1 - b2) * g * g               # second moment: moving average of g² (scale)
        m_hat, v_hat = m / (1 - b1**t), v / (1 - b2**t)   # bias correction
        step = -lr * m_hat / (np.sqrt(v_hat) + eps)       # each parameter: divide by its own gradient scale
        theta = theta + step
        hist.append((theta.copy(), step))
    return hist


def weight_decay_demo(decoupled: bool, steps=3000, lr=0.01, wd=0.1, n=1000, seed=0):
    """Two groups of n parameters each. All parameters start at 1.
    The loss gives no signal; the gradient is only noise:
    group A has noise std 0.01 (small gradients), group B has 10 (large gradients).
    Weight decay must pull both groups toward 0 by the same amount.
    Return the mean value of each group."""
    rng = np.random.default_rng(seed)
    sigma = np.repeat([0.01, 10.0], n)
    w, m, v = np.ones(2 * n), np.zeros(2 * n), np.zeros(2 * n)
    b1, b2 = 0.9, 0.95
    for t in range(1, steps + 1):
        g = rng.normal(0, sigma)
        if not decoupled:
            g = g + wd * w                          # Adam + L2: the decay goes into g, so √v divides it too
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        if decoupled:
            w = w - lr * wd * w                     # AdamW: a separate decay step, not divided by √v
        w = w - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + 1e-8)
    return w[:n].mean(), w[n:].mean()


if __name__ == "__main__":
    lr_sgd = 0.019                                  # critical learning rate of SGD: 2/100 = 0.02
    runs = {
        "SGD": sgd(200, lr_sgd),
        "Momentum": momentum(200, lr_sgd),
        "Adam": adam(200, 0.05),
    }
    print("Curvature of the bowl h =", H.tolist(),
          "(the steepest and the flattest directions differ by 1000×), start point θ = [1, 1, 1, 1]")
    print(f"SGD and momentum (β = 0.9): η = {lr_sgd} "
          f"(a little larger and the steepest direction diverges); Adam: η = 0.05\n")
    print("Step 1: how far each parameter moves |Δθᵢ|:")
    for name, hist in runs.items():
        print(f"  {name:<12s}" + "".join(f"{abs(s):>10.4f}" for s in hist[0][1]))
    print("\nAfter 200 steps:")
    for name, hist in runs.items():
        th = hist[-1][0]
        hit = next((i + 1 for i, (t, _) in enumerate(hist) if loss(t) < 1e-3), None)
        print(f"  {name:<12s} θ = [" + ", ".join(f"{x:8.4f}" for x in th) + f"]   loss = {loss(th):.2e}"
              f"   steps until loss < 0.001: {hit if hit else 'more than 200'}")
    print(f"  (loss at the start point = {loss(np.ones(4)):.2f})\n")

    print("Weight decay λ = 0.1, η = 0.01, 3000 steps, the gradient is only noise; mean value of each group:")
    print("              Group A (noise 0.01)    Group B (noise 10)")
    for name, dec in [("Adam + L2 term", False), ("AdamW decoupled", True)]:
        a, b = weight_decay_demo(dec)
        print(f"  {name:<16s}{a:>16.3f}{b:>22.3f}")
    print(f"  Decay term only, value from theory (1 − ηλ)^3000 = {(1 - 0.01 * 0.1) ** 3000:.3f}")
