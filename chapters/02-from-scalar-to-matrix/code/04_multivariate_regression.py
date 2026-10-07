"""Chapter 2 · Minimal code 4: multivariate linear regression with gradient descent in matrix form

Chapter 1: one input x, two parameters a and b, ŷ = a·x + b.
This chapter: 3 inputs (area, number of bedrooms, distance to the city center), ŷ = X @ W + b.
The four steps (model, loss, gradient, update) do not change. Only the scalars become matrices:
    ŷ     = X @ W + b                 (N, 3) @ (3, 1) + (1,) → (N, 1)
    L     = 1/N · Σ (ŷ − y)²
    ∂L/∂W = 2/N · Xᵀ @ (ŷ − y)        (3, N) @ (N, 1) → (3, 1), the same shape as W
    ∂L/∂b = 2/N · Σ (ŷ − y)
Run: uv run python chapters/02-from-scalar-to-matrix/code/04_multivariate_regression.py
"""

import numpy as np

FEATURES = ["area(m²)", "rooms", "dist(km)"]
W_TRUE = np.array([[0.8], [5.0], [-3.0]])   # true weights (unit: 10k yuan): 0.8 per m², 5 per bedroom, −3 per km of distance
B_TRUE = 20.0


def make_data(n: int = 200, noise: float = 5.0, seed: int = 0):
    """Make n houses. X: (n, 3), y: (n, 1). The unit of the price is 10k yuan (10,000 yuan)."""
    rng = np.random.default_rng(seed)
    area = rng.uniform(50, 150, size=n)
    rooms = rng.integers(1, 5, size=n).astype(float)
    dist = rng.uniform(1, 20, size=n)
    X = np.stack([area, rooms, dist], axis=1)
    y = X @ W_TRUE + B_TRUE + rng.normal(0, noise, size=(n, 1))
    return X, y


def standardize(X: np.ndarray):
    """Subtract the mean of each feature and divide by its standard deviation.
    Then the three features have the same scale (for the reason, see critical_lr)."""
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    return (X - mu) / sigma, mu, sigma


def critical_lr(X: np.ndarray) -> float:
    """The critical learning rate 2/λ_max from Chapter 1, for many features: Hessian = 2/N · [X 1]ᵀ[X 1]."""
    A = np.hstack([X, np.ones((len(X), 1))])
    lam_max = np.linalg.eigvalsh(2 / len(X) * A.T @ A).max()
    return 2 / lam_max


def mse(W, b, X, y) -> float:
    return float(np.mean((X @ W + b - y) ** 2))


def gradients(W, b, X, y):
    """The gradient in matrix form: one line of code calculates the gradients of all weights."""
    n = len(X)
    err = X @ W + b - y                  # (N, 1): residual ŷ − y
    grad_W = 2 / n * X.T @ err           # (3, N) @ (N, 1) → (3, 1)
    grad_b = 2 / n * err.sum(axis=0)     # (1,)
    return grad_W, grad_b


def gradient_descent(X, y, lr: float = 0.1, steps: int = 200):
    """Return (W, b, loss) for each step. Start point: all of W = 0, b = 0."""
    W = np.zeros((X.shape[1], 1))
    b = np.zeros(1)
    history = [(W.copy(), b.copy(), mse(W, b, X, y))]
    for _ in range(steps):
        grad_W, grad_b = gradients(W, b, X, y)
        W = W - lr * grad_W              # W ← W − η · ∂L/∂W
        b = b - lr * grad_b              # b ← b − η · ∂L/∂b
        history.append((W.copy(), b.copy(), mse(W, b, X, y)))
    return history


def to_original_units(W, b, mu, sigma):
    """Convert (W, b) from the standardized space back to the original units: ŷ = ((X − μ)/σ) @ W + b."""
    W_orig = W / sigma[:, None]
    b_orig = b - (mu / sigma) @ W
    return W_orig, b_orig


if __name__ == "__main__":
    X, y = make_data()
    Xs, mu, sigma = standardize(X)
    print(f"Shape of X {X.shape}, shape of y {y.shape}")
    print(f"Standard deviations of the three features: {np.round(sigma, 2)} (very different scales)")
    print(f"Critical learning rate 2/λ_max: original features {critical_lr(X):.2e}, standardized {critical_lr(Xs):.3f}\n")

    history = gradient_descent(Xs, y, lr=0.1, steps=200)
    print("Step    w_area   w_rooms    w_dist         b        Loss")
    for step in [0, 1, 5, 10, 20, 50, 100, 200]:
        W, b, loss = history[step]
        w = W.ravel()
        print(f"{step:>4}  {w[0]:8.3f}  {w[1]:8.3f}  {w[2]:8.3f}  {b[0]:8.3f}  {loss:10.3f}")

    W, b, _ = history[-1]
    W_o, b_o = to_original_units(W, b, mu, sigma)

    # For comparison: the closed-form least-squares solution
    A = np.hstack([X, np.ones((len(X), 1))])
    sol = np.linalg.lstsq(A, y, rcond=None)[0]

    print("\nConverted back to the original units (10k yuan):")
    print(f"{'':14s}{'Grad desc':>10s}{'Least sq':>10s}{'True':>10s}")
    for i, name in enumerate(FEATURES):
        print(f"w_{name:12s}{W_o[i, 0]:10.3f}{sol[i, 0]:10.3f}{W_TRUE[i, 0]:10.3f}")
    print(f"{'b':14s}{b_o[0]:10.3f}{sol[3, 0]:10.3f}{B_TRUE:10.3f}")

    new = np.array([[100.0, 3.0, 5.0]])
    pred = ((new - mu) / sigma) @ W + b
    print(f"\nNew house, 100 m², 3 bedrooms, 5 km to the city center: prediction {pred[0, 0]:.1f} (10k yuan)")
