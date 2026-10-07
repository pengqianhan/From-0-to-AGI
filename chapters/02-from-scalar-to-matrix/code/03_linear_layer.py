"""Chapter 2 · Minimal code 3: Y = X @ W + b. A linear layer is one matrix multiplication.

Example: 4 houses with 3 features each (area, number of bedrooms, distance to the city center).
First use one layer (3 → 2), then one layer (2 → 1). Look at the shape after each step.
At the end, the script shows that two linear layers in a stack are equal to one linear layer
(Chapter 3 starts from this result).
Run: uv run python chapters/02-from-scalar-to-matrix/code/03_linear_layer.py
"""

import numpy as np

# 4 houses with 3 features each. Shape (4, 3): 4 samples (batch dimension), 3 features for each sample
HOUSES = np.array([
    [80,  2, 5.0],   # 80 m², 2 bedrooms, 5 km to the city center
    [120, 3, 2.0],   # 120 m², 3 bedrooms, 2 km to the city center
    [60,  1, 8.0],   # 60 m², 1 bedroom, 8 km to the city center
    [100, 3, 3.5],   # 100 m², 3 bedrooms, 3.5 km to the city center
])


def linear(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Y = X @ W + b. X: (batch, k), W: (k, n), b: (n,) → Y: (batch, n)."""
    return X @ W + b


def init_params(seed: int = 42):
    """Parameters of the two layers. W1: (3, 2); column j holds the 3 weights for output j. b1: (2,), one bias for each output."""
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((3, 2)) * 0.1
    b1 = np.array([1.0, -1.0])
    W2 = rng.standard_normal((2, 1)) * 0.1
    b2 = np.array([0.5])
    return W1, b1, W2, b2


if __name__ == "__main__":
    X = HOUSES
    print(f"Input X: shape {X.shape} (4 houses × 3 features)")

    W1, b1, W2, b2 = init_params()

    # ── Layer 1: 3 dimensions → 2 dimensions ───────────────────────────
    H = linear(X, W1, b1)
    print("\nLayer 1: H = X @ W1 + b1")
    print(f"  {X.shape} @ {W1.shape} → {(X @ W1).shape}, then + b1 {b1.shape} (broadcast to each row) → {H.shape}")
    print(f"  H =\n{H.round(3)}")

    # ── Layer 2: 2 dimensions → 1 dimension ────────────────────────────
    Y = linear(H, W2, b2)
    print("\nLayer 2: Y = H @ W2 + b2")
    print(f"  {H.shape} @ {W2.shape} → {Y.shape}")
    print(f"  Y = {Y.ravel().round(4)}")

    print("\nData flow: (4, 3) --[@W1 + b1]--> (4, 2) --[@W2 + b2]--> (4, 1)")

    # ── Preview of Chapter 3: two linear layers = one linear layer ─────
    W = W1 @ W2              # (3, 1)
    b = b1 @ W2 + b2         # (1,)
    Y_one = linear(X, W, b)
    print("\nMerge the two layers into one layer: W = W1 @ W2, b = b1 @ W2 + b2")
    print(f"  Shape of the merged W {W.shape}, shape of b {b.shape}")
    print(f"  Maximum difference between the output of two layers and of one layer = {np.abs(Y - Y_one).max():.1e}")
    print("  Conclusion: a stack of linear layers is still one linear function. To fit a curve, we must add a \"nonlinearity\". Chapter 3 shows how.")
