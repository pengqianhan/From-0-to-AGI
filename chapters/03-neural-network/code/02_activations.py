"""Chapter 3 · Minimal code 2: activation functions, and "one ReLU = one kink"

1. The values of common activation functions at some points: ReLU, sigmoid, tanh (common in the past),
   SiLU, and GELU (the SwiGLU in modern large models uses SiLU).
2. Build shapes from ReLUs: |x| = ReLU(x) + ReLU(−x); "tent" = ReLU(x+1) − 2·ReLU(x) + ReLU(x−1).
   Each ReLU adds only one kink. A weighted sum of ReLUs makes a shape with kinks.
Run: uv run python chapters/03-neural-network/code/02_activations.py
"""

import math

import numpy as np


def relu(z):
    return np.maximum(0.0, z)                        # ReLU(z) = max(0, z)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))                  # σ(z) = 1 / (1 + e^(−z))


def tanh(z):
    return np.tanh(z)


def silu(z):
    return z * sigmoid(z)                            # SiLU(z) = z · σ(z), also known as Swish


def gelu(z):
    # GELU(z) = z · Φ(z). Φ is the cumulative distribution function of the standard normal distribution.
    return z * 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


ACTIVATIONS = {"ReLU": relu, "sigmoid": sigmoid, "tanh": tanh, "SiLU": silu, "GELU": gelu}


def hinge(x, w: float, b: float, v: float):
    """The contribution of one hidden unit to the output: v · ReLU(w·x + b). The kink is at x = −b / w."""
    return v * relu(w * x + b)


if __name__ == "__main__":
    zs = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    print("1) Values of the activation functions at some points")
    print("   z        " + "".join(f"{z:>8.1f}" for z in zs))
    for name, f in ACTIVATIONS.items():
        print(f"   {name:<8} " + "".join(f"{v:>8.3f}" for v in f(zs)))

    x = np.linspace(-2, 2, 9)
    print("\n2) Build shapes from ReLUs (x from −2 to 2)")
    rows = [
        ("x", x),
        ("|x| = ReLU(x)+ReLU(−x)", hinge(x, 1, 0, 1) + hinge(x, -1, 0, 1)),
        ("tent = ReLU(x+1)−2ReLU(x)+ReLU(x−1)",
         hinge(x, 1, 1, 1) + hinge(x, 1, 0, -2) + hinge(x, 1, -1, 1)),
    ]
    for name, vals in rows:
        print(f"   {name}")
        print("      " + "".join(f"{v:>6.1f}" for v in vals))
    print("   → Two ReLUs make |x| (1 kink). Three ReLUs make a tent (kinks at −1, 0, and 1).")
