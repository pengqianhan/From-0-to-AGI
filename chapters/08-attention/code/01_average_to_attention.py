"""Chapter 8 · Minimal code 1: from "average the earlier words" to "attention"

A bigram model looks only at the previous token. To see more context, the simplest method is
to average the vectors of all earlier tokens.
A classic trick does this: one matrix multiplication with a lower-triangular matrix
gives the "prefix average" at every position at the same time.
One more step: replace the fixed uniform weights with weights that the data sets
(dot-product similarity → softmax). The result is the first form of attention.

Uses only NumPy. It finishes at once on the CPU.
Run: uv run python chapters/08-attention/code/01_average_to_attention.py
"""

import numpy as np

np.set_printoptions(precision=2, suppress=True)

TOKENS = ["我", "爱", "吃", "苹", "果"]


def make_x(T: int = 5, C: int = 2, seed: int = 0) -> np.ndarray:
    """T tokens, each a C-dimensional vector (the embedding lookup in Chapter 7 gives such vectors)."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(T, C))


def prefix_mean_loop(x: np.ndarray) -> np.ndarray:
    """The most direct form: position t = the mean of the first t+1 vectors.

    The mean includes position t itself and no future positions.
    """
    out = np.zeros_like(x)
    for t in range(len(x)):
        out[t] = x[: t + 1].mean(axis=0)
    return out


def uniform_weights(T: int) -> np.ndarray:
    """Classic trick: a lower-triangular matrix of ones, with each row divided by its sum.

    Then each row is a set of uniform weights.
    """
    w = np.tril(np.ones((T, T)))
    return w / w.sum(axis=1, keepdims=True)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)  # the numerically stable form from Chapter 5
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def causal_softmax(scores: np.ndarray) -> np.ndarray:
    """Causal mask: fill the upper triangle (the future) with −∞.

    After softmax, the weights at those positions are exactly 0.
    """
    T = scores.shape[0]
    future = np.triu(np.ones((T, T), dtype=bool), k=1)
    return softmax(np.where(future, -np.inf, scores))


def dot_product_weights(x: np.ndarray) -> np.ndarray:
    """Weights that the data sets.

    How related positions t and s are = the dot product of the two vectors (Chapter 2).
    Then softmax.
    """
    scores = x @ x.T  # (T, T), scores[t, s] = x_t · x_s
    return causal_softmax(scores)


def main() -> None:
    x = make_x()
    T = len(x)
    print("Input: 5 tokens, each a 2-dimensional vector x (shape", x.shape, ")")
    for tok, row in zip(TOKENS, x):
        print(f"  {tok}: {row}")

    # ① Prefix average with a loop
    loop = prefix_mean_loop(x)
    # ② Lower-triangular matrix multiplication: one W @ x gives all positions
    w_uni = uniform_weights(T)
    mat = w_uni @ x
    print("\n① Uniform weight matrix W (lower-triangular, each row sums to 1):")
    print(w_uni)
    print("② Max difference between W @ x and the loop version:", f"{np.abs(mat - loop).max():.1e}")

    # ③ The same W also comes from: all-zero scores → mask the future → softmax
    w_soft = causal_softmax(np.zeros((T, T)))
    print("③ Max difference between softmax(all-zero scores + causal mask) and W:", f"{np.abs(w_soft - w_uni).max():.1e}")

    # ④ Replace the all-zero scores with dot-product similarity: now the data sets the weights
    w_dot = dot_product_weights(x)
    print("\n④ Dot-product scores x @ xᵀ:")
    print(x @ x.T)
    print("   Weights after causal mask + softmax (each row sums to 1):")
    print(w_dot)
    print("   Sum of each row:", w_dot.sum(axis=1))
    last = TOKENS[-1]
    top = int(np.argmax(w_dot[-1]))
    print(
        f"   The last position '{last}' attends most to '{TOKENS[top]}', weight {w_dot[-1, top]:.2f}"
        f" (with the uniform average, each weight is {w_uni[-1, 0]:.2f})"
    )
    print("   Last row of the weighted average w_dot @ x:", (w_dot @ x)[-1])
    self_top = sum(int(np.argmax(w_dot[t]) == t) for t in range(T))
    print(
        f"   {self_top}/{T} positions give the largest weight to themselves: x·x = |x|² is often the largest, "
        "so we need two different projections, Q and K"
    )


if __name__ == "__main__":
    main()
