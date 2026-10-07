"""Chapter 5 · Minimal code 1: softmax changes a set of scores into a probability distribution.

Uses only NumPy. It runs almost at once on a CPU.
Run: uv run python chapters/05-classification-probability/code/01_softmax.py
"""

import numpy as np


def softmax_naive(z: np.ndarray) -> np.ndarray:
    """Softmax from the definition: p_k = exp(z_k) / Σ_j exp(z_j). It overflows when the logits are large."""
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax: subtract the maximum first, then apply exp.

    softmax(z) = softmax(z − c) for all constants c. With c = max(z), the largest exponential is exactly e⁰ = 1.
    temperature T: divide the logits by T first. T < 1 makes the distribution sharper.
    T > 1 makes it flatter (sampling in Chapter 10 uses this).
    """
    z = np.asarray(z, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# The example for the whole chapter: does an image show a cat, a dog, or a bird?
# The model outputs three scores (logits).
# The class names (cat, dog, bird) stay in Chinese because the Chinese video imports CLASSES.
CLASSES = ["猫", "狗", "鸟"]
CLASSES_EN = {"猫": "cat", "狗": "dog", "鸟": "bird"}  # display names for the printed output
LOGITS = np.array([2.0, 1.0, -1.0])


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    p = softmax(LOGITS)
    print("logits (real numbers):", LOGITS)
    print("exp(logits)          :", np.exp(LOGITS))
    print("softmax (probability):", p, " sum =", p.sum())
    print("naive on logits + 100:", softmax_naive(LOGITS + 100),
          "(add one constant to all logits: the result does not change)")

    print("\n—— Numerical stability ——")
    big = LOGITS * 500  # [1000, 500, -500]
    with np.errstate(over="ignore", invalid="ignore"):
        print("naive  softmax([1000, 500, -500]) =", softmax_naive(big), " ← exp(1000) overflows to inf")
    print("stable softmax([1000, 500, -500]) =", softmax(big))
    print("Largest x before exp(x) overflows: float64 about", np.log(np.finfo(np.float64).max).round(1),
          "; float32 about", np.log(np.finfo(np.float32).max).round(1))

    print("\n—— Temperature T: divide the logits by T first ——")
    for t in [0.5, 1.0, 2.0, 10.0]:
        print(f"T = {t:>4}:", softmax(LOGITS, temperature=t))
