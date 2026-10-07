"""Chapter 5 · Minimal code 2: maximum likelihood → cross-entropy, and its gradient p − onehot.

1. Maximum likelihood: make the probability of the correct answer as large as possible.
   A product of many probabilities underflows, so we take the log.
2. Negative log-likelihood (NLL) = cross-entropy.
3. The gradient of softmax + cross-entropy for the logits is p − onehot. A numerical gradient checks it.
4. Why classification does not use MSE: when the model is "confidently wrong", the MSE gradient is almost 0.
Run: uv run python chapters/05-classification-probability/code/02_cross_entropy.py
"""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("softmax_mod", Path(__file__).with_name("01_softmax.py"))
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)
softmax = sm.softmax


def log_softmax(z: np.ndarray) -> np.ndarray:
    """log p_k = z_k − logsumexp(z). Subtract the maximum first here too: no overflow in exp, no log(0)."""
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def cross_entropy(logits: np.ndarray, y: np.ndarray) -> float:
    """Cross-entropy = mean negative log-likelihood: L = −1/N · Σ_i log p_{i, y_i}.

    logits has the shape (N, K). y has the shape (N,).
    """
    logits = np.atleast_2d(logits)
    y = np.atleast_1d(y)
    return float(-log_softmax(logits)[np.arange(len(y)), y].mean())


def ce_grad(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """∂L/∂z = (p − onehot(y)) / N. This is the most important line of the chapter."""
    logits = np.atleast_2d(logits)
    y = np.atleast_1d(y)
    p = softmax(logits)
    p[np.arange(len(y)), y] -= 1.0
    return p / len(y)


def mse_on_probs(logits: np.ndarray, y: np.ndarray) -> float:
    """Comparison: the mean squared error between the softmax probabilities and the onehot target.

    L = 1/N · Σ_i Σ_k (p_ik − t_ik)²
    """
    logits = np.atleast_2d(logits)
    y = np.atleast_1d(y)
    p = softmax(logits)
    t = np.eye(p.shape[1])[y]
    return float(((p - t) ** 2).sum(axis=1).mean())


def mse_grad(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of the MSE for the logits. The chain rule goes through the softmax Jacobian J = diag(p) − p pᵀ.

    ∂L/∂z = J · 2(p − t) = 2 · p ⊙ (g − Σ_k p_k g_k), where g = p − t.
    """
    logits = np.atleast_2d(logits)
    y = np.atleast_1d(y)
    p = softmax(logits)
    g = p - np.eye(p.shape[1])[y]
    return 2 * p * (g - (p * g).sum(axis=1, keepdims=True)) / len(y)


def numerical_grad(f, z: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central difference: (f(z + ε) − f(z − ε)) / 2ε, one element at a time.

    This is the gradient check from Chapters 1 and 4.
    """
    g = np.zeros_like(z)
    for idx in np.ndindex(z.shape):
        zp, zm = z.copy(), z.copy()
        zp[idx] += eps
        zm[idx] -= eps
        g[idx] = (f(zp) - f(zm)) / (2 * eps)
    return g


def confident_wrong_table(gaps=(0, 2, 4, 6, 8, 10)):
    """The true class is 0. The model raises the logit of the wrong class 1 to s.

    A larger s gives a more confident wrong answer.
    For each s, return: p(correct), CE loss, CE gradient norm, MSE loss, MSE gradient norm.
    """
    rows = []
    for s in gaps:
        z = np.array([[0.0, float(s), 0.0]])
        y = np.array([0])
        rows.append((
            s,
            softmax(z)[0, 0],
            cross_entropy(z, y),
            np.linalg.norm(ce_grad(z, y)),
            mse_on_probs(z, y),
            np.linalg.norm(mse_grad(z, y)),
        ))
    return rows


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    logits, classes = sm.LOGITS, sm.CLASSES
    p = softmax(logits)

    print("—— 1. Likelihood: the product underflows ——")
    for n in [10, 100, 1000, 10000]:
        with np.errstate(under="ignore"):
            prod = np.prod(np.full(n, 0.9))
        print(f"{n:>6} samples, each gives the correct answer p = 0.9: likelihood = 0.9^{n} = {prod:.3e}"
              f", log-likelihood = {n * np.log(0.9):.2f}")

    print("\n—— 2. Cross-entropy = −log p(correct class) ——")
    print("logits =", logits, " softmax =", p)
    for k, name in enumerate(classes):
        print(f'Correct answer "{sm.CLASSES_EN[name]}": p = {p[k]:.4f}, loss = −ln p = {cross_entropy(logits, np.array([k])):.4f}')
    print(f"A model that knows nothing (uniform guess over 3 classes): −ln(1/3) = {np.log(3):.4f}")

    print("\n—— 3. Gradient = p − onehot, numerical check ——")
    y = np.array([0])
    z = logits[None, :].copy()
    g_ana = ce_grad(z, y)
    g_num = numerical_grad(lambda t: cross_entropy(t, y), z)
    print("Analytic gradient (p − onehot):    ", g_ana[0])
    print("Numerical gradient (central diff.):", g_num[0])
    print(f"Max difference: {np.abs(g_ana - g_num).max():.2e}")
    rng = np.random.default_rng(0)
    zb, yb = rng.normal(0, 3, size=(8, 5)), rng.integers(0, 5, size=8)
    diff = np.abs(ce_grad(zb, yb) - numerical_grad(lambda t: cross_entropy(t, yb), zb)).max()
    diff_mse = np.abs(mse_grad(zb, yb) - numerical_grad(lambda t: mse_on_probs(t, yb), zb)).max()
    print(f"Random 8 samples × 5 classes: max difference, analytic vs numerical gradient: "
          f"CE {diff:.2e}; MSE {diff_mse:.2e}")

    print("\n—— 4. Confidently wrong: the gradient of CE vs MSE ——")
    print("wrong logit s    p(correct)   CE loss   |CE grad|   MSE loss   |MSE grad|")
    for s, pt, ce, gce, mse, gmse in confident_wrong_table():
        print(f"{s:>12}      {pt:9.2e}   {ce:7.3f}   {gce:9.4f}   {mse:8.4f}   {gmse:10.2e}")
