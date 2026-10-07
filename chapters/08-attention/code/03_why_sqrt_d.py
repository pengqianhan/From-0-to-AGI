"""Chapter 8 · Minimal code 3: why we divide by √d

Let each component of q and k be a random number with mean 0 and variance 1.
Then the dot product q·k = Σ qᵢkᵢ is a sum of d terms, each with variance 1, so its variance is d.
A larger d spreads the scores more, so softmax comes closer to one-hot (it looks at almost
only one position), and the gradient is almost 0. Division by √d brings the variance back to 1,
for every d.

Uses only NumPy. It finishes in one or two seconds on the CPU.
Run: uv run python chapters/08-attention/code/03_why_sqrt_d.py
"""

import numpy as np

DIMS = [16, 64, 256, 1024]
T = 16  # each query has 16 keys
TRIALS = 2000  # number of trials


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def stats(d: int, scale: bool, rng: np.random.Generator) -> dict:
    q = rng.normal(size=(TRIALS, 1, d))
    k = rng.normal(size=(TRIALS, T, d))
    scores = (q * k).sum(-1)  # (TRIALS, T): each row is the scores of one query for T keys
    if scale:
        scores = scores / np.sqrt(d)
    p = softmax(scores)
    entropy = -(p * np.log(p + 1e-30)).sum(-1)
    # The softmax Jacobian is J = diag(p) − ppᵀ. Its size sets how much gradient flows back to the scores
    jac = np.einsum("ni,ij->nij", p, np.eye(T)) - np.einsum("ni,nj->nij", p, p)
    return {
        "var": scores.var(),
        "max_w": p.max(-1).mean(),  # largest weight (closer to 1 = closer to one-hot)
        "eff_n": np.exp(entropy).mean(),  # "effective count" = e^entropy; = T when uniform
        "jac": np.linalg.norm(jac, axis=(1, 2)).mean(),
    }


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"Each component of q, k ~ N(0, 1); each query takes a softmax over T = {T} keys; mean over {TRIALS} trials\n")
    header = f"{'d':>6} | {'score var.':>8} {'max weight':>8} {'eff. count':>8} {'grad. size':>8}"
    for scale in (False, True):
        print("No scaling: softmax(q·k)" if not scale else "\nWith scaling: softmax(q·k / √d)")
        print(header)
        for d in DIMS:
            s = stats(d, scale, rng)
            print(
                f"{d:>6} | {s['var']:>10.1f} {s['max_w']:>10.3f} {s['eff_n']:>10.2f} {s['jac']:>10.3f}"
            )
    print(
        "\n(Effective count = e^entropy: 16 when the weights are spread evenly over 16 positions, "
        "1 when all weight is on one position. Gradient size = Frobenius norm of the softmax Jacobian.)"
    )


if __name__ == "__main__":
    main()
