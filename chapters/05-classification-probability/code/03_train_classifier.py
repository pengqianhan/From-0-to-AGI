"""Chapter 5 · Minimal code 3: classify three spirals with an MLP + softmax + cross-entropy.

The code makes the data: three spiral arms that cross each other (a classic CS231n toy data set).
You do not need to download anything.
Model: a two-layer MLP 2 → 64 (ReLU) → 3 (Chapter 3). Backpropagation by hand (Chapter 4).
Full-batch gradient descent.
We train the same network with the same learning rate two times, with cross-entropy and with MSE,
and compare the results.
Uses only NumPy. It runs in a few seconds on a CPU.
Run: uv run python chapters/05-classification-probability/code/03_train_classifier.py
"""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("ce_mod", Path(__file__).with_name("02_cross_entropy.py"))
ce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ce)


def make_spirals(n_per_class: int = 100, n_classes: int = 3, noise: float = 0.2, seed: int = 0):
    """Three spiral arms. The points of class k turn outward along the radius r ∈ [0, 1].

    The start angles of the classes differ by 4 radians.
    """
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for k in range(n_classes):
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(k * 4.0, (k + 1) * 4.0, n_per_class) + rng.normal(0, noise, n_per_class)
        xs.append(np.stack([r * np.sin(t), r * np.cos(t)], axis=1))
        ys.append(np.full(n_per_class, k))
    return np.concatenate(xs), np.concatenate(ys)


def init_params(hidden: int = 64, n_in: int = 2, n_out: int = 3, seed: int = 0,
                out_std: float = 0.01) -> dict:
    """out_std: the standard deviation of the output-layer weights.

    The default is very small → initial logits ≈ 0 → initial predictions ≈ uniform distribution.
    Make it larger (for example 10). Then the model gives very "confident" random predictions
    from the start. We use this to show the problem of MSE.
    """
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, np.sqrt(2 / n_in), (n_in, hidden)), "b1": np.zeros(hidden),
        "W2": rng.normal(0, out_std, (hidden, n_out)), "b2": np.zeros(n_out),
    }


def forward(params: dict, X: np.ndarray):
    h_pre = X @ params["W1"] + params["b1"]
    h = np.maximum(h_pre, 0)                 # ReLU
    logits = h @ params["W2"] + params["b2"]  # z: one score for each class
    return logits, (X, h_pre, h)


def backward(params: dict, cache, dlogits: np.ndarray) -> dict:
    X, h_pre, h = cache
    dh = dlogits @ params["W2"].T
    dh_pre = dh * (h_pre > 0)
    return {"W1": X.T @ dh_pre, "b1": dh_pre.sum(0), "W2": h.T @ dlogits, "b2": dlogits.sum(0)}


def accuracy(params: dict, X: np.ndarray, y: np.ndarray) -> float:
    return float((forward(params, X)[0].argmax(1) == y).mean())


def train(X, y, loss: str = "ce", lr: float = 1.0, steps: int = 3000, hidden: int = 64,
          snapshot_steps=(), seed: int = 0, out_std: float = 0.01):
    """Full-batch gradient descent.

    loss="ce" uses cross-entropy. loss="mse" uses the mean squared error between the softmax
    probabilities and the onehot target.
    Returns (final parameters, log [(step, cross-entropy, accuracy)], {step: parameter snapshot}).
    """
    params = init_params(hidden, seed=seed, out_std=out_std)
    grad_fn = ce.ce_grad if loss == "ce" else ce.mse_grad
    log, snaps = [], {}
    for step in range(steps + 1):
        logits, cache = forward(params, X)
        if step % 100 == 0 or step in snapshot_steps:
            log.append((step, ce.cross_entropy(logits, y), accuracy(params, X, y)))
        if step in snapshot_steps:
            snaps[step] = {k: v.copy() for k, v in params.items()}
        if step == steps:
            break
        grads = backward(params, cache, grad_fn(logits, y))  # dlogits = (p − onehot)/N
        for k in params:
            params[k] -= lr * grads[k]
    return params, log, snaps


def train_linear(X, y, lr: float = 1.0, steps: int = 3000):
    """Comparison: softmax regression without a hidden layer (logits = XW + b).

    It can only make straight-line boundaries. Returns (W, b).
    """
    W, b = np.zeros((2, 3)), np.zeros(3)
    for _ in range(steps):
        g = ce.ce_grad(X @ W + b, y)
        W -= lr * X.T @ g
        b -= lr * g.sum(0)
    return W, b


if __name__ == "__main__":
    X, y = make_spirals()
    print(f"Data: {len(X)} points, {X.shape[1]} dimensions, {y.max() + 1} classes ({np.sum(y == 0)} for each class)")
    print(f"The initial loss must be about ln 3 = {np.log(3):.4f} (a model that knows nothing guesses uniformly)\n")

    params, log, _ = train(X, y, loss="ce")
    print("Cross-entropy training (lr = 1.0, full batch)")
    print("  step   CE loss  accuracy")
    for step, loss, acc in log:
        if step in (0, 100, 200, 500, 1000, 2000, 3000):
            print(f"{step:>6}   {loss:7.4f}    {acc:6.1%}")

    Xt, yt = make_spirals(seed=1)  # same distribution, different random seed: a separate test set
    print(f"Test-set accuracy (a second set of spiral data): {accuracy(params, Xt, yt):.1%}")

    print("\nComparison 1: a linear softmax classifier without a hidden layer")
    W_lin, b_lin = train_linear(X, y)
    acc_lin = ((X @ W_lin + b_lin).argmax(1) == y).mean()
    print(f"  Training accuracy after 3000 steps: {acc_lin:.1%} (straight boundaries cannot separate the spirals)")

    for title, out_std in [("Comparison 2: normal initialization (initial predictions ≈ uniform)", 0.01),
                           ("Comparison 3: output-layer weight std increased to 10"
                            " (confident random guesses from the start)", 10.0)]:
        print(f"\n{title}. Same MLP, same lr = 1.0, cross-entropy vs MSE (softmax probabilities vs onehot)")
        print("              CE training       |     MSE training\n"
              "  step   cross-ent    accuracy  | cross-ent    accuracy")
        _, log_ce, _ = train(X, y, loss="ce", out_std=out_std)
        _, log_mse, _ = train(X, y, loss="mse", out_std=out_std)
        for (s, l1, a1), (_, l2, a2) in zip(log_ce, log_mse, strict=True):
            if s in (0, 100, 500, 800, 1000, 3000):
                print(f"{s:>6}   {l1:9.4f}   {a1:9.1%}  | {l2:9.4f}   {a2:9.1%}")
    # Why does MSE training get stuck? After 500 steps, count the samples that are still
    # "confidently wrong" (probability of the correct class < 1%).
    for loss in ["ce", "mse"]:
        p500, _, _ = train(X, y, loss=loss, steps=500, out_std=10.0)
        p_true = ce.softmax(forward(p500, X)[0])[np.arange(len(y)), y]
        print(f"Comparison 3 · {loss.upper():>3} training, after 500 steps: "
              f"{np.sum(p_true < 0.01)} samples with p(correct) < 1%")
