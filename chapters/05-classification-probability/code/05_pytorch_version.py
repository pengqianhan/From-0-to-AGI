"""Chapter 5 · From minimal code to production code: F.cross_entropy and the standard PyTorch training loop.

1. Parity check: torch.nn.functional.cross_entropy(logits, y) gives the same digits as our NumPy version.
   Note: its input is logits (not probabilities). Inside, it fuses log-softmax and the negative
   log-likelihood, so it is numerically stable.
2. Numerical stability: with logits = [1000, 500, −500], softmax and then log gives inf.
   F.cross_entropy does not.
3. label_smoothing: make the onehot target a little "flatter", and compare with the formula by hand.
4. Train the same spiral classifier with nn.Sequential + torch.optim.SGD. Compare with the NumPy
   version with manual backpropagation.
5. The code in a language model: the logits have the shape (B, T, V). Flatten them to (B·T, V),
   then calculate the cross-entropy.
Run: uv run python chapters/05-classification-probability/code/05_pytorch_version.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ce = _load("ce_mod", "02_cross_entropy.py")
clf = _load("clf_mod", "03_train_classifier.py")


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(1)  # the tensors are small: thread scheduling costs more than the calculation

    print("—— 1. Parity check: F.cross_entropy vs NumPy ——")
    rng = np.random.default_rng(0)
    z_np, y_np = rng.normal(0, 3, size=(8, 5)), rng.integers(0, 5, size=8)
    z, y = torch.tensor(z_np, requires_grad=True), torch.tensor(y_np)
    loss = F.cross_entropy(z, y)  # the input is logits, not probabilities after softmax
    loss.backward()
    print(f"PyTorch: {loss.item():.10f}   NumPy: {ce.cross_entropy(z_np, y_np):.10f}")
    print(f"autograd gradient vs manual (p − onehot)/N: max difference "
          f"{np.abs(z.grad.numpy() - ce.ce_grad(z_np, y_np)).max():.2e}")
    same = F.nll_loss(F.log_softmax(z, dim=-1), y).item()
    print(f"Equivalent code nll_loss(log_softmax(z)): {same:.10f}")

    print("\n—— 2. Numerical stability ——")
    zb = torch.tensor([[1000.0, 500.0, -500.0]])
    yb = torch.tensor([2])
    naive = -torch.log(torch.exp(zb[0, 2]) / torch.exp(zb).sum())
    print(f"Naive code −log(exp(z_y)/Σexp(z)): {naive.item()}")
    print(f"F.cross_entropy (fused log-softmax): {F.cross_entropy(zb, yb).item()}")

    print("\n—— 3. label_smoothing = 0.1 ——")
    eps, K = 0.1, z_np.shape[1]
    ls = F.cross_entropy(z.detach(), y, label_smoothing=eps).item()
    logp = ce.log_softmax(z_np)
    target = np.full_like(logp, eps / K)            # each class first gets ε/K
    target[np.arange(len(y_np)), y_np] += 1 - eps   # the correct class gets 1 − ε more
    manual = float(-(target * logp).sum(axis=1).mean())
    print(f"PyTorch: {ls:.10f}   manual −Σ q_k log p_k (q = smoothed target): {manual:.10f}")

    print("\n—— 4. Train the spiral classifier with PyTorch; parity check with the manual NumPy version ——")
    X_np, Y_np = clf.make_spirals()
    X, Y = torch.tensor(X_np), torch.tensor(Y_np)
    model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 3)).double()
    p0 = clf.init_params()  # use the same initial parameters as the NumPy version
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(p0["W1"].T))
        model[0].bias.copy_(torch.tensor(p0["b1"]))
        model[2].weight.copy_(torch.tensor(p0["W2"].T))
        model[2].bias.copy_(torch.tensor(p0["b2"]))
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    for step in range(3001):
        logits = model(X)                    # 1. forward pass: get the logits
        loss = F.cross_entropy(logits, Y)    # 2. cross-entropy (log-softmax inside)
        if step in (0, 500, 3000):
            acc = (logits.argmax(1) == Y).double().mean().item()
            print(f"step {step:>4}: cross-entropy = {loss.item():.4f}, accuracy = {acc:.1%}")
        if step == 3000:
            final_loss = loss.item()
            break
        optimizer.zero_grad()                # 3. set the gradients to zero
        loss.backward()                      # 4. backward pass: autograd calculates (p − onehot)/N and sends it back
        optimizer.step()                     # 5. update

    _, log_np, _ = clf.train(X_np, Y_np, loss="ce", steps=3000)
    print(f"Manual NumPy version, step 3000: cross-entropy = {log_np[-1][1]:.4f}, accuracy = {log_np[-1][2]:.1%}")
    print(f"Difference between the two cross-entropies at step 3000: {abs(final_loss - log_np[-1][1]):.2e}")

    print("\n—— 5. The code in a language model ——")
    B, T, V = 2, 4, 50257  # 2 sequences × 4 positions each × the GPT-2 vocabulary size
    logits = torch.zeros(B, T, V)  # all zeros = uniform guess
    targets = torch.randint(0, V, (B, T))
    lm_loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
    print(f"logits {tuple(logits.shape)} → flattened to {(B * T, V)}; loss = {lm_loss.item():.4f}, "
          f"ln V = {np.log(V):.4f}")


if __name__ == "__main__":
    main()
