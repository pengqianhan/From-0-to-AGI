"""Chapter 2 · From minimal code to production code: multivariate linear regression in standard PyTorch

This script does the same task as 04_multivariate_regression.py:
- The model is nn.Linear(3, 1). It takes the full batch at one time: input (N, 3) → output (N, 1).
- Note: PyTorch keeps the weight as (out, in) = (1, 3). The forward pass calculates x @ weight.T + bias.
- The training loop is the same as the five lines of Chapter 1:
  forward pass → loss → set gradients to zero → backward pass → update.
At the end, a parity check compares the result with the NumPy version (matrix gradients by hand).
Run: uv run python chapters/02-from-scalar-to-matrix/code/06_pytorch_version.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn

_spec = importlib.util.spec_from_file_location(
    "multivariate_regression", Path(__file__).with_name("04_multivariate_regression.py")
)
reg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reg)


def train_torch(Xs: np.ndarray, y: np.ndarray, dtype=torch.float64, lr=0.1, steps=200,
                verbose=False):
    x = torch.tensor(Xs, dtype=dtype)          # (N, 3)
    t = torch.tensor(y, dtype=dtype)           # (N, 1)

    model = nn.Linear(in_features=3, out_features=1, dtype=dtype)
    with torch.no_grad():                      # the same start point as the NumPy version: W = 0, b = 0
        model.weight.zero_()
        model.bias.zero_()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for step in range(steps):
        y_hat = model(x)                       # 1. forward pass: (N, 3) → (N, 1)
        loss = loss_fn(y_hat, t)               # 2. loss
        if verbose and step in (0, 10, 50):
            print(f"  Step {step:>3}: loss = {loss.item():.3f}")
        optimizer.zero_grad()                  # 3. set the gradients to zero
        loss.backward()                        # 4. backward pass: autograd calculates ∂L/∂W = 2/N·Xᵀ(ŷ−y)
        optimizer.step()                       # 5. update
    return model


def main() -> None:
    torch.manual_seed(0)
    X, y = reg.make_data()
    Xs, mu, sigma = reg.standardize(X)

    # The forward pass of nn.Linear is x @ weight.T + bias
    layer = nn.Linear(3, 1, dtype=torch.float64)
    xb = torch.tensor(Xs[:4])
    manual = xb @ layer.weight.T + layer.bias
    print(f"nn.Linear(3, 1): weight shape {tuple(layer.weight.shape)}, bias shape {tuple(layer.bias.shape)}")
    print(f"Input batch {tuple(xb.shape)} → output {tuple(layer(xb).shape)}; "
          f"maximum difference from x @ weight.T + bias {(layer(xb) - manual).abs().max().item():.1e}\n")

    print("PyTorch training (float64):")
    model = train_torch(Xs, y, verbose=True)
    W_t = model.weight.detach().numpy().T      # (3, 1), the same shape as in the NumPy version
    b_t = model.bias.detach().numpy()

    W_n, b_n, _ = reg.gradient_descent(Xs, y, lr=0.1, steps=200)[-1]
    print("\nAfter 200 steps (standardized space):")
    print(f"  PyTorch  W = {W_t.ravel().round(3)}, b = {b_t.round(3)}")
    print(f"  NumPy    W = {W_n.ravel().round(3)}, b = {b_n.round(3)}")
    diff64 = max(np.abs(W_t - W_n).max(), np.abs(b_t - b_n).max())
    print(f"  float64 maximum difference = {diff64:.1e}")

    m32 = train_torch(Xs, y, dtype=torch.float32)
    diff32 = max(np.abs(m32.weight.detach().numpy().T - W_n).max(),
                 np.abs(m32.bias.detach().numpy() - b_n).max())
    print(f"  float32 (default precision of PyTorch) maximum difference = {diff32:.1e}")

    W_o, b_o = reg.to_original_units(W_t, b_t, mu, sigma)
    print(f"\nConverted back to the original units: w = {W_o.ravel().round(3)}, b = {b_o.round(3)}")


if __name__ == "__main__":
    main()
