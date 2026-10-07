"""Chapter 3 · From minimal code to production code: the standard PyTorch code for the same two-layer MLP

This script does the same task as 03_mlp_numpy.py. The differences are:
- nn.Sequential(nn.Linear, nn.ReLU, nn.Linear) builds the network. We do not write the matrix products;
- loss.backward() calculates the gradients. We do not derive them by hand (in Chapter 4, you write it yourself);
- torch.optim.SGD does the update.
Two experiments:
1. Parity check: copy the initial parameters of the NumPy code into PyTorch. Use float64 and the same
   learning rate and number of steps. The losses must agree digit by digit;
2. Train from the start with the default PyTorch initialization and float32. Is the fit about as good?
Run: uv run python chapters/03-neural-network/code/04_pytorch_version.py
"""

import importlib.util
from pathlib import Path

import torch
from torch import nn

_spec = importlib.util.spec_from_file_location(
    "mlp_numpy", Path(__file__).with_name("03_mlp_numpy.py")
)
mlp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mlp)

HIDDEN = 64


def make_model(hidden: int = HIDDEN) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(1, hidden),    # Z = X·W1ᵀ + b1 (the PyTorch weight has the shape (out, in))
        nn.ReLU(),               # A = ReLU(Z)
        nn.Linear(hidden, 1),    # Ŷ = A·W2ᵀ + b2
    )


def fit(model: nn.Module, x: torch.Tensor, y: torch.Tensor, lr: float, steps: int,
        report=(0, 1000, 5000), verbose: bool = True) -> float:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for step in range(steps + 1):
        y_hat = model(x)              # 1. forward pass
        loss = loss_fn(y_hat, y)      # 2. loss
        if verbose and (step in report or step == steps):
            print(f"   step {step:>5}: loss = {loss.item():.4f}")
        if step == steps:
            return loss.item()
        optimizer.zero_grad()         # 3. set the gradients to zero
        loss.backward()               # 4. backward pass: autograd calculates the gradients of all parameters
        optimizer.step()              # 5. update


def run_match(verbose: bool = True):
    """Parity check: copy the NumPy initial parameters into PyTorch (float64). Returns (PyTorch loss, NumPy loss)."""
    x_np, y_np = mlp.make_data()
    torch.set_default_dtype(torch.float64)
    x, y = torch.tensor(x_np), torch.tensor(y_np)
    model = make_model()
    p0 = mlp.init_params(HIDDEN, seed=0)
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(p0["W1"].T))   # (1, H) → (H, 1)
        model[0].bias.copy_(torch.tensor(p0["b1"]))
        model[2].weight.copy_(torch.tensor(p0["W2"].T))   # (H, 1) → (1, H)
        model[2].bias.copy_(torch.tensor(p0["b2"]))
    loss_torch = fit(model, x, y, lr=mlp.LR, steps=mlp.STEPS, verbose=verbose)
    loss_numpy = mlp.train(x_np, y_np, HIDDEN)[1][-1]
    torch.set_default_dtype(torch.float32)
    return loss_torch, loss_numpy


def run_default(verbose: bool = True):
    """Default PyTorch initialization + float32, train from the start. Returns (final loss, number of parameters)."""
    x_np, y_np = mlp.make_data()
    torch.manual_seed(0)
    x, y = torch.tensor(x_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.float32)
    model = make_model()
    loss = fit(model, x, y, lr=mlp.LR, steps=mlp.STEPS, verbose=verbose)
    return loss, sum(p.numel() for p in model.parameters())


def main() -> None:
    print(f"1) Parity check: NumPy initial parameters copied into PyTorch (float64), width {HIDDEN}, SGD learning rate {mlp.LR}")
    loss_torch, loss_numpy = run_match()
    print(f"   PyTorch {loss_torch:.10f}  vs  NumPy with manual gradients {loss_numpy:.10f}, "
          f"difference = {abs(loss_torch - loss_numpy):.1e}")

    print(f"\n2) Default PyTorch initialization + float32, width {HIDDEN}, same training for {mlp.STEPS} steps")
    _, n = run_default()
    print(f"   {n} parameters (W1: {HIDDEN}, b1: {HIDDEN}, W2: {HIDDEN}, b2: 1)")


if __name__ == "__main__":
    main()
