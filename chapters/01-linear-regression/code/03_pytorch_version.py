"""Chapter 1 · From minimal code to production code: the standard PyTorch code for the same task.

This script does the same task as 01_fit_line.py. The differences are:
- autograd calculates the gradients. We do not derive them by hand (Chapter 4 shows how autograd works);
- the model is nn.Linear with one input and one output. The weight is a and the bias is b;
- torch.optim.SGD updates the parameters;
- the five-line training loop (forward → loss → zero the gradients → backward → update)
  is the same loop that trains a large model.
Run: uv run python chapters/01-linear-regression/code/03_pytorch_version.py
"""

import importlib.util
from pathlib import Path

import torch
from torch import nn

_spec = importlib.util.spec_from_file_location(
    "fit_line", Path(__file__).with_name("01_fit_line.py")
)
fit_line = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fit_line)


def main() -> None:
    x_np, y_np = fit_line.make_data()
    x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)  # shape (50, 1)
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

    model = nn.Linear(in_features=1, out_features=1)
    with torch.no_grad():  # use the same start point as the minimal code, so that we can compare
        model.weight.fill_(-1.0)
        model.bias.fill_(4.0)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()

    for step in range(201):
        y_hat = model(x)              # 1. forward pass: calculate the predictions
        loss = loss_fn(y_hat, y)      # 2. loss
        if step in (0, 10, 50, 200):  # print the parameters and the loss before this step's update
            a, b = model.weight.item(), model.bias.item()
            print(f"step {step:>3}: a = {a:.3f}, b = {b:.3f}, loss = {loss.item():.4f}")
        optimizer.zero_grad()         # 3. set the gradients from the last step to zero
        loss.backward()               # 4. backward pass: autograd calculates ∂L/∂a and ∂L/∂b
        optimizer.step()              # 5. update: a ← a − η·∂L/∂a

    # Parity check: the result must be the same as the minimal code with manual gradients
    a_ref, b_ref, _ = fit_line.gradient_descent(x_np, y_np, lr=0.05, steps=201)[-1]
    a, b = model.weight.item(), model.bias.item()
    print(f"\nAfter 201 updates: PyTorch a = {a:.3f}, b = {b:.3f}; manual gradients a = {a_ref:.3f}, b = {b_ref:.3f}")


if __name__ == "__main__":
    main()
