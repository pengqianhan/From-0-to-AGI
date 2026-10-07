"""Chapter 4 · From minimal code to production code: the same MLP again, with PyTorch autograd.

1. Gradient parity check: copy the initial weights of the Value MLP into PyTorch without change,
   calculate the same loss, and compare the gradients parameter by parameter.
2. Training parity check: train both versions for 500 steps with SGD (lr = 0.1) and compare the final loss.
3. Speed: scalar computational graph vs. tensor computational graph.
4. torch.autograd.gradcheck: the gradient check that PyTorch includes.
5. Vector-Jacobian product (VJP): what the tensor version of backpropagation calculates.

Run: uv run python chapters/04-backprop-autograd/code/04_pytorch_compare.py
"""

import importlib.util
import math
import random
import time
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("train_mlp", HERE / "03_train_mlp.py")
train_mlp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_mlp)
Value, MLP = train_mlp.Value, train_mlp.MLP

torch.set_num_threads(1)               # one thread is fastest for small matrices and gives more stable timing
torch.set_default_dtype(torch.float64)  # Value uses Python float (double precision), so both sides use the same precision in the parity check


def to_torch(net) -> nn.Sequential:
    """Copy the weights of the Value MLP into nn.Linear. The weight of nn.Linear has the shape (out, in)."""
    mods = []
    for i, layer in enumerate(net.layers):
        lin = nn.Linear(len(layer.neurons[0].w), len(layer.neurons))
        with torch.no_grad():
            lin.weight.copy_(torch.tensor([[w.data for w in n.w] for n in layer.neurons]))
            lin.bias.copy_(torch.tensor([n.b.data for n in layer.neurons]))
        mods.append(lin)
        if i < len(net.layers) - 1:
            mods.append(nn.Tanh())
    return nn.Sequential(*mods)


def value_grads_as_tensors(net) -> list[torch.Tensor]:
    """Put the gradients of the Value version in the order of the PyTorch parameters (weight, bias of each layer)."""
    out = []
    for layer in net.layers:
        out.append(torch.tensor([[w.grad for w in n.w] for n in layer.neurons]))
        out.append(torch.tensor([n.b.grad for n in layer.neurons]))
    return out


def time_value_step(net, xs, ys, reps: int) -> float:
    t0 = time.perf_counter()
    for _ in range(reps):
        loss = train_mlp.mse(net, xs, ys)
        net.zero_grad()
        loss.backward()
    return (time.perf_counter() - t0) / reps


def time_torch_step(model, x, y, reps: int) -> float:
    t0 = time.perf_counter()
    for _ in range(reps):
        loss = ((model(x) - y) ** 2).mean()
        model.zero_grad()
        loss.backward()
    return (time.perf_counter() - t0) / reps


if __name__ == "__main__":
    xs, ys = train_mlp.make_data()
    x = torch.tensor(xs).unsqueeze(1)          # shape (20, 1): one pass for the full batch of samples
    y = torch.tensor(ys).unsqueeze(1)

    # ── 1. Gradient parity check ─────────────────────────────────────────────
    random.seed(0)                             # the same initialization as train() in 03_train_mlp.py
    net = MLP(1, [8, 8, 1])
    model = to_torch(net)
    loss_v = train_mlp.mse(net, xs, ys)
    net.zero_grad()
    loss_v.backward()
    loss_t = ((model(x) - y) ** 2).mean()      # nn.MSELoss() calculates the same expression
    model.zero_grad()
    loss_t.backward()
    print(f"Initial loss: Value = {loss_v.data:.12f}, PyTorch = {loss_t.item():.12f}")
    names = [n for n, _ in model.named_parameters()]
    max_diff = 0.0
    for name, p, g in zip(names, model.parameters(), value_grads_as_tensors(net), strict=True):
        ok = torch.allclose(p.grad, g, rtol=1e-9, atol=1e-12)
        diff = (p.grad - g).abs().max().item()
        max_diff = max(max_diff, diff)
        print(f"  {name:<9} shape {str(tuple(p.shape)):<7} allclose = {ok}  max difference {diff:.1e}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params} parameters in total, max gradient difference {max_diff:.1e}")

    # ── 2. Training parity check ─────────────────────────────────────────────
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(500):
        loss_t = ((model(x) - y) ** 2).mean()
        opt.zero_grad()
        loss_t.backward()
        opt.step()
    final_t = ((model(x) - y) ** 2).mean().item()
    _, losses, _ = train_mlp.train(steps=500, lr=0.1)
    print(f"\nLoss after 500 SGD steps: Value = {losses[-1]:.10f}, PyTorch = {final_t:.10f}")

    # ── 3. Speed: one forward pass + one backward pass ──────────────────────
    print("\nTime for one forward pass + one backward pass:")
    print(" samples  Value (scalar)  PyTorch (tensor)       ratio")
    for n in (20, 200):
        xs_n = [-3 + 6 * i / (n - 1) for i in range(n)]
        ys_n = [math.sin(v) for v in xs_n]
        tv = time_value_step(net, xs_n, ys_n, reps=10)
        xt, yt = torch.tensor(xs_n).unsqueeze(1), torch.tensor(ys_n).unsqueeze(1)
        time_torch_step(model, xt, yt, reps=20)   # warm-up
        tt = time_torch_step(model, xt, yt, reps=500)
        print(f"  {n:>6}   {tv * 1000:10.1f} ms   {tt * 1000:12.3f} ms   {tv / tt:8.0f}×")

    # ── 4. torch.autograd.gradcheck: check autograd with numerical gradients, the same idea as 02_grad_check.py ──
    params = {k: v.detach().clone().requires_grad_(True) for k, v in model.named_parameters()}

    def loss_of_params(*ps):
        out = torch.func.functional_call(model, dict(zip(params, ps, strict=True)), (x,))
        return ((out - y) ** 2).mean()
    ok = torch.autograd.gradcheck(loss_of_params, tuple(params.values()), eps=1e-6, atol=1e-6)
    print(f"\ntorch.autograd.gradcheck (all {n_params} parameters): {ok}")

    # ── 5. Vector-Jacobian product: the backward pass of Y = X·W multiplies the upstream gradient G by Wᵀ ──
    torch.manual_seed(0)
    X = torch.randn(4, 3, requires_grad=True)
    W = torch.randn(3, 2)
    Y = X @ W
    G = torch.randn(4, 2)                      # the gradient ∂L/∂Y that comes back from upstream
    (gX,) = torch.autograd.grad(Y, X, grad_outputs=G)
    print(f"VJP: the ∂L/∂X from autograd is equal to G·Wᵀ: {torch.allclose(gX, G @ W.T)}"
          f" (the full Jacobian has {Y.numel()}×{X.numel()} = {Y.numel() * X.numel()} numbers;"
          f" autograd never builds it)")
