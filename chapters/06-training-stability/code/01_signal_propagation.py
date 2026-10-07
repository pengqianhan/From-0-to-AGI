"""Chapter 6 · Minimal code 1: how does the signal move through the layers of a plain 30-layer MLP?

There is no training. We look only at the moment after initialization. We send a batch of
random inputs through the network and print, for each layer:
  - forward pass: the standard deviation of the layer output (the activations);
  - backward pass: the standard deviation of the gradient of the loss for the activations
    of that layer (the error signal).
We compare these initializations: weight std 1.0, 0.01, and 0.02, plus two that scale with
the input size: Xavier (sqrt(1/fan_in)) and Kaiming (sqrt(2/fan_in)).
Run: uv run python chapters/06-training-stability/code/01_signal_propagation.py
"""

import math

import torch

torch.set_num_threads(1)  # one thread is faster for small matrices and easier to reproduce

DEPTH, WIDTH, BATCH = 30, 256, 512


def make_weights(std: float, depth: int = DEPTH, width: int = WIDTH, seed: int = 0):
    """Make depth weight matrices of size width×width. Each element ~ N(0, std²)."""
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(width, width, generator=g) * std).requires_grad_() for _ in range(depth)]


def layer_stats(std: float, depth: int = DEPTH, width: int = WIDTH, seed: int = 0):
    """Return (list of activation std per layer, list of gradient std per layer).

    Network: h_l = ReLU(h_{l-1} · W_l). No bias, no normalization, no residual connection.
    Loss: the dot product of each output with one fixed random direction, summed over the
    samples. Its only purpose is to send a gradient back through the network.
    "Gradient" means ∂loss/∂h_l: the size of the error signal when it arrives at layer l.
    """
    ws = make_weights(std, depth, width, seed)
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(BATCH, width, generator=g)
    h, hs = x, []
    for w in ws:
        h = torch.relu(h @ w)          # one layer: linear + ReLU
        h.retain_grad()                # keep the gradients of the hidden layers
        hs.append(h)
    probe = torch.randn(width, generator=g) / math.sqrt(width)
    loss = (h @ probe).sum()
    loss.backward()
    act_std = [t.std().item() for t in hs]
    grad_std = [t.grad.std().item() for t in hs]
    return act_std, grad_std


def kaiming_std(fan_in: int = WIDTH) -> float:
    """The correct initialization for a ReLU network: Var(W) = 2 / fan_in."""
    return math.sqrt(2 / fan_in)


INITS = {
    "std = 1.0": 1.0,
    "std = 0.01": 0.01,
    "std = 0.02": 0.02,                       # a common initializer_range in large-model configs
    "Xavier": math.sqrt(1 / WIDTH),           # Var = 1/fan_in: ignores that ReLU sets half the values to 0
    "Kaiming": kaiming_std(),                 # Var = 2/fan_in
}


if __name__ == "__main__":
    shown = [1, 5, 10, 15, 20, 25, 30]
    print(f"ReLU MLP with {DEPTH} layers, width {WIDTH} per layer; input std = 1\n")
    for name, std in INITS.items():
        act, grad = layer_stats(std)
        print(f"── Initialization {name} (weight std {std:.4g})")
        print("   layer     " + "".join(f"{i:>10d}" for i in shown))
        print("   act std   " + "".join(f"{act[i - 1]:>10.3g}" for i in shown))
        print("   grad std  " + "".join(f"{grad[i - 1]:>10.3g}" for i in shown))
        print(f"   grad at layer 1 / grad at layer 30 = {grad[0] / grad[-1]:.3g}\n")

    # Why does Kaiming use 2/fan_in? Each layer multiplies the std by about std · sqrt(fan_in / 2)
    for name, std in INITS.items():
        print(f"{name:>11s}: gain per layer ≈ {std * math.sqrt(WIDTH / 2):.3f}, "
              f"after 30 layers ≈ {(std * math.sqrt(WIDTH / 2)) ** DEPTH:.3g}")
