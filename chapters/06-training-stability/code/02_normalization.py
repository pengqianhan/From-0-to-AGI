"""Chapter 6 · Minimal code 2: normalization — LayerNorm and RMSNorm

1. Write the two normalizations by hand. Use a 4-dimensional vector to see what each one does.
2. Put RMSNorm into the 30-layer network of script 1 (each layer normalizes, then multiplies
   by the matrix). Even with a "wrong" initialization, the signal does not collapse or explode.
Run: uv run python chapters/06-training-stability/code/02_normalization.py
"""

import importlib.util
from pathlib import Path

import torch

torch.set_num_threads(1)

_spec = importlib.util.spec_from_file_location(
    "signal", Path(__file__).with_name("01_signal_propagation.py"))
signal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(signal)


def layer_norm(x, gamma, beta, eps=1e-6):
    """LayerNorm: subtract the mean, divide by the std, then multiply by γ and add β.
    The calculation is separate for each sample (the last dimension)."""
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu) ** 2).mean(-1, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps) * gamma + beta


def rms_norm(x, gamma, eps=1e-6):
    """RMSNorm: do not subtract the mean. Only divide by the root mean square
    RMS(x) = sqrt(mean(x²)), then multiply by γ."""
    rms = torch.sqrt((x * x).mean(-1, keepdim=True) + eps)
    return x / rms * gamma


def normed_layer_stats(std: float, depth: int = signal.DEPTH, width: int = signal.WIDTH,
                       seed: int = 0):
    """The same network as in script 01, but each layer is h_l = ReLU(RMSNorm(h_{l-1}) · W_l)."""
    ws = signal.make_weights(std, depth, width, seed)
    gamma = torch.ones(width)
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(signal.BATCH, width, generator=g)
    h, hs = x, []
    for w in ws:
        h = torch.relu(rms_norm(h, gamma) @ w)
        h.retain_grad()
        hs.append(h)
    probe = torch.randn(width, generator=g) / width ** 0.5
    (rms_norm(h, gamma) @ probe).sum().backward()  # also normalize before the output (the final norm of Pre-Norm)
    # After normalization, the scale (the input that the next layer "sees") is always 1.
    # Thus we report the input to the norm, h_l
    return [t.std().item() for t in hs], [t.grad.std().item() for t in hs]


if __name__ == "__main__":
    x = torch.tensor([2.0, 4.0, 6.0, 8.0])
    one, zero = torch.ones(4), torch.zeros(4)
    ln, rn = layer_norm(x, one, zero), rms_norm(x, one)
    print("input x                =", x.tolist())
    print(f"LayerNorm(x)           = {[round(v, 3) for v in ln.tolist()]}"
          f"   mean {ln.mean():.3f}, RMS {ln.pow(2).mean().sqrt():.3f}")
    print(f"RMSNorm(x)             = {[round(v, 3) for v in rn.tolist()]}"
          f"   mean {rn.mean():.3f}, RMS {rn.pow(2).mean().sqrt():.3f}")
    xc = x - x.mean()
    print(f"x minus its mean: max difference between LayerNorm and RMSNorm = "
          f"{(layer_norm(xc, one, zero) - rms_norm(xc, one)).abs().max():.2e}")
    print(f"x × 100: max change of the RMSNorm output = "
          f"{(rms_norm(100 * x, one) - rn).abs().max():.2e} (scale invariance)")
    print("Learnable parameters (width d): LayerNorm has γ and β, 2d in total; RMSNorm has only γ, d in total\n")

    shown = [1, 10, 20, 30]
    print(f"{signal.DEPTH}-layer ReLU MLP with RMSNorm before each layer:")
    print("   init         " + "".join(f"{'   layer ' + str(i) + ' act':>12s}" for i in shown)
          + "   grad1/grad30")
    for name in ["std = 1.0", "std = 0.01", "std = 0.02", "Kaiming"]:
        act, grad = normed_layer_stats(signal.INITS[name])
        print(f"   {name:<10s}  " + "".join(f"{act[i - 1]:>15.3g}" for i in shown)
              + f"   {grad[0] / grad[-1]:>12.3g}")
