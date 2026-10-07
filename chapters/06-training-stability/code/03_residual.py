"""Chapter 6 · Minimal code 3: residual connections — a direct path for the signal

The same 30 blocks. Each block is f(h) = ReLU(RMSNorm(h) · W1) · W2 (the Pre-Norm form).
We compare:
  - plain stack: h ← f(h)
  - residual connection: h ← h + f(h)
We look at three things:
  1. Can the deep layers still tell different inputs apart? (the mean pairwise cosine
     similarity: a value near 1 means that the inputs look the same);
  2. How does the scale of the residual stream change with depth? Must we make the
     output projection W2 smaller as the depth increases?
  3. How much of the error signal arrives back at block 1? Does it still point in the
     same direction as the signal at block 30?
Run: uv run python chapters/06-training-stability/code/03_residual.py
"""

import importlib.util
import math
from pathlib import Path

import torch

torch.set_num_threads(1)

_spec = importlib.util.spec_from_file_location(
    "norm", Path(__file__).with_name("02_normalization.py"))
norm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(norm)

DEPTH, WIDTH, BATCH = 30, 256, 256


def mean_cosine(h: torch.Tensor) -> float:
    """The mean cosine similarity between all pairs of samples in a batch."""
    u = h / h.norm(dim=-1, keepdim=True)
    sim = u @ u.T
    n = h.shape[0]
    return ((sim.sum() - n) / (n * (n - 1))).item()


def run(residual: bool, w2_scale: float = 1.0, depth: int = DEPTH, seed: int = 0):
    """Return, after each block: (cosine similarity, residual-stream std, error-signal std),
    and the direction similarity of the error signals at block 1 and block 30."""
    g = torch.Generator().manual_seed(seed)
    w1s = [torch.randn(WIDTH, WIDTH, generator=g) * math.sqrt(2 / WIDTH) for _ in range(depth)]
    w2s = [torch.randn(WIDTH, WIDTH, generator=g) * math.sqrt(1 / WIDTH) * w2_scale
           for _ in range(depth)]
    gamma = torch.ones(WIDTH)
    x = torch.randn(BATCH, WIDTH, generator=torch.Generator().manual_seed(seed + 1))
    x.requires_grad_()
    h, hs = x, []
    for w1, w2 in zip(w1s, w2s):
        f = torch.relu(norm.rms_norm(h, gamma) @ w1) @ w2
        h = h + f if residual else f       # ← the only difference: "h +" or not
        h.retain_grad()
        hs.append(h)
    probe = torch.randn(WIDTH, generator=torch.Generator().manual_seed(seed + 2)) / WIDTH ** 0.5
    (norm.rms_norm(h, gamma) @ probe).sum().backward()
    cos = [mean_cosine(t.detach()) for t in hs]
    std = [t.std().item() for t in hs]
    grad = [t.grad.std().item() for t in hs]
    # For the same sample: how similar in direction are the error signals at block 1 and block 30?
    g1, gL = hs[0].grad, hs[-1].grad
    same_dir = torch.nn.functional.cosine_similarity(g1, gL, dim=-1).mean().item()
    return cos, std, grad, same_dir


# video/scenes.py looks up the results by these keys, so the keys stay in Chinese.
# VARIANT_EN below gives the English names for the printed output.
VARIANTS = {
    "普通堆叠": dict(residual=False),
    "残差（W2 不缩放）": dict(residual=True),
    "残差（W2 × 1/√(2L)）": dict(residual=True, w2_scale=1 / math.sqrt(2 * DEPTH)),
}
# Display names for the printed output
VARIANT_EN = {
    "普通堆叠": "plain stack",
    "残差（W2 不缩放）": "residual (W2 not scaled)",
    "残差（W2 × 1/√(2L)）": "residual (W2 × 1/√(2L))",
}


if __name__ == "__main__":
    x = torch.randn(BATCH, WIDTH, generator=torch.Generator().manual_seed(1))
    print(f"Inputs: mean pairwise cosine similarity = {mean_cosine(x):.3f} (random vectors are almost orthogonal)\n")
    shown = [1, 10, 20, 30]
    for name, kw in VARIANTS.items():
        cos, std, grad, same_dir = run(**kw)
        print(f"── {VARIANT_EN[name]}")
        print("   block         " + "".join(f"{i:>9d}" for i in shown))
        print("   cosine sim    " + "".join(f"{cos[i - 1]:>9.3f}" for i in shown))
        print("   residual std  " + "".join(f"{std[i - 1]:>9.3f}" for i in shown))
        print(f"   error signal: size at block 1 / size at block 30 = {grad[0] / grad[-1]:.3f}, "
              f"direction similarity = {same_dir:.3f}\n")
