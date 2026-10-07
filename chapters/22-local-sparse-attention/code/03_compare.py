"""Chapter 22 · Minimal code 3: full attention vs sliding window vs local-global interleaving, three sets of real numbers.

  1. Validation loss of language modeling (character-level Shakespeare): the three are about the same.
  2. KV cache: how many positions each configuration must cache, at the training length 128 and at an
     imagined context of 4096.
  3. Needle in a haystack: the accuracy in three ranges of the distance d between the "needle" and the
     question. Only here do we see the problem of the sliding window.

Run 02_swa_model.py first to train. (Or run this script directly: it trains the missing models.)
Run: uv run python chapters/22-local-sparse-attention/code/03_compare.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)
_spec = importlib.util.spec_from_file_location(
    "swa_model", Path(__file__).resolve().parent / "02_swa_model.py"
)
m = importlib.util.module_from_spec(_spec)
sys.modules["swa_model"] = m  # dataclass must find its module in sys.modules
_spec.loader.exec_module(m)


def cached_positions(windows: list, T: int) -> int:
    """Sum of the cached positions over all layers: a full-attention layer stores T, a sliding-window layer stores min(W, T)."""
    return sum(T if w is None else min(w, T) for w in windows)


def bucket_means(acc: torch.Tensor, n_layers: int) -> list[float]:
    """Split the accuracies for d = 1..95 into three ranges: in the window, reachable only by relay across layers, and out of reach of the relay."""
    rf = n_layers * (m.W - 1)  # receptive field of L layers with sliding windows only
    d = torch.arange(1, len(acc) + 1)
    segs = [d < m.W, (d >= m.W) & (d <= rf), d > rf]
    return [acc[s].mean().item() for s in segs]


def main() -> None:
    rf = 4 * (m.W - 1)
    print(f"Window W = {m.W}, 4 layers; theoretical receptive field with sliding windows only = 4 × (W − 1) = {rf}\n")
    print(
        f"{'Config':<11}{'Windows per layer':<22}{'LM val loss':>12}{'KV pos@128':>12}{'KV pos@4096':>13}"
    )
    for variant, windows in m.VARIANTS.items():
        model = m.load_or_train("lm", variant)
        loss = m.lm_val_loss(model)
        print(
            f"{variant:<11}{str(windows):<22}{loss:>12.3f}"
            f"{cached_positions(windows, 128):>12}{cached_positions(windows, 4096):>13}"
        )

    print(f"\nNeedle-in-a-haystack accuracy (1 of 8, random guess = 12.5%; sequence length {m.NEEDLE_T}, 64 samples at each distance)")
    print(
        f"{'Config':<11}{f'd < {m.W} (window)':>16}{f'{m.W} ≤ d ≤ {rf} (relay)':>20}"
        f"{f'd > {rf} (far)':>16}"
    )
    curves = {}
    for variant in m.VARIANTS:
        model = m.load_or_train("needle", variant)
        acc = m.needle_accuracy(model)
        curves[variant] = acc
        a, b, c = bucket_means(acc, 4)
        print(f"{variant:<11}{a:>16.1%}{b:>20.1%}{c:>16.1%}")

    # A finer curve: the mean over each group of 8 distances, for plots (the video uses these numbers)
    print("\nBy distance (mean accuracy over each group of 8 distances):")
    edges = list(range(1, m.NEEDLE_T, 8))
    print("  d from    :", " ".join(f"{e:>4}" for e in edges))
    for variant, acc in curves.items():
        vals = [acc[e - 1 : e + 7].mean().item() for e in edges]
        print(f"  {variant:<10}:", " ".join(f"{v:>4.0%}" for v in vals))


if __name__ == "__main__":
    main()
