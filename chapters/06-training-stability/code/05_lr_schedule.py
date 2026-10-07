"""Chapter 6 · Minimal code 5: learning-rate schedules (warmup + cosine / WSD) and gradient clipping

This script only defines the functions and prints values. Scripts 06 and 07 use them to train a network.
Run: uv run python chapters/06-training-stability/code/05_lr_schedule.py
"""

import math


def warmup_cosine(step: int, total: int, peak: float, warmup: int, min_ratio: float = 0.1) -> float:
    """Linear warmup to peak, then a cosine curve down to peak × min_ratio."""
    if step < warmup:
        return peak * (step + 1) / warmup                       # warmup: 0 → peak
    p = (step - warmup) / max(1, total - warmup)                # decay progress 0 → 1
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def wsd(step: int, total: int, peak: float, warmup: int, decay_frac: float = 0.2,
        min_ratio: float = 0.0) -> float:
    """Warmup-Stable-Decay: warmup to peak, keep it constant, then a linear decay to peak × min_ratio
    in the last decay_frac of the steps."""
    decay_start = int(total * (1 - decay_frac))
    if step < warmup:
        return peak * (step + 1) / warmup                       # W: warmup
    if step < decay_start:
        return peak                                             # S: stable, constant
    p = (step - decay_start) / max(1, total - decay_start)      # D: decay, linear
    return peak * (1 - (1 - min_ratio) * p)


def clip_by_global_norm(grads, max_norm: float = 1.0):
    """Gradient clipping: put the gradients of all parameters into one long vector.
    If its norm is more than max_norm, scale all gradients down by the same factor.

    The direction does not change; only the length is limited. Return the norm before
    clipping (training code often writes it to the log).
    """
    total = math.sqrt(sum(float((g * g).sum()) for g in grads))
    scale = min(1.0, max_norm / (total + 1e-6))
    for g in grads:
        g *= scale
    return total


if __name__ == "__main__":
    total, peak, warm = 1000, 3e-3, 100
    print(f"Total steps {total}, peak learning rate {peak}, warmup {warm} steps")
    print("  step   warmup+cosine WSD (last 20%: linear decay to 0)")
    for s in [0, 50, 99, 100, 300, 500, 700, 799, 800, 900, 999]:
        print(f"{s:>6d}   {warmup_cosine(s, total, peak, warm):.2e}      {wsd(s, total, peak, warm):.2e}")

    import numpy as np
    g = [np.array([3.0, 4.0]), np.array([12.0])]               # global norm = √(9+16+144) = 13
    before = clip_by_global_norm(g, max_norm=1.0)
    after = math.sqrt(sum(float((x * x).sum()) for x in g))
    print(f"\nGradient clipping: global norm {before:.1f} before clipping, {after:.3f} after; "
          f"components {[round(float(v), 4) for x in g for v in x]} (same direction)")
