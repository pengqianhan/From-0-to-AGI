"""Paired bootstrap confidence intervals and the "ahead / tie / behind" decision (Chapters 11 and 20; GOAL.md 3.2 item 5).

Two models have per-item scores a_i and b_i on **the same set of items** (correct / wrong = 1 / 0,
or a continuous score). We want to know how reliable the difference of the mean scores
d = mean(a) − mean(b) is. The paired bootstrap:

1. Draw n items with replacement from the n items. Both models use **the same indices**. This is the
   "paired" part: it cancels the variation in item difficulty.
2. Compute d* on these items.
3. Do this n_boot times. The 2.5% and 97.5% quantiles of d* are the 95% confidence interval
   [lo, hi] (percentile method).

The decision (GOAL.md 3.2 item 5: "the lead must be outside the bootstrap 95% confidence interval
to count as 'ahead'; if not, we can only say 'tie'"):

- lo > 0: the full interval is to the right of 0 → **ahead**.
- hi < 0: the full interval is to the left of 0 → **behind**.
- In all other cases, the interval contains 0 → **tie**.

If the opponent has a thinking mode and a non-thinking mode, compare with the mode in which the
opponent has the **higher score** (GOAL.md 3.2 item 4). See `compare_to_opponent`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

AHEAD, TIE, BEHIND = "ahead", "tie", "behind"


@dataclass
class BootstrapResult:
    mean_a: float
    mean_b: float
    diff: float  # mean_a - mean_b
    ci_low: float
    ci_high: float
    n: int
    n_boot: int
    confidence: float
    decision: str  # ahead / tie / behind (a compared with b)


def decide(ci_low: float, ci_high: float) -> str:
    if ci_low > 0:
        return AHEAD
    if ci_high < 0:
        return BEHIND
    return TIE


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
    chunk: int = 1000,
) -> BootstrapResult:
    """a and b are the per-item scores on the same set of items (in the same order)."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"The paired bootstrap needs 1-D scores of equal length: {x.shape} vs {y.shape}")
    n = len(x)
    if n == 0:
        raise ValueError("No items")
    d = x - y
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for s in range(0, n_boot, chunk):
        m = min(chunk, n_boot - s)
        idx = rng.integers(0, n, size=(m, n))
        stats[s : s + m] = d[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(
        mean_a=float(x.mean()),
        mean_b=float(y.mean()),
        diff=float(d.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        n=n,
        n_boot=n_boot,
        confidence=confidence,
        decision=decide(float(lo), float(hi)),
    )


def compare_to_opponent(
    ours: Sequence[float],
    opponent_modes: Mapping[str, Sequence[float]],
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[str, BootstrapResult]:
    """If the opponent has several modes (for example thinking / non-thinking), compare with the mode that has the highest mean score.

    Returns (mode name, result).
    """
    if not opponent_modes:
        raise ValueError("At least one opponent mode is necessary")
    best = max(opponent_modes, key=lambda k: float(np.mean(opponent_modes[k])))
    return best, paired_bootstrap(ours, opponent_modes[best], n_boot=n_boot, seed=seed)


def _weighted(means: Mapping[str, float], weights: Mapping[str, float]) -> float:
    total = sum(weights.values())
    return sum(weights[k] * means[k] for k in weights) / total


def stratified_paired_bootstrap(
    a_by: Mapping[str, Sequence[float]],
    b_by: Mapping[str, Sequence[float]],
    weights: Mapping[str, float],
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Stratified paired bootstrap: total score = weighted sum of the mean scores of the categories.

    BFCL, for example, weights its categories (see E1 in eval/PREREGISTRATION.md).
    In each category, the items are resampled with replacement independently (both models use the
    same indices). Then the weights combine the differences into a difference of the total scores.
    With one category and weight 1, the result is identical to paired_bootstrap.
    """
    keys = list(weights)
    if set(keys) != set(a_by) or set(keys) != set(b_by):
        raise ValueError(f"The categories do not match: weights {sorted(keys)}, a {sorted(a_by)}, b {sorted(b_by)}")
    total_w = float(sum(weights.values()))
    if total_w <= 0:
        raise ValueError("The sum of the weights must be positive")
    rng = np.random.default_rng(seed)
    stats = np.zeros(n_boot)
    n_total = 0
    for k in keys:
        x = np.asarray(a_by[k], dtype=np.float64)
        y = np.asarray(b_by[k], dtype=np.float64)
        if x.shape != y.shape or x.ndim != 1 or len(x) == 0:
            raise ValueError(f"Category {k}: needs non-empty 1-D scores of equal length")
        d = x - y
        n = len(d)
        n_total += n
        idx = rng.integers(0, n, size=(n_boot, n))
        stats += weights[k] / total_w * d[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    mean_a = _weighted({k: float(np.mean(a_by[k])) for k in keys}, weights)
    mean_b = _weighted({k: float(np.mean(b_by[k])) for k in keys}, weights)
    return BootstrapResult(
        mean_a=mean_a,
        mean_b=mean_b,
        diff=mean_a - mean_b,
        ci_low=float(lo),
        ci_high=float(hi),
        n=n_total,
        n_boot=n_boot,
        confidence=confidence,
        decision=decide(float(lo), float(hi)),
    )


def overall_verdict(decisions: Mapping[tuple[str, str], str]) -> str:
    """Overall verdict (intersection-union test). Keys are (opponent, endpoint); values are "ahead / tie / behind".

    We can claim "ahead" only if the decision is "ahead" against **every** opponent on **every**
    preregistered endpoint. If any combination is "behind", the overall verdict is "behind".
    All other cases are "tie". This is the conservative reading of GOAL.md 3.2: "better than all
    models of the same size".
    """
    if not decisions:
        raise ValueError("No comparison results")
    values = set(decisions.values())
    if values == {AHEAD}:
        return AHEAD
    if BEHIND in values:
        return BEHIND
    return TIE
