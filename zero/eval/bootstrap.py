"""配对 bootstrap 置信区间与"超过 / 持平 / 落后"判定（对应第 11、20 章；GOAL.md 3.2 第 5 条）。

两个模型在**同一组题**上各有逐题得分 a_i、b_i（对 / 错记 1 / 0，或连续分数）。我们关心平均分之差
d = mean(a) − mean(b) 有多可信。配对 bootstrap：

1. 从 n 道题里有放回地抽 n 道（两个模型用**同一组下标**——这就是"配对"，题目难度的波动被抵消）；
2. 算这组题上的 d*；
3. 重复 n_boot 次，取 d* 的 2.5% 与 97.5% 分位数作为 95% 置信区间 [lo, hi]（百分位法）。

判定（GOAL.md 3.2 第 5 条："领先幅度要超出 bootstrap 95% 置信区间才算'超过'，否则只能说'持平'"）：

- lo > 0：整个区间都在 0 右边 → **超过**；
- hi < 0：整个区间都在 0 左边 → **落后**；
- 否则区间跨过 0 → **持平**。

对手有思考 / 非思考两种模式时，取对手**较高分**的模式来比（GOAL.md 3.2 第 4 条），见 `compare_to_opponent`。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

AHEAD, TIE, BEHIND = "超过", "持平", "落后"


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
    decision: str  # 超过 / 持平 / 落后（a 相对 b）


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
    """a、b 是同一组题上的逐题得分（顺序对齐）。"""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"配对 bootstrap 需要等长的一维得分：{x.shape} vs {y.shape}")
    n = len(x)
    if n == 0:
        raise ValueError("没有题目")
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
    """对手有多种模式（如 thinking / non-thinking）时，取平均分最高的模式来比较。返回 (模式名, 结果)。"""
    if not opponent_modes:
        raise ValueError("至少要有一种对手模式")
    best = max(opponent_modes, key=lambda k: float(np.mean(opponent_modes[k])))
    return best, paired_bootstrap(ours, opponent_modes[best], n_boot=n_boot, seed=seed)
