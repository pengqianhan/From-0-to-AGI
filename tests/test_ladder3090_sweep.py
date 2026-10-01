"""runs/ladder-3090/sweep.py 的步数与采样（阶梯实验的衰减分叉必须恰好从第 k/8 个 checkpoint 开始）。"""

from __future__ import annotations

import importlib.util
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from zero.train.schedule import wsd

_spec = importlib.util.spec_from_file_location(
    "ladder_sweep", Path(__file__).resolve().parents[1] / "runs" / "ladder-3090" / "sweep.py"
)
sweep = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sweep  # @dataclass 要从 sys.modules 里找到模块
_spec.loader.exec_module(sweep)


@pytest.mark.parametrize("c", [8, 18, 36, 74, 294, 588, 2352, 9999])
def test_decay_starts_exactly_at_checkpoint(c: int) -> None:
    m, frac = sweep.with_decay(c)
    assert m == c + round(c / 4)
    # c 之前学习率是峰值（稳定段），从 c 开始下降，最后一步降到 0
    assert wsd(c - 1, 1, m, frac) == 1.0
    assert wsd(c, 1, m, frac) < 1.0
    assert wsd(m - 1, 1, m, frac) == 0.0


def test_plan_keeps_budget_and_checkpoint_grid() -> None:
    neff = 4.83e6
    for tps in [2**15, 2**18, 2**21]:
        cfg = {"scale": "e5m", "tokens_per_step": tps, "warmup_frac": 0.01}
        p = sweep.plan_for(cfg, neff)
        assert p.stable_steps % 8 == 0 and p.every * 8 == p.stable_steps
        assert p.micro * p.accum * sweep.SEQ == tps
        # 量化到 8 步的整数倍后，token 数与 32 × N_eff 的差不超过 4 步
        assert abs(p.stable_steps * tps - 32 * neff) <= 4 * tps
        assert p.step_at(Fraction(2, 8)) == 2 * p.every


def test_sampling_is_deterministic_and_in_range() -> None:
    a = sweep.sample_full(random.Random("e5m/7/full"))
    b = sweep.sample_full(random.Random("e5m/7/full"))
    assert a == b
    for i in range(200):
        hp = sweep.sample_full(random.Random(i))
        assert 2**15 <= hp["tokens_per_step"] <= 2**21
        assert 1e-5 <= hp["lr"] <= 1e-1
        assert 0.7 <= hp["beta1"] <= 0.999 and 0.9 <= hp["beta2"] <= 0.9999
        assert 1e-3 <= hp["warmup_frac"] <= 1 / 8
        assert 1e-4 <= hp["weight_decay"] <= 1.0
        assert 2**10 <= hp["rope_theta"] <= 2**20
