"""学习率调度：warmup、cosine、WSD 的关键点。"""

from __future__ import annotations

import math

import pytest
import torch

from zero.config import ScheduleConfig
from zero.train.schedule import LRScheduler, cosine_with_warmup, lr_multiplier_fn, wsd


def test_cosine() -> None:
    assert cosine_with_warmup(0, 10, 100, 0.1) == pytest.approx(0.1)
    assert cosine_with_warmup(9, 10, 100, 0.1) == pytest.approx(1.0)
    assert cosine_with_warmup(10, 10, 100, 0.1) == pytest.approx(1.0)
    assert cosine_with_warmup(55, 10, 100, 0.1) == pytest.approx(0.1 + 0.9 * 0.5)
    assert cosine_with_warmup(100, 10, 100, 0.1) == pytest.approx(0.1)
    assert cosine_with_warmup(1000, 10, 100, 0.1) == pytest.approx(0.1)
    vals = [cosine_with_warmup(s, 10, 100, 0.1) for s in range(10, 101)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


@pytest.mark.parametrize("shape", ["linear", "cosine", "sqrt"])
def test_wsd(shape: str) -> None:
    total, warm = 100, 10
    f = [
        wsd(s, warm, total, decay_frac=0.2, min_lr_ratio=0.0, decay_shape=shape)
        for s in range(total)
    ]
    assert f[0] == pytest.approx(0.1)
    assert all(v == 1.0 for v in f[warm:80])  # 稳定段
    decay = f[80:]
    assert all(a >= b for a, b in zip(decay, decay[1:]))
    assert decay[0] < 1.0
    assert f[-1] == pytest.approx(0.0)


def test_wsd_pure_decay_for_midtrain() -> None:
    f = [wsd(s, 0, 50, decay_frac=1.0, min_lr_ratio=0.0) for s in range(50)]
    assert f[0] == pytest.approx(1 - 1 / 50) and f[-1] == pytest.approx(0.0)


def test_no_decay_wsd_is_constant() -> None:
    assert [wsd(s, 5, 100, decay_frac=0.0) for s in range(5, 100)] == [1.0] * 95


def test_scheduler_sets_optimizer_lr() -> None:
    p = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW([p], lr=123.0)
    sched = LRScheduler(
        opt,
        ScheduleConfig(kind="cosine", warmup_steps=4, min_lr_ratio=0.0),
        base_lr=1e-3,
        total_steps=20,
    )
    assert sched.apply(1) == pytest.approx(0.5e-3)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.5e-3)
    assert sched.state_dict()["step_count"] == 1
    fn = lr_multiplier_fn(ScheduleConfig(kind="constant", warmup_steps=0), 10)
    assert fn(5) == 1.0
    assert not math.isnan(sched.lr_at(10**6))
