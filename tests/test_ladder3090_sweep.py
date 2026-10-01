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


def test_narrow_ranges_follow_trend_and_stay_in_bounds(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    # 两个小档：最优学习率随 N 变小（N 翻 4 倍、lr 减半）；收窄的区间应该顺着这个趋势外推到大档
    import csv
    import json
    import math

    neffs = {"a": 4e6, "b": 16e6, "c": 64e6}
    monkeypatch.setattr(sweep, "HERE", tmp_path)
    monkeypatch.setattr(sweep, "n_eff", lambda s: neffs[s])
    (tmp_path / "configs").mkdir()
    (tmp_path / "results").mkdir()
    for scale, best_lr in (("a", 4e-3), ("b", 2e-3)):
        cfgs, rows = [], []
        for i in range(60):
            hp = sweep.sample_full(random.Random(f"{scale}/{i}"))
            cfg = {"id": f"{scale}-{i:03d}", "scale": scale, "space": "full", **hp}
            cfgs.append(cfg)
            loss = 1 + (math.log(hp["lr"] / best_lr)) ** 2  # 只有学习率重要
            rows.append({"id": cfg["id"], "phase": "decay", "k": 8, "val_bpb": loss})
        (tmp_path / "configs" / f"{scale}.jsonl").write_text("\n".join(json.dumps(c) for c in cfgs) + "\n")
        with (tmp_path / "results" / f"{scale}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    r = sweep.narrow_ranges("c", ["a", "b"])
    for h, (lo, hi) in r.items():
        full_lo, full_hi = sweep.FULL_Z[h]
        assert full_lo - 1e-9 <= lo < hi <= full_hi + 1e-9, h
        assert hi - lo >= 2 * sweep.MIN_HALF[h] - 1e-9, h
    center = math.exp(sum(r["lr"]) / 2)
    assert 3e-4 < center < 2e-3  # 外推到 c 档：比两个小档的最优值都小
    for i in range(50):
        hp = sweep.sample_in(random.Random(i), r)
        assert math.exp(r["lr"][0]) - 1e-12 <= hp["lr"] <= math.exp(r["lr"][1]) + 1e-12
        assert hp["tokens_per_step"] in {2**k for k in range(15, 22)}
