"""Steps and sampling of runs/ladder-3090/sweep.py.

Each decay branch of the ladder experiment must start exactly at checkpoint k/8.
"""

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
sys.modules[_spec.name] = sweep  # @dataclass must find the module in sys.modules
_spec.loader.exec_module(sweep)


@pytest.mark.parametrize("c", [8, 18, 36, 74, 294, 588, 2352, 9999])
def test_decay_starts_exactly_at_checkpoint(c: int) -> None:
    m, frac = sweep.with_decay(c)
    assert m == c + round(c / 4)
    # Before c, the learning rate is at the peak (stable phase). It decreases from c and is 0 at the last step
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
        # After rounding to a multiple of 8 steps, the token count is within 4 steps of 32 × N_eff
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
    # Two small sizes: the best learning rate decreases with N (4× N, half the lr).
    # The narrowed ranges must extrapolate this trend to the large size
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
            loss = 1 + (math.log(hp["lr"] / best_lr)) ** 2  # Only the learning rate matters
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
    assert 3e-4 < center < 2e-3  # Extrapolation to size c: smaller than the best values of both small sizes
    for i in range(50):
        hp = sweep.sample_in(random.Random(i), r)
        assert math.exp(r["lr"][0]) - 1e-12 <= hp["lr"] <= math.exp(r["lr"][1]) + 1e-12
        assert hp["tokens_per_step"] in {2**k for k in range(15, 22)}


def test_ext_sampling_extends_batch_downward(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    # Boundary extension: batch is 2^13–2^15, and the other hyperparameters stay near the best config of this size.
    # Only append; do not change the existing configs
    import argparse
    import csv
    import json

    monkeypatch.setattr(sweep, "HERE", tmp_path)
    monkeypatch.setattr(sweep, "n_eff", lambda s: 4e6)
    (tmp_path / "configs").mkdir()
    (tmp_path / "results").mkdir()
    cfgs = [{"id": f"a-{i:03d}", "scale": "a", "space": "full", **sweep.sample_full(random.Random(i))} for i in range(40)]
    (tmp_path / "configs" / "a.jsonl").write_text("\n".join(json.dumps(c) for c in cfgs) + "\n")
    with (tmp_path / "results" / "a.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "phase", "k", "val_bpb"])
        w.writeheader()
        w.writerows({"id": c["id"], "phase": "decay", "k": 8, "val_bpb": 1 + c["tokens_per_step"] / 2**21} for c in cfgs)
    sweep.cmd_sample(argparse.Namespace(scale="a", n=52, space="ext", sources=["a"]))
    out = sweep.load_configs("a")
    assert out[:40] == cfgs and len(out) == 52
    for c in out[40:]:
        assert c["space"] == "ext" and c["tokens_per_step"] in {2**13, 2**14, 2**15}
        for h, (lo, hi) in sweep.FULL_Z.items():
            if h != "tokens_per_step":
                assert lo - 1e-9 <= sweep.to_z(h, c[h]) <= hi + 1e-9, h
    assert (tmp_path / "results" / "a_ext_ranges.json").exists()


def test_pruned_run_counts_as_reached(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    # After prune deletes the checkpoints, an evaluation of this step in the log means that the run reached the step.
    # The run must not train again from the start
    import json

    monkeypatch.setattr(sweep, "OUT", tmp_path)
    cfg = {"id": "a-000", "scale": "a", "tokens_per_step": 2**16, "warmup_frac": 0.01}
    plan = sweep.Plan(stable_steps=80, micro=4, accum=8, warmup_steps=1)
    d = sweep.run_dir(cfg)
    d.mkdir(parents=True)
    (d / "log.jsonl").write_text("\n".join(json.dumps({"step": s, "val_bpb": 2.0}) for s in (10, 20, 30, 40)) + "\n")
    assert sweep.reached(cfg, plan, 20) and sweep.reached(cfg, plan, 40)
    assert not sweep.reached(cfg, plan, 50)
