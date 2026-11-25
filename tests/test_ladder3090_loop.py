"""runs/ladder-3090/loop.py and `analyze.py loop` (README stage 7).

- The configs of the looped scales have the same steps, warmup and batch as the dense run.
- `analyze.py loop` must not crash when some results are still missing (it reports them), and it
  computes the preregistered comparisons when the results are there.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "runs" / "ladder-3090"
sys.path.insert(0, str(HERE))

import analyze  # noqa: E402
import loop  # noqa: E402
import sweep  # noqa: E402


def test_plan_matches_preregistration() -> None:
    loop.check_prereg()
    for dense, (S, w) in loop.PREREG_PLAN.items():
        for st in loop.STRUCTS:
            c = loop.make(dense, st, 0, 1.0)
            assert c["scale"] == f"{dense}_{st}" and c["id"] == f"{dense}_{st}-000"
            pl = sweep.plan_for(c, sweep.n_eff(c["scale"]))
            dense_pl = sweep.plan_for(loop.base_cfg(dense), sweep.n_eff(dense))
            assert (pl.stable_steps, pl.warmup_steps, pl.micro * pl.accum) == (S, w, 32)
            assert (pl.stable_steps, pl.warmup_steps, pl.accum) == (
                dense_pl.stable_steps, dense_pl.warmup_steps, dense_pl.accum)
            assert sweep.run_dir(c) == sweep.OUT / f"{dense}_{st}" / f"{dense}_{st}-000"  # not a dense run dir
            # all hyperparameters of the dense base config
            b = loop.base_cfg(dense)
            assert all(c[h] == b[h] for h in loop.HP)
            cmd = sweep.train_cmd(c, pl, sweep.run_dir(c), 10, 0.2, 0, 1, 1)
            assert cmd[cmd.index("--config") + 1].endswith(f"configs/ladder3090/{dense}_{st}.toml")
    assert loop.make("e24m", "lm3", 2, 2.0)["lr"] == 2 * loop.base_cfg("e24m")["lr"]


def _log(d: Path, step: int, val: float) -> None:
    d.mkdir(parents=True, exist_ok=True)
    with (d / "log.jsonl").open("a") as f:
        f.write(json.dumps({"step": step, "val_bpb": val, "tok_per_s": 1e5}) + "\n")


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    res, cfgs = tmp_path / "results", tmp_path / "configs"
    res.mkdir()
    cfgs.mkdir()
    for s in ("e11m", "e24m", "e44m", "e83m"):
        shutil.copy(HERE / "results" / f"{s}.csv", res)
        shutil.copy(HERE / "configs" / f"{s}.jsonl", cfgs)
    monkeypatch.setattr(analyze, "RES", res)
    monkeypatch.setattr(sweep, "HERE", tmp_path)
    monkeypatch.setattr(sweep, "OUT", tmp_path / "out")
    return tmp_path


def _run() -> dict:
    analyze.cmd_loop(argparse.Namespace())
    return json.loads((analyze.RES / "loop_result.json").read_text())


def test_analyze_loop_with_missing_and_partial_results(sandbox: Path) -> None:
    # 1. nothing yet
    out = _run()
    assert out["law_matches_prereg"] and len(out["missing"]) == 8
    # 2. e24m_lf3: the three lr-check branches at 4/8, no decision yet
    cfgs = [loop.make("e24m", "lf3", i, m) for i, m in loop.LR_CHECK]
    loop.write("e24m_lf3", cfgs)
    for c, v in zip(cfgs, (1.40, 1.41, 1.45)):
        _log(sweep.run_dir(c), 5840, v + 0.03)
        _log(sweep.run_dir(c, 4), 7300, v)
    sweep.cmd_collect(argparse.Namespace(scale="e24m_lf3"))
    out = _run()
    s = out["scales"]["e24m_lf3"]
    assert s["lr_check_k4"] == {"x1.0": 1.40, "x0.5": 1.41, "x2.0": 1.45}
    assert any("loop_lr_lf3" in m for m in out["missing"])
    # 3. decision + the chosen trunk at 8/8 (2% below the dense prediction 1.2790)
    (analyze.RES / "decisions.json").write_text(json.dumps({"loop_lr_lf3": {"value": 1.0}}))
    _log(sweep.run_dir(cfgs[0]), 14600, 1.2790 * 0.98)
    sweep.cmd_collect(argparse.Namespace(scale="e24m_lf3"))
    out = _run()
    s = out["scales"]["e24m_lf3"]
    assert s["config"] == "e24m_lf3-000"
    k8 = s["k8"]
    assert k8["same_compute"]["pred_matches_prereg"] and k8["same_compute"]["verdict"] == "循环更好"
    assert k8["same_params"]["D_matches"] and k8["same_params"]["dense_val_bpb"] == pytest.approx(1.3458, abs=1e-4)
    assert s["k4"]["same_params"]["dense_val_bpb"] == pytest.approx(1.3821, abs=1e-4)
    assert k8["dense_equivalent_N_over_n_eff_loop"] > 1  # better than the prediction → a larger dense model
    # 4. e44m_lf3 at 8/8 (equal to the e83m interpolation)
    c = loop.make("e44m", "lf3", 0, 1.0)
    loop.write("e44m_lf3", [c])
    _log(sweep.run_dir(c), 27060, 1.1538)
    sweep.cmd_collect(argparse.Namespace(scale="e44m_lf3"))
    out = _run()
    e = out["scales"]["e44m_lf3"]["k8"]
    assert e["e83m_interp"]["value"] == pytest.approx(1.1538, abs=5e-5)
    assert e["same_compute"]["verdict"] == "持平" and e["same_compute"]["pred_matches_prereg"]
    assert any("e44m_lf3" in m and "k=4" in m for m in out["missing"])


def _fake_sweep(calls: list[tuple], k4: dict) -> object:
    """Fake training: record the sweep calls, write the val_bpb the rules read (None = diverges)."""

    def fake_sweep(*a: object) -> bool:
        a = tuple(map(str, a))
        calls.append(a)
        cmd, scale = a[0], a[a.index("--scale") + 1]
        if cmd == "collect":
            return True
        cfgs = {c["id"]: c for c in sweep.load_configs(scale)}
        i = a.index("--ids") + 1
        while i < len(a) and not a[i].startswith("--"):
            c = cfgs[a[i]]
            i += 1
            S = c["stable_steps"]
            if cmd == "run" and sweep.failed(sweep.run_dir(c)):
                continue
            v = k4.get(scale, {}).get(c["lr_mult"], 1.2)
            diverge = v is None or (scale == "e44m_lf3" and c["id"].endswith("-000"))
            d = sweep.run_dir(c, 4 if cmd == "branch" else None)
            if diverge:
                d.mkdir(parents=True, exist_ok=True)
                (d / "FAILED").write_text("FloatingPointError: Loss diverged: nan\n")
                continue
            if cmd == "branch":
                _log(d, sweep.with_decay(S // 2)[0], v)
            elif a[a.index("--until") + 1] == "1":
                _log(d, sweep.with_decay(S)[0], v - 0.05)
            else:
                _log(d, S // 2, v + 0.03)
        return True

    return fake_sweep


def test_loop_schedule_order_lr_rule_and_retry(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """e24m_lf3: x0.5 wins by more than 0.5%; e24m_lm3: x2 diverges (not adopted, x1 stays);
    e44m_lf3 diverges once (lr halved, new id)."""
    import orchestrate as orc

    monkeypatch.setattr(orc, "RES", analyze.RES)
    calls: list[tuple] = []
    k4 = {"e24m_lf3": {1.0: 1.40, 0.5: 1.39, 2.0: 1.41}, "e24m_lm3": {1.0: 1.40, 0.5: 1.401, 2.0: None}}
    fake_sweep = _fake_sweep(calls, k4)
    monkeypatch.setattr(orc, "sweep", fake_sweep)
    monkeypatch.setattr(orc, "analyze", lambda *a: True)
    monkeypatch.setattr(sys, "argv", ["loop.py"])
    loop.main()
    dec = json.loads((analyze.RES / "decisions.json").read_text())
    assert dec["loop_lr_lf3"]["value"] == 0.5 and dec["loop_lr_lm3"]["value"] == 1.0
    assert dec["loop_lr_retry_e44m_lf3"]["value"] == 0.25 and "loop_lr_retry_e44m_lm3" not in dec
    trained = [c for c in calls if c[0] in ("run", "branch")]
    scales = [c[2] for c in trained]
    assert scales.index("e24m_lm3") > max(i for i, s in enumerate(scales) if s == "e24m_lf3")
    assert scales.index("e44m_lm3") > max(i for i, s in enumerate(scales) if s == "e24m_lm3")
    assert scales.index("e44m_lf3") > max(i for i, s in enumerate(scales) if s == "e44m_lm3")
    # the chosen e24m_lf3 trunk (x0.5 = -001) is the only one trained to 8/8
    lf3_full = [c for c in calls if c[0] == "run" and c[2] == "e24m_lf3" and c[c.index("--until") + 1] == "1"]
    assert len(lf3_full) == 1 and lf3_full[0][lf3_full[0].index("--ids") + 1] == "e24m_lf3-001"
    # e44m_lf3: retried with half the lr under a new id, and the retry got its branch
    e44 = {c["id"]: c for c in sweep.load_configs("e44m_lf3")}
    assert e44["e44m_lf3-001"]["lr_mult"] == 0.25 and e44["e44m_lf3-001"]["retry_of"] == "e44m_lf3-000"
    assert ("branch", "e44m_lf3") in [(c[0], c[2]) for c in trained if c[c.index("--ids") + 1] == "e44m_lf3-001"]
    assert [c[0] for c in calls[-4:]] == ["collect"] * 4
    # rerun: idempotent, the decisions are reused
    loop.main()
    assert json.loads((analyze.RES / "decisions.json").read_text()) == dec


def test_loop_stop_in_one_structure_does_not_block_the_other(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """e24m_lf3 x1 diverges (no rule: stop). e44m_lf3 is skipped, lm3 still runs, collect and analyze
    still run, and the script ends with a non-zero exit."""
    import orchestrate as orc

    monkeypatch.setattr(orc, "RES", analyze.RES)
    calls: list[tuple] = []
    analyzed: list[tuple] = []
    k4 = {"e24m_lf3": {1.0: None, 0.5: 1.39, 2.0: 1.41}, "e24m_lm3": {1.0: 1.40, 0.5: 1.401, 2.0: 1.402}}
    monkeypatch.setattr(orc, "sweep", _fake_sweep(calls, k4))
    monkeypatch.setattr(orc, "analyze", lambda *a: analyzed.append(a) or True)
    monkeypatch.setattr(sys, "argv", ["loop.py"])
    with pytest.raises(SystemExit) as e:
        loop.main()
    assert "e24m_lf3" in str(e.value) and "e44m_lf3" in str(e.value) and "e44m_lm3" not in str(e.value)
    assert not any(c[2] == "e44m_lf3" for c in calls if c[0] in ("run", "branch"))
    assert any(c[2] == "e44m_lm3" and c[0] == "branch" for c in calls)
    assert [c[0] for c in calls[-4:]] == ["collect"] * 4 and analyzed == [("loop",)]
    assert "loop_lr_lf3" not in json.loads((analyze.RES / "decisions.json").read_text())
