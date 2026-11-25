"""runs/ladder-3090/arloop.py and `analyze.py arloop` (README stage 8), and the seed key of sweep.py.

- The seed key only adds `--set train.seed=...`; without it the training commands are the
  same as before stage 8.
- arloop.py runs the blocks in the preregistered order, applies the lr rule, retries e44m once
  with half the lr, and is idempotent.
- `analyze.py arloop` reports what is missing and never crashes; with all results it computes
  s, t and the verdicts of the preregistration.
- The seed replicates in the e24m_lf3 scale do not change the stage-7 analysis.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# the last commit before stage 8 (the stage-8 preregistration): the code without attn_res and the seed key
BEFORE_STAGE8 = "e3a5ec7"
HERE = REPO / "runs" / "ladder-3090"
sys.path.insert(0, str(HERE))

import analyze  # noqa: E402
import arloop  # noqa: E402
import loop  # noqa: E402
import sweep  # noqa: E402

from zero.config import load_config  # noqa: E402

# ---------------------------------------------------------------- seed key of sweep.py


def _head_sweep() -> object:
    src = subprocess.run(["git", "show", f"{BEFORE_STAGE8}:runs/ladder-3090/sweep.py"], cwd=REPO, capture_output=True,
                         text=True, check=True).stdout
    spec = importlib.util.spec_from_loader("_head_sweep", loader=None)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["__file__"] = str(HERE / "sweep.py")
    sys.modules["_head_sweep"] = mod
    exec(compile(src, f"{BEFORE_STAGE8}:sweep.py", "exec"), mod.__dict__)
    return mod


def test_existing_configs_give_byte_identical_commands() -> None:
    old = _head_sweep()
    n = 0
    for path in sorted((HERE / "configs").glob("*.jsonl")):
        for c in sweep.load_configs(path.stem):
            if c["scale"] not in old.MICRO or "seed" in c:  # type: ignore[attr-defined]
                continue  # (the seed replicates of stage 8 are new: their command adds train.seed)
            neff = 1e7 if "stable_steps" not in c else 1.0  # only the plan inputs matter here
            pl = sweep.plan_for(c, neff)
            assert dataclasses.astuple(pl) == dataclasses.astuple(old.plan_for(c, neff))  # type: ignore[attr-defined]
            args = (pl, sweep.run_dir(c), 100, 0.2, 50, 10, 10)
            assert sweep.train_cmd(c, *args) == old.train_cmd(c, *args)  # type: ignore[attr-defined]
            n += 1
    assert n > 300  # every existing config of every scale (dense, ot, lf3, lm3)


def test_seed_key_sets_train_seed() -> None:
    c = arloop.make_seed(1338)
    base = loop.make("e24m", "lf3", 0, 1.0)
    assert c["id"] == "e24m_lf3-s1338" and c["scale"] == "e24m_lf3" and c["seed"] == 1338
    assert {k for k in c if c.get(k) != base.get(k)} == {"id", "space", "seed"}
    pl = sweep.plan_for(c, sweep.n_eff(c["scale"]))
    cmd = sweep.train_cmd(c, pl, sweep.run_dir(c), 100, 0.2, 50, 10, 10)
    ref = sweep.train_cmd(base, pl, sweep.run_dir(c), 100, 0.2, 50, 10, 10)
    assert cmd[:-2] == ref and cmd[-2:] == ["--set", "train.seed=1338"]
    sets = [cmd[i + 1] for i, x in enumerate(cmd) if x == "--set"]
    cfg = load_config(cmd[cmd.index("--config") + 1], sets)
    assert cfg.train.seed == 1338 and isinstance(cfg.train.seed, int)
    assert load_config(REPO / "configs" / "ladder3090" / "e24m_lf3.toml").train.seed == arloop.ORIG_SEED


def test_plan_matches_preregistration() -> None:
    arloop.check_prereg()
    for scale in arloop.AR_SCALES:
        dense = scale.split("_")[0]
        c = arloop.make(scale, 0, 1.0)
        assert c["id"] == f"{scale}-000" and c["space"] == "arloop"
        pl = sweep.plan_for(c, sweep.n_eff(scale))
        S, w = loop.PREREG_PLAN[dense]
        assert (pl.stable_steps, pl.warmup_steps, pl.micro * pl.accum) == (S, w, 32)
        assert all(c[h] == loop.base_cfg(dense)[h] for h in loop.HP)
        cmd = sweep.train_cmd(c, pl, sweep.run_dir(c), 10, 0.2, 0, 1, 1)
        assert cmd[cmd.index("--config") + 1].endswith(f"configs/ladder3090/{scale}.toml")
        assert "train.seed" not in " ".join(cmd)
    steps = arloop.plan_steps()
    assert steps["e24m_ar3"] == loop.plan_steps()["e24m_lf3"] and steps["e44m_ar3"] == loop.plan_steps()["e44m_lf3"]


def test_plan_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    before = {p: p.stat().st_mtime_ns for p in (HERE / "configs").glob("*.jsonl")}
    monkeypatch.setattr(sys, "argv", ["arloop.py", "--plan"])
    arloop.main()
    assert "e24m_lf3-s1339" in capsys.readouterr().out
    assert {p: p.stat().st_mtime_ns for p in (HERE / "configs").glob("*.jsonl")} == before


# ---------------------------------------------------------------- sandbox


def _copy_configs_without_seeds(scale: str, dst: Path) -> list[dict]:
    """Copy configs/<scale>.jsonl without the stage-8 seed replicates (the real run appends them)."""
    lines = [ln for ln in (HERE / "configs" / f"{scale}.jsonl").read_text().splitlines()
             if ln.strip() and "seed" not in json.loads(ln)]
    (dst / f"{scale}.jsonl").write_text("".join(ln + "\n" for ln in lines))
    return [json.loads(ln) for ln in lines]


def _log(d: Path, step: int, val: float, tok: float = 8e4, mem: float = 7.5) -> None:
    d.mkdir(parents=True, exist_ok=True)
    with (d / "log.jsonl").open("a") as f:
        f.write(json.dumps({"step": step, "val_bpb": val, "tok_per_s": tok, "max_mem_gb": mem}) + "\n")


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    res, cfgs = tmp_path / "results", tmp_path / "configs"
    res.mkdir()
    cfgs.mkdir()
    for s in ("e11m", "e24m", "e44m", "e83m", "e24m_lf3", "e44m_lf3"):
        shutil.copy(HERE / "results" / f"{s}.csv", res)
    for s in ("e24m", "e44m", "e24m_lf3", "e44m_lf3"):
        _copy_configs_without_seeds(s, cfgs)
    shutil.copy(HERE / "results" / "loop_result.json", res)
    monkeypatch.setattr(analyze, "RES", res)
    monkeypatch.setattr(sweep, "HERE", tmp_path)
    monkeypatch.setattr(sweep, "OUT", tmp_path / "out")
    import orchestrate as orc

    monkeypatch.setattr(orc, "RES", res)
    return tmp_path


def _run() -> dict:
    analyze.cmd_arloop(argparse.Namespace())
    return json.loads((analyze.RES / "arloop_result.json").read_text())


def _k(cfg: dict, k: int) -> tuple[Path, int]:
    S = cfg["stable_steps"]
    return (sweep.run_dir(cfg), sweep.with_decay(S)[0]) if k == 8 else (sweep.run_dir(cfg, k), sweep.with_decay(S * k // 8)[0])


def test_analyze_arloop_nothing_partial_full(sandbox: Path) -> None:
    # 1. nothing yet: no crash, the lf3 references come from the stage-7 csv
    out = _run()
    assert out["law_matches_prereg"]
    assert out["lf3"]["s"] is None and out["lf3"]["t"] is None
    assert out["lf3"]["e24m_seeds"]["e24m_lf3-000"]["k8"]["val_bpb"] == pytest.approx(1.3282, abs=5e-5)
    assert out["lf3"]["e44m"]["k8"]["val_bpb"] == pytest.approx(1.2140, abs=5e-5)
    assert any("s 还不能算" in m for m in out["missing"])
    assert sum(m.startswith(("e24m_ar3", "e44m_ar3", "e24m_arf3", "e44m_arf3")) for m in out["missing"]) == 4
    # 2. partial: e24m_ar3 lr check + decision + 8/8, but only one seed replicate
    cfgs = [arloop.make("e24m_ar3", i, m) for i, m in arloop.LR_CHECK]
    arloop.write("e24m_ar3", cfgs)
    for c, v in zip(cfgs, (1.36, 1.37, 1.39)):
        _log(*_k(c, 4), v)
    (analyze.RES / "decisions.json").write_text(json.dumps({"arloop_lr": {"value": 1.0}}))
    _log(*_k(cfgs[0], 8), 1.3150, tok=7.6e4, mem=7.45)
    s1 = arloop.make_seed(1338)
    arloop.write("e24m_lf3", [s1])
    _log(*_k(s1, 8), 1.3300)
    out = _run()
    r = out["scales"]["e24m_ar3"]
    assert r["config"] == "e24m_ar3-000" and r["lr_check_k4"] == {"x1.0": 1.36, "x0.5": 1.37, "x2.0": 1.39}
    assert r["k8"]["vs_lf3"]["verdict"] is None and out["lf3"]["t"] is None
    assert r["k8"]["vs_lf3"]["rel_diff_vs_e24m_lf3-000_only"] == pytest.approx(1.3150 / 1.328182358335454 - 1)
    assert r["k8"]["vs_dense"]["pred_matches_prereg"] and r["k8"]["vs_dense"]["verdict"] == "ARLoop 更差"
    assert r["k4"]["vs_lf3_000"]["D_matches"] and r["k8"]["vs_lf3"]["D_matches"]
    assert r["tok_per_s"] == 7.6e4 and r["max_mem_gb"] == 7.45
    assert r["time_vs_lf3"] == pytest.approx(83863.79197741288 / 7.6e4) and r["time_vs_dense"] == pytest.approx(139945 / 7.6e4)
    assert r["queries"] is None  # no checkpoint
    # 3. full: second seed and e44m_ar3 + the arf3 rows
    s2 = arloop.make_seed(1339)
    arloop.write("e24m_lf3", [s2])
    _log(*_k(s2, 8), 1.3250)
    for scale, v8 in (("e44m_ar3", 1.2140 * 0.99), ("e24m_arf3", 1.3270), ("e44m_arf3", 1.2300)):
        c = arloop.make(scale, 0, 1.0)
        arloop.write(scale, [c])
        _log(*_k(c, 4), v8 + 0.04)
        _log(*_k(c, 8), v8)
    out = _run()
    vals = [1.328182358335454, 1.3300, 1.3250]
    s = max(vals) / min(vals) - 1
    assert out["lf3"]["s"] == pytest.approx(s) and out["lf3"]["t"] == pytest.approx(max(0.005, 2 * s))
    t = out["lf3"]["t"]
    e24 = out["scales"]["e24m_ar3"]["k8"]["vs_lf3"]
    assert e24["rel_diff"] == pytest.approx(1.3150 / (sum(vals) / 3) - 1)
    assert e24["verdict"] == ("ARLoop 更好" if e24["rel_diff"] <= -t else "分不出")
    e44 = out["scales"]["e44m_ar3"]["k8"]
    assert e44["vs_lf3"]["rel_diff"] == pytest.approx(1.2140 * 0.99 / 1.2139913936199074 - 1)
    assert e44["vs_lf3"]["verdict"] == ("ARLoop 更好" if -0.01 <= -t else "分不出")
    assert e44["vs_dense"]["pred_matches_prereg"] and e44["e83m_interp"]["value"] == pytest.approx(1.1538, abs=5e-5)
    assert out["scales"]["e44m_arf3"]["k8"]["vs_lf3"]["verdict"] == "ARLoop 更差"
    assert not out["scales"]["e24m_arf3"]["primary"] and out["scales"]["e24m_ar3"]["primary"]
    assert not out["missing"]


def test_verdict_thresholds() -> None:
    assert analyze.ar_verdict(-0.006, 0.005) == "ARLoop 更好"
    assert analyze.ar_verdict(0.005, 0.005) == "ARLoop 更差"
    assert analyze.ar_verdict(0.0049, 0.005) == "分不出"
    assert analyze.ar_dense_verdict(-0.01) == "ARLoop 更好" and analyze.ar_dense_verdict(0.0099) == "持平"


def test_query_stats_from_checkpoint(sandbox: Path) -> None:
    import torch

    c = arloop.make("e24m_ar3", 0, 1.0)
    ck = sweep.run_dir(c) / "ckpt" / "step_00000010"
    ck.mkdir(parents=True)
    (ck.parent / "latest").write_text(ck.name)
    sd = {f"_orig_mod.attn_res_queries.{i}": torch.full((4,), float(i)) for i in range(24)}
    sd["final_attn_res_query"] = torch.tensor([1.0, -1.0, 3.0, -3.0])
    torch.save(sd, ck / "model.pt")
    (ck / "meta.json").write_text("{}")
    q = analyze.ar_query_stats(c)
    assert q is not None and [x["i"] for x in q["sublayers"]] == list(range(24))
    assert q["sublayers"][5]["mean_abs"] == 5.0 and q["sublayers"][5]["kind"] == "ffn"
    assert q["final"]["mean_abs"] == 2.0 and q["final"]["max_abs"] == 3.0


# ---------------------------------------------------------------- stage 7 unchanged


def test_stage7_analysis_unchanged_by_seed_replicates(sandbox: Path) -> None:
    """Collect + analyze.py loop with the seed replicates in e24m_lf3 give the same stage-7 output."""
    real = REPO / "out" / "ladder3090"
    scales = ("e24m_lf3", "e24m_lm3", "e44m_lf3", "e44m_lm3")
    if not all((real / s).is_dir() for s in scales):
        pytest.skip("the stage-7 run directories are not on this machine")
    for s in scales:
        shutil.copy(HERE / "results" / f"{s}.csv", analyze.RES)
        # copies (no links) of the logs of the stage-7 runs only: the test never writes into out/
        for c in _copy_configs_without_seeds(s, sandbox / "configs"):
            for log in [real / s / c["id"] / "log.jsonl", *(real / s / c["id"]).glob("branch_*/log.jsonl")]:
                if log.exists():
                    dst = sweep.OUT / log.relative_to(real)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(log, dst)
    shutil.copy(HERE / "results" / "decisions.json", analyze.RES)

    def stage7() -> dict:
        analyze.cmd_loop(argparse.Namespace())
        return json.loads((analyze.RES / "loop_result.json").read_text())

    sweep.cmd_collect(argparse.Namespace(scale="e24m_lf3"))
    before_csv = (analyze.RES / "e24m_lf3.csv").read_text()
    a = stage7()
    assert a["scales"]["e24m_lf3"]["k8"]["val_bpb"] == pytest.approx(1.3282, abs=5e-5)
    for seed, v in ((1338, 1.30), (1339, 1.31)):  # better than -000 at 4/8 and 8/8: would win if not excluded
        c = arloop.make_seed(seed)
        arloop.write("e24m_lf3", [c])
        _log(*_k(c, 4), v + 0.03)
        _log(*_k(c, 8), v)
    sweep.cmd_collect(argparse.Namespace(scale="e24m_lf3"))
    after_csv = (analyze.RES / "e24m_lf3.csv").read_text()
    assert "e24m_lf3-s1338" in after_csv and after_csv != before_csv
    b = stage7()
    assert a == b
    assert b["scales"]["e24m_lf3"]["config"] == "e24m_lf3-000"
    assert b["scales"]["e24m_lf3"]["lr_check_k4"]["x1.0"] == pytest.approx(1.3667, abs=5e-5)


# ---------------------------------------------------------------- arloop.py control flow


def _fake_sweep(calls: list[tuple], k4: dict, diverge_ids: set[str] = frozenset()) -> object:  # type: ignore[assignment]
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
            d = sweep.run_dir(c, 4 if cmd == "branch" else None)
            if v is None or c["id"] in diverge_ids:
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


def _blocks(calls: list[tuple]) -> list[str]:
    """The sequence of trained blocks (scale, or 'seeds' for the e24m_lf3 replicates), without repeats."""
    out: list[str] = []
    for c in calls:
        if c[0] not in ("run", "branch"):
            continue
        name = "seeds" if c[2] == "e24m_lf3" else c[2]
        if not out or out[-1] != name:
            out.append(name)
    return out


def test_arloop_order_lr_rule_retry_and_rerun(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """x2 wins by more than 0.5%; e44m_ar3 diverges once (lr halved, new id); arf3 use arloop_lr."""
    import orchestrate as orc

    calls: list[tuple] = []
    analyzed: list[tuple] = []
    k4 = {"e24m_ar3": {1.0: 1.40, 0.5: 1.41, 2.0: 1.39}}
    monkeypatch.setattr(orc, "sweep", _fake_sweep(calls, k4, {"e44m_ar3-000"}))
    monkeypatch.setattr(orc, "analyze", lambda *a: analyzed.append(a) or True)
    monkeypatch.setattr(sys, "argv", ["arloop.py"])
    arloop.main()
    assert _blocks(calls) == ["e24m_ar3", "e44m_ar3", "seeds", "e24m_arf3", "e44m_arf3"]
    dec = json.loads((analyze.RES / "decisions.json").read_text())
    assert dec["arloop_lr"]["value"] == 2.0 and dec["arloop_lr_retry_e44m_ar3"]["value"] == 1.0
    assert "arloop_lr_retry_e44m_arf3" not in dec
    # only the chosen e24m_ar3 trunk (x2 = -002) is trained to 8/8
    full = [c for c in calls if c[0] == "run" and c[2] == "e24m_ar3" and c[c.index("--until") + 1] == "1"]
    assert len(full) == 1 and full[0][full[0].index("--ids") + 1] == "e24m_ar3-002"
    e44 = {c["id"]: c for c in sweep.load_configs("e44m_ar3")}
    assert e44["e44m_ar3-001"]["lr_mult"] == 1.0 and e44["e44m_ar3-001"]["retry_of"] == "e44m_ar3-000"
    for s in ("e24m_arf3", "e44m_arf3"):
        assert [c["lr_mult"] for c in sweep.load_configs(s)] == [2.0]
    seeds = [c for c in sweep.load_configs("e24m_lf3") if "seed" in c]
    assert [(c["id"], c["seed"], c["lr_mult"]) for c in seeds] == [("e24m_lf3-s1338", 1338, 1.0),
                                                                    ("e24m_lf3-s1339", 1339, 1.0)]
    assert ("branch", "e24m_lf3") in [(c[0], c[2]) for c in calls]
    assert [c[2] for c in calls if c[0] == "collect"] == list(arloop.AR_SCALES) and analyzed == [("arloop",)]
    # rerun: idempotent (no new configs, same decisions)
    cfg_text = {s: sweep.configs_path(s).read_text() for s in (*arloop.AR_SCALES, "e24m_lf3")}
    arloop.main()
    assert json.loads((analyze.RES / "decisions.json").read_text()) == dec
    assert {s: sweep.configs_path(s).read_text() for s in cfg_text} == cfg_text


def test_arloop_lr_rule_keeps_x1_and_x2_divergence_is_not_adopted(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import orchestrate as orc

    calls: list[tuple] = []
    k4 = {"e24m_ar3": {1.0: 1.40, 0.5: 1.397, 2.0: None}}  # x0.5 better by only 0.21%
    monkeypatch.setattr(orc, "sweep", _fake_sweep(calls, k4))
    monkeypatch.setattr(orc, "analyze", lambda *a: True)
    monkeypatch.setattr(sys, "argv", ["arloop.py"])
    arloop.main()
    assert json.loads((analyze.RES / "decisions.json").read_text())["arloop_lr"]["value"] == 1.0


def test_arloop_stop_in_lr_check_skips_dependants_but_not_seeds(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """x1 diverges: no arloop_lr. e44m_ar3 and both arf3 are skipped; the seeds still run; collect and
    analyze still run; non-zero exit."""
    import orchestrate as orc

    calls: list[tuple] = []
    analyzed: list[tuple] = []
    monkeypatch.setattr(orc, "sweep", _fake_sweep(calls, {"e24m_ar3": {1.0: None, 0.5: 1.39, 2.0: 1.41}}))
    monkeypatch.setattr(orc, "analyze", lambda *a: analyzed.append(a) or True)
    monkeypatch.setattr(sys, "argv", ["arloop.py"])
    with pytest.raises(SystemExit) as e:
        arloop.main()
    msg = str(e.value)
    assert all(s in msg for s in ("e24m_ar3", "e44m_ar3", "e24m_arf3", "e44m_arf3")) and "种子" not in msg
    assert _blocks(calls) == ["e24m_ar3", "seeds"]
    assert [c[0] for c in calls[-4:]] == ["collect"] * 4 and analyzed == [("arloop",)]
    dec = analyze.RES / "decisions.json"
    assert not dec.exists() or "arloop_lr" not in json.loads(dec.read_text())


def test_arloop_seed_divergence_stops_only_the_seeds(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A diverged seed replicate (e24m, no retry rule) stops that block; e44m_arf3 diverges twice → stop."""
    import orchestrate as orc

    calls: list[tuple] = []
    diverge = {"e24m_lf3-s1339", "e44m_arf3-000", "e44m_arf3-001"}
    monkeypatch.setattr(orc, "sweep", _fake_sweep(calls, {}, diverge))
    monkeypatch.setattr(orc, "analyze", lambda *a: True)
    monkeypatch.setattr(sys, "argv", ["arloop.py"])
    with pytest.raises(SystemExit) as e:
        arloop.main()
    msg = str(e.value)
    assert "e24m_lf3-s1339" in msg and "e44m_arf3-001" in msg and "e24m_arf3（" not in msg
    assert _blocks(calls) == ["e24m_ar3", "e44m_ar3", "seeds", "e24m_arf3", "e44m_arf3"]
    dec = json.loads((analyze.RES / "decisions.json").read_text())
    assert dec["arloop_lr_retry_e44m_arf3"]["value"] == 0.5


def test_analyze_cli_runs_arloop(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real run calls `analyze.py arloop` (orc.analyze) through the command line."""
    monkeypatch.setattr(sys, "argv", ["analyze.py", "arloop"])
    analyze.main()
    assert json.loads((analyze.RES / "arloop_result.json").read_text())["missing"]
