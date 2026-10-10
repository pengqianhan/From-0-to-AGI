"""Launch check (Stage 10): resolved configs, input check, log parsing, projection, report."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from zero.tools.launch_check import (
    measure,
    missing_inputs,
    plan_stages,
    project,
    render_report,
    run_launch_check,
    step_seconds,
    to_toml,
)


def test_to_toml_round_trip_of_every_post_config() -> None:
    from zero.config import _read_toml_with_base

    for track in ("tiny", "main", "proxy", "weak"):
        for st in ("sft", "distill", "dpo", "grpo", "opd"):
            d = _read_toml_with_base(Path(f"configs/{track}/{st}.toml"))
            assert tomllib.loads(to_toml(d)) == d, (track, st)


def test_plan_chains_stages_into_the_launch_dir(tmp_path: Path) -> None:
    plans = plan_stages(
        "proxy", ["sft", "distill", "dpo", "grpo", "opd"], tmp_path / "proxy", {"grpo": 2}
    )
    by = {p.stage: p for p in plans}
    assert by["sft"].resolved["train"]["init_from"] == "out/proxy/base"  # the base is not moved
    assert by["distill"].resolved["train"]["init_from"] == str(tmp_path / "proxy" / "sft" / "ckpt")
    assert by["grpo"].resolved["train"]["init_from"] == str(tmp_path / "proxy" / "dpo" / "ckpt")
    assert [t["path"] for t in by["opd"].resolved["opd"]["teachers"]] == [
        str(tmp_path / "proxy" / "distill" / "ckpt"),
        str(tmp_path / "proxy" / "grpo" / "ckpt"),
    ]
    assert by["grpo"].resolved["train"]["max_steps"] == 2 and by["grpo"].full_steps == 500
    assert (
        by["sft"].resolved["checkpoint"]["every"] == 0
        and by["sft"].resolved["logging"]["every"] == 1
    )
    d = by["distill"].resolved["distill"]
    assert (
        d["out_jsonl"].startswith(str(tmp_path)) and d["n_tasks"] <= 50 and d["overwrite"] is True
    )
    # the resolved configs parse with the stage loaders
    from zero.post.common import load_post_config
    from zero.post.opd import OPDConfig

    cfg, sec = load_post_config(tomllib.loads(to_toml(by["opd"].resolved)), {"opd": OPDConfig})
    assert cfg.train.out_dir == str(tmp_path / "proxy" / "opd") and len(sec["opd"].teachers) == 2
    # without DPO, GRPO starts from distill
    plans2 = plan_stages("proxy", ["sft", "distill", "grpo"], tmp_path / "p2", {})
    assert (
        plans2[-1].resolved["train"]["init_from"] == "out/proxy/dpo/ckpt"
    )  # not moved: no DPO run here
    miss = missing_inputs(plans)
    assert any("tokenizer" in m for m in miss) and any("SFT data" in m for m in miss)


def test_measure_and_project() -> None:
    trainer_log = [
        {"step": s, "elapsed_s": 2.0 * s, "tok_per_s": 1000.0, "max_mem_gb": 30 + s}
        for s in range(1, 9)
    ]
    assert step_seconds("sft", trainer_log) == [2.0] * 7
    m = measure("sft", trainer_log)
    assert m["s_per_step"] == 2.0 and m["max_mem_gb"] == 38 and m["tok_per_s"] == 1000.0
    loop_log = [
        {"step": 1, "step_s": 100.0},
        {"step": 2, "step_s": 10.0},
        {"step": 3, "step_s": 12.0},
    ]
    assert (
        measure("grpo", loop_log)["s_per_step"] == 11.0
    )  # the first step (warm-up) is not counted
    p = project(11.0, 500, 8, 2.5)
    assert p["gpu_hours"] == pytest.approx(11 * 500 / 3600 * 8) and p["needs_approval"] is False
    assert project(100.0, 500, 8, 2.5)["needs_approval"] is True
    assert project(None, 500, 8, 2.5)["usd"] is None
    rep = render_report(
        "proxy",
        8,
        2.5,
        [
            {
                "stage": "grpo",
                "status": "ok",
                "full_steps": 500,
                "measure": measure("grpo", loop_log),
                "projection": project(100.0, 500, 8, 2.5),
            }
        ],
        None,
    )
    assert "ask first" in rep and "| grpo | ok |" in rep


def test_dry_run_writes_configs_and_commands(tmp_path: Path) -> None:
    res = run_launch_check(
        "proxy",
        ["sft", "grpo"],
        8,
        2.5,
        {},
        tmp_path / "launch",
        tmp_path / "runs",
        dry_run=True,
        log=lambda _: None,
    )
    assert res["commands"][0].startswith(
        "torchrun --standalone --nproc_per_node=8 -m zero.post.sft --config"
    )
    assert (tmp_path / "launch" / "proxy" / "configs" / "grpo.toml").exists()
    assert res["missing"] and not (tmp_path / "runs").exists()  # a dry run writes no report


def test_run_on_tiny_track(tmp_path: Path, chat_tok, chat_tok_path, tiny_ckpt) -> None:  # noqa: ANN001
    """A real (CPU, 1 process) run of GRPO through the launch check, with a tiny config track."""
    from tests.conftest import post_config

    cfgs = tmp_path / "configs" / "t"
    cfgs.mkdir(parents=True)
    d = post_config(
        tmp_path,
        chat_tok_path,
        tiny_ckpt,
        chat_tok.vocab_size,
        grpo={"group_size": 2, "prompts_per_step": 2, "max_new_tokens": 6, "n_train_tasks": 10},
    )
    d["train"]["max_steps"] = 400
    (cfgs / "grpo.toml").write_text(to_toml(d))
    res = run_launch_check(
        "t",
        ["grpo"],
        1,
        2.5,
        {"grpo": 3},
        tmp_path / "launch",
        tmp_path / "runs",
        configs_dir=tmp_path / "configs",
        log=lambda _: None,
    )
    row = res["stages"][0]
    assert row["status"] == "ok", (tmp_path / "launch" / "t" / "grpo" / "stdout.log").read_text()[
        -2000:
    ]
    assert row["measure"]["steps_logged"] == 3 and row["measure"]["s_per_step"] > 0
    assert row["projection"]["gpu_hours"] == pytest.approx(
        row["measure"]["s_per_step"] * 400 / 3600
    )
    reports = list((tmp_path / "runs").glob("*-t-launch/README.md"))
    assert reports and "| grpo | ok | 3 |" in reports[0].read_text()
    assert (
        json.loads((reports[0].parent / "launch.json").read_text())["stages"][0]["stage"] == "grpo"
    )
