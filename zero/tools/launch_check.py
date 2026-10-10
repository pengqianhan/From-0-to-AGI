"""Launch check: short runs of every post-training stage on the rented GPUs → time, memory, cost (Stage 10).

    # on the GPU machine, first thing:
    uv run python -m zero.tools.launch_check --track proxy --nproc 8 --price 2.5
    # only write the resolved configs and print the commands (no GPU needed):
    uv run python -m zero.tools.launch_check --track proxy --dry-run

GPU time is billed by the hour, so the first hour on a new machine should answer three questions
before any long run: does every stage start, how much memory does it use, and what will the full run
cost? For each stage of the track (`configs/<track>/<stage>.toml`) this tool:

1. writes a **resolved config** (the `base` file merged) to `<launch-dir>/<track>/configs/`, with a few
   changes: a few steps (`--steps`), a log record every step, one checkpoint at the end, and every path
   under the out_dir of an earlier stage moved to the launch directory (so the stages chain:
   SFT → distill → DPO → GRPO → OPD, including the OPD teacher paths). Distillation generates a small
   sample of teacher data (`--teacher-jobs`) into the launch directory, which also measures the teacher.
2. checks that the inputs exist (tokenizer, base checkpoint, SFT / preference / task files) and stops
   before any run if one is missing (`--force` runs anyway);
3. runs the stage (`torchrun --nproc_per_node=N -m zero.post.<stage>`; one process: `python -m`);
4. reads the log: median seconds per step (after the first steps, which include compilation and
   warm-up), tokens per second, peak GPU memory;
5. projects the full run: seconds/step × max_steps of the real config × N GPUs × $/GPU-hour. A stage
   above $100 needs approval first (RUNBOOK). RL stages are less certain: their responses get longer
   or shorter during training, and the projection uses the lengths of the first steps.

The report goes to `<report-dir>/<date>-<track>-launch/` (README.md + launch.json): copy the numbers into
`runs/ledger.md` and the budget of `runs/POSTTRAIN_PLAN.md`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
STAGES = ("sft", "distill", "dpo", "grpo", "opd")
DEFAULT_STEPS = {"sft": 20, "distill": 20, "dpo": 20, "grpo": 3, "opd": 3}
WARMUP = {"sft": 3, "distill": 3, "dpo": 2, "grpo": 1, "opd": 1}  # steps not counted in the median
APPROVAL_USD = 100.0


# ---------------------------------------------------------------------------
# A small TOML writer for resolved configs (tomllib can only read)
# ---------------------------------------------------------------------------


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float):
        if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
            raise ValueError("inf / nan cannot be written")
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_toml_key(k)} = {_toml_value(x)}" for k, x in v.items()) + "}"
    raise TypeError(f"Cannot write {type(v).__name__} to TOML")


def _toml_key(k: str) -> str:
    return k if k.replace("_", "").replace("-", "").isalnum() else json.dumps(k)


def _is_table_list(v: Any) -> bool:
    return isinstance(v, list) and bool(v) and all(isinstance(x, dict) for x in v)


def _is_subtable(v: Any) -> bool:
    return isinstance(v, dict) and any(isinstance(x, dict) or _is_table_list(x) for x in v.values())


def to_toml(d: dict[str, Any], prefix: str = "") -> str:
    """dict → TOML text. Plain dicts become inline tables; dicts with nested tables become [sections]."""
    lines = []
    tables, arrays = [], []
    for k, v in d.items():
        if _is_table_list(v):
            arrays.append((k, v))
        elif isinstance(v, dict) and (prefix == "" or _is_subtable(v)):
            tables.append((k, v))
        else:
            lines.append(f"{_toml_key(k)} = {_toml_value(v)}")
    out = "\n".join(lines) + ("\n" if lines else "")
    for k, v in tables:
        name = f"{prefix}{_toml_key(k)}"
        out += f"\n[{name}]\n" + to_toml(v, name + ".")
    for k, v in arrays:
        name = f"{prefix}{_toml_key(k)}"
        for item in v:
            out += f"\n[[{name}]]\n" + to_toml(item, name + ".")
    return out


# ---------------------------------------------------------------------------
# Resolved configs
# ---------------------------------------------------------------------------


def _remap(v: Any, moves: dict[str, str]) -> Any:
    """Move every path under an earlier stage's out_dir to the launch directory."""
    if isinstance(v, str):
        for old, new in moves.items():
            if v == old or v.startswith(old.rstrip("/") + "/"):
                return new + v[len(old.rstrip("/")) :]
        return v
    if isinstance(v, list):
        return [_remap(x, moves) for x in v]
    if isinstance(v, dict):
        return {k: _remap(x, moves) for k, x in v.items()}
    return v


@dataclass
class StagePlan:
    stage: str
    config: str  # the original config
    resolved: dict[str, Any]  # the launch config (merged, remapped, shortened)
    full_steps: int  # max_steps of the original config
    out_dir: str
    inputs: list[tuple[str, str]] = field(default_factory=list)  # (what, path)


def plan_stages(
    track: str,
    stages: list[str],
    launch_dir: Path,
    steps: dict[str, int],
    teacher_jobs: int = 50,
    configs_dir: Path = REPO / "configs",
) -> list[StagePlan]:
    from zero.config import _read_toml_with_base

    plans: list[StagePlan] = []
    moves: dict[str, str] = {}
    for st in stages:
        cfg_path = configs_dir / track / f"{st}.toml"
        d = _read_toml_with_base(cfg_path)
        full_steps = int(d["train"]["max_steps"])
        orig_out = d["train"]["out_dir"]
        d = _remap(d, moves)
        out = str(launch_dir / st)
        d["train"]["out_dir"] = out
        d["train"]["max_steps"] = steps.get(st, DEFAULT_STEPS[st])
        d.setdefault("checkpoint", {}).update({"every": 0, "keep_last": 1})
        d.setdefault("logging", {})["every"] = 1
        if st == "distill":  # a small sample of teacher data, in the launch directory
            dd = d.setdefault("distill", {})
            dd["out_jsonl"] = str(launch_dir / "distill" / "teacher_sample.jsonl")
            dd["shard_dir"] = str(launch_dir / "distill" / "packed")
            dd["n_tasks"] = min(int(dd.get("n_tasks", 0)), teacher_jobs)
            dd["max_tasks"] = teacher_jobs
            dd["max_prompts"] = teacher_jobs
            dd["overwrite"] = True
        moves[orig_out] = out
        plans.append(StagePlan(st, str(cfg_path), d, full_steps, out, stage_inputs(st, d)))
    return plans


def stage_inputs(stage: str, d: dict[str, Any]) -> list[tuple[str, str]]:
    """The files that must exist before the stage starts (outputs of earlier launch stages are not checked)."""
    t, data = d["train"], d.get("data", {})
    out = [("tokenizer", data.get("tokenizer", ""))]
    init = t.get("init_from", "")
    if init and "/launch/" not in init.replace(os.sep, "/"):
        out.append(("init checkpoint", init))
    sec = d.get(stage, {})
    if stage == "sft":
        if not sec.get("generate_train"):
            out.append(("SFT data", sec.get("train_jsonl", "")))
    elif stage == "distill":
        if sec.get("mix_sft_jsonl"):
            out.append(("SFT data to mix in", sec["mix_sft_jsonl"]))
        out += [("task file", p) for p in sec.get("task_files", [])]
        out += [("prompt file", p) for p in sec.get("prompt_files", [])]
    elif stage == "dpo":
        if not sec.get("generate_pairs"):
            out.append(("preference pairs", sec.get("train_jsonl", "")))
        out += [("task file", p) for p in sec.get("task_files", [])]
    elif stage == "grpo":
        out += [("task file", p) for p in sec.get("task_files", [])]
    elif stage == "opd":
        for tch in sec.get("teachers", []):
            if tch.get("prompts", "tool_env") != "tool_env":
                out.append((f"prompts of teacher {tch.get('name', '')}", tch["prompts"]))
    return out


def missing_inputs(plans: list[StagePlan]) -> list[str]:
    return [
        f"{p.stage}: {what} not found: {path or '(empty)'}"
        for p in plans
        for what, path in p.inputs
        if not path or not Path(path).exists()
    ]


def command(plan: StagePlan, cfg_file: Path, nproc: int) -> list[str]:
    mod = f"zero.post.{plan.stage}"
    if nproc > 1:
        return [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={nproc}",
            "-m",
            mod,
            "--config",
            str(cfg_file),
        ]
    return [sys.executable, "-m", mod, "--config", str(cfg_file)]


# ---------------------------------------------------------------------------
# Measurements and projection
# ---------------------------------------------------------------------------


def read_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def step_seconds(stage: str, log: list[dict[str, Any]]) -> list[float]:
    """Seconds of each logged step. Trainer stages log elapsed_s; the other stages log step_s."""
    if any("step_s" in r for r in log):
        return [r["step_s"] for r in log if "step_s" in r]
    el = [(r["step"], r["elapsed_s"]) for r in log if "elapsed_s" in r]
    return [(b[1] - a[1]) / max(b[0] - a[0], 1) for a, b in zip(el, el[1:])]


def measure(stage: str, log: list[dict[str, Any]]) -> dict[str, Any]:
    secs = step_seconds(stage, log)
    skip = WARMUP.get(stage, 1)
    used = secs[skip:] if len(secs) > skip else secs
    m: dict[str, Any] = {
        "steps_logged": len(log),
        "s_per_step": statistics.median(used) if used else None,
        "s_per_step_all": [round(x, 3) for x in secs],
    }
    for k in ("tok_per_s", "mfu"):
        vals = [r[k] for r in log[skip:] if r.get(k) is not None]
        if vals:
            m[k] = statistics.median(vals)
    mem = [r["max_mem_gb"] for r in log if r.get("max_mem_gb") is not None]
    if mem:
        m["max_mem_gb"] = max(mem)
    for k in ("resp_len", "reward_mean", "loss"):
        if log and k in log[-1]:
            m[f"last_{k}"] = log[-1][k]
    return m


def project(s_per_step: float | None, full_steps: int, nproc: int, price: float) -> dict[str, Any]:
    if s_per_step is None:
        return {"hours": None, "gpu_hours": None, "usd": None, "needs_approval": None}
    hours = s_per_step * full_steps / 3600
    gpu_hours = hours * nproc
    usd = gpu_hours * price
    return {
        "hours": hours,
        "gpu_hours": gpu_hours,
        "usd": usd,
        "needs_approval": usd > APPROVAL_USD,
    }


def teacher_projection(
    launch_dir: Path, full_cfg: dict[str, Any], teacher_gpus: int, price: float
) -> dict[str, Any] | None:
    """Seconds per teacher job in the sample × all jobs of the real config."""
    meta_p = launch_dir / "distill" / "teacher_sample.jsonl.meta.json"
    if not meta_p.exists():
        return None
    meta = json.loads(meta_p.read_text())
    n_sample = sum(v.get("jobs", 0) for v in meta.get("by_source", {}).values())
    if not n_sample:
        return None
    from zero.post.common import read_jsonl

    dd = full_cfg.get("distill", {})
    n_full = int(dd.get("n_tasks", 0))
    for p in dd.get("task_files", []):
        n_full += sum(1 for _ in open(p, encoding="utf-8")) if Path(p).exists() else 0
    for p in dd.get("prompt_files", []):
        if Path(p).exists():
            n = len(read_jsonl(p))
            n_full += min(n, dd["max_prompts"]) if dd.get("max_prompts") else n
    s_per_job = meta["seconds"] / n_sample
    hours = s_per_job * n_full / 3600
    return {
        "sample_jobs": n_sample,
        "sample_pass_rate": meta.get("pass_rate"),
        "s_per_job": s_per_job,
        "full_jobs": n_full,
        "hours": hours,
        "teacher_gpus": teacher_gpus,
        "usd": hours * teacher_gpus * price if teacher_gpus else None,
    }


def render_report(
    track: str, nproc: int, price: float, rows: list[dict[str, Any]], teacher: dict[str, Any] | None
) -> str:
    def f(x: Any, fmt: str) -> str:
        return "—" if x is None else format(x, fmt)

    out = [
        f"# Launch check: track {track}",
        "",
        f"{time.strftime('%Y-%m-%d %H:%M')}, {nproc} GPU(s), ${price}/GPU-hour. "
        "Generated by `zero.tools.launch_check`; projections = median s/step × max_steps of the real config.",
        "",
        "| Stage | Status | Steps run | s/step | Peak mem (GB) | tok/s | Full steps | Hours | GPU-hours | Cost ($) | > $100 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    total = 0.0
    for r in rows:
        m, p = r.get("measure", {}), r.get("projection", {})
        total += p.get("usd") or 0.0
        out.append(
            f"| {r['stage']} | {r['status']} | {m.get('steps_logged', 0)} | {f(m.get('s_per_step'), '.2f')} | "
            f"{f(m.get('max_mem_gb'), '.1f')} | {f(m.get('tok_per_s'), ',.0f')} | {r['full_steps']} | "
            f"{f(p.get('hours'), '.1f')} | {f(p.get('gpu_hours'), '.1f')} | {f(p.get('usd'), ',.0f')} | "
            f"{'**yes: ask first**' if p.get('needs_approval') else ''} |"
        )
    out += ["", f"**Total of the measured stages: ${total:,.0f}** (training only)."]
    if teacher:
        out += [
            "",
            f"Teacher data: {teacher['sample_jobs']} sample jobs, {teacher['s_per_job']:.2f} s/job, pass rate "
            f"{teacher['sample_pass_rate']:.1%} → {teacher['full_jobs']} jobs ≈ {teacher['hours']:.1f} h"
            + (
                f" on {teacher['teacher_gpus']} teacher GPU(s) ≈ ${teacher['usd']:,.0f}."
                if teacher.get("usd") is not None
                else " (set --teacher-gpus for a cost)."
            ),
        ]
    out += [
        "",
        "RL stages (GRPO, OPD): the response length changes during training; the projection uses the first steps.",
        "Copy the numbers into runs/ledger.md and the budget table of runs/POSTTRAIN_PLAN.md.",
    ]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_launch_check(
    track: str,
    stages: list[str],
    nproc: int,
    price: float,
    steps: dict[str, int],
    launch_root: Path,
    report_root: Path,
    teacher_jobs: int = 50,
    teacher_gpus: int = 0,
    dry_run: bool = False,
    force: bool = False,
    configs_dir: Path = REPO / "configs",
    log: Any = print,
) -> dict[str, Any]:
    launch_dir = launch_root / track
    plans = plan_stages(track, stages, launch_dir, steps, teacher_jobs, configs_dir)
    cfg_dir = launch_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cmds = []
    for p in plans:
        cfg_file = cfg_dir / f"{p.stage}.toml"
        cfg_file.write_text(
            f"# Resolved by zero.tools.launch_check from {p.config}\n" + to_toml(p.resolved)
        )
        cmds.append(command(p, cfg_file, nproc))
    missing = missing_inputs(plans)
    result: dict[str, Any] = {
        "track": track,
        "nproc": nproc,
        "price": price,
        "missing": missing,
        "commands": [" ".join(c) for c in cmds],
        "stages": [],
    }
    for m in missing:
        log(f"[launch] MISSING {m}")
    if dry_run:
        for c in result["commands"]:
            log(c)
        return result
    if missing and not force:
        raise SystemExit("[launch] inputs are missing (see above); fix them or use --force")
    for p, cmd in zip(plans, cmds):
        out = Path(p.out_dir)
        if out.exists():
            shutil.rmtree(out)  # a stale checkpoint would be resumed
        out.mkdir(parents=True, exist_ok=True)
        log(f"[launch] {p.stage}: {' '.join(cmd)}")
        t0 = time.time()
        with open(out / "stdout.log", "w") as fo:
            rc = subprocess.run(cmd, cwd=REPO, stdout=fo, stderr=subprocess.STDOUT).returncode
        lg = read_log(out / "log.jsonl")
        meas = measure(p.stage, lg)
        row = {
            "stage": p.stage,
            "status": "ok" if rc == 0 else f"failed ({rc})",
            "wall_s": round(time.time() - t0, 1),
            "full_steps": p.full_steps,
            "measure": meas,
            "projection": project(meas.get("s_per_step"), p.full_steps, nproc, price),
        }
        result["stages"].append(row)
        log(
            f"[launch] {p.stage}: {row['status']}, {meas.get('s_per_step')} s/step, projection {row['projection']}"
        )
        if rc != 0:
            log(
                f"[launch] {p.stage} failed: see {out / 'stdout.log'}; later stages need its checkpoint, so the check stops here"
            )
            break
    teacher = None
    if "distill" in stages:
        from zero.config import _read_toml_with_base

        teacher = teacher_projection(
            launch_dir,
            _read_toml_with_base(configs_dir / track / "distill.toml"),
            teacher_gpus,
            price,
        )
    result["teacher"] = teacher
    rep = report_root / f"{time.strftime('%Y-%m-%d')}-{track}-launch"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "launch.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (rep / "README.md").write_text(render_report(track, nproc, price, result["stages"], teacher))
    log(f"[launch] report: {rep / 'README.md'}")
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Launch check: short runs, time, memory, cost projection"
    )
    ap.add_argument("--track", required=True, help="proxy | weak | main (configs/<track>/)")
    ap.add_argument("--stages", default=",".join(STAGES), help="comma-separated, in pipeline order")
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--price", type=float, default=2.5, help="US dollars per GPU-hour")
    ap.add_argument("--steps", default="", help=f"e.g. sft=20,grpo=3 (defaults: {DEFAULT_STEPS})")
    ap.add_argument(
        "--teacher-jobs", type=int, default=50, help="teacher jobs in the distillation sample"
    )
    ap.add_argument(
        "--teacher-gpus", type=int, default=0, help="GPUs of the teacher server (for its cost)"
    )
    ap.add_argument("--launch-dir", default="out/launch")
    ap.add_argument("--report-dir", default="runs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="run even if inputs are missing")
    a = ap.parse_args(argv)
    stages = [s for s in a.stages.split(",") if s]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise SystemExit(f"Unknown stages {bad}; known: {STAGES}")
    steps = {k: int(v) for k, v in (x.split("=") for x in a.steps.split(",") if x)}
    run_launch_check(
        a.track,
        stages,
        a.nproc,
        a.price,
        steps,
        Path(a.launch_dir),
        Path(a.report_dir),
        a.teacher_jobs,
        a.teacher_gpus,
        a.dry_run,
        a.force,
    )


if __name__ == "__main__":
    main()
