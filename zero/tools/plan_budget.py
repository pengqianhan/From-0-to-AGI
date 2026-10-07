"""Budget planning: for a budget in US dollars, how many tokens each candidate model can train on.

Chapter 12. The budget table of Gate 1 uses this tool.

    tokens          = budget / cost per token    (estimate_cost gives the cost per token: FLOPs/token ÷ (peak × MFU) × price)
    tokens/param    = tokens / total parameters
    wall-clock time = GPU-hours / number of GPUs
    predicted loss  = L(N, D) from the ladder fit (only with --fit; the fit must come from ladder experiments with the same recipe)

Use `--speedup` for "what if" estimates. For example, if FP8 training is 1.3 times faster in a
measurement, use 1.3. This is the same as multiplying the MFU by that factor.
zero does not implement an FP8 path yet, and it is not verified on GPUs. Use this flag only for
discussion. It is not a promise.

Usage:

    uv run python -m zero.tools.plan_budget --config configs/main/pretrain.toml --budget 5000 \\
        --mfu 0.4 --mfu 0.5 --candidate "0.5B:dim=1024,ffn_dim=3072" --fit runs/ladder/fit.json
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

from zero.config import ModelConfig, load_model_config, read_toml
from zero.tools.estimate_cost import GPUS, estimate_cost
from zero.tools.fit_scaling import ChinchillaFit, model_size


@dataclass
class PlanRow:
    name: str
    params_total: int
    params_non_embedding: int
    mfu: float
    tokens: float
    tokens_per_param: float
    gpu_hours: float
    wall_days: float
    cost_usd: float
    predicted_loss: float | None = None


def tokens_for_budget(
    config: ModelConfig,
    budget_usd: float,
    seq_len: int,
    gpu: str = "h100-sxm",
    price_per_gpu_hour: float = 2.5,
    mfu: float = 0.4,
    num_gpus: int = 8,
    speedup: float = 1.0,
) -> float:
    """How many tokens the budget buys. This uses the "cost per token" from estimate_cost."""
    per_token = estimate_cost(
        config, 1.0, seq_len, gpu, price_per_gpu_hour, min(mfu * speedup, 1.0), num_gpus
    ).cost_usd
    return budget_usd / per_token


def plan(
    config: ModelConfig,
    budget_usd: float,
    seq_len: int,
    name: str = "",
    gpu: str = "h100-sxm",
    price_per_gpu_hour: float = 2.5,
    mfu: float = 0.4,
    num_gpus: int = 8,
    speedup: float = 1.0,
    fit: ChinchillaFit | None = None,
) -> PlanRow:
    tokens = tokens_for_budget(
        config, budget_usd, seq_len, gpu, price_per_gpu_hour, mfu, num_gpus, speedup
    )
    est = estimate_cost(
        config, tokens, seq_len, gpu, price_per_gpu_hour, min(mfu * speedup, 1.0), num_gpus
    )
    pred = None
    if fit is not None:
        pred = float(fit.predict(model_size(config, fit.param_count, seq_len), tokens))
    return PlanRow(
        name=name,
        params_total=est.params_total,
        params_non_embedding=est.params_non_embedding,
        mfu=mfu,
        tokens=tokens,
        tokens_per_param=tokens / est.params_total,
        gpu_hours=est.gpu_hours,
        wall_days=est.wall_hours / 24,
        cost_usd=est.cost_usd,
        predicted_loss=pred,
    )


def apply_candidate(base: ModelConfig, spec: str) -> tuple[str, ModelConfig]:
    """ "name:dim=1024,ffn_dim=3072,n_layers=24" → change these fields of base."""
    name, _, kv = spec.partition(":")
    cfg = copy.deepcopy(base)
    for item in filter(None, kv.split(",")):
        k, _, v = item.partition("=")
        if not hasattr(cfg, k.strip()):
            raise ValueError(f"Candidate {spec!r}: ModelConfig has no field {k!r}")
        cur = getattr(cfg, k.strip())
        setattr(cfg, k.strip(), type(cur)(float(v)) if isinstance(cur, int | float) else v)
    cfg.validate()
    return name or kv, cfg


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="For a budget: how many tokens each candidate can train on, how long it takes, and the predicted loss")
    ap.add_argument("--config", action="append", required=True, help="TOML with [model]; can repeat")
    ap.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="a candidate that changes the shape of the first --config, for example '0.5B:dim=1024,ffn_dim=3072'; can repeat",
    )
    ap.add_argument(
        "--budget", type=float, default=5000.0, help="budget in US dollars (default 5000, the pretraining line in GOAL.md 3.4)"
    )
    ap.add_argument("--gpu", default="h100-sxm", choices=sorted(GPUS))
    ap.add_argument("--price", type=float, default=2.5, help="US dollars per GPU-hour")
    ap.add_argument("--mfu", type=float, action="append", default=[], help="can repeat; default 0.4")
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=0, help="default: data.seq_len from the config")
    ap.add_argument("--speedup", type=float, default=1.0, help="assumed extra speedup (for example FP8); not verified")
    ap.add_argument("--fit", help="JSON that fit_scaling --out writes; used to predict the loss")
    ap.add_argument("--json", help="write the table as JSON")
    args = ap.parse_args(argv)

    fit = (
        ChinchillaFit.from_dict(json.loads(Path(args.fit).read_text())["fit"]) if args.fit else None
    )
    candidates: list[tuple[str, ModelConfig, int]] = []
    for path in args.config:
        cfg = load_model_config(path)
        seq = args.seq_len or int(read_toml(path).get("data", {}).get("seq_len", cfg.max_seq_len))
        candidates.append((Path(path).stem, cfg, seq))
    base_cfg, base_seq = candidates[0][1], candidates[0][2]
    for spec in args.candidate:
        name, cfg = apply_candidate(base_cfg, spec)
        candidates.append((name, cfg, base_seq))

    rows = []
    for mfu in args.mfu or [0.4]:
        for name, cfg, seq in candidates:
            rows.append(
                plan(
                    cfg,
                    args.budget,
                    seq,
                    name,
                    args.gpu,
                    args.price,
                    mfu,
                    args.num_gpus,
                    args.speedup,
                    fit,
                )
            )
    spec = GPUS[args.gpu]
    print(
        f"Budget ${args.budget:,.0f}, {args.num_gpus}×{spec.name} (peak {spec.bf16_dense_tflops} TFLOPS"
        f"{'' if spec.verified else ', not verified'}), ${args.price}/GPU-hour"
        + (f", assumed extra speedup ×{args.speedup} (not verified)" if args.speedup != 1.0 else "")
    )
    head = f"{'candidate':<14} {'MFU':>4} {'params':>8} {'non-emb':>8} {'token':>8} {'D/N':>8} {'GPU-h':>7} {'days':>7}"
    print(head + (f" {'L(N,D)':>8}" if fit else ""))
    for r in rows:
        line = (
            f"{r.name[:14]:<14} {r.mfu:>4.2f} {r.params_total / 1e6:>7.1f}M {r.params_non_embedding / 1e6:>7.1f}M "
            f"{r.tokens / 1e9:>7.0f}B {r.tokens_per_param:>8.0f} {r.gpu_hours:>7.0f} {r.wall_days:>7.1f}"
        )
        print(line + (f" {r.predicted_loss:>8.4f}" if r.predicted_loss is not None else ""))
    if args.json:
        Path(args.json).write_text(
            json.dumps([r.__dict__ for r in rows], indent=2, ensure_ascii=False)
        )


if __name__ == "__main__":
    main()
