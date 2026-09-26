"""预算规划：给定美元预算，候选模型各能训多少 token（对应第 12 章；闸门 1 的预算表用它）。

    token 数   = 预算 / 每 token 的费用        （每 token 费用由 estimate_cost 算：FLOPs/token ÷ (峰值 × MFU) × 单价）
    token/参数 = token 数 / 总参数
    墙钟时间   = 卡时 / 卡数
    预测 loss  = 阶梯拟合的 L(N, D)（只有给了 --fit 才算；拟合必须来自同一配方的阶梯实验）

`--speedup` 用来做"如果……会怎样"的估算，比如 FP8 训练实测提速 1.3 倍：它等价于把 MFU 乘上这个倍数。
FP8 路径在 zero 里还没有实现，也尚未在 GPU 上验证——这个参数只用于讨论，不代表承诺。

用法：

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
    """预算能买多少 token：直接复用 estimate_cost 的"每 token 费用"。"""
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
    """ "名字:dim=1024,ffn_dim=3072,n_layers=24" → 在 base 上改这几个字段。"""
    name, _, kv = spec.partition(":")
    cfg = copy.deepcopy(base)
    for item in filter(None, kv.split(",")):
        k, _, v = item.partition("=")
        if not hasattr(cfg, k.strip()):
            raise ValueError(f"候选 {spec!r}：ModelConfig 没有字段 {k!r}")
        cur = getattr(cfg, k.strip())
        setattr(cfg, k.strip(), type(cur)(float(v)) if isinstance(cur, int | float) else v)
    cfg.validate()
    return name or kv, cfg


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="给定预算，候选模型能训多少 token、要多久、预测 loss")
    ap.add_argument("--config", action="append", required=True, help="含 [model] 的 TOML；可重复")
    ap.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="在第一个 --config 上改形状的候选，如 '0.5B:dim=1024,ffn_dim=3072'；可重复",
    )
    ap.add_argument(
        "--budget", type=float, default=5000.0, help="美元预算（默认 5000，GOAL.md 3.4 的预训练线）"
    )
    ap.add_argument("--gpu", default="h100-sxm", choices=sorted(GPUS))
    ap.add_argument("--price", type=float, default=2.5, help="每卡时美元")
    ap.add_argument("--mfu", type=float, action="append", default=[], help="可重复，默认 0.4")
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=0, help="默认读配置 data.seq_len")
    ap.add_argument("--speedup", type=float, default=1.0, help="假设的额外提速（如 FP8），未验证")
    ap.add_argument("--fit", help="fit_scaling --out 写出的 JSON，用来预测 loss")
    ap.add_argument("--json", help="把表格写成 JSON")
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
        f"预算 ${args.budget:,.0f}，{args.num_gpus}×{spec.name}（峰值 {spec.bf16_dense_tflops} TFLOPS"
        f"{'' if spec.verified else '，待核实'}），${args.price}/卡时"
        + (f"，假设额外提速 ×{args.speedup}（未验证）" if args.speedup != 1.0 else "")
    )
    head = f"{'候选':<14} {'MFU':>4} {'总参数':>8} {'非emb':>8} {'token':>8} {'tok/参数':>8} {'卡时':>7} {'8卡天数':>7}"
    print(head + (f" {'预测loss':>8}" if fit else ""))
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
