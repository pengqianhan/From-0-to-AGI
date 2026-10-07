"""Training cost estimate: FLOPs → GPU-hours → US dollars (Chapter 12; GOAL.md 9.1 "Step 2 ready").

    total FLOPs     = FLOPs per token × tokens        (FLOPs per token = 6N + attention term, see zero/model.py)
    GPU-hours       = total FLOPs / (peak FLOPs/s of one GPU × MFU) / 3600
    cost            = GPU-hours × price per GPU-hour
    wall-clock time = GPU-hours / number of GPUs

MFU (Model FLOPs Utilization) = useful compute that we get / peak of the hardware. For a dense
Transformer on H100 GPUs, a well-tuned open-source implementation usually gets 0.3–0.5.
The default is 0.4. The GPU verification runs of Step 2, stage 6, will measure it and update it.

Usage:

    uv run python -m zero.tools.estimate_cost --config configs/main/pretrain.toml \\
        --tokens 500B --gpu h100-sxm --price 2.5 --num-gpus 8 --mfu 0.4

GPU peak table: it contains only the "dense BF16 Tensor Core" compute. It does not include the
doubled numbers for 2:4 structured sparsity. In the NVIDIA data sheets, the BF16 column shows the
value "with sparsity"; the dense value is half of it. The values in this table come from the
official pages below. But the environment that wrote this table could not open nvidia.com to
check each value. Thus each entry has `verified=False` (not verified). Before you spend money in
Step 2, compare the values with the data sheets, and correct them with the measured MFU.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

from zero.config import ModelConfig, load_model_config, read_toml
from zero.model import count_params, estimate_flops_per_token


@dataclass(frozen=True)
class GPUSpec:
    name: str
    bf16_dense_tflops: float
    source: str
    verified: bool = False  # True after a check against the official data sheet
    note: str = ""


GPUS: dict[str, GPUSpec] = {
    "h100-sxm": GPUSpec(
        "NVIDIA H100 SXM5",
        989.5,
        "https://www.nvidia.com/en-us/data-center/h100/ (data sheet: BF16 1,979 TFLOPS with sparsity → dense 989.5)",
        note="not verified",
    ),
    "h100-pcie": GPUSpec(
        "NVIDIA H100 PCIe",
        756.5,
        "https://www.nvidia.com/en-us/data-center/h100/ (data sheet: BF16 1,513 TFLOPS with sparsity → dense 756.5)",
        note="not verified",
    ),
    "h200-sxm": GPUSpec(
        "NVIDIA H200 SXM",
        989.5,
        "https://www.nvidia.com/en-us/data-center/h200/ (data sheet: BF16 1,979 TFLOPS with sparsity → dense 989.5)",
        note="not verified",
    ),
    "a100": GPUSpec(
        "NVIDIA A100 (SXM/PCIe)",
        312.0,
        "https://www.nvidia.com/en-us/data-center/a100/ (data sheet: BF16 312 TFLOPS, 624 with sparsity)",
        note="not verified",
    ),
    "l40s": GPUSpec(
        "NVIDIA L40S",
        366.0,
        "https://www.nvidia.com/en-us/data-center/l40s/ (data sheet: BF16 733 TFLOPS with sparsity → dense about 366)",
        note="not verified: check if the dense value in the data sheet is 362 or 366",
    ),
    "rtx4090": GPUSpec(
        "NVIDIA GeForce RTX 4090",
        165.2,
        "NVIDIA Ada Lovelace architecture white paper (BF16 Tensor, FP32 accumulate 165.2 TFLOPS, 330.4 with sparsity)",
        note="not verified",
    ),
}


def peak_tflops_for_device_name(device_name: str) -> float | None:
    """Find an approximate peak from the value of `torch.cuda.get_device_name()` (for the MFU log).

    Return None if no GPU matches.
    """
    n = device_name.lower()
    if "h100" in n:
        return GPUS["h100-pcie" if "pcie" in n else "h100-sxm"].bf16_dense_tflops
    if "h200" in n:
        return GPUS["h200-sxm"].bf16_dense_tflops
    if "a100" in n:
        return GPUS["a100"].bf16_dense_tflops
    if "l40s" in n:
        return GPUS["l40s"].bf16_dense_tflops
    if "4090" in n:
        return GPUS["rtx4090"].bf16_dense_tflops
    return None


_SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9, "g": 1e9, "t": 1e12}


def parse_count(s: str) -> float:
    """'500B' / '1.2T' / '3e11' → number."""
    m = re.fullmatch(r"\s*([0-9.eE+-]+)\s*([kKmMbBgGtT]?)\s*", s)
    if not m:
        raise ValueError(f"Cannot read the count: {s!r} (examples: 500B, 1.2T, 3e11)")
    return float(m.group(1)) * _SUFFIX.get(m.group(2).lower(), 1.0)


@dataclass
class CostEstimate:
    params_total: int
    params_non_embedding: int
    flops_per_token: float
    total_flops: float
    gpu_hours: float
    wall_hours: float
    cost_usd: float


def estimate_cost(
    config: ModelConfig,
    tokens: float,
    seq_len: int,
    gpu: str = "h100-sxm",
    price_per_gpu_hour: float = 2.5,
    mfu: float = 0.4,
    num_gpus: int = 8,
    peak_tflops: float | None = None,
) -> CostEstimate:
    if not 0 < mfu <= 1:
        raise ValueError("mfu must be in (0, 1]")
    peak = peak_tflops if peak_tflops is not None else GPUS[gpu].bf16_dense_tflops
    fpt = estimate_flops_per_token(config, seq_len)
    total = fpt * tokens
    gpu_hours = total / (peak * 1e12 * mfu) / 3600
    counts = count_params(config)
    return CostEstimate(
        params_total=counts["total"],
        params_non_embedding=counts["non_embedding"],
        flops_per_token=fpt,
        total_flops=total,
        gpu_hours=gpu_hours,
        wall_hours=gpu_hours / num_gpus,
        cost_usd=gpu_hours * price_per_gpu_hour,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Estimate the FLOPs, GPU-hours, and cost of a training run")
    ap.add_argument("--config", required=True, help="TOML config with [model]")
    ap.add_argument("--tokens", required=True, help="number of training tokens, for example 500B or 1T")
    ap.add_argument(
        "--seq-len",
        type=int,
        default=0,
        help="sequence length (default: data.seq_len from the config, else max_seq_len)",
    )
    ap.add_argument("--gpu", default="h100-sxm", choices=sorted(GPUS))
    ap.add_argument(
        "--peak-tflops", type=float, default=None, help="dense BF16 peak of one GPU; overrides the value from the table"
    )
    ap.add_argument(
        "--price", type=float, default=2.5, help="price in US dollars per GPU-hour (default 2.5, the assumption in GOAL.md 3.4)"
    )
    ap.add_argument("--mfu", type=float, default=0.4)
    ap.add_argument("--num-gpus", type=int, default=8)
    args = ap.parse_args(argv)

    cfg = load_model_config(args.config)
    seq_len = args.seq_len
    if not seq_len:
        seq_len = read_toml(args.config).get("data", {}).get("seq_len", cfg.max_seq_len)
    tokens = parse_count(args.tokens)
    est = estimate_cost(
        cfg, tokens, seq_len, args.gpu, args.price, args.mfu, args.num_gpus, args.peak_tflops
    )
    spec = GPUS[args.gpu]
    peak = args.peak_tflops or spec.bf16_dense_tflops
    print(f"Config          {args.config}")
    print(
        f"Parameters      {est.params_total / 1e6:,.1f}M ({est.params_non_embedding / 1e6:,.1f}M non-embedding)"
    )
    print(f"Tokens          {tokens:,.3g} ({tokens / est.params_total:,.1f} tokens/parameter)")
    print(f"Sequence length {seq_len}")
    print(f"FLOPs per token {est.flops_per_token:,.4g}")
    print(f"Total FLOPs     {est.total_flops:,.4g}")
    print(
        f"GPU             {spec.name}, dense BF16 peak {peak} TFLOPS{'' if spec.verified else ' (not verified)'}"
    )
    print(f"MFU             {args.mfu}")
    print(f"GPU-hours       {est.gpu_hours:,.1f} GPU·h")
    print(
        f"Wall-clock time {est.wall_hours:,.1f} h ({args.num_gpus} GPUs) = {est.wall_hours / 24:,.2f} days"
    )
    print(f"Cost            ${est.cost_usd:,.0f} (${args.price}/GPU-hour)")


if __name__ == "__main__":
    main()
