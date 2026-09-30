"""训练成本估算：FLOPs → 卡时 → 美元（对应第 12 章，GOAL.md 9.1"第二步就绪"）。

    总 FLOPs  = 每 token FLOPs × token 数                （每 token FLOPs = 6N + 注意力项，见 zero/model.py）
    卡时      = 总 FLOPs / (单卡峰值 FLOPs/s × MFU) / 3600
    费用      = 卡时 × 每卡时单价
    墙钟时间  = 卡时 / 卡数

MFU（Model FLOPs Utilization）= 实际有效算力 / 硬件峰值。稠密 Transformer 在 H100 上训练，
调得好的开源实现一般在 0.3–0.5 之间，默认取 0.4；第二步阶段 6 的 GPU 验证运行会实测后更新。

用法：

    uv run python -m zero.tools.estimate_cost --config configs/main/pretrain.toml \\
        --tokens 500B --gpu h100-sxm --price 2.5 --num-gpus 8 --mfu 0.4

GPU 峰值表：只收"稠密 BF16 Tensor Core"算力（不含 2:4 结构化稀疏的翻倍数字）。NVIDIA 数据手册里
BF16 一栏标注的是"带稀疏"的数值，稠密值是它的一半。本表数值出自下列官方页面，但编写时的环境
无法访问 nvidia.com 逐条复核，所以每一项都标了 `verified=False`（待核实）——第二步花钱之前请对照
数据手册确认，并用实测 MFU 校正。
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
    verified: bool = False  # 是否已对照官方数据手册核实
    note: str = ""


GPUS: dict[str, GPUSpec] = {
    "h100-sxm": GPUSpec(
        "NVIDIA H100 SXM5",
        989.5,
        "https://www.nvidia.com/en-us/data-center/h100/（手册 BF16 1,979 TFLOPS 带稀疏 → 稠密 989.5）",
        note="待核实",
    ),
    "h100-pcie": GPUSpec(
        "NVIDIA H100 PCIe",
        756.5,
        "https://www.nvidia.com/en-us/data-center/h100/（手册 BF16 1,513 TFLOPS 带稀疏 → 稠密 756.5）",
        note="待核实",
    ),
    "h200-sxm": GPUSpec(
        "NVIDIA H200 SXM",
        989.5,
        "https://www.nvidia.com/en-us/data-center/h200/（手册 BF16 1,979 TFLOPS 带稀疏 → 稠密 989.5）",
        note="待核实",
    ),
    "a100": GPUSpec(
        "NVIDIA A100 (SXM/PCIe)",
        312.0,
        "https://www.nvidia.com/en-us/data-center/a100/（手册 BF16 312 TFLOPS，带稀疏 624）",
        note="待核实",
    ),
    "l40s": GPUSpec(
        "NVIDIA L40S",
        366.0,
        "https://www.nvidia.com/en-us/data-center/l40s/（手册 BF16 带稀疏 733 TFLOPS → 稠密约 366）",
        note="待核实：手册的稠密值是否为 362 或 366 需复核",
    ),
    "rtx4090": GPUSpec(
        "NVIDIA GeForce RTX 4090",
        165.2,
        "NVIDIA Ada Lovelace 架构白皮书（BF16 Tensor、FP32 累加 165.2 TFLOPS，带稀疏 330.4）",
        note="待核实",
    ),
}


def peak_tflops_for_device_name(device_name: str) -> float | None:
    """根据 `torch.cuda.get_device_name()` 的返回值粗略匹配峰值（MFU 日志用）；匹配不上返回 None。"""
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
    """'500B' / '1.2T' / '3e11' → 数字。"""
    m = re.fullmatch(r"\s*([0-9.eE+-]+)\s*([kKmMbBgGtT]?)\s*", s)
    if not m:
        raise ValueError(f"看不懂的数量：{s!r}（例：500B、1.2T、3e11）")
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
        raise ValueError("mfu 必须在 (0, 1]")
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
    ap = argparse.ArgumentParser(description="估算训练的 FLOPs、卡时和费用")
    ap.add_argument("--config", required=True, help="含 [model] 的 TOML 配置")
    ap.add_argument("--tokens", required=True, help="训练 token 数，如 500B、1T")
    ap.add_argument(
        "--seq-len",
        type=int,
        default=0,
        help="序列长度（默认读配置里的 data.seq_len，否则 max_seq_len）",
    )
    ap.add_argument("--gpu", default="h100-sxm", choices=sorted(GPUS))
    ap.add_argument(
        "--peak-tflops", type=float, default=None, help="手动指定单卡稠密 BF16 峰值，覆盖查表"
    )
    ap.add_argument(
        "--price", type=float, default=2.5, help="每卡时美元单价（默认 2.5，GOAL.md 3.4 的假设）"
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
    print(f"配置            {args.config}")
    print(
        f"参数量          {est.params_total / 1e6:,.1f}M（非 embedding {est.params_non_embedding / 1e6:,.1f}M）"
    )
    print(f"token 数        {tokens:,.3g}（{tokens / est.params_total:,.1f} token/参数）")
    print(f"序列长度        {seq_len}")
    print(f"每 token FLOPs  {est.flops_per_token:,.4g}")
    print(f"总 FLOPs        {est.total_flops:,.4g}")
    print(
        f"GPU             {spec.name}，稠密 BF16 峰值 {peak} TFLOPS{'' if spec.verified else '（待核实）'}"
    )
    print(f"MFU             {args.mfu}")
    print(f"卡时            {est.gpu_hours:,.1f} GPU·h")
    print(
        f"墙钟时间        {est.wall_hours:,.1f} h（{args.num_gpus} 卡）= {est.wall_hours / 24:,.2f} 天"
    )
    print(f"费用            ${est.cost_usd:,.0f}（${args.price}/卡时）")


if __name__ == "__main__":
    main()
