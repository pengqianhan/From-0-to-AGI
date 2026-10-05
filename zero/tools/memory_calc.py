"""训练显存计算器（对应第 14 章"预训练工程"）。

    uv run python -m zero.tools.memory_calc configs/main/pretrain.toml
    uv run python -m zero.tools.memory_calc configs/main/pretrain.toml --micro-batch 4 --checkpointing
    uv run python -m zero.tools.memory_calc configs/main/pretrain.toml --strategy fsdp --num-gpus 8

一张卡上训练时，显存里有四类东西（P = 参数个数）：

| 类别 | zero 的写法（BF16 autocast + AdamW） | 每参数字节 |
|---|---|---:|
| 参数 | FP32 主权重 | 4 |
| 梯度 | FP32（和参数同 dtype） | 4 |
| 优化器状态 | AdamW 的一阶矩 m、二阶矩 v，各 FP32 | 8 |
| 合计（"16 字节/参数"） | 与 ZeRO 论文混合精度 Adam 的 2 + 2 + 12 相同 | 16 |

再加上：
- **BF16 权重副本**：autocast 在前向里把矩阵乘的权重转成 BF16，这些副本被保存下来给反向用
  （每个参与矩阵乘的参数 2 字节）；
- **DDP 通信桶**：PyTorch DDP 默认 `gradient_as_bucket_view=False`，梯度在桶里还有一份拷贝（4 字节/参数）；
- **激活值**：前向时为反向保存的中间张量，和 micro batch × 序列长度成正比，是最大的变量；
- **logits 的瞬时峰值**：词表很大时，前向结束那一刻 BF16 logits + FP32 拷贝同时存在。

激活值的公式不是抄来的，而是按 zero 的 `Block` 逐个张量数出来的：在 CPU 上用
`torch.autograd.graph.saved_tensors_hooks` 记录反向要用的每个张量（按存储去重），
`tests/test_memory_calc.py` 保证公式和真实记录的字节数**逐字节相等**（FP32 与 CPU 上的 BF16 autocast
两种情况，注意力走 SDPA 的 flash 内核，不保存 T×T 的注意力矩阵）。
每层每 token（d = dim，q = q_dim，kv = kv_dim，f = ffn_dim，h / h_kv = 查询 / KV 头数）：

    BF16 autocast：26d + 10q + 10kv + 8f + 8h + 4h_kv + 8 字节
    FP32：         24d + 16q + 16kv + 16f + 8h + 4h_kv + 8 字节
    最后的 norm + lm_head + 交叉熵：BF16 10d + 4 + 4V，FP32 12d + 4 + 4V；输入 token id 8 字节

**口径与局限（务必读）**：
- 按"所有东西同时在显存里"相加，是偏保守的上界；实际峰值出现在反向途中，部分激活已释放；
- CUDA 上 autocast 的算子清单与 CPU 略有差别，`torch.compile` 会融合逐元素运算、少存很多中间张量。
  单张 RTX 3090 上的实测（2026-10，见 runs/2026-10-01-gpu0-check/，单卡、eager）：主线配置本估算比 PyTorch
  实际分配的峰值高 2.1–2.4 GiB，缓存分配器的 reserved 峰值又可能比本估算再高约 2 GiB；"放不放得下"的判断
  （T=4096 不开检查点 0 条、开检查点 3 条；T=16K/32K 开检查点也放不下）全部与实测一致。多卡数字尚未在 GPU 上验证；
- 不含 CUDA 上下文、NCCL 缓冲、内存碎片（通常再留 2–5 GB 余量）；
- 激活检查点（activation checkpointing）：每层只存块的输入（FP32 残差流，4d 字节/token），反向时重算
  整层 —— 峰值再加一层的完整激活。zero 已实现（`train.activation_checkpointing`，见 zero/model.py）；
- FSDP / ZeRO 的切分按理想情况（均分到每张卡）计算，FSDP 另加"正在用的那一层 + 预取的下一层"的
  BF16 完整参数。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path

from zero.config import ModelConfig, load_model_config, read_toml
from zero.model import count_params

STRATEGIES = ("ddp", "zero1", "zero2", "fsdp")
GiB = 1024**3


def _dims(cfg: ModelConfig) -> tuple[int, int, int, int, int, int]:
    assert cfg.head_dim is not None
    return cfg.dim, cfg.q_dim, cfg.kv_dim, cfg.ffn_dim, cfg.n_heads, cfg.n_kv_heads


def matmul_params(cfg: ModelConfig) -> tuple[int, int]:
    """(每层参与矩阵乘的参数, lm_head 参数)。与 zero.model.estimate_flops_per_token 的 N_matmul 口径一致。"""
    d, q, kv, f, _, _ = _dims(cfg)
    return d * q + 2 * d * kv + q * d + 3 * d * f, d * cfg.vocab_size


def layer_activation_bytes_per_token(cfg: ModelConfig, dtype: str = "bf16") -> int:
    """一个 Block 为反向保存的激活，每个 token 多少字节（不含权重）。

    BF16 autocast 时逐项（与 zero/model.py 的前向一一对应）：
      attn_norm：输入 x、归一化后的 x（FP32，各 4d）+ rstd（4）
      wq / wk / wv：各自把输入转成 BF16 再乘（3 份 2d）
      q_norm：q 的 FP32 拷贝 4q + rstd 4h + 归一化后的 BF16 2q；k_norm 同理（kv、h_kv）
      SDPA（flash）：q 2q、k 2kv、v 2kv、输出 2q（BF16）+ 每行 logsumexp 4h（FP32）
      ffn_norm：同 attn_norm（8d + 4）；w_gate / w_up 输入各 2d
      SwiGLU：gate 输出、silu 输出、up 输出、乘积（w_down 的输入）各 2f
    """
    d, q, kv, f, h, hkv = _dims(cfg)
    if dtype == "bf16":
        return 26 * d + 10 * q + 10 * kv + 8 * f + 8 * h + 4 * hkv + 8
    if dtype == "fp32":
        # FP32 时同一个张量喂给 wq/wk/wv 只存一份；归一化后的 q/k 以 FP32 存
        return 24 * d + 16 * q + 16 * kv + 16 * f + 8 * h + 4 * hkv + 8
    raise ValueError(f"dtype 只能是 bf16 / fp32，当前 {dtype!r}")


def head_activation_bytes_per_token(cfg: ModelConfig, dtype: str = "bf16") -> int:
    """最后的 RMSNorm + lm_head + 交叉熵（FP32 log-probs）+ 输入 token id（int64）。"""
    d, V = cfg.dim, cfg.vocab_size
    head = (10 * d + 4 if dtype == "bf16" else 12 * d + 4) + 4 * V
    return head + 8


def weight_copy_bytes(cfg: ModelConfig, dtype: str = "bf16", layers: int | None = None) -> int:
    """autocast 为反向保存的 BF16 权重副本；layers=None 表示全部层（不做激活检查点时）。"""
    if dtype != "bf16":
        return 0
    per_layer, head = matmul_params(cfg)
    n = cfg.n_layers if layers is None else layers
    return 2 * (n * per_layer + head)


@dataclass
class MemoryEstimate:
    """一张卡上的显存（字节）。"""

    params: int
    grads: int
    optimizer: int
    ddp_buckets: int
    weight_copies: int  # autocast 的 BF16 权重副本，或 FSDP all-gather 出来的 BF16 完整参数
    activations: int
    logits_transient: int

    @property
    def static(self) -> int:
        return self.params + self.grads + self.optimizer + self.ddp_buckets

    @property
    def total(self) -> int:
        return sum(getattr(self, f.name) for f in fields(self))


def estimate_memory(
    cfg: ModelConfig,
    micro_batch_size: int,
    seq_len: int,
    num_gpus: int = 1,
    strategy: str = "ddp",
    dtype: str = "bf16",
    checkpointing: bool = False,
) -> MemoryEstimate:
    """估算每张卡的训练显存。strategy：ddp / zero1（切优化器）/ zero2（再切梯度）/ fsdp（再切参数）。"""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy 只能是 {STRATEGIES}，当前 {strategy!r}")
    P = count_params(cfg)["total"]
    N = num_gpus
    params, grads, optim = 4 * P, 4 * P, 8 * P
    buckets = 4 * P if (strategy in ("ddp", "zero1") and N > 1) else 0
    if strategy in ("zero1", "zero2", "fsdp"):
        optim = optim // N
    if strategy in ("zero2", "fsdp"):
        grads = grads // N
    if strategy == "fsdp":
        params = params // N

    tokens = micro_batch_size * seq_len
    layer = layer_activation_bytes_per_token(cfg, dtype)
    head = head_activation_bytes_per_token(cfg, dtype)
    if checkpointing:
        # 每层只存块输入（FP32 残差流 4d），反向时再加一层完整激活（重算那一层）
        act = tokens * (cfg.n_layers * 4 * cfg.dim + layer + head)
        copies = weight_copy_bytes(cfg, dtype, layers=1)
    else:
        act = tokens * (cfg.n_layers * layer + head)
        copies = weight_copy_bytes(cfg, dtype)
    if strategy == "fsdp":
        # 参数以 BF16 all-gather：当前层 + 预取的下一层，外加 embedding / lm_head
        per_layer = count_params(cfg)["per_layer"]
        copies = 2 * (2 * per_layer + cfg.vocab_size * cfg.dim)
    # 前向结束时：BF16 logits（2V）+ 交叉熵前转出的 FP32 拷贝（4V）；FP32 训练只有 logits 本身（4V）
    logits = tokens * cfg.vocab_size * (6 if dtype == "bf16" else 4)
    return MemoryEstimate(params, grads, optim, buckets, copies, act, logits)


def max_micro_batch(cfg: ModelConfig, seq_len: int, gpu_gib: float, **kw: object) -> int:
    """在 gpu_gib 以内能放下的最大 micro batch（0 表示一条都放不下）。"""
    b = 0
    while estimate_memory(cfg, b + 1, seq_len, **kw).total <= gpu_gib * GiB:  # type: ignore[arg-type]
        b += 1
        if b > 4096:
            break
    return b


def _fmt(n: int) -> str:
    return f"{n / GiB:8.2f}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="估算训练时每张卡的显存")
    ap.add_argument(
        "config", help="含 [model] 的 TOML 配置（有 [train]/[data] 时读 micro batch 和 seq_len）"
    )
    ap.add_argument(
        "--micro-batch", type=int, default=0, help="每张卡每次前向的序列数（默认读配置）"
    )
    ap.add_argument("--seq-len", type=int, default=0, help="序列长度（默认读配置的 data.seq_len）")
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--strategy", choices=STRATEGIES + ("all",), default="all")
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument(
        "--checkpointing",
        action="store_true",
        help="估算开启激活检查点（对应 train.activation_checkpointing；单卡实测见 runs/2026-10-01-gpu0-check/）",
    )
    ap.add_argument(
        "--gpu-mem", type=float, default=80.0, help="单卡显存 GiB（用来判断放不放得下）"
    )
    args = ap.parse_args(argv)

    cfg = load_model_config(args.config)
    raw = read_toml(args.config)
    mb = args.micro_batch or int(raw.get("train", {}).get("micro_batch_size", 1))
    T = args.seq_len or int(raw.get("data", {}).get("seq_len", cfg.max_seq_len))
    P = count_params(cfg)["total"]
    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)

    print(f"配置      {Path(args.config)}（{P / 1e6:,.1f}M 参数）")
    print(
        f"每卡      micro batch {mb} × seq_len {T} = {mb * T:,} token；{args.num_gpus} 卡；{args.dtype}"
        f"{'；激活检查点' if args.checkpointing else ''}"
    )
    print(
        f"每层激活  {layer_activation_bytes_per_token(cfg, args.dtype):,} 字节/token；"
        f"头部 {head_activation_bytes_per_token(cfg, args.dtype):,} 字节/token"
    )
    print()
    cols = ["参数", "梯度", "优化器", "DDP桶", "权重副本", "激活", "logits", "合计"]
    print(f"{'策略':<6}" + "".join(f"{c:>9}" for c in cols) + "   单位 GiB")
    for s in strategies:
        est = estimate_memory(cfg, mb, T, args.num_gpus, s, args.dtype, args.checkpointing)
        vals = [
            est.params,
            est.grads,
            est.optimizer,
            est.ddp_buckets,
            est.weight_copies,
            est.activations,
            est.logits_transient,
            est.total,
        ]
        fits = "放得下" if est.total <= args.gpu_mem * GiB else f"超过 {args.gpu_mem:g} GiB"
        max_b = max_micro_batch(
            cfg,
            T,
            args.gpu_mem,
            num_gpus=args.num_gpus,
            strategy=s,
            dtype=args.dtype,
            checkpointing=args.checkpointing,
        )
        print(
            f"{s:<6}"
            + "".join(f"{_fmt(v):>9}" for v in vals)
            + f"   {fits}；最大 micro batch ≈ {max_b}"
        )
    print(
        "\n口径：各项同时在显存里的保守上界；不含 CUDA 上下文与碎片；torch.compile 会更省。"
        "GPU 上尚未验证，以阶段 6 实测为准。"
    )


if __name__ == "__main__":
    main()
