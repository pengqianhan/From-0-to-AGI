"""Training memory calculator (Chapter 14, "Pretraining engineering").

    uv run python -m zero.tools.memory_calc configs/main/pretrain.toml
    uv run python -m zero.tools.memory_calc configs/main/pretrain.toml --micro-batch 4 --checkpointing
    uv run python -m zero.tools.memory_calc configs/main/pretrain.toml --strategy fsdp --num-gpus 8

When you train on one GPU, the GPU memory holds four kinds of data (P = number of parameters):

| Kind | How zero stores it (BF16 autocast + AdamW) | Bytes per parameter |
|---|---|---:|
| Parameters | FP32 master weights | 4 |
| Gradients | FP32 (the same dtype as the parameters) | 4 |
| Optimizer state | AdamW first moment m and second moment v, each FP32 | 8 |
| Total ("16 bytes/parameter") | The same as the 2 + 2 + 12 of mixed-precision Adam in the ZeRO paper | 16 |

Add these items:
- **BF16 weight copies**: in the forward pass, autocast converts the matmul weights to BF16.
  The backward pass needs these copies, so they stay in memory
  (2 bytes for each parameter that is part of a matmul).
- **DDP communication buckets**: PyTorch DDP uses `gradient_as_bucket_view=False` by default.
  Thus the buckets hold one more copy of the gradients (4 bytes/parameter).
- **Activations**: intermediate tensors that the forward pass saves for the backward pass.
  They are proportional to micro batch × sequence length. They are the largest variable item.
- **Transient peak of the logits**: with a large vocabulary, the BF16 logits and an FP32 copy
  exist at the same time at the end of the forward pass.

We did not copy the activation formula from a paper. We counted it tensor by tensor in the zero
`Block`. On the CPU, `torch.autograd.graph.saved_tensors_hooks` records each tensor that the
backward pass needs (one entry for each storage). `tests/test_memory_calc.py` makes sure that the
formula and the recorded bytes are **equal to the byte**. The test covers two cases: FP32, and BF16
autocast on the CPU. Attention uses the SDPA flash kernel, which does not save the T×T attention
matrix.
For each layer and each token (d = dim, q = q_dim, kv = kv_dim, f = ffn_dim,
h / h_kv = number of query / KV heads):

    BF16 autocast: 26d + 10q + 10kv + 8f + 8h + 4h_kv + 8 bytes
    FP32:          24d + 16q + 16kv + 16f + 8h + 4h_kv + 8 bytes
    Final norm + lm_head + cross-entropy: BF16 10d + 4 + 4V, FP32 12d + 4 + 4V; input token ids 8 bytes

**What the estimate includes, and its limits (read this)**:
- The estimate adds all items as if they are in memory at the same time. Thus it is a
  conservative upper bound. The real peak occurs during the backward pass, when some activations
  are already free.
- On CUDA, the list of autocast operators is a little different from the CPU list.
  `torch.compile` fuses element-wise operations and saves many fewer intermediate tensors.
  We measured on one RTX 3090 (2026-10, see runs/2026-10-01-gpu0-check/, one GPU, eager mode).
  For the main-line config, this estimate is 2.1–2.4 GiB higher than the peak that PyTorch
  really allocates. The reserved peak of the caching allocator can be about 2 GiB higher than
  this estimate. All "does it fit" results agree with the measurements: at T=4096, 0 sequences
  fit without checkpointing and 3 fit with checkpointing; at T=16K/32K, nothing fits even with
  checkpointing. The numbers for more than one GPU are not verified on GPUs yet.
- The estimate does not include the CUDA context, the NCCL buffers, or memory fragmentation.
  Usually, keep 2–5 GB more as a margin.
- Activation checkpointing: each layer stores only the input of the block (the FP32 residual
  stream, 4d bytes/token). The backward pass computes the full layer again. Thus the peak
  adds the full activations of one layer. zero implements this
  (`train.activation_checkpointing`, see zero/model.py).
- The FSDP / ZeRO sharding uses the ideal case (an equal split across all GPUs). FSDP also adds
  the full BF16 parameters of "the layer in use + the next layer that it prefetches".
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
    """(matmul parameters per layer, lm_head parameters).

    The same definition as N_matmul in zero.model.estimate_flops_per_token.
    """
    d, q, kv, f, _, _ = _dims(cfg)
    return d * q + 2 * d * kv + q * d + 3 * d * f, d * cfg.vocab_size


def layer_activation_bytes_per_token(cfg: ModelConfig, dtype: str = "bf16") -> int:
    """Bytes per token of the activations that one Block saves for the backward pass (no weights).

    The items for BF16 autocast (each one matches a step of the forward pass in zero/model.py):
      attn_norm: input x, normalized x (FP32, 4d each) + rstd (4)
      wq / wk / wv: each one converts its input to BF16, then multiplies (3 copies of 2d)
      q_norm: FP32 copy of q 4q + rstd 4h + normalized BF16 2q; k_norm the same (kv, h_kv)
      SDPA (flash): q 2q, k 2kv, v 2kv, output 2q (BF16) + logsumexp of each row 4h (FP32)
      ffn_norm: the same as attn_norm (8d + 4); inputs of w_gate / w_up 2d each
      SwiGLU: gate output, silu output, up output, product (the input of w_down) 2f each
    """
    d, q, kv, f, h, hkv = _dims(cfg)
    if dtype == "bf16":
        return 26 * d + 10 * q + 10 * kv + 8 * f + 8 * h + 4 * hkv + 8
    if dtype == "fp32":
        # In FP32, wq/wk/wv get the same tensor, so it is stored once; normalized q/k are stored in FP32
        return 24 * d + 16 * q + 16 * kv + 16 * f + 8 * h + 4 * hkv + 8
    raise ValueError(f"dtype must be bf16 / fp32, got {dtype!r}")


def head_activation_bytes_per_token(cfg: ModelConfig, dtype: str = "bf16") -> int:
    """Final RMSNorm + lm_head + cross-entropy (FP32 log-probs) + input token ids (int64)."""
    d, V = cfg.dim, cfg.vocab_size
    head = (10 * d + 4 if dtype == "bf16" else 12 * d + 4) + 4 * V
    return head + 8


def weight_copy_bytes(cfg: ModelConfig, dtype: str = "bf16", layers: int | None = None) -> int:
    """BF16 weight copies that autocast saves for the backward pass.

    layers=None means all layers (no activation checkpointing).
    """
    if dtype != "bf16":
        return 0
    per_layer, head = matmul_params(cfg)
    n = cfg.n_layers if layers is None else layers
    return 2 * (n * per_layer + head)


@dataclass
class MemoryEstimate:
    """GPU memory on one GPU (bytes)."""

    params: int
    grads: int
    optimizer: int
    ddp_buckets: int
    weight_copies: int  # BF16 weight copies from autocast, or full BF16 parameters from the FSDP all-gather
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
    """Estimate the training memory on each GPU.

    strategy: ddp / zero1 (shard the optimizer) / zero2 (also shard the gradients) /
    fsdp (also shard the parameters).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
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
        # Each layer stores only the block input (FP32 residual stream, 4d).
        # The backward pass adds the full activations of one layer (the layer that it computes again).
        act = tokens * (cfg.n_layers * 4 * cfg.dim + layer + head)
        copies = weight_copy_bytes(cfg, dtype, layers=1)
    else:
        act = tokens * (cfg.n_layers * layer + head)
        copies = weight_copy_bytes(cfg, dtype)
    if strategy == "fsdp":
        # The all-gather gives BF16 parameters: current layer + prefetched next layer, plus embedding / lm_head
        per_layer = count_params(cfg)["per_layer"]
        copies = 2 * (2 * per_layer + cfg.vocab_size * cfg.dim)
    # At the end of the forward pass: BF16 logits (2V) + the FP32 copy made before cross-entropy (4V).
    # FP32 training has only the logits (4V).
    logits = tokens * cfg.vocab_size * (6 if dtype == "bf16" else 4)
    return MemoryEstimate(params, grads, optim, buckets, copies, act, logits)


def max_micro_batch(cfg: ModelConfig, seq_len: int, gpu_gib: float, **kw: object) -> int:
    """The largest micro batch that fits in gpu_gib (0 means that not even one sequence fits)."""
    b = 0
    while estimate_memory(cfg, b + 1, seq_len, **kw).total <= gpu_gib * GiB:  # type: ignore[arg-type]
        b += 1
        if b > 4096:
            break
    return b


def _fmt(n: int) -> str:
    return f"{n / GiB:8.2f}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Estimate the training memory on each GPU")
    ap.add_argument(
        "config", help="TOML config with [model] (micro batch and seq_len come from [train]/[data] if present)"
    )
    ap.add_argument(
        "--micro-batch", type=int, default=0, help="sequences per forward pass on each GPU (default: from the config)"
    )
    ap.add_argument("--seq-len", type=int, default=0, help="sequence length (default: data.seq_len from the config)")
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--strategy", choices=STRATEGIES + ("all",), default="all")
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument(
        "--checkpointing",
        action="store_true",
        help="estimate with activation checkpointing (train.activation_checkpointing; one-GPU measurements: runs/2026-10-01-gpu0-check/)",
    )
    ap.add_argument(
        "--gpu-mem", type=float, default=80.0, help="memory of one GPU in GiB (to decide if the job fits)"
    )
    args = ap.parse_args(argv)

    cfg = load_model_config(args.config)
    raw = read_toml(args.config)
    mb = args.micro_batch or int(raw.get("train", {}).get("micro_batch_size", 1))
    T = args.seq_len or int(raw.get("data", {}).get("seq_len", cfg.max_seq_len))
    P = count_params(cfg)["total"]
    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)

    print(f"Config       {Path(args.config)} ({P / 1e6:,.1f}M parameters)")
    print(
        f"Per GPU      micro batch {mb} × seq_len {T} = {mb * T:,} tokens; {args.num_gpus} GPUs; {args.dtype}"
        f"{'; activation checkpointing' if args.checkpointing else ''}"
    )
    print(
        f"Activations  {layer_activation_bytes_per_token(cfg, args.dtype):,} bytes/token per layer; "
        f"head {head_activation_bytes_per_token(cfg, args.dtype):,} bytes/token"
    )
    print()
    cols = ["params", "grads", "optim", "buckets", "w-copies", "activ.", "logits", "total"]
    print(f"{'method':<6}" + "".join(f"{c:>9}" for c in cols) + "   (GiB)")
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
        fits = "fits" if est.total <= args.gpu_mem * GiB else f"over {args.gpu_mem:g} GiB"
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
            + f"   {fits}; max micro batch ≈ {max_b}"
        )
    print(
        "\nBasis: a conservative upper bound (all items in memory at the same time); no CUDA context or fragmentation; torch.compile uses less. "
        "Not verified on GPUs yet; the stage 6 measurements are the reference."
    )


if __name__ == "__main__":
    main()
