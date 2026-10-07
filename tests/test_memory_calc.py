"""Training memory calculator (zero/tools/memory_calc.py): hand-calculated numbers + a byte-exact parity check against the activations that are really saved (Chapter 14)."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from zero.config import ModelConfig
from zero.model import Transformer, count_params, cross_entropy_loss
from zero.tools.memory_calc import (
    GiB,
    estimate_memory,
    head_activation_bytes_per_token,
    layer_activation_bytes_per_token,
    main,
    max_micro_batch,
    weight_copy_bytes,
)

TINY = dict(
    vocab_size=512,
    dim=128,
    n_layers=2,
    n_heads=4,
    n_kv_heads=2,
    head_dim=32,
    ffn_dim=384,
    max_seq_len=256,
    tie_embeddings=True,
)


def _tiny() -> ModelConfig:
    return ModelConfig(**TINY)


def test_per_token_formulas_hand_checked() -> None:
    cfg = _tiny()  # d=128, q=128, kv=64, f=384, h=4, h_kv=2, V=512
    # BF16: 26·128 + 10·128 + 10·64 + 8·384 + 8·4 + 4·2 + 8 = 3328+1280+640+3072+32+8+8
    assert layer_activation_bytes_per_token(cfg, "bf16") == 8368
    # FP32: 24·128 + 16·128 + 16·64 + 16·384 + 8·4 + 4·2 + 8 = 3072+2048+1024+6144+32+8+8
    assert layer_activation_bytes_per_token(cfg, "fp32") == 12336
    # Head: BF16 10·128 + 4 + 4·512 + 8 = 3340; FP32 12·128 + 4 + 2048 + 8 = 3596
    assert head_activation_bytes_per_token(cfg, "bf16") == 3340
    assert head_activation_bytes_per_token(cfg, "fp32") == 3596
    # BF16 weight copies: (128·128 + 2·128·64 + 128·128 + 3·128·384) = 196,608 per layer, 65,536 for lm_head, × 2 bytes
    assert weight_copy_bytes(cfg) == 2 * (2 * 196_608 + 65_536)
    assert weight_copy_bytes(cfg, "fp32") == 0


def test_static_memory_16_bytes_per_param_and_sharding() -> None:
    cfg = _tiny()
    P = count_params(cfg)["total"]
    one = estimate_memory(cfg, 1, 128, num_gpus=1, strategy="ddp")
    assert (one.params, one.grads, one.optimizer, one.ddp_buckets) == (4 * P, 4 * P, 8 * P, 0)
    ddp = estimate_memory(cfg, 1, 128, num_gpus=8, strategy="ddp")
    assert ddp.static == 20 * P  # 16 bytes/parameter + one FP32 copy of the gradients in the DDP buckets
    z1 = estimate_memory(cfg, 1, 128, num_gpus=8, strategy="zero1")
    assert z1.optimizer == 8 * P // 8 and z1.grads == 4 * P
    z2 = estimate_memory(cfg, 1, 128, num_gpus=8, strategy="zero2")
    assert (z2.grads, z2.optimizer, z2.ddp_buckets) == (4 * P // 8, 8 * P // 8, 0)
    fsdp = estimate_memory(cfg, 1, 128, num_gpus=8, strategy="fsdp")
    assert fsdp.static == 4 * P // 8 + 4 * P // 8 + 8 * P // 8
    # Activations do not depend on the sharding method, only on the tokens per GPU
    assert fsdp.activations == ddp.activations
    with pytest.raises(ValueError):
        estimate_memory(cfg, 1, 128, strategy="tp")


def test_main_config_numbers() -> None:
    """Main-line model (689.5M): parameter count by hand; 16 bytes/parameter; micro batch 8 does not fit in 80 GiB (eager estimate)."""
    cfg = ModelConfig(
        vocab_size=65536,
        dim=1280,
        n_layers=28,
        n_heads=16,
        n_kv_heads=8,
        head_dim=128,
        ffn_dim=3584,
        max_seq_len=4096,
        tie_embeddings=True,
    )
    P = count_params(cfg)["total"]
    # Each layer: attention 1280·2048 + 2·1280·1024 + 2048·1280 = 7,864,320; QK-Norm 256;
    # SwiGLU 3·1280·3584 = 13,762,560; two RMSNorms 2,560 → 21,629,696 × 28 = 605,631,488
    # embedding 65,536·1280 = 83,886,080 (shared with lm_head), final norm 1,280
    assert P == 605_631_488 + 83_886_080 + 1_280 == 689_518_848
    assert layer_activation_bytes_per_token(cfg) == 92_840
    est = estimate_memory(cfg, 8, 4096, num_gpus=8, strategy="ddp")
    assert est.params + est.grads + est.optimizer == 16 * P
    assert est.activations == 8 * 4096 * (28 * 92_840 + head_activation_bytes_per_token(cfg))
    ckpt = estimate_memory(cfg, 8, 4096, num_gpus=8, strategy="ddp", checkpointing=True)
    assert ckpt.activations < est.activations / 5
    assert max_micro_batch(cfg, 4096, 80.0, num_gpus=8, strategy="ddp") < 8


def _saved_bytes(model: nn.Module, tokens: torch.Tensor, autocast: bool) -> int:
    """Record the tensors that the forward pass really saves for the backward pass.

    Count each storage once. Do not count the parameters and buffers.
    """
    own = {p.untyped_storage().data_ptr() for p in model.parameters()}
    own |= {b.untyped_storage().data_ptr() for b in model.buffers()}
    seen: dict[int, int] = {}

    def pack(t: torch.Tensor) -> torch.Tensor:
        key = t.untyped_storage().data_ptr()
        if key not in own and key not in seen:
            seen[key] = t.untyped_storage().nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            logits = model(tokens)
        loss = cross_entropy_loss(logits, tokens)
    loss.backward()
    return sum(seen.values()) - 4  # subtract the one scalar that cross-entropy saves


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_activation_formula_matches_saved_tensors(dtype: str) -> None:
    torch.manual_seed(0)
    cfg = _tiny()
    model = Transformer(cfg)
    B, T = 2, 128
    tokens = torch.randint(0, cfg.vocab_size, (B, T))
    measured = _saved_bytes(model, tokens, autocast=dtype == "bf16")
    est = estimate_memory(cfg, B, T, dtype=dtype)
    assert measured == est.activations + est.weight_copies


class _CheckpointedBlock(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, x, cos, sin, kv_cache=None, start_pos=0):  # noqa: ANN001, ANN201
        return checkpoint(self.block, x, cos, sin, use_reentrant=True)


def test_checkpointing_keeps_only_block_inputs() -> None:
    """Activation checkpointing: each layer keeps only the block input (FP32 residual stream, 4d bytes/token).

    The one extra layer in the formula is the transient peak during the recomputation.
    """
    torch.manual_seed(0)
    cfg = _tiny()
    model = Transformer(cfg)
    model.layers = nn.ModuleList([_CheckpointedBlock(b) for b in model.layers])
    B, T = 2, 128
    tokens = torch.randint(0, cfg.vocab_size, (B, T))
    measured = _saved_bytes(model, tokens, autocast=True)
    lm_head_copy = 2 * cfg.dim * cfg.vocab_size
    expected = (
        B * T * (cfg.n_layers * 4 * cfg.dim + head_activation_bytes_per_token(cfg)) + lm_head_copy
    )
    assert measured == expected
    est = estimate_memory(cfg, B, T, checkpointing=True)
    assert est.activations == expected - lm_head_copy + B * T * layer_activation_bytes_per_token(
        cfg
    )


def test_cli_runs(capsys: pytest.CaptureFixture[str]) -> None:
    main(["configs/tiny/pretrain.toml", "--num-gpus", "2"])
    out = capsys.readouterr().out
    assert "fsdp" in out and "GiB" in out
    assert GiB == 1024**3
