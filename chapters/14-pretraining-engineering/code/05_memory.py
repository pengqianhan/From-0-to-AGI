"""Chapter 14 · Minimal code 5: what is in GPU memory during training? Count it, do not memorize a formula.

One training step keeps four types of data in memory:
  parameters (FP32 master weights, 4 bytes) + gradients (4 bytes) + AdamW m and v (4 bytes each)
  = 16 bytes per parameter,
  plus the activations that the forward pass saves for the backward pass.
  The activations are proportional to micro batch × sequence length.

This script counts each item on the CPU with the real model of zero (tiny shape):
  ① after one AdamW step, count the bytes of the parameters, the gradients, and the optimizer state;
  ② record each tensor that the backward pass needs with torch.autograd.graph.saved_tensors_hooks.
     Compare three cases: FP32, BF16 autocast, and BF16 + activation checkpointing ("ckpt");
  ③ do the same calculation for the main-line model (689.5M, configs/main/pretrain.toml)
     and see how to select the micro batch.

Run: uv run python chapters/14-pretraining-engineering/code/05_memory.py   (a few seconds)
"""

import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))  # so that `import zero` works from any folder of the repository

from zero.config import ModelConfig, load_model_config  # noqa: E402
from zero.model import Transformer, count_params, cross_entropy_loss  # noqa: E402
from zero.tools.memory_calc import GiB, estimate_memory  # noqa: E402

TINY = ModelConfig(vocab_size=2048, dim=128, n_layers=4, n_heads=4, n_kv_heads=2, head_dim=32,
                   ffn_dim=384, max_seq_len=256, tie_embeddings=True)
B, T = 4, 128


def nbytes(ts) -> int:
    return sum(t.numel() * t.element_size() for t in ts)


def saved_activation_bytes(model: nn.Module, tokens: torch.Tensor, autocast: bool) -> int:
    """Which tensors does autograd save for the backward pass during the forward pass?

    Count each storage once. Do not count the parameters and buffers.
    """
    own = {p.untyped_storage().data_ptr() for p in model.parameters()}
    own |= {b.untyped_storage().data_ptr() for b in model.buffers()}
    seen = {}

    def pack(t):
        key = t.untyped_storage().data_ptr()
        if key not in own and key not in seen:
            seen[key] = t.untyped_storage().nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            logits = model(tokens)
        loss = cross_entropy_loss(logits, tokens)
    loss.backward()
    return sum(seen.values())


class CheckpointedBlock(nn.Module):
    """Activation checkpointing: the forward pass saves only the input of the block.

    The backward pass calculates the full block again.
    """

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x, cos, sin, kv_cache=None, start_pos=0):
        return checkpoint(self.block, x, cos, sin, use_reentrant=True)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = Transformer(TINY)
    P = sum(p.numel() for p in model.parameters())
    tokens = torch.randint(0, TINY.vocab_size, (B, T))

    print(f"① Parameters + gradients + optimizer state (tiny: {P:,} parameters, after one AdamW step)")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cross_entropy_loss(model(tokens), tokens).backward()
    opt.step()
    rows = [
        ("parameter (FP32)", nbytes(model.parameters())),
        ("gradient (FP32)", nbytes(p.grad for p in model.parameters())),
        ("AdamW m (FP32)", nbytes(s["exp_avg"] for s in opt.state.values())),
        ("AdamW v (FP32)", nbytes(s["exp_avg_sq"] for s in opt.state.values())),
    ]
    for name, b in rows:
        print(f"  {name:<16} {b:>12,} bytes   {b / P:.0f} bytes/parameter")
    total = sum(b for _, b in rows)
    print(f"  {'total':<16} {total:>12,} bytes   {total / P:.0f} bytes/parameter")
    opt.zero_grad(set_to_none=True)

    print(f"\n② Activations saved for the backward pass (micro batch {B} × sequence {T} = {B * T} tokens, {TINY.n_layers} layers)")
    cases = [("FP32", model, False), ("BF16 autocast", model, True)]
    ckpt_model = Transformer(TINY)
    ckpt_model.load_state_dict(model.state_dict())
    ckpt_model.layers = nn.ModuleList([CheckpointedBlock(b) for b in ckpt_model.layers])
    cases.append(("BF16 + ckpt", ckpt_model, True))
    base = None
    for name, m, ac in cases:
        b = saved_activation_bytes(m, tokens, ac)
        base = base or b
        print(f"  {name:<18} {b:>11,} bytes = {b / (B * T):>8,.0f} bytes/token   ({b / base:.0%})")
    print("  BF16 autocast also includes the BF16 weight copies that the forward pass makes. With activation checkpointing (ckpt),")
    print("  each layer keeps only the block input (4·dim bytes/token). The cost: in the backward pass, each layer does its forward pass again (about 1/3 more compute)")

    print("\n③ The same calculation for the main-line model (8×H100 80GB, BF16, seq 4096; zero/tools/memory_calc.py)")
    main_cfg = load_model_config(ROOT / "configs/main/pretrain.toml")
    Pm = count_params(main_cfg)["total"]
    print(f"  Parameters {Pm / 1e6:.1f}M × 16 bytes = {16 * Pm / GiB:.1f} GiB (each GPU stores a full copy: DDP)")
    print(f"  {'micro batch':>11} {'activ.':>8} {'DDP total':>9} {'FSDP total':>10} {'DDP + ckpt':>12}")
    for mb in (1, 2, 4, 8):
        ddp = estimate_memory(main_cfg, mb, 4096, num_gpus=8, strategy="ddp")
        fsdp = estimate_memory(main_cfg, mb, 4096, num_gpus=8, strategy="fsdp")
        ck = estimate_memory(main_cfg, mb, 4096, num_gpus=8, strategy="ddp", checkpointing=True)
        print(f"  {mb:>11} {ddp.activations / GiB:>7.1f}G {ddp.total / GiB:>8.1f}G "
              f"{fsdp.total / GiB:>9.1f}G {ck.total / GiB:>11.1f}G")
    print("  (Conservative upper bound, eager mode, not verified on a GPU. Micro batch 8 needs more than 80 GB → use gradient accumulation to make a large batch)")


if __name__ == "__main__":
    main()
