"""Chapter 15 · Minimal code 5 (GPU measurement): how much more does each token of a 32K sequence
cost? A measurement on one Block of the main-line model.

Section 4 of the chapter uses the formula of zero (PaLM Appendix B, no halving for the causal mask).
Each training token of the main-line model costs 6.96 GFLOP at 4K and 26.69 GFLOP at 32K: 3.84× more.
FlashAttention skips the upper triangle of the causal mask, so it does about half of the attention
operations. This script measures the real ratio on a GPU:

  Take one Block of the long-context configuration (configs/main/longctx.toml), as zero implements it:
  RMSNorm, GQA (16 query heads, 8 KV heads × 128), QK-Norm, RoPE (base 1M), SwiGLU (FFN 3584),
  BF16 autocast, SDPA attention. Each call gets the same number of tokens (T × micro batch = 32,768).
  T increases from 4096 to 32768:
  ① Time per token for forward + backward, as a ratio to T = 4096. Compare with two formulas
     (not halved / causal halved). Then calculate the compute utilization in two ways: with the
     operations that the GPU really does (causal halved), and with the not-halved formula of the
     MFU in the zero log.
  ② The memory that this layer keeps for the backward pass, in bytes per token. FlashAttention does
     not keep the T×T matrix, so this value must not depend on T.

The script measures only one layer. The full model has 28 layers (28×), plus an output layer that
does not depend on T. The input is random vectors.

Run: uv run python chapters/15-midtraining-long-context/code/05_gpu_long_context_cost.py   (needs a CUDA GPU; about 20 s on an RTX 3090)
"""

import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]   # repository root, to import zero
sys.path.insert(0, str(ROOT))

from zero.config import load_model_config  # noqa: E402
from zero.model import Block, RotaryEmbedding  # noqa: E402
from zero.tools.memory_calc import layer_activation_bytes_per_token, matmul_params  # noqa: E402

TOKENS = 32768            # tokens in each forward + backward (T × micro batch)
SEQ_LENS = [4096, 8192, 16384, 32768]
SPEC_BF16 = {"3090": 71.0}  # dense BF16 Tensor Core peak of the RTX 3090 data sheet (TFLOPS, FP32 accumulate)


def main():
    if not torch.cuda.is_available():
        print("This script needs a CUDA GPU. Without a GPU, skip it: the chapter shows a result from an RTX 3090.")
        sys.exit(0)
    name = torch.cuda.get_device_name()
    peak = next((v for k, v in SPEC_BF16.items() if k in name), None)
    print(f"GPU: {name}; PyTorch {torch.__version__}, CUDA {torch.version.cuda}")

    cfg = load_model_config(ROOT / "configs/main/longctx.toml")
    torch.manual_seed(0)
    with torch.device("cuda"):
        block = Block(cfg, layer_idx=0)
        rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta, cfg.rope_scaling)
    per_layer, _ = matmul_params(cfg)
    print(f"One Block: dim {cfg.dim}, {cfg.n_heads} query heads / {cfg.n_kv_heads} KV heads × {cfg.head_dim}, "
          f"FFN {cfg.ffn_dim}, RoPE base {cfg.rope_theta:g}; {TOKENS:,} tokens per call, BF16 autocast")

    def flops(T, causal_half):
        """Training operations per token for one layer: 6·N_layer + 12·q_dim·T (with causal halving,
        use half of the attention term)."""
        attn = 12 * cfg.q_dim * T
        return 6 * per_layer + (attn / 2 if causal_half else attn)

    def measure(T):
        """Return (seconds for each forward + backward, extra memory in bytes at the end of the forward pass)."""
        mb = TOKENS // T
        cos, sin = rope(0, T)
        x = torch.randn(mb, T, cfg.dim, device="cuda", requires_grad=True)
        grad = torch.randn(mb, T, cfg.dim, device="cuda")

        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = block(x, cos, sin)
            y.backward(grad)
            x.grad = None
            block.zero_grad(set_to_none=True)

        for _ in range(3):                        # warm up
            step()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = block(x, cos, sin)
        torch.cuda.synchronize()
        saved = torch.cuda.memory_allocated() - base   # includes the output of this layer (the input of the next layer)
        del y
        times = []
        for _ in range(10):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            step()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1e3)
        return statistics.median(times), saved

    rows = [(T, TOKENS // T, *measure(T)) for T in SEQ_LENS]

    def pct(tflops):
        return f"{tflops / peak:.0%}" if peak else "—"

    t0 = rows[0][2]
    print("\n① Forward + backward, 32,768 tokens per call (median of 10 runs)")
    print(f"  {'T':>6} {'micro batch':>11} {'time':>8} {'per tok':>8} {'measured':>7} "
          f"{' not halved':>9} {'causal halved':>10} {'real TFLOPS':>11} {'% peak':>6} {'MFU (zero rule)':>16}")
    for T, mb, sec, _ in rows:
        real = TOKENS * flops(T, True) / sec / 1e12       # operations the GPU really does (causal halved)
        booked = TOKENS * flops(T, False) / sec / 1e12    # the accounting rule of zero / PaLM (not halved)
        print(f"  {T:>6} {mb:>11} {sec * 1e3:>6.1f}ms {sec / TOKENS * 1e9:>6.0f}ns {sec / t0:>7.2f}× "
              f"{flops(T, False) / flops(SEQ_LENS[0], False):>10.2f}× "
              f"{flops(T, True) / flops(SEQ_LENS[0], True):>12.2f}× {real:>11.1f} {pct(real):>6} {pct(booked):>16}")
    print("  (real TFLOPS uses the operations done after causal halving; the MFU in the zero log uses the not-halved formula, so it is too high, more so for a longer T)")

    print("\n② Extra memory of this layer at the end of the forward pass (activations kept for backward + BF16 weight copies + output)")
    formula = layer_activation_bytes_per_token(cfg, "bf16")
    copies = 2 * per_layer                        # BF16 weight copies that autocast keeps for backward; they do not depend on the token count
    print(f"  memory_calc formula: {formula:,} bytes per token × {TOKENS:,} + weight copies {copies / 2**20:.0f} MiB"
          f" = {(formula * TOKENS + copies) / 2**20:,.0f} MiB")
    for T, _, _, saved in rows:
        print(f"  T = {T:>6}: measured {saved / 2**20:>6,.0f} MiB; without the weight copies, per token "
              f"{(saved - copies) / TOKENS:>8,.0f} bytes")


if __name__ == "__main__":
    main()
