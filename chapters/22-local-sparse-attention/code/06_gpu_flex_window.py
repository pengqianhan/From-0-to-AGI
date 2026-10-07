"""Chapter 22 · GPU measurements: a sliding window is faster only if the kernel really skips the blocks outside the window.

The same sliding-window attention (mask = (j ≤ i) & (i − j < W), sliding_mask in 01), with three
algorithms on the GPU:
  1. Dense causal SDPA (FlashAttention kernel): it also calculates everything outside the window.
     It is the baseline.
  2. Boolean-mask SDPA (memory-efficient kernel + attn_mask): it calculates the scores outside the window
     and then masks them. It must also build a T × T boolean mask first.
  3. Block-sparse sliding window of FlexAttention: create_block_mask divides the table into blocks of
     128 × 128 and marks the blocks that are fully outside the window. The kernel skips these blocks and
     calculates only the blocks on the band.
First, fix T and change the window W. Then fix W and change T. Look at how the time changes.
At the end, make sure that the three algorithms calculate the same result.
FlexAttention uses torch.compile to generate a Triton kernel. The first call must compile it
(a few seconds on this machine; later calls read the cache).
If you have no GPU, skip this script. The README shows one result on an RTX 3090.
Run: uv run python chapters/22-local-sparse-attention/code/06_gpu_flex_window.py
"""

from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

B, H, D = 1, 16, 128  # the number of query heads and the head_dim of the main-line model
WINDOW = None  # current window (an integer tensor on the GPU); mask_mod reads it, so a new window needs no new compilation


def sliding_mask_mod(b, h, q_idx, kv_idx):
    """The mask_mod for FlexAttention: the same condition as sliding_mask in 01."""
    return (kv_idx <= q_idx) & (q_idx - kv_idx < WINDOW)


def bool_mask(T: int, W: int) -> torch.Tensor:
    i = torch.arange(T, device="cuda")[:, None]
    j = torch.arange(T, device="cuda")[None, :]
    return (j <= i) & (i - j < W)


def cuda_ms(fn, reps: int = 10) -> float:
    for _ in range(2):  # warmup
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)[reps // 2]


def peak_extra_mib(fn) -> float:
    """The extra GPU memory (MiB) that fn uses while it runs, in addition to the tensors that already exist."""
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def main() -> None:
    global WINDOW
    if not torch.cuda.is_available():
        print("This script needs a CUDA GPU. If you have no GPU, skip it. The README shows one result on an RTX 3090.")
        sys.exit(0)
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}, PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"Shape: batch {B}, {H} heads, head_dim {D}, BF16; FlexAttention block size 128 × 128")
    flex = torch.compile(flex_attention)
    # Compile the block-mask builder too. It builds the mask block by block and does not put the full
    # T × T table in GPU memory (at T = 65,536, that table needs 32 GiB).
    make_block_mask = torch.compile(create_block_mask)
    WINDOW = torch.tensor(1, device="cuda")

    def run_all(T: int, W: int, with_mask: bool = True):
        q, k, v = (torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        WINDOW.fill_(W)
        bm = make_block_mask(sliding_mask_mod, None, None, T, T, device="cuda")
        n_blocks = (T // 128) * (T // 128 + 1) // 2  # the number of blocks in the causal lower triangle
        frac = (1 - bm.sparsity() / 100) * (T // 128) ** 2 / n_blocks  # blocks to calculate / all causal blocks

        def dense():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        def flex_run():
            return flex(q, k, v, block_mask=bm)

        r = dict(frac=frac, dense=cuda_ms(dense), flex=cuda_ms(flex_run))
        if with_mask:
            m = bool_mask(T, W)  # one T × T boolean table; all layers of a forward pass can share it, so we do not time it
            r["mask_mib"] = m.nbytes / 2**20

            def masked():
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    return F.scaled_dot_product_attention(q, k, v, attn_mask=m)

            r["masked"] = cuda_ms(masked)
            r["masked_mem"] = peak_extra_mib(masked)
            ref = masked()
            r["diff_flex"] = (flex_run() - ref).abs().max().item()
            if W >= T:
                r["diff_dense"] = (dense() - ref).abs().max().item()
        r["flex_mem"] = peak_extra_mib(flex_run)
        r["dense_mem"] = peak_extra_mib(dense)
        return r

    t0 = time.time()
    run_all(1024, 256, with_mask=False)  # start one compilation
    print(f"(The first compilation of FlexAttention took {time.time() - t0:.0f} s)")

    T = 16384
    print(f"\n1. Fix T = {T:,} and change the window W (milliseconds, median of 10 runs)")
    print("        W     % blocks   dense SDPA    masked SDPA  FlexAttention     speedup")
    diffs = []
    for W in (128, 512, 1024, 4096, T):
        r = run_all(T, W)
        diffs.append((W, r))
        print(f"   {W:6,d}   {r['frac']:10.1%}   {r['dense']:10.2f}   {r['masked']:12.2f}"
              f"   {r['flex']:12.2f}   {r['dense'] / r['flex']:8.1f}×")
    r = diffs[2][1]
    print(f"   Extra GPU memory at run time, in addition to q, k, v (W = 1,024): dense {r['dense_mem']:.0f} MiB; "
          f"boolean-mask SDPA first needs a mask of {r['mask_mib']:.0f} MiB, then {r['masked_mem']:.0f} MiB more at run time; "
          f"flex {r['flex_mem']:.0f} MiB")

    W = 1024
    print(f"\n2. Fix W = {W:,} and change the sequence length T (milliseconds)")
    print("        T   dense SDPA    masked SDPA  FlexAttention     speedup")
    for T in (4096, 16384, 65536):
        r = run_all(T, W, with_mask=T <= 16384)
        masked = f"{r['masked']:12.2f}" if "masked" in r else f"{'   (skipped)':>10}"
        print(f"   {T:6,d}   {r['dense']:10.2f}   {masked}   {r['flex']:12.2f}"
              f"   {r['dense'] / r['flex']:8.1f}×")
    print("   (At T = 65,536, the boolean mask alone needs 4 GiB, and the conversion to an additive bias doubles it. We skip it.)")

    print("\n3. The three algorithms calculate the same result (maximum absolute difference from boolean-mask SDPA, BF16)")
    for W, r in diffs:
        extra = f"; at W ≥ T, dense causal SDPA {r['diff_dense']:.1e}" if "diff_dense" in r else ""
        print(f"   T = 16,384, W = {W:>6,d}: FlexAttention {r['diff_flex']:.1e}{extra}")


if __name__ == "__main__":
    main()
