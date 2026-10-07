"""Chapter 8 · GPU script 6: handwritten attention vs the three SDPA backends.
We compare speed, and even more, GPU memory.

The handwritten softmax(QKᵀ/√d) in 02 keeps the full (B, H, T, T) score matrix in GPU memory.
When T doubles, this memory becomes 4 times larger.
PyTorch's F.scaled_dot_product_attention (SDPA) has three backends on CUDA. Their math is the same:
  - math: the PyTorch reference implementation. Like the handwritten version, it stores T × T
    (and it first converts BF16 inputs to float32);
  - efficient: memory-efficient attention (the xFormers kernel). It works in blocks
    and does not store T × T;
  - flash: FlashAttention-2. Blocks + online softmax; it also does not store T × T
    (Chapter 14 explains how).

The script does three things:
  1. Parity check: do the four methods give the same output for the same q, k, v?
  2. Sweep the sequence length T = 512 … 32K. Record the time of each method (CUDA event timing, median)
     and the peak GPU memory.
  3. GQA (Chapter 10): with fewer K/V heads than query heads,
     which backends can run with enable_gqa=True?

The shapes are those of the main-line model's attention:
batch 1, 16 query heads, head_dim 128, BF16, causal mask.
Needs a CUDA GPU (at least 20 GB of GPU memory for the handwritten version at 16K).
It takes about 10–20 seconds on an RTX 3090.
Run: uv run python chapters/08-attention/code/06_gpu_sdpa_backends.py
"""

from __future__ import annotations

import importlib.util
import os
import statistics
import sys
import warnings
from pathlib import Path

# Let the GPU memory allocator grow segments as needed, with less fragmentation.
# Then "does it fit?" depends only on the memory that the computation really needs.
# This must be set before import torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("attn02", HERE / "02_attention_from_scratch.py")
attn02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(attn02)

B, H, D = 1, 16, 128  # main-line model: 16 query heads, head_dim 128
H_KV = 8  # GQA: 8 K/V heads
DTYPE = torch.bfloat16
LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
MiB = 2**20


def handwritten(q, k, v):
    """Call attention from 02 without changes.

    torch.device("cuda") also puts the causal mask that it creates on the GPU.
    """
    with torch.device(q.device):
        return attn02.attention(q, k, v, causal=True)[0]


def sdpa(backend: SDPBackend):
    def fn(q, k, v):
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    return fn


METHODS = {
    "handwritten(02)": handwritten,
    "SDPA math": sdpa(SDPBackend.MATH),
    "SDPA efficient": sdpa(SDPBackend.EFFICIENT_ATTENTION),
    "SDPA flash": sdpa(SDPBackend.FLASH_ATTENTION),
}


def qkv(T: int, kv_heads: int = H, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, H, T, D, device="cuda", dtype=DTYPE, generator=g)
    k = torch.randn(B, kv_heads, T, D, device="cuda", dtype=DTYPE, generator=g)
    v = torch.randn(B, kv_heads, T, D, device="cuda", dtype=DTYPE, generator=g)
    return q, k, v


@torch.no_grad()
def measure(fn, q, k, v, reps: int):
    """Return (median time in ms, peak extra GPU memory in MiB).

    Return None if the GPU memory is not sufficient.

    Peak extra GPU memory = the highest memory use during the call − the memory use before the call
    (q, k, v are not included; the output is included).
    """
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    try:
        out = fn(q, k, v)  # first call: measure the peak memory, and warm up
        torch.cuda.synchronize()
        peak = (torch.cuda.max_memory_allocated() - base) / MiB
        del out
        fn(q, k, v)  # warm up once more
        times = []
        for _ in range(reps):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn(q, k, v)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    return statistics.median(times), peak


def fmt_len(T: int) -> str:
    return f"{T // 1024}K" if T >= 1024 else str(T)


def main() -> None:
    if not torch.cuda.is_available():
        print("This script needs a CUDA GPU. Without a GPU, skip it: the chapter text shows the results of one run on an RTX 3090.")
        sys.exit(0)
    torch.manual_seed(0)
    free, total = (x / 2**30 for x in torch.cuda.mem_get_info())
    print(f"GPU: {torch.cuda.get_device_name(0)} ({total:.1f} GiB, {free:.1f} GiB free)"
          f"  torch {torch.__version__}  CUDA {torch.version.cuda}")
    print(f"Shapes: B={B}, H={H} query heads, head_dim={D}, {str(DTYPE).replace('torch.', '')}, causal mask\n")

    # ① Parity check: the float32 handwritten version is the reference
    q, k, v = qkv(1024)
    ref = handwritten(q.float(), k.float(), v.float())
    print("1. Parity check (T=1024, max difference from the float32 handwritten version; BF16 has only 8 mantissa bits, so about 1e-2 is normal)")
    with torch.no_grad():
        for name, fn in METHODS.items():
            print(f"   {name:<15} {float((fn(q, k, v).float() - ref).abs().max()):.1e}")

    # ② Sweep the sequence length
    results = {}
    for T in LENGTHS:
        q, k, v = qkv(T)
        reps = 20 if T <= 4096 else 5
        results[T] = {name: measure(fn, q, k, v, reps) for name, fn in METHODS.items()}
        del q, k, v

    names = list(METHODS)
    print("\n2a. Time (ms, median; OOM = not enough GPU memory)")
    print("   T     " + "".join(f"{n:>16}" for n in names))
    for T, row in results.items():
        cells = "".join(f"{'OOM' if r is None else f'{r[0]:.2f}':>16}" for r in row.values())
        print(f"   {fmt_len(T):<6}{cells}")

    flash = {T: row["SDPA flash"] for T, row in results.items() if row["SDPA flash"]}
    tflops = ", ".join(f"{fmt_len(T)} {2 * B * H * T * T * D / r[0] / 1e9:.1f}" for T, r in flash.items())
    print(f"   Effective compute of flash (causal attention is about 2·H·T²·d floating-point operations; TFLOPS): {tflops}")

    print(f"\n2b. Peak extra GPU memory (MiB; the second column is the size of the T×T score matrices of {H} heads in BF16)")
    print("   T     " + f"{'T×T scores':>14}" + "".join(f"{n:>16}" for n in names))
    for T, row in results.items():
        scores = B * H * T * T * 2 / MiB
        cells = "".join(f"{'OOM' if r is None else f'{r[1]:.0f}':>16}" for r in row.values())
        print(f"   {fmt_len(T):<6}{scores:>14.0f}{cells}")

    # ③ GQA: 16 query heads share 8 K/V groups
    T = 4096
    q, k, v = qkv(T, kv_heads=H_KV, seed=1)
    g = H // H_KV
    k_rep, v_rep = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)  # the manual copy from Chapter 10
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
    print(f"\n3. GQA: q has {H} heads, k and v have only {H_KV} heads (T={T}), enable_gqa=True")
    for name, be in [("math", SDPBackend.MATH), ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                     ("flash", SDPBackend.FLASH_ATTENTION)]:
        with warnings.catch_warnings(record=True) as caught:  # a rejected backend gives its reason in a warning
            warnings.simplefilter("always")
            try:
                with torch.no_grad(), sdpa_kernel(be):
                    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
                diff = float((out.float() - ref.float()).abs().max())
                print(f"   {name:<10} runs     max difference from 'copy K/V, then compute': {diff:.1e}")
            except RuntimeError as e:
                why = [str(w.message).split(". ")[0] for w in caught if "num_heads" in str(w.message)]
                print(f"   {name:<10} fails    {str(e).splitlines()[0]}")
                if why:
                    print(f"              reason: {why[0]}")
    if hasattr(torch, "_fused_sdp_choice"):  # private function; shows only the default backend choice
        for label, (kk, vv, gqa) in {"MHA": (k_rep, v_rep, False), "GQA": (k, v, True)}.items():
            choice = SDPBackend(torch._fused_sdp_choice(q, kk, vv, is_causal=True, enable_gqa=gqa))
            print(f"   No backend given: the default for {label} is {choice.name}")

    def flash_gqa(q, k, v):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

    def flash_copy(q, k, v):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            k2, v2 = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
            return F.scaled_dot_product_attention(q, k2, v2, is_causal=True)

    del k_rep, v_rep, ref
    t_gqa, m_gqa = measure(flash_gqa, q, k, v, 20)
    t_copy, m_copy = measure(flash_copy, q, k, v, 20)
    print(f"   flash + enable_gqa: {t_gqa:.2f} ms, peak extra GPU memory {m_gqa:.0f} MiB")
    print(f"   flash + copy K/V first: {t_copy:.2f} ms, peak extra GPU memory {m_copy:.0f} MiB")


if __name__ == "__main__":
    main()
