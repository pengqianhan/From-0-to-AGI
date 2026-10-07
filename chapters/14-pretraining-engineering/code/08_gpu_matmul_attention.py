"""Chapter 14 · Minimal code 8 (GPU measurement): matmul and attention on a real GPU. How fast are they, and how much memory do they use?

Sections 3 and 4 of the chapter give these results with the CPU:
  - BF16 lets the Tensor Cores calculate faster, but it has only 7 mantissa bits:
    a precision of about 2–3 significant digits;
  - naive attention writes the T×T matrices S and P to GPU memory. FlashAttention uses tiles + online
    softmax and never writes them out.
This script measures them on one CUDA GPU:
  ① square matmul C = A·B: the measured TFLOPS of FP32, TF32, BF16, and FP16 (CUDA event timing,
     median), and the relative error of each against the FP64 result;
  ② the same Linear and the same inputs as 02_precision.py ⑥, with BF16 autocast on the CPU and on the GPU;
  ③ attention forward + backward pass (1 sequence, 16 heads × head_dim 128, BF16, causal): naive code vs
     the FlashAttention backend of SDPA. How do the peak memory and the time grow with the sequence length T?
  ④ which kernel the attention of zero (GQA: 16 query heads, 8 KV heads, enable_gqa=True) really calls
     on the GPU, once with BF16 autocast and once with FP32.

Run: uv run python chapters/14-pretraining-engineering/code/08_gpu_matmul_attention.py   (needs a CUDA GPU; about 15 s on an RTX 3090)
"""

import contextlib
import dataclasses
import math
import re
import statistics
import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))  # so that `import zero` works from any folder of the repository
warnings.filterwarnings("ignore", message=".*no current CUDA context")    # a harmless message; hide it to keep the output clean
warnings.filterwarnings("ignore", message=".*Profiler clears events")

# Dense peak values from the spec sheets (TFLOPS). RTX 3090: NVIDIA Ampere GA102 whitepaper; Tensor Cores
# with FP32 accumulation. The spec peak of TF32 and of FP32 is 35.6 (on GeForce cards, TF32 is not faster
# than FP32). For other cards, the script does not use a table and shows only the measured values.
SPEC = {"3090": {"FP32": 35.6, "TF32": 35.6, "BF16": 71.0, "FP16": 71.0}}
FORMATS = [("FP32", torch.float32, False), ("TF32", torch.float32, True),
           ("BF16", torch.bfloat16, False), ("FP16", torch.float16, False)]
H, D = 16, 128  # main-line model: 16 query heads × head_dim 128


def cuda_time(fn, warmup: int = 3, reps: int = 10) -> float:
    """Time of one call of fn() in seconds: warm up first, then time reps calls with CUDA events and take the median."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    return statistics.median(times)


def matmul_tflops(n: int, dtype: torch.dtype, tf32: bool) -> float:
    torch.backends.cuda.matmul.allow_tf32 = tf32
    a = torch.randn(n, n, device="cuda").to(dtype)
    b = torch.randn(n, n, device="cuda").to(dtype)
    inner = max(1, int(1e11 // (2 * n**3)))  # each timing has at least about 10^11 operations, to make the timing error small

    def run():
        for _ in range(inner):
            a @ b

    t = cuda_time(run)
    torch.backends.cuda.matmul.allow_tf32 = False
    return 2 * n**3 * inner / t / 1e12


def naive_attention(q, k, v, mask):
    s = q @ k.transpose(-2, -1) / math.sqrt(D)       # (1, H, T, T): written to GPU memory in full
    s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ v               # P is also (1, H, T, T) and is saved for the backward pass


def flash_attention(q, k, v, mask):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):     # allow only the FlashAttention backend; if it cannot run, raise an error
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def attention_cost(fn, T: int) -> tuple[float | None, int | None]:
    """Time (s) of one forward + backward pass and the increase of peak memory (bytes, without q, k, v).

    Return None if there is not sufficient memory.
    """
    torch.cuda.empty_cache()
    q, k, v = (torch.randn(1, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
               for _ in range(3))
    grad_out = torch.randn_like(q)
    mask = torch.ones(T, T, dtype=torch.bool, device="cuda").triu(1)

    def step():
        fn(q, k, v, mask).backward(grad_out)
        q.grad = k.grad = v.grad = None

    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    try:
        t = cuda_time(step, warmup=2, reps=5)
    except torch.OutOfMemoryError:
        return None, None
    return t, torch.cuda.max_memory_allocated() - base


def attention_kernels(cfg, T: int, bf16: bool) -> tuple[list[str], bool]:
    """Run one layer of the zero Transformer (forward + backward pass).

    Return (the fused attention kernels that it calls, whether there is a separate softmax kernel).
    """
    from zero.model import Transformer

    model = Transformer(cfg).cuda()
    tokens = torch.randint(0, cfg.vocab_size, (1, T), device="cuda")

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else contextlib.nullcontext():
            logits = model(tokens)
        logits.float().square().mean().backward()    # do not use cross-entropy here: it has its own softmax, which would confuse the result

    step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()
    keys = ("flash", "fmha", "efficient", "attention", "cudnn")
    names = {e.name for e in prof.events() if e.device_type.name == "CUDA"}
    fused = {re.sub(r"<.*", "", n).removeprefix("void ") for n in names if any(k in n.lower() for k in keys)}
    return sorted(fused), any("softmax" in n.lower() for n in names)   # keep only the function name, without template arguments


def main():
    if not torch.cuda.is_available():
        print("This script needs a CUDA GPU. Without a GPU, skip it: the chapter shows one result from an RTX 3090.")
        sys.exit(0)
    torch.manual_seed(0)
    name = torch.cuda.get_device_name()
    spec = next((v for k, v in SPEC.items() if k in name), None)
    print(f"GPU: {name}, {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB; "
          f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")

    print("\n① Measured throughput of square matmul C = A·B (TFLOPS, CUDA event timing, median of 10)")
    sizes = (1024, 2048, 4096, 8192)
    print(f"  {'format':<6}" + "".join(f"{f'n={n}':>10}" for n in sizes) + f"{'spec peak':>10}{'best/peak':>10}")
    best = {}
    for fmt, dtype, tf32 in FORMATS:
        row = [matmul_tflops(n, dtype, tf32) for n in sizes]
        best[fmt] = max(row)
        peak = spec[fmt] if spec else None
        tail = f"{peak:>10.1f}{best[fmt] / peak:>10.0%}" if peak else f"{'—':>10}{'—':>10}"
        print(f"  {fmt:<6}" + "".join(f"{x:>10.1f}" for x in row) + tail)
    print(f"  BF16 is {best['BF16'] / best['FP32']:.1f} times faster than FP32 (fastest size of each)")

    n = 4096
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")
    ref = a.double() @ b.double()
    print(f"\n  The same pair of {n}×{n} random matrices. Relative error against the FP64 result, ‖C − C₆₄‖ / ‖C₆₄‖:")
    for fmt, dtype, tf32 in FORMATS:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        c = (a.to(dtype) @ b.to(dtype)).double()
        torch.backends.cuda.matmul.allow_tf32 = False
        print(f"    {fmt:<5} {((c - ref).norm() / ref.norm()).item():.1e}")

    print("\n② The same Linear and inputs as 02_precision.py ⑥: BF16 autocast on the CPU and on the GPU")
    torch.manual_seed(0)
    torch.randn(100_000)                         # before ⑥, 02_precision.py draws global random numbers only this one time
    lin = torch.nn.Linear(1024, 1024, bias=False)
    xin = torch.randn(64, 1024)
    ref_cpu = lin(xin)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out_cpu = lin(xin)
    lin_gpu, x_gpu = lin.cuda(), xin.cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_gpu = lin_gpu(x_gpu)
    def rel(out):
        return ((out.float().cpu() - ref_cpu).norm() / ref_cpu.norm()).item()

    same = (out_gpu.cpu() == out_cpu).float().mean().item()
    print(f"  Relative error against the FP32 result: CPU {rel(out_cpu):.2e}, GPU {rel(out_gpu):.2e}; "
          f"output dtype {out_gpu.dtype}; fraction of BF16 outputs that are the same on both: {same:.1%}")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False  # the default is True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_gpu2 = lin_gpu(x_gpu)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    print(f"  On the GPU, do not let cuBLAS use BF16 for intermediate reductions (allow_bf16_reduced_precision_reduction=False): "
          f"relative error {rel(out_gpu2):.2e}")

    print(f"\n③ Attention forward + backward pass (1 sequence, {H} heads × {D}, BF16, causal; median of 5)")
    print(f"  {'T':>6} {'S+P size':>9} {'naive peak':>13} {'Flash peak':>14} "
          f"{'naive time':>10} {'Flash time':>10} {'speedup':>8}")
    for T in (1024, 2048, 4096, 8192, 16384):
        sp = 2 * H * T * T * 2                                   # S and P, each H·T² BF16 values
        tn, mn = attention_cost(naive_attention, T)
        tf, mf = attention_cost(flash_attention, T)
        naive_mem = f"{mn / 2**20:>12,.0f}M" if mn is not None else f"{'out of memory':>9}"
        naive_t = f"{tn * 1e3:>8.2f}ms" if tn is not None else f"{'—':>10}"
        speed = f"{tn / tf:>7.1f}×" if tn is not None else f"{'—':>8}"
        print(f"  {T:>6} {sp / 2**20:>8,.0f}M {naive_mem} {mf / 2**20:>13,.0f}M "
              f"{naive_t} {tf * 1e3:>8.2f}ms {speed}")
    print("  (S+P size = theoretical size. Peak = the highest memory above q, k, v in one forward + backward pass; M = MiB)")

    print("\n④ Which kernel the attention of zero uses on the GPU (main-line shape, 1 layer: 16 query heads, 8 KV heads, enable_gqa=True)")
    from zero.config import load_model_config

    cfg = dataclasses.replace(load_model_config(ROOT / "configs/main/pretrain.toml"), n_layers=1)
    for label, bf16 in (("BF16 autocast", True), ("FP32", False)):
        fused, softmax = attention_kernels(cfg, 2048, bf16)
        print(f"  {label}: separate softmax kernel: {'yes' if softmax else 'no'}; fused attention kernels{':' if fused else ': none'}")
        for kname in fused:
            print(f"    {kname}")
    q = torch.randn(1, H, 256, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, H // 2, 256, D, device="cuda", dtype=torch.bfloat16)
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION):
        try:
            with warnings.catch_warnings(), sdpa_kernel(backend):
                warnings.simplefilter("ignore")        # if a backend is not supported, PyTorch first prints a list of reasons
                F.scaled_dot_product_attention(q, kv, kv, is_causal=True, enable_gqa=True)
            status = "OK"
        except RuntimeError:
            status = "not supported (error)"
        print(f"  Only {backend.name:<20} + enable_gqa=True: {status}")


if __name__ == "__main__":
    main()
