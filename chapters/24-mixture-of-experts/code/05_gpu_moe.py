"""Chapter 24 · GPU measurement: double the experts and keep the compute per token. Also: why a
Python loop over the experts is slow on a GPU.

The main code of the chapter runs on a CPU. This script moves the MoE layer of 02_moe_layer.py to one GPU
without changes (BF16, needs CUDA). It sends 8192 tokens at a time (like one batch in training or prefill),
with d = 1024, an expert width of 2048, and 2 experts for each token. The active width is 2 × 2048 = 4096,
so the compute is the same as one dense SwiGLU of width 4096. The number of experts E grows from 8 to 128:
the total parameters grow 16×, and the FLOPs per token do not change. The script compares the time of three versions:
  - a dense SwiGLU (width 4096, the baseline with the same active compute);
  - MoE.forward of 02: a Python loop over the experts. Each expert does one nonzero (it must wait for
    the GPU to finish before it knows its tokens) + three small matrix multiplications + index_add_;
  - sort into segments + F.grouped_mm: sort the tokens by expert. One grouped GEMM call for each matrix
    calculates all experts, and the full forward pass never waits for the GPU. (The MoEFFN in the production
    code zero/arch/moe.py uses the same idea: "stack the weights by expert + sort into segments".)

The router and the experts have random initialization, because we only measure time. For each E, a parity
check first compares the outputs of the two MoE versions to make sure that they calculate the same thing.
Run: uv run python chapters/24-mixture-of-experts/code/05_gpu_moe.py (about 15 seconds)
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
DEV = "cuda"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


moe_mod = _load("moe_ch24", HERE / "02_moe_layer.py")


def build(cls, *args, **kw):
    """Build the module directly on the GPU (this saves the time to initialize hundreds of millions of
    parameters on the CPU). Then convert it to BF16."""
    with torch.device(DEV):
        m = cls(*args, **kw)
    return m.to(torch.bfloat16).eval()


def stack_experts(m) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack the weights of the E experts into 3D tensors: (E, d, h), (E, d, h), (E, h, d).
    The production code stores the weights in this form."""
    wg = torch.stack([e.w_gate.weight.t() for e in m.experts]).contiguous()
    wu = torch.stack([e.w_up.weight.t() for e in m.experts]).contiguous()
    wd = torch.stack([e.w_down.weight.t() for e in m.experts]).contiguous()
    return wg, wu, wd


def grouped_moe(m, x: torch.Tensor, w) -> torch.Tensor:
    """The same routing and gates as MoE.forward in 02. The experts use sort into segments + grouped GEMM,
    so the code never waits for the GPU."""
    wg, wu, wd = w
    s = m.router(x).sigmoid() if m.score == "sigmoid" else m.router(x).softmax(-1)
    idx = (s + m.bias).topk(m.K, dim=-1).indices  # (T, K)
    g = s.gather(-1, idx)
    g = g / g.sum(-1, keepdim=True)
    flat = idx.flatten()  # (T·K,) the expert of each "assignment"
    order = flat.argsort(stable=True)  # sort by expert: the tokens of one expert form one segment
    offs = torch.bincount(flat, minlength=m.E).cumsum(0).to(torch.int32)  # the end of each segment
    tok = order // m.K  # the token that row i comes from, after the sort
    xs = x[tok]  # dispatch: (T·K, d)
    h = F.silu(F.grouped_mm(xs, wg, offs=offs)) * F.grouped_mm(xs, wu, offs=offs)
    y = F.grouped_mm(h, wd, offs=offs)  # each segment uses the weights of its own expert
    out = torch.zeros_like(x)
    out.index_add_(0, tok, y * g.flatten()[order, None])  # combine: add back to the original positions, times the gates
    return out


@torch.no_grad()
def gpu_ms(fn, reps: int = 30, warmup: int = 5) -> float:
    """Time with CUDA events. Warm up first, then take the median (milliseconds)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def launch_us(n: int = 2000) -> float:
    """Microseconds that Python needs on this machine to launch the smallest GPU operation
    (an addition of one element; the GPU itself needs almost no time)."""
    x = torch.zeros(1, device=DEV)
    for _ in range(100):
        x.add_(1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        x.add_(1)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


@torch.no_grad()
def kernels_per_grouped_mm(E: int = 128, d: int = 1024, h: int = 2048, T: int = 1024) -> dict:
    """Use the profiler to see which GPU kernels one F.grouped_mm call (E experts) launches on this card,
    and how many of each."""
    from torch.profiler import ProfilerActivity, profile

    xs = torch.randn(T, d, device=DEV, dtype=torch.bfloat16)
    wg = torch.randn(E, d, h, device=DEV, dtype=torch.bfloat16)
    offs = (torch.arange(1, E + 1, device=DEV) * (T // E)).to(torch.int32)
    F.grouped_mm(xs, wg, offs=offs)
    torch.cuda.synchronize()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            F.grouped_mm(xs, wg, offs=offs)
            torch.cuda.synchronize()
    counts: dict = {}
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            counts[ev.name] = counts.get(ev.name, 0) + 1
    return counts


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


@torch.no_grad()
def main(d: int = 1024, h: int = 2048, K: int = 2, T: int = 8192) -> None:
    print(
        f"{T} tokens at a time, d = {d}, expert width {h}, {K} experts for each token; "
        "BF16, CUDA event timing, median of 30 runs"
    )
    x = torch.randn(T, d, device=DEV, dtype=torch.bfloat16)
    dense = build(moe_mod.Expert, d, K * h)
    flops_tok = 2 * 3 * d * K * h  # per token: three matrices in each active expert, one multiply-add per parameter
    t_dense = gpu_ms(lambda dense=dense: dense(x))
    print(
        f"{'':13}{'params':>8}{'GFLOP/token':>15}{' tokens per expert':>14}{'loop ms':>14}"
        f"{'grouped ms':>12}{'loop/grp':>12}{'grouped TFLOPS':>16}"
    )
    print(
        f"{'dense ' + str(K * h):13}{n_params(dense) / 1e6:7.0f}M{flops_tok / 1e9:15.4f}"
        f"{'':>18}{'':>14}{t_dense:12.2f}{'':>13}{flops_tok * T / t_dense / 1e9:16.1f}"
    )
    del dense
    for E in (8, 16, 32, 64, 128):
        m = build(moe_mod.MoE, d, E, K, h, score="sigmoid")
        w = stack_experts(m)
        ref, out = m(x), grouped_moe(m, x, w)
        rel = float((ref - out).abs().max() / ref.abs().max())
        assert rel < 2e-2, f"the two versions do not match: relative difference {rel:.1e}"
        fl = flops_tok + 2 * d * E  # plus the router (small)
        t_loop = gpu_ms(lambda m=m: m(x))
        t_grp = gpu_ms(lambda m=m, w=w: grouped_moe(m, x, w))
        print(
            f"{'MoE E=' + str(E):13}{n_params(m) / 1e6:7.0f}M{fl / 1e9:15.4f}{T * K / E:18.0f}"
            f"{t_loop:14.2f}{t_grp:12.2f}{t_loop / t_grp:12.1f}×{fl * T / t_grp / 1e9:16.1f}"
        )
        del m, w, ref, out
        torch.cuda.empty_cache()
    print(
        "(Parity check first for each E: the max difference between the outputs of the two MoE versions "
        "is < 2% of the max output, the size of BF16 rounding errors.)"
    )
    print(f"On this machine, Python needs about {launch_us():.1f} µs to launch the smallest GPU operation")
    kern = kernels_per_grouped_mm()
    print(
        f"GPU kernels from one F.grouped_mm call (128 experts) on this card: {sum(kern.values())} in total; "
        f"the most frequent kind: {max(kern.values())} ({max(kern, key=kern.get)[:40]}…)"
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("This script needs a CUDA GPU. Without a GPU, skip it: the README shows one result from an RTX 3090.")
        sys.exit(0)
    t0 = time.perf_counter()
    torch.manual_seed(0)
    print(
        f"GPU: {torch.cuda.get_device_name(0)}, PyTorch {torch.__version__}, CUDA {torch.version.cuda}\n"
    )
    main()
    print(f"Total time {time.perf_counter() - t0:.0f} s")
