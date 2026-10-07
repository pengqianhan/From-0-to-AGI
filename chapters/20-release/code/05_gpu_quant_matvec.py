"""Chapter 20 · GPU measurement: with half the weight bytes, is decode almost 2× faster?

Section 3.1 says: in token-by-token generation (decode), each step reads all weights from memory once.
The bottleneck is bandwidth. Thus, with half the weight bytes, the read is almost 2× faster.
This script measures this on one GPU (it needs CUDA).

We build one decode step from all matrices of the main-line model (configs/main/pretrain.toml:
28 layers, width 1280, FFN 3584, 16 query heads / 8 K/V heads, output layer 65,536 × 1280 with a shared
embedding): batch 1, the vector of one token goes through 28 × 7 matrices and the output layer.
We keep only the matrix multiplications (attention, KV cache, and norm read few bytes; Chapter 21
calculates them separately). The weights are random (N(0, 0.02²), the same as the initialization).
We measure only the speed here; the error column is only for comparison. Formats and kernels:

  BF16, cuBLAS              the default path of F.linear                                   16 bits/weight
  BF16, torch.compile       the same matrix–vector product written as "multiply elementwise, then sum",
                            so that torch.compile makes one kernel
  INT8, 1 scale per row     the same code, but the weights are int8: the compiled kernel reads int8,
                            converts to BF16 on the chip, multiplies and adds, then multiplies by
                            the scale (dequantization is fused into the matrix multiplication)  ≈ 8 bits
  INT4, groups of 32        _weight_int4pack_mm (the tinygemm kernel that torchao / gpt-fast use):
                            each group has one bf16 scale + one bf16 zero, ŵ = (q − 8)·scale + zero  5 bits
  INT8, dequantize first    w.to(bf16) * scale first writes a full BF16 copy of the weights,
                            then F.linear (not fused)

One step has almost 200 small kernels. The cost of launching each kernel from Python (a few
microseconds each) is larger than the time to read the GPU memory. Thus, we record the full step with
a CUDA Graph and submit it in one call (inference engines do the same). Then we look at how the
comparison of BF16 and INT4 changes when the batch becomes larger (B sequences at the same time).
At the end, we look at matrices of different sizes and the bandwidth that each one reaches.
If you do not have a GPU, skip this script; the chapter text shows the results of one run on an RTX 3090.
Run: uv run python chapters/20-release/code/05_gpu_quant_matvec.py
"""

from __future__ import annotations

import statistics
import sys
import time
import tomllib
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
CFG = ROOT / "configs" / "main" / "pretrain.toml"
HBM_BW = 936e9  # RTX 3090 memory bandwidth (data sheet), bytes/s
GROUP = 32  # INT4: one group for each 32 numbers


def shapes(m: dict) -> list[tuple[str, int, int]]:
    """All matrices of one decode step: (name, output dim N, input dim K). Each is N × K, the same as the weight of nn.Linear."""
    d, hd, f = m["dim"], m["head_dim"], m["ffn_dim"]
    q, kv = m["n_heads"] * hd, m["n_kv_heads"] * hd
    layer = [("wq", q, d), ("wk", kv, d), ("wv", kv, d), ("wo", d, q),
             ("w_gate", f, d), ("w_up", f, d), ("w_down", d, f)]
    return layer * m["n_layers"] + [("lm_head", m["vocab_size"], d)]


# ── Quantization: INT8 with 1 scale per row; INT4 in groups of 32 in the tinygemm format ──
def int8_rowwise(w: torch.Tensor):
    scale = w.float().abs().amax(1) / 127                                   # scale = max|w| / 127
    q = torch.round(w.float() / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.bfloat16)


def int4_groupwise(w: torch.Tensor):
    """One group for each GROUP numbers: q = round((w − min) / scale) ∈ [0, 15], ŵ = (q − 8)·scale + zero, zero = min + 8·scale."""
    N, K = w.shape
    g = w.float().view(N, K // GROUP, GROUP)
    mn, mx = g.amin(-1), g.amax(-1)
    scale = ((mx - mn) / 15).clamp(min=1e-8)
    q = torch.round((g - mn[..., None]) / scale[..., None]).clamp(0, 15).to(torch.int32)
    zero = mn + 8 * scale
    deq = ((q - 8) * scale[..., None] + zero[..., None]).view(N, K)         # reference: the dequantized weights
    q = q.view(N, K)
    packed = torch.ops.aten._convert_weight_to_int4pack((q[:, ::2] << 4 | q[:, 1::2]).to(torch.uint8), 8)
    sz = torch.stack([scale, zero], -1).transpose(0, 1).contiguous().to(torch.bfloat16)  # (K/32, N, 2)
    return packed, sz, deq


# ── Batch-1 matrix–vector product as "multiply elementwise, then sum over K"; torch.compile makes one kernel ──
@torch.compile(dynamic=False)
def matvec_bf16(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:  # x (1, K), w (N, K) → (N,)
    return (x * w).sum(-1)


@torch.compile(dynamic=False)
def matvec_int8(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return (x * q.to(x.dtype)).sum(-1) * s  # read int8 → convert to BF16 on the chip → multiply-add → multiply by scale, in one kernel


# ── Timing: record the full step with a CUDA Graph, replay it many times, take the median ──
def graph_ms(fn, budget_s: float = 0.3, rounds: int = 5) -> float:
    """Record fn as a CUDA Graph. In each of `rounds` rounds, replay it several times and take the mean.
    Return the median milliseconds per step. The number of replays in a round depends on the time of
    one step, so that each round takes about budget_s seconds (slow formats replay fewer times, so the
    total time stays under control)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # before recording, run twice on a side stream (warm up, compile, let cuBLAS select its algorithm)
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()  # warm up once
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    graph.replay()
    b.record()
    torch.cuda.synchronize()
    reps = max(2, min(50, int(budget_s * 1e3 / a.elapsed_time(b))))
    times = []
    for _ in range(rounds):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            graph.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) / reps)
    del graph
    return statistics.median(times)


def copy_bandwidth(dev: str) -> float:
    """Copy a 1 GiB BF16 tensor: read 1 GiB + write 1 GiB. Return bytes/s."""
    src = torch.empty(2**29, device=dev, dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    return 2 * src.numel() * 2 / (graph_ms(lambda: dst.copy_(src)) / 1e3)


def main() -> None:
    if not torch.cuda.is_available():
        print("This script needs a CUDA GPU. Without a GPU, skip it; the chapter text shows the results of one run on an RTX 3090.")
        sys.exit(0)
    T0 = time.time()
    torch.manual_seed(0)
    dev = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}, PyTorch {torch.__version__}, CUDA {torch.version.cuda}")

    m = tomllib.loads(CFG.read_text(encoding="utf-8"))["model"]
    mats = shapes(m)
    n_params = sum(n * k for _, n, k in mats)
    n_rows = sum(n for _, n, _ in mats)
    W, Q8, Q4 = [], [], []
    err = {"int8": [], "int4": []}
    xs = {k: torch.randn(1, k, device=dev, dtype=torch.bfloat16) for k in {k for _, _, k in mats}}
    for _, n, k in mats:
        w = (torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)
        q8, s8 = int8_rowwise(w)
        p4, sz, deq4 = int4_groupwise(w)
        x = xs[k]
        ref = x.float() @ w.float().T                                        # exact output of the BF16 weights
        y8 = (x.float() @ q8.float().T) * s8.float()
        y4 = torch.ops.aten._weight_int4pack_mm(x, p4, GROUP, sz).float()
        y4_ref = x.float() @ deq4.T                                          # calculate by hand with the dequantized weights
        assert (y4 - y4_ref).norm() / y4_ref.norm() < 1e-2, "INT4 packing format is wrong"
        err["int8"].append(float((y8 - ref).norm() / ref.norm()))
        err["int4"].append(float((y4 - ref).norm() / ref.norm()))
        W.append(w)
        Q8.append((q8, s8))
        Q4.append((p4, sz))
    print(f"All matrices of the main-line model: 28 layers × 7 + output layer = {len(mats)}, {n_params / 1e6:.1f}M parameters"
          " (without the 1D norm weights)")

    def run(fmt: str, B: int = 1, sel: list[int] | None = None):
        """Record "multiply by the matrices in sel" (by default all 197) as one step. Return milliseconds."""
        x = {k: torch.randn(B, k, device=dev, dtype=torch.bfloat16) for k in xs}
        idx = range(len(mats)) if sel is None else sel

        def step():
            for i in idx:
                (_, _, k), w, (q8, s8), (p4, sz) = mats[i], W[i], Q8[i], Q4[i]
                if fmt == "bf16":
                    F.linear(x[k], w)
                elif fmt == "bf16c":
                    matvec_bf16(x[k], w)
                elif fmt == "int8c":
                    matvec_int8(x[k], q8, s8)
                elif fmt == "int4":
                    torch.ops.aten._weight_int4pack_mm(x[k], p4, GROUP, sz)
                else:  # first dequantize to a full BF16 copy of the weights, then multiply
                    F.linear(x[k], q8.to(torch.bfloat16) * s8[:, None])

        return graph_ms(step)

    b8 = n_params + 2 * n_rows  # int8 weights + one bf16 scale per row
    b4 = n_params // 2 + n_params // GROUP * 4  # 4-bit weights + two bf16 per group
    rows = [  # (name, format, weight bytes read per step, error)
        ("BF16, cuBLAS (F.linear)", "bf16", 2 * n_params, None),
        ("BF16, torch.compile kernel", "bf16c", 2 * n_params, None),
        ("INT8, compiled fused dequant", "int8c", b8, "int8"),
        ("INT4, tinygemm kernel", "int4", b4, "int4"),
        ("INT8, dequant to BF16 first", "naive", b8, "int8"),
    ]
    copy_bw = copy_bandwidth(dev)
    print(f"Reference: a copy of a 1 GiB tensor on this card (read 1 GiB + write 1 GiB) measured {copy_bw / 1e9:.0f} GB/s"
          f" ({100 * copy_bw / HBM_BW:.0f}% of the 936 GB/s in the data sheet). This is a reference for the highest bandwidth in practice")
    print("\n① One decode step with batch 1 (matrix multiplications only):")
    print(f"   {'Format and kernel':28s} {'bits/w':>8s} {'W bytes':>9s} {'Step time':>9s} {'Eff. BW':>10s} "
          f"{'% of peak':>9s} {'Speedup':>9s} {'Output err':>10s}")
    base = None
    for name, fmt, nbytes, e in rows:
        ms = run(fmt)
        base = base or ms
        bw = nbytes / (ms / 1e3)
        es = "—" if e is None else f"{100 * statistics.mean(err[e]):.2f}%"
        print(f"   {name:28s} {8 * nbytes / n_params:8.2f} {nbytes / 1e9:7.3f}GB {ms:7.3f}ms "
              f"{bw / 1e9:6.0f}GB/s {100 * bw / HBM_BW:8.0f}% {base / ms:8.2f}× {es:>10s}")
    print(f"   (Theoretical lower bound = weight bytes ÷ 936 GB/s: BF16 {2 * n_params / HBM_BW * 1e3:.3f} ms, "
          f"INT8 {b8 / HBM_BW * 1e3:.3f} ms, INT4 {b4 / HBM_BW * 1e3:.3f} ms. "
          "Effective bandwidth = weight bytes ÷ time. The last row also writes and reads one extra BF16 copy of the weights.)")

    print("\n② Generate B sequences at the same time (batch B), time for one step (ms):")
    print(f"   {'B':>4s} {'BF16 cuBLAS':>12s} {'INT4 tinygemm':>14s} {'INT4 speedup':>14s}")
    for B in (1, 8, 32, 128, 512):
        t16, t4 = run("bf16", B), run("int4", B)
        print(f"   {B:4d} {t16:12.3f} {t4:14.3f} {t16 / t4:13.2f}×")
    print("\n③ Why the time is far from the lower bound: one matrix type at a time (one product in each of the 28 layers, mean), effective bandwidth for batch 1 (GB/s)")
    print(f"   {'Matrix':8s} {'Shape N×K':>12s} {'BF16 size':>10s} {'BF16 cuBLAS':>12s} {'INT8 comp.':>10s} "
          f"{'INT4 tinygemm':>14s}")
    for nm in ("wk", "w_up", "lm_head"):
        sel = [i for i, (name, _, _) in enumerate(mats) if name == nm]
        _, n, k = mats[sel[0]]
        gbs = []
        for fmt, nbytes in (("bf16", 2 * n * k), ("int8c", n * k + 2 * n), ("int4", n * k // 2 + n * k // GROUP * 4)):
            ms = run(fmt, sel=sel) / len(sel)
            gbs.append(nbytes / (ms / 1e3) / 1e9)
        print(f"   {nm:8s} {f'{n}×{k}':>12s} {2 * n * k / 2**20:8.1f}MiB {gbs[0]:12.0f} {gbs[1]:10.0f} "
              f"{gbs[2]:14.0f}")

    print(f"\nPeak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
          f"total time {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
