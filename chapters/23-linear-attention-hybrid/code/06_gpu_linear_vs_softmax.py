"""Chapter 23 · GPU measurements: softmax attention vs the chunkwise and recurrent forms of linear attention

The script runs the functions of 03 and 04 of this chapter on the GPU without changes.
It calls them in `with torch.device("cuda")`, so the tensors that the functions create are also on the GPU.
It compares them with the FlashAttention of PyTorch:
  1. Numbers: chunkwise form == recurrent form for Gated DeltaNet (GPU, FP32).
  2. Full-sequence processing (training / prefill): softmax attention uses the FlashAttention kernel
     of SDPA (BF16). Linear attention uses linear_chunked from 04. Gated DeltaNet uses gated_delta_chunked
     from 03 (pure PyTorch, FP32, chunk length 64, a Python loop over the chunks).
     The token-by-token recurrent form runs only on short sequences. We compare time and peak GPU memory.
  3. One decode step: softmax attention reads all T K and V vectors. Gated DeltaNet first compresses
     T tokens into the state with the chunkwise form, then does one recurrent step.
     We show the size of the state and the time of the step.
The shapes are those of the Gated DeltaNet layer of Qwen3.5-0.8B: batch 1, 16 heads, d_k = d_v = 128.
Softmax attention also uses 16 × 128.
Without a GPU, skip this script. The chapter text shows the results of one run on an RTX 3090.
Run: uv run python chapters/23-linear-attention-hybrid/code/06_gpu_linear_vs_softmax.py
"""

from __future__ import annotations

import importlib.util
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

HERE = Path(__file__).resolve().parent
B, H, D = 1, 16, 128
C = 64  # chunk length of the chunkwise form


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def inputs(T: int, dtype=torch.float32):
    """q and k are L2-normalized (as from Section 3 of this chapter).
    g is the log decay; beta is the write strength."""
    q = F.normalize(torch.randn(B, H, T, D), dim=-1).to(dtype)
    k = F.normalize(torch.randn(B, H, T, D), dim=-1).to(dtype)
    v = torch.randn(B, H, T, D).to(dtype)
    g, beta = -torch.rand(B, H, T) * 0.05, torch.rand(B, H, T)
    return q, k, v, g, beta


def cuda_ms(fn, reps: int) -> float:
    fn()  # warmup
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


def peak_mib(fn) -> float:
    """Peak extra GPU memory of fn, on top of the inputs (MiB)."""
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def softmax_prefill(q, k, v):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def main() -> None:
    if not torch.cuda.is_available():
        print("This script needs a CUDA GPU. Without a GPU, skip it. The chapter text shows the results of one run on an RTX 3090.")
        sys.exit(0)
    lm = _load("hybrid_lm", HERE / "04_hybrid_lm.py")  # 04 also loads 03 (lm.delta)
    delta = lm.delta
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}, PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"Shape: batch {B}, {H} heads, d_k = d_v = {D}; linear attention FP32, chunk length {C}; softmax attention BF16")

    with torch.device("cuda"):
        print("\n1. Gated DeltaNet: chunkwise form == recurrent form (GPU, FP32, T = 1,024)")
        q, k, v, g, beta = inputs(1024)
        o1, s1 = delta.gated_delta_recurrent(q, k, v, g, beta)
        o2, s2 = delta.gated_delta_chunked(q, k, v, g, beta, C=C)
        print(f"   max|output diff| = {(o1 - o2).abs().max():.1e}   max|final state diff| = {(s1 - s2).abs().max():.1e}")

        print("\n2. Process all T tokens at once (ms, median; in parentheses: peak extra GPU memory on top of the inputs)")
        print("        T     softmax (FlashAttn)         linear, chunked"
              "            GDN, chunked   GDN, recurrent")
        for T in (1024, 4096, 16384, 65536):
            q, k, v, g, beta = inputs(T)
            fns = [partial(softmax_prefill, q.bfloat16(), k.bfloat16(), v.bfloat16()),
                   partial(lm.linear_chunked, q, k, v, C=C),
                   partial(delta.gated_delta_chunked, q, k, v, g, beta, C=C)]
            reps = 5 if T >= 16384 else 10
            cells = [f"{cuda_ms(fn, reps):9.1f} ({peak_mib(fn):5.0f} MiB)" for fn in fns]
            if T <= 4096:  # token-by-token loop: T steps, a few small kernels per step; too slow for longer sequences
                rec = partial(delta.gated_delta_recurrent, q, k, v, g, beta)
                cells.append(f"{cuda_ms(rec, 3):9.1f}")
            else:
                cells.append(f"{'—':>9}")
            print(f"   {T:6,d}   " + "   ".join(cells))

        print("\n3. One decode step: a context of T tokens, then 1 more token (ms, median of 50 runs)")
        print("          T           KV cache   attn step   GDN state                 GDN step")
        for T in (1024, 16384, 65536, 262144):
            kv_k = torch.randn(B, H, T, D, dtype=torch.bfloat16)  # softmax: K and V for T positions
            kv_v = torch.randn(B, H, T, D, dtype=torch.bfloat16)
            q1 = torch.randn(B, H, 1, D, dtype=torch.bfloat16)
            soft_step = partial(F.scaled_dot_product_attention, q1, kv_k, kv_v)
            _, S = delta.gated_delta_chunked(*inputs(T), C=C)  # prefill: compress T tokens into the state S
            gdn_step = partial(delta.gated_delta_recurrent, *inputs(1), S=S)  # one more step from S
            kv_mib = (kv_k.nbytes + kv_v.nbytes) / 2**20
            print(f"   {T:8,d}   {kv_mib:12,.0f} MiB   {cuda_ms(soft_step, 50):9.3f}"
                  f"   {tuple(S.shape)} {S.nbytes / 2**20:.0f} MiB   {cuda_ms(gdn_step, 50):7.3f}")


if __name__ == "__main__":
    main()
