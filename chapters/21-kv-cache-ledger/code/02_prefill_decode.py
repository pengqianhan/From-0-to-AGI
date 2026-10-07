"""Chapter 21 · Minimal code 2: why long context is expensive.

Prefill compute grows as T². Decode is limited by memory bandwidth.
Two ledgers. Both use the real configuration of the main-line model (configs/main):

1. Forward floating-point operations of prefill (process the full prompt of T tokens at once):
       matmul part:    2 × N × T                (N = parameters in matmuls, lm_head included)
       attention part: 2 × layers × q_dim × T²  (two matmuls QKᵀ and AV; the causal mask keeps half)
   When T is long, the T² term becomes larger than the parameter term.

2. Decode (each step makes 1 token; the batch has B conversations, each with T tokens already):
       bytes to read   ≈ weights (read once in full, shared by all B) + B × KV cache (each its own)
       operations      ≈ B × (2N + 4 × layers × q_dim × T)
   The minimum time of one step = max(compute time, memory read time). When the arithmetic
   intensity (operations per byte) is far below the "ridge point" of the hardware, the GPU waits
   for data. This is memory-bound.

The hardware numbers are rough values from the public H100 SXM specification:
dense BF16 989.5 TFLOPS, HBM3 bandwidth 3.35 TB/s, 80 GB of memory.
These are theoretical lower bounds, not measurements. Real systems have more costs:
kernel efficiency, activations, fragmentation, and others.
Run: uv run python chapters/21-kv-cache-ledger/code/02_prefill_decode.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from zero.config import load_model_config  # noqa: E402
from zero.model import count_params  # noqa: E402

PEAK_FLOPS = 989.5e12  # H100 SXM dense BF16 (the same value as zero/tools/estimate_cost.py)
HBM_BW = 3.35e12  # bytes/s
HBM_BYTES = 80e9
BF16 = 2


def model_numbers():
    c = load_model_config(ROOT / "configs" / "main" / "pretrain.toml")
    p = count_params(c)
    n_matmul = c.n_layers * (p["attention_per_layer"] + p["ffn_per_layer"]) + c.dim * c.vocab_size
    kv_per_token = 2 * c.n_layers * c.n_kv_heads * c.head_dim * BF16
    return c, p["total"], n_matmul, kv_per_token


def prefill_flops(c, n_matmul: int, T: int) -> tuple[float, float]:
    linear = 2 * n_matmul * T
    attn = 2 * c.n_layers * c.q_dim * T * T  # 4·q_dim per pair (i, j); causal: only T²/2 pairs
    return linear, attn


def decode_step(c, n_params: int, n_matmul: int, kv_per_token: int, T: int, B: int) -> dict:
    flops = B * (2 * n_matmul + 4 * c.n_layers * c.q_dim * T)
    w_bytes = n_params * BF16
    kv = B * T * kv_per_token
    t_compute, t_memory = flops / PEAK_FLOPS, (w_bytes + kv) / HBM_BW
    t = max(t_compute, t_memory)
    return dict(flops=flops, weights=w_bytes, kv=kv, intensity=flops / (w_bytes + kv),
                t=t, bound="memory" if t_memory > t_compute else "compute", tok_s=B / t)


if __name__ == "__main__":
    c, n_params, n_matmul, kv_tok = model_numbers()
    print(f"Main-line model: {n_params / 1e6:.1f}M parameters in total, N = {n_matmul / 1e6:.1f}M "
          f"in matmuls, q_dim = {c.q_dim}, KV per token {kv_tok:,} bytes")

    print("\n1. Forward operations of prefill (T tokens in one pass)")
    print("         T         Matmul      Attention Attn share  H100 theory")
    for T in (1024, 4096, 32768, 131072):
        lin, att = prefill_flops(c, n_matmul, T)
        print(f"   {T:7,d}   {lin:12.3e}   {att:12.3e}   {att / (lin + att):8.1%}   "
              f"{(lin + att) / PEAK_FLOPS * 1e3:9.1f} ms")

    ridge = PEAK_FLOPS / HBM_BW
    print(f"\n2. One decode step (H100 ridge point: {ridge:.0f} operations/byte; "
          "a lower arithmetic intensity is memory-bound)")
    print("        T    B      Weights      KV cache Intensity     Bound       Step     token/s"
          "   Fits in 80 GB?")
    for T in (4096, 32768):
        for B in (1, 16, 64):
            d = decode_step(c, n_params, n_matmul, kv_tok, T, B)
            print(f"   {T:6,d} {B:4d}   {d['weights'] / 2**30:6.2f} GiB  {d['kv'] / 2**30:8.2f} GiB"
                  f"   {d['intensity']:7.1f}    {d['bound']}    {d['t'] * 1e3:6.2f} ms  "
                  f"{d['tok_s']:10,.0f}   {'yes' if d['weights'] + d['kv'] <= HBM_BYTES else 'no'}")

    print("\n3. How many conversations fit in memory? "
          "(80 GB minus the weights, all for the KV cache; no activations or fragmentation)")
    free = HBM_BYTES - n_params * BF16
    for T in (4096, 32768, 131072):
        for name, per in (("GQA main-line", kv_tok), ("if MHA", kv_tok * 2),
                          ("if MLA 512+64", c.n_layers * 576 * BF16)):
            print(f"   context {T:7,d}  {name:14} at most {int(free // (T * per)):5d} conversations")
