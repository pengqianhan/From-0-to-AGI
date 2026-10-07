"""Chapter 21 · GPU measurement: one decode step ≈ (weights + KV cache) ÷ memory bandwidth.

This script moves the ledger of Sections 1.2 and 3 onto a real GPU (CUDA is necessary):
  1. Build a 28-layer decode step (BF16) with random weights in the shape of the main-line model
     (configs/main). Fill the KV cache with T positions first. Then measure the time to generate
     1 more token. The theoretical value comes from decode_step in 02_prefill_decode.py.
     Only the hardware changes, to the RTX 3090 specification:
         one step ≥ max(operations ÷ peak, (weights + B × T × KV per token) ÷ memory bandwidth)
  2. At the same context, change to MHA / GQA / MQA / MLA (absorbed path, 512 + 64).
     Measure how much GPU memory the KV cache uses and how long one decode step takes.

One decode step has hundreds of small kernels. The overhead to launch each one from Python
(some microseconds each) is larger than the memory read time. Thus we use a CUDA Graph:
it records the full step and submits it at once. Inference engines such as vLLM do the same.
Standard attention reads the KV cache with PyTorch SDPA (the FlashAttention kernel). MLA has no
ready-made kernel, so we write the absorbed path with a few matrix multiplications.
We omit element-wise operations such as RoPE and QK-Norm. They read almost no memory,
so they do not change the ledger.
If you have no GPU, skip this script. The README shows one result from an RTX 3090.
Run: uv run python chapters/21-kv-cache-ledger/code/05_gpu_decode_ledger.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PEAK_FLOPS = 71e12  # RTX 3090 dense BF16 tensor-core peak (spec sheet)
HBM_BW = 936e9  # RTX 3090 memory bandwidth, bytes/s (spec sheet)
BF16 = 2


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build(c, kind: str, B: int, T: int, dev: str = "cuda") -> dict:
    """Random weights + a cache filled with T positions. kind: MHA / GQA / MQA / MLA."""
    D, H, hd, L = c.dim, c.n_heads, c.head_dim, c.n_layers
    r, dr = 512, 64  # MLA: kv_lora_rank 512 + qk_rope_head_dim 64 (the same shape as DeepSeek-V3)

    def w(*shape):
        return torch.randn(*shape, device=dev, dtype=torch.bfloat16) * 0.02

    m = dict(kind=kind, B=B, T=T, H=H, hd=hd, D=D, layers=[])
    m["lm_head"] = w(c.vocab_size, D)  # shared with the embedding; decode reads all of it once
    before = torch.cuda.memory_allocated()
    if kind == "MLA":
        m["cache"] = torch.randn(L, B, T, r + dr, device=dev, dtype=torch.bfloat16)  # [c_KV ; k_R]
    else:
        m["n_kv"] = {"MHA": H, "GQA": c.n_kv_heads, "MQA": 1}[kind]
        m["cache"] = torch.randn(L, 2, B, m["n_kv"], T, hd, device=dev, dtype=torch.bfloat16)
    m["cache_alloc"] = torch.cuda.memory_allocated() - before  # bytes that the allocator actually gives
    for _ in range(L):
        lay = dict(n1=torch.ones(D, device=dev, dtype=torch.bfloat16),
                   n2=torch.ones(D, device=dev, dtype=torch.bfloat16),
                   w_gu=w(D, 2 * c.ffn_dim), w_down=w(c.ffn_dim, D), wo=w(H * hd, D))
        if kind == "MLA":  # q per head: 128 (no position) + 64 (RoPE); W_UK and W_UV are each (H, 128, 512)
            lay.update(wq=w(D, H * (hd + dr)), wkv_a=w(D, r + dr), w_uk=w(H, hd, r),
                       w_uv=w(H, r, hd))
        else:
            lay.update(wqkv=w(D, (H + 2 * m["n_kv"]) * hd))
        m["layers"].append(lay)
    m["weight_bytes"] = sum(t.numel() * BF16 for lay in m["layers"] for k, t in lay.items()
                            if k.startswith("w")) + m["lm_head"].numel() * BF16
    return m


def attend_gqa(m, lay, h, cache_l):
    """Decode for standard attention.

    Write the new K/V into the last slot, then read all T keys and values with SDPA (FlashAttention).
    """
    B, H, hd, n_kv = m["B"], m["H"], m["hd"], m["n_kv"]
    q, k, v = (h @ lay["wqkv"]).split([H * hd, n_kv * hd, n_kv * hd], dim=-1)
    cache_l[0, :, :, -1] = k.view(B, n_kv, hd)
    cache_l[1, :, :, -1] = v.view(B, n_kv, hd)
    o = F.scaled_dot_product_attention(q.view(B, H, 1, hd), cache_l[0], cache_l[1],
                                       enable_gqa=n_kv < H)  # query heads that share one K/V group do not copy K/V
    return o.reshape(B, H * hd)


def attend_mla(m, lay, h, cache_l, n_split: int = 32):
    """MLA absorbed path (the same formula as 03_mla.py).

    Project the query into the latent space. Then take the dot product with the cached
    576-dim latent vectors directly.
    """
    B, H, hd, T = m["B"], m["H"], m["hd"], m["T"]
    q = (h @ lay["wq"]).view(B, H, hd + 64)
    q_nope, q_pe = q.split([hd, 64], dim=-1)
    cache_l[:, -1] = h @ lay["wkv_a"]  # write [c_KV ; k_R] of the new position into the last slot
    q_lat = (q_nope.transpose(0, 1) @ lay["w_uk"]).transpose(0, 1)  # (B, H, 512): absorb W_UK into the query
    q_full = torch.cat([q_lat, q_pe], dim=-1)  # (B, H, 576)
    s = (q_full @ cache_l.transpose(1, 2)).float() / (hd + 64) ** 0.5  # (B, H, T): all heads read the same cache
    p = s.softmax(-1).to(torch.bfloat16)
    # Weighted average in the latent space. Split T into n_split parts, calculate each part,
    # then add them (split-KV, as in flash-decoding). Without the split, one sum over length T
    # goes to only a few SMs, and the memory read stays slow.
    p = p.view(B, H, n_split, T // n_split).transpose(1, 2)  # (B, n_split, H, T/n_split)
    c = cache_l[..., :512].view(B, n_split, T // n_split, 512)
    o_lat = (p @ c).sum(1)  # (B, H, 512)
    o = (o_lat.transpose(0, 1) @ lay["w_uv"]).transpose(0, 1)  # (B, H, 128): project back with W_UV
    return o.reshape(B, H * hd)


def decode_step(m, x):
    D = m["D"]
    attend = attend_mla if m["kind"] == "MLA" else attend_gqa
    for li, lay in enumerate(m["layers"]):
        h = F.rms_norm(x, (D,), lay["n1"])
        x = x + attend(m, lay, h, m["cache"][li]) @ lay["wo"]
        h = F.rms_norm(x, (D,), lay["n2"])
        gate, up = (h @ lay["w_gu"]).chunk(2, dim=-1)
        x = x + (F.silu(gate) * up) @ lay["w_down"]
    return x @ m["lm_head"].T  # (B, vocab)


def cuda_ms(run, reps: int = 30) -> float:
    """Time with CUDA events. Return the median (ms)."""
    for _ in range(3):  # warmup
        run()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        run()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)[reps // 2]


def time_step(m, graph: bool = True) -> float:
    """The median time of one decode step (ms)."""
    x = torch.randn(m["B"], m["D"], device="cuda", dtype=torch.bfloat16)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # warmup (CUDA Graph needs some runs on a side stream first)
        for _ in range(3):
            decode_step(m, x)
    torch.cuda.current_stream().wait_stream(side)
    if not graph:
        return cuda_ms(lambda: decode_step(m, x))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        decode_step(m, x)
    return cuda_ms(g.replay)


def free(m) -> None:
    m.clear()
    torch.cuda.empty_cache()


def main() -> None:
    if not torch.cuda.is_available():
        print("This script needs a CUDA GPU. If you have no GPU, skip it. "
              "The README shows one result from an RTX 3090.")
        sys.exit(0)
    torch.manual_seed(0)
    pd = _load("prefill_decode", HERE / "02_prefill_decode.py")
    pd.PEAK_FLOPS, pd.HBM_BW = PEAK_FLOPS, HBM_BW  # the ledger formula stays the same; only the hardware changes
    c, n_params, n_matmul, kv_tok = pd.model_numbers()
    print(f"GPU: {torch.cuda.get_device_name(0)}, PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"Hardware specification: BF16 peak {PEAK_FLOPS / 1e12:.0f} TFLOPS, "
          f"memory bandwidth {HBM_BW / 1e9:.0f} GB/s "
          f"(ridge point {PEAK_FLOPS / HBM_BW:.0f} operations/byte)")

    buf = torch.empty(2**29, device="cuda", dtype=torch.bfloat16)  # 1 GiB
    read_bw = buf.nbytes / (cuda_ms(buf.sum, 10) / 1e3)
    print(f"Reference: measured bandwidth to read 1 GiB once (a sum) on the same card: "
          f"{read_bw / 1e9:.0f} GB/s")
    del buf

    print("\n1. Main-line model (GQA 16/8, 28 layers, BF16), one decode step: "
          "measured vs ledger (CUDA Graph, median of 30 runs)")
    print("         T    B     Weights    KV cache      Theory   Measured    Ratio    Bandwidth"
          "    token/s")
    rows = [(1024, 1), (4096, 1), (4096, 16), (32768, 1), (32768, 4), (131072, 1)]
    got = {}
    for T, B in rows:
        m = build(c, "GQA", B, T)
        d = pd.decode_step(c, n_params, n_matmul, kv_tok, T, B)
        ms = time_step(m)
        got[T, B] = (ms, m["cache"].nbytes)
        bw = (m["weight_bytes"] + m["cache"].nbytes) / (ms / 1e3)
        print(f"   {T:7,d} {B:4d}   {d['weights'] / 2**30:5.2f} GiB  {d['kv'] / 2**30:6.2f} GiB"
              f"   {d['t'] * 1e3:6.2f} ms  {ms:6.2f} ms   {ms / (d['t'] * 1e3):5.2f}×"
              f"   {bw / 1e9:5.0f} GB/s  {B / ms * 1e3:9,.0f}")
        if (T, B) == (1024, 1):
            eager = time_step(m, graph=False)
            print(f"           (the same step without CUDA Graph, one kernel launch at a time: "
                  f"{eager:.2f} ms)")
        free(m)
    (t0, b0), (t1, b1) = got[1024, 1], got[131072, 1]
    print(f"   batch 1 from T = 1,024 to 131,072: the KV cache reads {(b1 - b0) / 2**30:.2f} GiB more, "
          f"one step takes {t1 - t0:.2f} ms more, "
          f"which is {(b1 - b0) / ((t1 - t0) / 1e3) / 1e9:.0f} GB/s")

    T, B = 32768, 1
    print(f"\n2. The same context T = {T:,}, batch {B}: KV cache and one decode step "
          "for each kind of attention")
    print("   Kind   Per layer/pos   Ledger KV   Allocated    Weights     Theory   Measured"
          "   Bandwidth")
    for kind in ("MHA", "GQA", "MQA", "MLA"):
        m = build(c, kind, B, T)
        per = 576 if kind == "MLA" else 2 * m["n_kv"] * c.head_dim
        ledger = per * c.n_layers * T * B * BF16
        # Bytes actually read: standard attention reads K and V once each. MLA reads 576 dims
        # for the scores, then reads the first 512 dims again for the weighted average.
        read_kv = ledger if kind != "MLA" else (576 + 512) * c.n_layers * T * B * BF16
        t_theory = (m["weight_bytes"] + read_kv) / HBM_BW * 1e3
        ms = time_step(m)
        print(f"   {kind:4}  {per:6,d} numbers  {ledger / 2**30:6.2f} GiB  {m['cache_alloc'] / 2**30:6.2f} GiB"
              f"  {m['weight_bytes'] / 2**30:5.2f} GiB  {t_theory:6.2f} ms  {ms:6.2f} ms"
              f"  {(m['weight_bytes'] + read_kv) / (ms / 1e3) / 1e9:5.0f} GB/s")
        free(m)
    print("   (The MLA lower bound uses the bytes actually read: the scores read the 576-dim latent "
          "vectors, and the weighted average reads the first 512 dims again. "
          "A fused kernel such as FlashMLA reads them only once.)")


if __name__ == "__main__":
    main()
