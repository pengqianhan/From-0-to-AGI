"""Chapter 10 · GPU script 6: the KV cache on a GPU, and "prefill uses compute, decode uses bandwidth"

On the CPU (03_kv_cache.py), we saw two things. The KV cache made generation several times
faster. The prefill throughput was tens of times the decode throughput.
Here we look again with a GPU and a much larger model:
  1. Naive generation vs KV cache: the mean time to generate one token. The script calls
     generate_naive / generate_cached from 03 directly.
  2. One forward pass with T tokens (this is a prefill of T tokens), with T from 1 to 2048.
     For small T, the time almost does not change: the bottleneck is to read all weights from
     GPU memory once. For large T, the time increases linearly with T: compute becomes the bottleneck.
  3. Decode: each sequence gets only 1 token per step, and the batch increases from 1 to 64.
     One read of the weights serves 64 sequences.

There are two kinds of timing:
  - eager (one launch at a time): the usual call of the model. For each operator call, Python
    launches one kernel on the GPU.
  - CUDA graph: first record all kernels of one forward pass. Then replay the full graph at once,
    without Python. This measures only the time of the GPU work. (vLLM and other inference
    engines use the same method in the decode phase to remove the launch overhead.)

Model: TinyLM from 01_tiny_model.py. importlib loads the code without changes. Only the size
changes: the shape of one layer of Llama 3 8B (d=4096, 32 query heads, 8 KV heads, FFN 14336),
4 layers, about 0.87B parameters, BF16.
Few, wide layers make "read the weights" the main cost, not "Python launches kernels".
The weights are random: this script measures only the speed, not the generated content.
Needs a CUDA GPU (about 10 GB of GPU memory). It takes about 1 minute.
Run: uv run python chapters/10-inference/code/06_gpu_prefill_decode.py
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
import time
from pathlib import Path

import torch


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tiny = _load("tiny_model", "01_tiny_model.py")
kvc = _load("kv_cache_demo", "03_kv_cache.py")

SIZE = dict(dim=4096, n_layers=4, n_heads=32, n_kv_heads=8, ffn_dim=14336, max_seq_len=2560)
GEN_LENGTHS = [64, 256, 512]
PREFILL_LENGTHS = [1, 8, 32, 128, 512, 2048]
BATCHES = [1, 8, 32, 64]
CONTEXTS = [64, 512]  # positions already in the cache for the decode timing
SPEC = {"RTX 3090": (71.0, 936.0)}  # spec sheet: dense BF16 tensor-core peak TFLOPS, memory bandwidth GB/s


def cuda_time(fn, reps: int) -> float:
    """Time with CUDA events. Warm up two times first. Return the median (ms)."""
    fn()
    fn()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def graph_time(fn, reps: int) -> float:
    """Record all kernels of fn as one CUDA graph, then time the replay (median, ms)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # warm up on a side stream before the recording
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    ms = cuda_time(graph.replay, reps)
    del graph
    torch.cuda.empty_cache()
    return ms


def wall_time(fn, reps: int) -> float:
    """Time a full generation with the wall clock and return the median (s).

    Each step copies the token back to the CPU.
    """
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def pct(x: float, peak: float | None) -> str:
    return f" ({x / peak:4.0%})" if peak else ""


def main() -> None:
    if not torch.cuda.is_available():
        print("This script needs a CUDA GPU. Without a GPU, skip it: "
              "the README shows the results of one run on an RTX 3090.")
        sys.exit(0)
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name}  torch {torch.__version__}  CUDA {torch.version.cuda}")
    peak_tf, peak_bw = next((v for k, v in SPEC.items() if k in name), (None, None))
    torch.manual_seed(0)

    with torch.device("cuda"), torch.no_grad():  # new tensors in 01/03 (ids, masks) also go to the GPU
        data = tiny.CharData()
        c = tiny.Config(vocab_size=data.vocab_size, **SIZE)
        model = tiny.TinyLM(c).to(torch.bfloat16).eval()
        n_params = sum(p.numel() for p in model.parameters())
        w_bytes = n_params * 2
        d_attn = c.n_heads * c.head_dim
        print(f"Model: {c.n_layers} layers, d={c.dim}, {c.n_heads} query heads / {c.n_kv_heads} KV heads, "
              f"head_dim={c.head_dim}, FFN {c.ffn_dim}; {n_params / 1e9:.2f}B parameters, "
              f"BF16 weights {w_bytes / 2**30:.2f} GiB")
        if peak_tf:
            print(f"Spec-sheet peak: BF16 {peak_tf:.0f} TFLOPS, memory bandwidth {peak_bw:.0f} GB/s; "
                  f"ratio ≈ {peak_tf * 1e12 / (peak_bw * 1e9):.0f} FLOP/byte. "
                  f"One read of the weights takes at least {w_bytes / (peak_bw * 1e9) * 1e3:.2f} ms")

        # ① naive vs KV cache. The generation functions of 03 copy each token back to the CPU,
        # so only eager timing is possible.
        prompt = data.encode(kvc.PROMPT)
        kvc.generate_naive(model, prompt, 8, temperature=0)  # warmup
        kvc.generate_cached(model, prompt, 8, temperature=0)
        print(f"\n1. Naive generation vs KV cache (greedy, prompt of {len(prompt)} characters, eager, "
              f"median of 3 runs)")
        print("      new  naive ms/token   KV cache ms/token  speedup")
        for n in GEN_LENGTHS:
            t_naive = wall_time(lambda n=n: kvc.generate_naive(model, prompt, n, temperature=0), 3)
            t_cache = wall_time(lambda n=n: kvc.generate_cached(model, prompt, n, temperature=0), 3)
            print(f"   {n:6d}   {t_naive / n * 1e3:13.2f}   {t_cache / n * 1e3:17.2f}"
                  f"   {t_naive / t_cache:5.1f}×")

        # ② one forward pass with T tokens (prefill)
        print("\n2. One forward pass with T tokens (prefill; median of 10 runs; the last three columns "
              "use the CUDA graph time; read = read the weights once)")
        print("       T  eager ms  graph ms      token/s   TFLOPS (% peak)  read GB/s (% peak)")
        for T in PREFILL_LENGTHS:
            ids = torch.randint(0, c.vocab_size, (1, T))
            eager = cuda_time(lambda ids=ids: model(ids), 10)
            ms = graph_time(lambda ids=ids: model(ids), 10)
            flops = 2 * n_params * T + 2 * c.n_layers * d_attn * T * T  # matrix multiplications + causal attention
            tflops, gbps = flops / ms / 1e9, w_bytes / ms / 1e6
            print(f"   {T:5d}  {eager:8.2f}  {ms:8.2f}  {T / ms * 1e3:11,.0f}   {tflops:8.1f}{pct(tflops, peak_tf)}"
                  f"   {gbps:10.0f}{pct(gbps, peak_bw)}")

        # ③ decode: the cache already has ctx positions. Each sequence gets 1 new token per step.
        print("\n3. decode: each sequence gets 1 token per step, batch from 1 to 64 (CUDA graph, "
              "median of 20 runs; read = all weights + all of the KV cache)")
        print("  context  batch   ms/step      token/s   KV cache MiB  read GB/s (% peak)")
        for ctx in CONTEXTS:
            for B in BATCHES:
                cache = tiny.KVCache(c.n_layers)
                model(torch.randint(0, c.vocab_size, (B, ctx)), cache)
                k0, v0 = list(cache.k), list(cache.v)
                x = torch.randint(0, c.vocab_size, (B, 1))

                def step(cache=cache, k0=k0, v0=v0, x=x):
                    cache.k, cache.v = list(k0), list(v0)  # each run starts from the same ctx positions
                    model(x, cache)

                ms = graph_time(step, 20)
                kv_bytes = sum(t.numel() * t.element_size() for t in k0 + v0)
                gbps = (w_bytes + kv_bytes) / ms / 1e6  # minimum read: all weights + all of the KV cache
                print(f"   {ctx:6d}  {B:5d}  {ms:8.2f}  {B / ms * 1e3:11,.0f}   {kv_bytes / 2**20:12.0f}"
                      f"   {gbps:10.0f}{pct(gbps, peak_bw)}")
                del cache, k0, v0


if __name__ == "__main__":
    main()
