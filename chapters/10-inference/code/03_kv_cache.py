"""Chapter 10 · Minimal code 3: KV cache (the same output with much less calculation)

Naive generation: for each new character, the model processes "the prompt + all generated text"
again. Step t processes t positions. To generate n characters, the model processes about n²/2
positions in total. Most of this work is repeated (the K and V of past positions do not change).
KV cache: keep the K and V that each layer calculated. After that, give the model only 1 new
character at each step, and calculate only its own q, k, v.

This script does three things:
  1. Parity check: in greedy mode and in sampling mode, the cached version and the naive version
     generate exactly the same characters.
  2. Speed: the time that each version needs to generate 64 / 128 / 256 / 512 characters.
  3. Two phases: the throughput difference between prefill (give the full prompt at once)
     and decode (one character at a time).
Run: uv run python chapters/10-inference/code/03_kv_cache.py
"""

from __future__ import annotations

import importlib.util
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
samp = _load("sampling", "02_sampling.py")


@torch.no_grad()
def generate_naive(model, prompt: list[int], n: int, seed: int = 0, **kw):
    g = torch.Generator().manual_seed(seed)
    ids, processed = list(prompt), 0
    for _ in range(n):
        logits = model(torch.tensor([ids]))[0, -1]  # calculate the full sequence again
        processed += len(ids)
        ids.append(samp.sample_next(logits, g=g, **kw))
    return ids[len(prompt):], processed


@torch.no_grad()
def generate_cached(model, prompt: list[int], n: int, seed: int = 0, **kw):
    g = torch.Generator().manual_seed(seed)
    cache = tiny.KVCache(model.c.n_layers)
    logits = model(torch.tensor([prompt]), cache)[0, -1]  # prefill: give the full prompt at once
    processed, out = len(prompt), []
    for i in range(n):
        nxt = samp.sample_next(logits, g=g, **kw)
        out.append(nxt)
        if i < n - 1:
            logits = model(torch.tensor([[nxt]]), cache)[0, -1]  # decode: give only 1 new character
            processed += 1
    return out, processed, cache


def best_time(fn, repeats: int = 2) -> float:
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


PROMPT = "ROMEO:\nI will "
LENGTHS = [64, 128, 256, 512]


def speed_table(model, data):
    prompt = data.encode(PROMPT)
    rows = []
    for n in LENGTHS:
        t_naive = best_time(lambda n=n: generate_naive(model, prompt, n, temperature=0))
        t_cache = best_time(lambda n=n: generate_cached(model, prompt, n, temperature=0))
        proc_naive = sum(len(prompt) + i for i in range(n))
        proc_cache = len(prompt) + n - 1
        rows.append(dict(n=n, naive=t_naive, cached=t_cache, speedup=t_naive / t_cache,
                         proc_naive=proc_naive, proc_cache=proc_cache))
    return rows


@torch.no_grad()
def prefill_vs_decode(model, data, n: int = 256):
    """Process the same n positions: all at once (prefill) vs one at a time (decode)."""
    ids = torch.tensor([(data.encode(PROMPT) * 100)[:n]])

    def prefill():
        model(ids, tiny.KVCache(model.c.n_layers))

    def decode():
        cache = tiny.KVCache(model.c.n_layers)
        for t in range(n):
            model(ids[:, t : t + 1], cache)

    tp, td = best_time(prefill, 3), best_time(decode, 2)
    return dict(n=n, prefill=tp, decode=td, prefill_tps=n / tp, decode_tps=n / td)


if __name__ == "__main__":
    model, data = tiny.load_or_train(4), tiny.CharData()
    prompt = data.encode(PROMPT)

    print("1. Parity check: do the cached and the naive versions generate the same 200 characters?")
    for name, kw in [("greedy", dict(temperature=0)), ("sample T=1.0 top-p=0.9", dict(temperature=1.0, top_p=0.9))]:
        a, _ = generate_naive(model, prompt, 200, seed=0, **kw)
        b, _, cache = generate_cached(model, prompt, 200, seed=0, **kw)
        print(f"  {name:22} same: {a == b}   start: {data.decode(b[:40])!r}")
    with torch.no_grad():  # how much the logits at the last position differ between the two methods
        full = model(torch.tensor([prompt + b]))[0, -1]
        c2 = tiny.KVCache(model.c.n_layers)
        model(torch.tensor([prompt + b[:-1]]), c2)
        inc = model(torch.tensor([[b[-1]]]), c2)[0, -1]
    print(f"  max logits difference at the same position: {float((full - inc).abs().max()):.1e} "
          f"(the size of floating-point rounding)")
    L, H, D, T = model.c.n_layers, model.c.n_kv_heads, model.c.head_dim, len(prompt) + 199
    print(f"  cache size {cache.nbytes()} bytes = 2 × {L} layers × {H} KV heads × {D} × {T} positions"
          f" × 4 bytes = {2 * L * H * D * T * 4}")

    print("\n2. Speed (greedy, 1 CPU thread, the faster of 2 runs; pos. = positions processed)")
    print("     new naive (s)   cache (s)  speedup     naive pos.    cached pos.")
    for r in speed_table(model, data):
        print(f"  {r['n']:6d}   {r['naive']:7.2f}   {r['cached']:9.2f}    {r['speedup']:4.1f}×   "
              f"{r['proc_naive']:12,d}   {r['proc_cache']:12,d}")

    print("\n3. prefill vs decode: process the same 256 positions")
    r = prefill_vs_decode(model, data)
    print(f"  prefill, 256 at a time: {r['prefill'] * 1000:6.1f} ms  → {r['prefill_tps']:8.0f} positions/s")
    print(f"  decode,  1 at a time:   {r['decode'] * 1000:6.1f} ms  → {r['decode_tps']:8.0f} positions/s")
    print(f"  prefill throughput is {r['prefill_tps'] / r['decode_tps']:.0f}× the decode throughput")
