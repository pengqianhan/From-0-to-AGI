"""第 10 章 · 极简代码 3：KV cache——同样的输出，少算很多

朴素生成：每生成一个字，都把"提示词 + 已生成的全部内容"重新过一遍模型。第 t 步要处理 t 个位置，
生成 n 个字总共处理约 n²/2 个位置，其中绝大部分是重复劳动（过去位置的 K、V 不会变）。
KV cache：把每层算过的 K、V 存起来，之后每步只喂 1 个新字，只算它自己的 q、k、v。

这个脚本做三件事：
  1. 对拍：贪心和采样两种模式下，缓存版和朴素版生成的字符完全一致；
  2. 测速：生成 64 / 128 / 256 / 512 个字，两种写法各要多久；
  3. 两个阶段：prefill（一次喂进整段提示词）和 decode（一次一个字）的吞吐差别。
运行：uv run python chapters/10-inference/code/03_kv_cache.py
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
        logits = model(torch.tensor([ids]))[0, -1]  # 整段重算
        processed += len(ids)
        ids.append(samp.sample_next(logits, g=g, **kw))
    return ids[len(prompt):], processed


@torch.no_grad()
def generate_cached(model, prompt: list[int], n: int, seed: int = 0, **kw):
    g = torch.Generator().manual_seed(seed)
    cache = tiny.KVCache(model.c.n_layers)
    logits = model(torch.tensor([prompt]), cache)[0, -1]  # prefill：整段提示词一次喂进去
    processed, out = len(prompt), []
    for i in range(n):
        nxt = samp.sample_next(logits, g=g, **kw)
        out.append(nxt)
        if i < n - 1:
            logits = model(torch.tensor([[nxt]]), cache)[0, -1]  # decode：只喂 1 个新字
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
    """同样处理 n 个位置：一次性喂进去（prefill） vs 一个一个喂（decode）。"""
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

    print("1. 对拍：缓存版和朴素版生成的 200 个字符是否完全一致")
    for name, kw in [("贪心", dict(temperature=0)), ("采样 T=1.0 top-p=0.9", dict(temperature=1.0, top_p=0.9))]:
        a, _ = generate_naive(model, prompt, 200, seed=0, **kw)
        b, _, cache = generate_cached(model, prompt, 200, seed=0, **kw)
        print(f"  {name:22} 一致：{a == b}   开头：{data.decode(b[:40])!r}")
    with torch.no_grad():  # 最后一个位置的 logits，两种算法差多少
        full = model(torch.tensor([prompt + b]))[0, -1]
        c2 = tiny.KVCache(model.c.n_layers)
        model(torch.tensor([prompt + b[:-1]]), c2)
        inc = model(torch.tensor([[b[-1]]]), c2)[0, -1]
    print(f"  同一位置的 logits 最大差异 {float((full - inc).abs().max()):.1e}（浮点舍入量级）")
    L, H, D, T = model.c.n_layers, model.c.n_kv_heads, model.c.head_dim, len(prompt) + 199
    print(f"  缓存大小 {cache.nbytes()} 字节 = 2 × {L} 层 × {H} 个 KV 头 × {D} × {T} 个位置 × 4 字节"
          f" = {2 * L * H * D * T * 4}")

    print("\n2. 测速（贪心，单线程 CPU，取两次中较快的一次）")
    print("  新生成   朴素(秒)  KV cache(秒)  加速    朴素共处理位置  缓存共处理位置")
    for r in speed_table(model, data):
        print(f"  {r['n']:6d}   {r['naive']:7.2f}   {r['cached']:9.2f}    {r['speedup']:4.1f}×   "
              f"{r['proc_naive']:12,d}   {r['proc_cache']:12,d}")

    print("\n3. prefill vs decode：同样处理 256 个位置")
    r = prefill_vs_decode(model, data)
    print(f"  prefill 一次喂 256 个：{r['prefill'] * 1000:6.1f} ms  → {r['prefill_tps']:8.0f} 位置/秒")
    print(f"  decode  一次喂 1 个： {r['decode'] * 1000:6.1f} ms  → {r['decode_tps']:8.0f} 位置/秒")
    print(f"  prefill 的吞吐是 decode 的 {r['prefill_tps'] / r['decode_tps']:.0f} 倍")
