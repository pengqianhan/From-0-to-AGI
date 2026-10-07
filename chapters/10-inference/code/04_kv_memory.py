"""Chapter 10 · Minimal code 4: the memory ledger of the KV cache

Each layer keeps one K and one V for each position. Each of them has n_kv_heads × head_dim numbers:

    KV cache bytes = 2 × layers × KV heads × head_dim × sequence length × bytes per number (× batch)

This script calculates three ledgers:
  1. The main-line model (configs/main): the memory for 1 token, for a 4K context, and for a
     32K context. It compares MHA / GQA / MQA.
  2. Some public models. The numbers come from the config.json of each model on Hugging Face
     (see the source table in the README).
  3. It allocates a real cache with the production class zero.kv_cache.KVCache and makes sure
     that nbytes() agrees with the formula.
Run: uv run python chapters/10-inference/code/04_kv_memory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))  # `import zero` then works from any directory
BF16 = 2  # bytes


def kv_bytes(layers: int, kv_heads: int, head_dim: int, seq_len: int, bytes_per: int = BF16,
             batch: int = 1) -> int:
    return 2 * layers * kv_heads * head_dim * seq_len * bytes_per * batch


def fmt(n: float) -> str:
    for unit, k in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if n >= k:
            return f"{n / k:.2f} {unit}"
    return f"{n:.0f} B"


# name, layers, query heads, KV heads, head_dim, note.
# We checked the numbers against the config.json of each model (2026-09).
PUBLIC = [
    ("Qwen3-0.6B", 28, 16, 8, 128, ""),
    ("SmolLM3-3B", 36, 16, 4, 128, ""),
    ("Qwen2.5-7B", 28, 28, 4, 128, ""),
    ("Llama-3.1-8B", 32, 32, 8, 128, ""),
    ("Qwen3-8B", 36, 32, 8, 128, ""),
    ("Mistral-7B-v0.1", 32, 32, 8, 128, "sliding window 4096: the real cache has at most 4096 positions"),
    ("gpt-oss-20b", 24, 64, 8, 64, "half of the layers use sliding attention (window 128): the real cache is smaller"),
    ("Gemma-3-27B", 62, 32, 16, 128, "5/6 of the layers use local attention (window 1024): the real cache is smaller"),
    ("Llama-3.3-70B", 80, 64, 8, 128, ""),
]


def main_config():
    from zero.config import load_config

    return load_config(ROOT / "configs" / "main" / "pretrain.toml").model


if __name__ == "__main__":
    m = main_config()
    L, Hq, Hkv, D = m.n_layers, m.n_heads, m.n_kv_heads, m.head_dim
    print(f"1. Main-line model configs/main: {L} layers, {Hq} query heads, {Hkv} KV heads, "
          f"head_dim {D}, BF16")
    per_tok = kv_bytes(L, Hkv, D, 1)
    print(f"   per token: 2 × {L} × {Hkv} × {D} × 2 bytes = {per_tok:,} bytes = {fmt(per_tok)}")
    print("   scheme       KV heads    per token   4K context  32K context")
    for name, h in (("MHA (unshared)", Hq), ("GQA (main)", Hkv), ("MQA (shared)", 1)):
        print(f"   {name:14} {h:6d}   {fmt(kv_bytes(L, h, D, 1)):>10}   "
              f"{fmt(kv_bytes(L, h, D, 4096)):>10}   {fmt(kv_bytes(L, h, D, 32768)):>10}")
    weights = 689.5e6 * BF16  # count_params gives 689.5M parameters in total (see configs/main/pretrain.toml)
    print(f"   compare: model weights 689.5M parameters × 2 bytes = {fmt(weights)}; "
          f"for one 32K conversation, the GQA cache is {kv_bytes(L, Hkv, D, 32768) / weights:.1f}× "
          f"the weights, the MHA cache is {kv_bytes(L, Hq, D, 32768) / weights:.1f}×")
    print(f"   serve 16 conversations of 32K at the same time (batch 16): "
          f"GQA {fmt(kv_bytes(L, Hkv, D, 32768, batch=16))}, "
          f"MHA {fmt(kv_bytes(L, Hq, D, 32768, batch=16))}")

    print("\n2. Public models (BF16, batch 1; the formula counts all layers as full attention)")
    print("   model          layers    Q   KV head_dim  per token  32K context  MHA (no GQA) saved")
    for name, l, hq, hkv, d, note in PUBLIC:
        gqa, mha = kv_bytes(l, hkv, d, 32768), kv_bytes(l, hq, d, 32768)
        print(f"   {name:17} {l:3d} {hq:4d} {hkv:4d} {d:6d}   {fmt(kv_bytes(l, hkv, d, 1)):>10}   "
              f"{fmt(gqa):>10}   {fmt(mha):>11}   {hq // hkv:2d}×  {note}")

    print("\n3. Parity check with the production code: zero.kv_cache.KVCache allocates "
          "1024 positions in advance for the main-line config (BF16)")
    from zero.kv_cache import KVCache

    cache = KVCache.from_config(m, batch_size=1, max_seq_len=1024, dtype=torch.bfloat16)
    print(f"   KVCache.nbytes() = {cache.nbytes():,}   formula = {kv_bytes(L, Hkv, D, 1024):,}   "
          f"same: {cache.nbytes() == kv_bytes(L, Hkv, D, 1024)}")
