"""第 10 章 · 极简代码 4：KV cache 的显存账

每个位置、每一层都要存一份 K 和一份 V，每份是 n_kv_heads × head_dim 个数：

    KV cache 字节数 = 2 × 层数 × KV 头数 × head_dim × 序列长度 × 每个数的字节数（× batch）

这里算三组账：
  1. 主线模型（configs/main）：每个 token、4K、32K 上下文各占多少，对比 MHA / GQA / MQA；
  2. 几个公开模型（数字来自各自 Hugging Face 上的 config.json，见 README 的来源表）；
  3. 用生产级的 zero.kv_cache.KVCache 真实分配一块缓存，确认 nbytes() 和公式一致。
运行：uv run python chapters/10-inference/code/04_kv_memory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))  # 让 `import zero` 在仓库任意位置运行都能找到
BF16 = 2  # 字节


def kv_bytes(layers: int, kv_heads: int, head_dim: int, seq_len: int, bytes_per: int = BF16,
             batch: int = 1) -> int:
    return 2 * layers * kv_heads * head_dim * seq_len * bytes_per * batch


def fmt(n: float) -> str:
    for unit, k in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if n >= k:
            return f"{n / k:.2f} {unit}"
    return f"{n:.0f} B"


# 名称, 层数, 查询头数, KV 头数, head_dim, 备注。数字核对自各模型 config.json（2026-09）
PUBLIC = [
    ("Qwen3-0.6B", 28, 16, 8, 128, ""),
    ("SmolLM3-3B", 36, 16, 4, 128, ""),
    ("Qwen2.5-7B", 28, 28, 4, 128, ""),
    ("Llama-3.1-8B", 32, 32, 8, 128, ""),
    ("Qwen3-8B", 36, 32, 8, 128, ""),
    ("Mistral-7B-v0.1", 32, 32, 8, 128, "滑动窗口 4096：实际缓存最多 4096 个位置"),
    ("gpt-oss-20b", 24, 64, 8, 64, "一半层是 128 窗口的滑动注意力：实际更小"),
    ("Gemma-3-27B", 62, 32, 16, 128, "5/6 的层是 1024 窗口的局部注意力：实际更小"),
    ("Llama-3.3-70B", 80, 64, 8, 128, ""),
]


def main_config():
    from zero.config import load_config

    return load_config(ROOT / "configs" / "main" / "pretrain.toml").model


if __name__ == "__main__":
    m = main_config()
    L, Hq, Hkv, D = m.n_layers, m.n_heads, m.n_kv_heads, m.head_dim
    print(f"1. 主线模型 configs/main：{L} 层，{Hq} 个查询头，{Hkv} 个 KV 头，head_dim {D}，BF16")
    per_tok = kv_bytes(L, Hkv, D, 1)
    print(f"   每个 token：2 × {L} × {Hkv} × {D} × 2 字节 = {per_tok:,} 字节 = {fmt(per_tok)}")
    print("   方案              KV 头   每 token      4K 上下文     32K 上下文")
    for name, h in (("MHA（不共享）", Hq), ("GQA（主线）", Hkv), ("MQA（全共享）", 1)):
        print(f"   {name:14} {h:6d}   {fmt(kv_bytes(L, h, D, 1)):>10}   "
              f"{fmt(kv_bytes(L, h, D, 4096)):>10}   {fmt(kv_bytes(L, h, D, 32768)):>10}")
    weights = 689.5e6 * BF16  # count_params 给出的总参数 689.5M（见 configs/main/pretrain.toml 注释）
    print(f"   对照：模型权重 689.5M 参数 × 2 字节 = {fmt(weights)}；"
          f"一条 32K 的对话，GQA 缓存是权重的 {kv_bytes(L, Hkv, D, 32768) / weights:.1f} 倍，"
          f"MHA 是 {kv_bytes(L, Hq, D, 32768) / weights:.1f} 倍")
    print(f"   同时服务 16 条 32K 对话（batch 16）：GQA {fmt(kv_bytes(L, Hkv, D, 32768, batch=16))}，"
          f"MHA {fmt(kv_bytes(L, Hq, D, 32768, batch=16))}")

    print("\n2. 公开模型（BF16，batch 1，按公式把所有层都当全注意力算）")
    print("   模型               层  Q头 KV头 head_dim  每 token     32K 上下文   若不共享(MHA)  省下")
    for name, l, hq, hkv, d, note in PUBLIC:
        gqa, mha = kv_bytes(l, hkv, d, 32768), kv_bytes(l, hq, d, 32768)
        print(f"   {name:17} {l:3d} {hq:4d} {hkv:4d} {d:6d}   {fmt(kv_bytes(l, hkv, d, 1)):>10}   "
              f"{fmt(gqa):>10}   {fmt(mha):>11}   {hq // hkv:2d}×  {note}")

    print("\n3. 生产级代码对拍：zero.kv_cache.KVCache 按主线配置预分配 1024 个位置（BF16）")
    from zero.kv_cache import KVCache

    cache = KVCache.from_config(m, batch_size=1, max_seq_len=1024, dtype=torch.bfloat16)
    print(f"   KVCache.nbytes() = {cache.nbytes():,}   公式 = {kv_bytes(L, Hkv, D, 1024):,}   "
          f"一致：{cache.nbytes() == kv_bytes(L, Hkv, D, 1024)}")
