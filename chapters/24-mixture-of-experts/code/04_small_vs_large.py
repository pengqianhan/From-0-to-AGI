"""第 24 章 · 极简代码 4：为什么大模型都用 MoE，小模型却很少用

三笔账：
  1. 看公开的产品线：同一家族里，多大开始换成 MoE？
  2. 同一个模型，"按总参数比"和"按激活参数比"结论完全相反（Qwen3 技术报告的同数据对照）；
  3. 显存装的是总参数；而解码时每一步要读多少专家，取决于同时处理多少个 token（batch）。

所有模型数字来自官方模型卡 / 技术报告 / config.json（链接见本章 README）。
第 3 笔账假设各 token 独立、均匀地选专家（真实路由并不均匀，只作量级估计）。
运行：uv run python chapters/24-mixture-of-experts/code/04_small_vs_large.py
"""

from __future__ import annotations

# ── 1. 产品线（Hugging Face 上的官方仓库名，2026-09）──────────────────────────
LINEUP = {
    "Qwen3": dict(dense=[0.6, 1.7, 4, 8, 14, 32], moe=[("30B-A3B", 30.5, 3.3), ("235B-A22B", 235, 22)]),
    "Qwen3.5": dict(dense=[0.8, 2, 4, 9, 27],
                    moe=[("35B-A3B", 35, 3), ("122B-A10B", 122, 10), ("397B-A17B", 397, 17)]),
    "gpt-oss": dict(dense=[], moe=[("20b", 20.9, 3.6), ("120b", 116.8, 5.1)]),
    "Llama 4": dict(dense=[], moe=[("Scout", 109, 17), ("Maverick", 400, 17)]),
    "Nemotron 3": dict(dense=[], moe=[("Nano 30B-A3B", 31.6, 3.2)]),
}

# ── 2. Qwen3 技术报告表 5：同样的预训练数据，Qwen3-30B-A3B-Base vs Qwen3-14B-Base ──────
QWEN3_TABLE5 = {
    "benchmarks": ["MMLU", "MMLU-Pro", "GSM8K", "MATH", "EvalPlus", "MGSM"],
    "Qwen3-14B（稠密）": dict(total=14, active=14, scores=[81.05, 61.03, 92.49, 62.02, 72.23, 79.20]),
    "Qwen3-30B-A3B（MoE）": dict(total=30, active=3, scores=[81.38, 61.49, 91.81, 59.04, 71.45, 79.11]),
}

# ── 3. 解码一步要读多少专家 ────────────────────────────────────────────────
MOE_CFG = {  # (专家数 E, 每 token 选 K, 共享专家数)
    "Qwen3-30B-A3B": (128, 8, 0),
    "DeepSeek-V3": (256, 8, 1),
    "Mixtral-8x7B": (8, 2, 0),
}


def touched_fraction(E: int, K: int, batch: int) -> float:
    """B 个 token 各自均匀地选 K 个专家，一层里至少被一个 token 选中的专家比例：1 − (1 − K/E)^B。"""
    return 1 - (1 - K / E) ** batch


def weight_gb(params_b: float, bits: float) -> float:
    return params_b * 1e9 * bits / 8 / 1e9


if __name__ == "__main__":
    print("1) 产品线：稠密尺寸 | MoE（总参数 / 激活参数，B = 十亿）")
    for fam, v in LINEUP.items():
        dense = ", ".join(f"{x:g}B" for x in v["dense"]) or "—"
        moe = ", ".join(f"{n}（{t:g}/{a:g}）" for n, t, a in v["moe"])
        smallest = min(t for _, t, _ in v["moe"])
        print(f"  {fam:11} 稠密：{dense:28} MoE：{moe}   最小的 MoE 总参数 {smallest:g}B")

    print("\n2) 同样的数据训出来：MoE 该和谁比？（Qwen3 技术报告表 5，Base 模型）")
    names = [k for k in QWEN3_TABLE5 if k != "benchmarks"]
    print("  " + " " * 22 + "".join(f"{b:>10}" for b in QWEN3_TABLE5["benchmarks"]))
    for n in names:
        r = QWEN3_TABLE5[n]
        print(f"  {n:20}" + "".join(f"{s:10.2f}" for s in r["scores"])
              + f"   总 {r['total']}B / 激活 {r['active']}B")
    a, b = (QWEN3_TABLE5[n] for n in names)
    print(f"  → 成绩相当：MoE 的激活参数只有稠密的 {b['active'] / a['active']:.0%}（算力便宜 ~4.7 倍），"
          f"但总参数是它的 {b['total'] / a['total']:.1f} 倍（显存贵一倍多）")

    print("\n3) 显存：权重必须全部装下（只算权重，不含 KV cache 与运行时开销）")
    for name, total, active in [("Qwen3.5-0.8B（稠密）", 0.8, 0.8), ("Qwen3.5-9B（稠密）", 9, 9),
                                ("Qwen3-30B-A3B", 30.5, 3.3), ("Qwen3.5-35B-A3B", 35, 3)]:
        print(f"  {name:22} 激活 {active:5.1f}B   BF16 {weight_gb(total, 16):6.1f} GB   "
              f"4-bit {weight_gb(total, 4):5.1f} GB")
    print("  → 一台 8 GB 内存的手机，4-bit 量化也装不下 30B 级的 MoE；而 0.8B 的稠密模型只要约 0.4 GB")

    print("\n4) 解码一步，一层里要把多少比例的专家权重从显存读出来（均匀路由的估计）")
    batches = (1, 4, 16, 64, 256)
    print(f"  {'模型':15}" + "".join(f"{'batch ' + str(b):>11}" for b in batches))
    for name, (E, K, _) in MOE_CFG.items():
        print(f"  {name:15}" + "".join(f"{touched_fraction(E, K, b):11.1%}" for b in batches))
    print("  → 单用户（batch 1）只读激活的那一小部分，所以本地跑 MoE 很快——前提是内存装得下全部专家；")
    print("    服务端大 batch 时几乎每个专家都要读，省下的是每个 token 的算力，不是显存")
