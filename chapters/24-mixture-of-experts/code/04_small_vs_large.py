"""Chapter 24 · Minimal code 4: why all large models use MoE, but small models seldom use it.

Three ledgers:
  1. Look at the public model lineups. In one model family, at what size does the family change to MoE?
  2. For the same models, a comparison "by total parameters" and a comparison "by active parameters"
     give opposite conclusions (the same-data comparison in the Qwen3 technical report).
  3. The memory must hold the total parameters. The number of experts that one decode step reads
     depends on the number of tokens that the model processes at the same time (the batch).

All model numbers come from the official model cards, technical reports, and config.json files
(the links are in the chapter README).
Ledger 3 assumes that each token selects experts independently and uniformly. Real routing is not
uniform, so the result is only an estimate of the order of magnitude.
Run: uv run python chapters/24-mixture-of-experts/code/04_small_vs_large.py
"""

from __future__ import annotations

# ── 1. Lineups (official repository names on Hugging Face, 2026-09) ──────────────
LINEUP = {
    "Qwen3": dict(dense=[0.6, 1.7, 4, 8, 14, 32], moe=[("30B-A3B", 30.5, 3.3), ("235B-A22B", 235, 22)]),
    "Qwen3.5": dict(dense=[0.8, 2, 4, 9, 27],
                    moe=[("35B-A3B", 35, 3), ("122B-A10B", 122, 10), ("397B-A17B", 397, 17)]),
    "gpt-oss": dict(dense=[], moe=[("20b", 20.9, 3.6), ("120b", 116.8, 5.1)]),
    "Llama 4": dict(dense=[], moe=[("Scout", 109, 17), ("Maverick", 400, 17)]),
    "Nemotron 3": dict(dense=[], moe=[("Nano 30B-A3B", 31.6, 3.2)]),
}

# ── 2. Qwen3 technical report, Table 5: the same pretraining data, Qwen3-30B-A3B-Base vs Qwen3-14B-Base ──
# The keys stay in Chinese (稠密 = dense) because the video shows them.
QWEN3_TABLE5 = {
    "benchmarks": ["MMLU", "MMLU-Pro", "GSM8K", "MATH", "EvalPlus", "MGSM"],
    "Qwen3-14B（稠密）": dict(total=14, active=14, scores=[81.05, 61.03, 92.49, 62.02, 72.23, 79.20]),
    "Qwen3-30B-A3B（MoE）": dict(total=30, active=3, scores=[81.38, 61.49, 91.81, 59.04, 71.45, 79.11]),
}
# Display names for the printed output
TABLE5_EN = {"Qwen3-14B（稠密）": "Qwen3-14B (dense)", "Qwen3-30B-A3B（MoE）": "Qwen3-30B-A3B (MoE)"}

# ── 3. How many experts one decode step reads ─────────────────────────────────
MOE_CFG = {  # (experts E, experts per token K, shared experts)
    "Qwen3-30B-A3B": (128, 8, 0),
    "DeepSeek-V3": (256, 8, 1),
    "Mixtral-8x7B": (8, 2, 0),
}


def touched_fraction(E: int, K: int, batch: int) -> float:
    """B tokens each select K experts uniformly. Return the fraction of experts in one layer that
    at least one token selects: 1 − (1 − K/E)^B."""
    return 1 - (1 - K / E) ** batch


def weight_gb(params_b: float, bits: float) -> float:
    return params_b * 1e9 * bits / 8 / 1e9


if __name__ == "__main__":
    print("1) Lineups: dense sizes | MoE (total / active parameters, B = billion)")
    for fam, v in LINEUP.items():
        dense = ", ".join(f"{x:g}B" for x in v["dense"]) or "—"
        moe = ", ".join(f"{n} ({t:g}/{a:g})" for n, t, a in v["moe"])
        smallest = min(t for _, t, _ in v["moe"])
        print(f"  {fam:11} dense: {dense:28} MoE: {moe}   smallest MoE total {smallest:g}B")

    print("\n2) Trained on the same data: which model should the MoE be compared with? (Qwen3 technical report, Table 5, Base models)")
    names = [k for k in QWEN3_TABLE5 if k != "benchmarks"]
    print("  " + " " * 20 + "".join(f"{b:>10}" for b in QWEN3_TABLE5["benchmarks"]))
    for n in names:
        r = QWEN3_TABLE5[n]
        print(f"  {TABLE5_EN[n]:20}" + "".join(f"{s:10.2f}" for s in r["scores"])
              + f"   total {r['total']}B / active {r['active']}B")
    a, b = (QWEN3_TABLE5[n] for n in names)
    print(f"  → Similar scores: the MoE has only {b['active'] / a['active']:.0%} of the active parameters "
          f"of the dense model (~4.7× less compute), "
          f"but {b['total'] / a['total']:.1f}× its total parameters (more than 2× the memory)")

    print("\n3) Memory: all weights must fit (weights only, without the KV cache and the runtime overhead)")
    for name, total, active in [("Qwen3.5-0.8B (dense)", 0.8, 0.8), ("Qwen3.5-9B (dense)", 9, 9),
                                ("Qwen3-30B-A3B", 30.5, 3.3), ("Qwen3.5-35B-A3B", 35, 3)]:
        print(f"  {name:22} active {active:5.1f}B   BF16 {weight_gb(total, 16):6.1f} GB   "
              f"4-bit {weight_gb(total, 4):5.1f} GB")
    print("  → A phone with 8 GB of memory cannot hold a 30B-class MoE, even with 4-bit quantization. "
          "A 0.8B dense model needs only about 0.4 GB.")

    print("\n4) One decode step: the fraction of the expert weights in one layer that must be read from memory "
          "(estimate for uniform routing)")
    batches = (1, 4, 16, 64, 256)
    print(f"  {'model':15}" + "".join(f"{'batch ' + str(b):>11}" for b in batches))
    for name, (E, K, _) in MOE_CFG.items():
        print(f"  {name:15}" + "".join(f"{touched_fraction(E, K, b):11.1%}" for b in batches))
    print("  → One user (batch 1) reads only the small active part. Thus a local MoE is fast, "
          "if the memory can hold all experts.")
    print("    With a large batch on a server, almost every expert must be read. "
          "The MoE saves compute per token, not memory reads.")
