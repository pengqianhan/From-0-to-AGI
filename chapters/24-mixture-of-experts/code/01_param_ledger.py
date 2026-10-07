"""Chapter 24 · Minimal code 1: the parameter ledger. How large is the FFN, and where do the
"total parameters" and the "active parameters" of an MoE come from?

The script does two things:
  1. It shows that in a dense model, the FFN has most of the non-embedding parameters
     (and thus most of the matrix-multiply compute for each token).
  2. It counts the total and active parameters of some MoE models from their official
     config.json, and compares them with the official numbers.

All configuration fields come from the config.json of each model on Hugging Face
(read in 2026-09; the links are in the chapter README).
Counting rules: count the embedding and the lm_head (these models do not tie them).
Count the attention matrices of GQA or MLA. Count 2 RMSNorms in each layer.
An MoE layer = the router + all routed experts + the shared experts.
"Active" = the total minus the experts that a token does not select.
We do not count MTP layers or vision encoders. Each model family counts them in a different way
(see the notes at the end of the output).
Run: uv run python chapters/24-mixture-of-experts/code/01_param_ledger.py
"""

from __future__ import annotations

# ── Configurations (only the fields that we need to count parameters) ───────────
# Dense: Qwen3-8B, and the main-line model (configs/main/pretrain.toml).
# The key "主线模型" (main-line model) stays in Chinese because the video shows it.
DENSE = {
    "Qwen3-8B": dict(d=4096, L=36, H=32, KV=8, hd=128, I=12288, V=151936, tied=False),
    "主线模型": dict(d=1280, L=28, H=16, KV=8, hd=128, I=3584, V=65536, tied=True),
}
DENSE_EN = {"主线模型": "main-line"}  # display names for the printed output

MOE = {
    "Mixtral-8x7B": dict(
        d=4096, L=32, H=32, KV=8, hd=128, V=32000, E=8, K=2, I_moe=14336,
        official=(47, 13), src="Mixtral paper: 47B / 13B (rounded)"),
    "DeepSeek-V3": dict(
        d=7168, L=61, H=128, V=129280, E=256, K=8, I_moe=2048, shared=1, dense_layers=3,
        I_dense=18432, mla=dict(q_lora=1536, kv_lora=512, nope=128, rope=64, v=128),
        router_bias=True, official=(671, 37),
        src="technical report: 671B / 37B (without the MTP module)"),
    "Kimi-K2": dict(
        d=7168, L=61, H=64, V=163840, E=384, K=8, I_moe=2048, shared=1, dense_layers=1,
        I_dense=18432, mla=dict(q_lora=1536, kv_lora=512, nope=128, rope=64, v=128),
        router_bias=True, official=(1040, 32.6),
        src="technical report, Table 2: 1.04T / 32.6B (the model card gives 1T / 32B); "
            "this table is about 14B lower, cause to be verified"),
    "Qwen3-235B-A22B": dict(
        d=4096, L=94, H=64, KV=4, hd=128, V=151936, E=128, K=8, I_moe=1536, qk_norm=True,
        official=(235, 22), src="technical report: 235B / 22B"),
    "Qwen3-30B-A3B": dict(
        d=2048, L=48, H=32, KV=4, hd=128, V=151936, E=128, K=8, I_moe=768, qk_norm=True,
        official=(30.5, 3.3), src="model card: 30.5B / 3.3B"),
    "GLM-4.5": dict(
        d=5120, L=92, H=96, KV=8, hd=128, V=151552, E=160, K=8, I_moe=1536, shared=1,
        dense_layers=3, I_dense=12288, qkv_bias=True, qk_norm=True, router_bias=True,
        official=(355, 32),
        src="technical report: 355B / 32B (with the MTP layer, without the embedding and "
            "the output layer)"),
    "Llama-4-Scout": dict(
        d=5120, L=48, H=40, KV=8, hd=128, V=202048, E=16, K=1, I_moe=8192, shared=1,
        official=(109, 17),
        src="model card: 109B / 17B (multimodal; the total includes the vision encoder)"),
    "gpt-oss-120b": dict(
        d=2880, L=36, H=64, KV=8, hd=64, V=201088, E=128, K=4, I_moe=2880, gptoss=True,
        official=(116.83, 5.13),
        src="model card, Table 1: 116.83B / 5.13B (the active count includes the unembedding, "
            "but not the embedding)"),
    "gpt-oss-20b": dict(
        d=2880, L=24, H=64, KV=8, hd=64, V=201088, E=32, K=4, I_moe=2880, gptoss=True,
        official=(20.91, 3.61), src="model card, Table 1: 20.91B / 3.61B (as above)"),
}


def attention_params(c: dict) -> int:
    d = c["d"]
    if "mla" in c:  # MLA (Chapter 21): low-rank q compression + joint kv compression + decoupled RoPE key
        m, H = c["mla"], c["H"]
        q = d * m["q_lora"] + m["q_lora"] + m["q_lora"] * H * (m["nope"] + m["rope"])
        kv = d * (m["kv_lora"] + m["rope"]) + m["kv_lora"] + m["kv_lora"] * H * (m["nope"] + m["v"])
        return q + kv + H * m["v"] * d
    H, KV, hd = c["H"], c["KV"], c["hd"]
    n = d * H * hd + 2 * d * KV * hd + H * hd * d  # Wq, Wk, Wv, Wo
    if c.get("qkv_bias"):
        n += H * hd + 2 * KV * hd
    if c.get("gptoss"):  # q/k/v/o all have a bias, plus one attention sink for each head
        n += H * hd + 2 * KV * hd + d + H
    if c.get("qk_norm"):
        n += 2 * hd
    return n


def expert_params(c: dict, width: int) -> int:
    """One SwiGLU expert: three matrices (gate, up, and down)."""
    n = 3 * c["d"] * width
    if c.get("gptoss"):  # gpt-oss experts have biases: 2I for gate_up, d for down
        n += 2 * width + c["d"]
    return n


def count(c: dict) -> dict:
    d, L, E, K = c["d"], c["L"], c["E"], c["K"]
    emb = c["V"] * d
    head = c["V"] * d
    attn = attention_params(c) + 2 * d  # attention + two RMSNorms
    n_dense = c.get("dense_layers", 0)
    dense_ffn = 3 * d * c.get("I_dense", 0)
    one_expert = expert_params(c, c["I_moe"])
    router = d * E + (E if (c.get("router_bias") or c.get("gptoss")) else 0)
    shared = c.get("shared", 0) * 3 * d * c["I_moe"]
    routed_total = (L - n_dense) * E * one_expert
    routed_active = (L - n_dense) * K * one_expert
    rest = L * attn + n_dense * dense_ffn + (L - n_dense) * (router + shared) + d
    return dict(total=emb + head + rest + routed_total, active=emb + head + rest + routed_active,
                routed=routed_total, emb=emb, head=head, attn=L * attn)


def dense_breakdown(c: dict) -> dict:
    attn = c["d"] * c["H"] * c["hd"] * 2 + 2 * c["d"] * c["KV"] * c["hd"]
    ffn = 3 * c["d"] * c["I"]
    return dict(attn=attn, ffn=ffn, share=ffn / (attn + ffn),
                emb=c["V"] * c["d"] * (1 if c["tied"] else 2), L=c["L"])


def moe_rows() -> list[dict]:
    rows = []
    for name, c in MOE.items():
        n = count(c)
        rows.append(dict(name=name, total=n["total"] / 1e9, active=n["active"] / 1e9,
                         off_total=c["official"][0], off_active=c["official"][1],
                         routed_share=n["routed"] / n["total"], E=c["E"], K=c["K"],
                         shared=c.get("shared", 0), I=c["I_moe"], d=c["d"], src=c["src"],
                         emb=n["emb"] / 1e9, head=n["head"] / 1e9))
    return rows


if __name__ == "__main__":
    print("1) Dense models: the FFN share of the matrix parameters in each layer "
          "(non-embedding part; matrix-multiply compute per token ≈ 2 × parameters)")
    for name, c in DENSE.items():
        b = dense_breakdown(c)
        print(f"  {DENSE_EN.get(name, name):10} attention per layer {b['attn'] / 1e6:6.1f}M  FFN {b['ffn'] / 1e6:6.1f}M  "
              f"→ FFN share {b['share']:.1%}")

    print("\n2) MoE models: total / active parameters that we count from config.json "
          "(B = billion), compared with the official numbers\n"
          "   (E = experts, K = experts per token, shr = shared experts, width = expert width; "
          "calc = our count, off = official)")
    print(f"  {'model':17}{'E':>6}{'K':>6}{'shr':>5}{'width':>7}{'d':>6}"
          f"{'tot calc':>9}{'act calc':>9}{'tot off':>9}{'act off':>10}{'routed/tot':>11}{'act/tot':>8}")
    for r in moe_rows():
        print(f"  {r['name']:17}{r['E']:6d}{r['K']:6d}{r['shared']:5d}{r['I']:7d}{r['d']:6d}"
              f"{r['total']:9.1f}{r['active']:9.2f}{r['off_total']:9.1f}{r['off_active']:10.2f}"
              f"{r['routed_share']:11.1%}{r['active'] / r['total']:8.1%}")
    print("\n  Counting differences (why our numbers do not match the official numbers exactly):")
    for r in moe_rows():
        print(f"  - {r['name']}: {r['src']}")
    rows = {r["name"]: r for r in moe_rows()}
    g = rows["GLM-4.5"]
    print(f"  GLM-4.5 without the embedding and the output layer (official method): "
          f"{g['total'] - g['emb'] - g['head']:.1f}B / "
          f"{g['active'] - g['emb'] - g['head']:.1f}B (the official number also includes 1 MTP layer; "
          f"this table does not)")
    for n in ("gpt-oss-120b", "gpt-oss-20b"):
        o = rows[n]
        print(f"  {n} active count without the embedding (official method): {o['active'] - o['emb']:.2f}B")
