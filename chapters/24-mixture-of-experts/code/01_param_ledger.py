"""第 24 章 · 极简代码 1：参数账本 —— FFN 占了多少，MoE 的"总参数"和"激活参数"怎么来

两件事：
  1. 稠密模型里，FFN 占了非 embedding 参数（也就是每个 token 的矩阵乘算力）的大头；
  2. 按官方 config.json 数一遍几个 MoE 模型的总参数与激活参数，和官方公布的数字对照。

所有配置字段抄自各模型在 Hugging Face 上的 config.json（2026-09 读取，链接见本章 README）。
数法：embedding 与 lm_head 都算（这些模型都不共享）；注意力按 GQA 或 MLA 各自的矩阵数；
每层 2 个 RMSNorm；MoE 层 = 路由器 + 全部路由专家 + 共享专家。"激活" = 去掉每个 token 没被选中的专家。
不计 MTP 层和视觉编码器（各家口径不同，见输出末尾的说明）。
运行：uv run python chapters/24-mixture-of-experts/code/01_param_ledger.py
"""

from __future__ import annotations

# ── 配置（只抄数参数需要的字段）──────────────────────────────────────────────
# 稠密：Qwen3-8B；主线模型（configs/main/pretrain.toml）
DENSE = {
    "Qwen3-8B": dict(d=4096, L=36, H=32, KV=8, hd=128, I=12288, V=151936, tied=False),
    "主线模型": dict(d=1280, L=28, H=16, KV=8, hd=128, I=3584, V=65536, tied=True),
}

MOE = {
    "Mixtral-8x7B": dict(
        d=4096, L=32, H=32, KV=8, hd=128, V=32000, E=8, K=2, I_moe=14336,
        official=(47, 13), src="Mixtral 论文：47B / 13B（取整）"),
    "DeepSeek-V3": dict(
        d=7168, L=61, H=128, V=129280, E=256, K=8, I_moe=2048, shared=1, dense_layers=3,
        I_dense=18432, mla=dict(q_lora=1536, kv_lora=512, nope=128, rope=64, v=128),
        router_bias=True, official=(671, 37), src="技术报告：671B / 37B（不含 MTP 模块）"),
    "Kimi-K2": dict(
        d=7168, L=61, H=64, V=163840, E=384, K=8, I_moe=2048, shared=1, dense_layers=1,
        I_dense=18432, mla=dict(q_lora=1536, kv_lora=512, nope=128, rope=64, v=128),
        router_bias=True, official=(1040, 32.6), src="技术报告表 2：1.04T / 32.6B（模型卡写 1T / 32B）；本表少约 14B，原因待核实"),
    "Qwen3-235B-A22B": dict(
        d=4096, L=94, H=64, KV=4, hd=128, V=151936, E=128, K=8, I_moe=1536, qk_norm=True,
        official=(235, 22), src="技术报告：235B / 22B"),
    "Qwen3-30B-A3B": dict(
        d=2048, L=48, H=32, KV=4, hd=128, V=151936, E=128, K=8, I_moe=768, qk_norm=True,
        official=(30.5, 3.3), src="模型卡：30.5B / 3.3B"),
    "GLM-4.5": dict(
        d=5120, L=92, H=96, KV=8, hd=128, V=151552, E=160, K=8, I_moe=1536, shared=1,
        dense_layers=3, I_dense=12288, qkv_bias=True, qk_norm=True, router_bias=True,
        official=(355, 32), src="技术报告：355B / 32B（含 MTP 层、不含 embedding 与输出层）"),
    "Llama-4-Scout": dict(
        d=5120, L=48, H=40, KV=8, hd=128, V=202048, E=16, K=1, I_moe=8192, shared=1,
        official=(109, 17), src="模型卡：109B / 17B（多模态，总数含视觉编码器）"),
    "gpt-oss-120b": dict(
        d=2880, L=36, H=64, KV=8, hd=64, V=201088, E=128, K=4, I_moe=2880, gptoss=True,
        official=(116.83, 5.13), src="模型卡表 1：116.83B / 5.13B（激活数含 unembedding、不含 embedding）"),
    "gpt-oss-20b": dict(
        d=2880, L=24, H=64, KV=8, hd=64, V=201088, E=32, K=4, I_moe=2880, gptoss=True,
        official=(20.91, 3.61), src="模型卡表 1：20.91B / 3.61B（同上）"),
}


def attention_params(c: dict) -> int:
    d = c["d"]
    if "mla" in c:  # MLA（第 21 章）：q 低秩压缩 + kv 联合压缩 + 解耦 RoPE key
        m, H = c["mla"], c["H"]
        q = d * m["q_lora"] + m["q_lora"] + m["q_lora"] * H * (m["nope"] + m["rope"])
        kv = d * (m["kv_lora"] + m["rope"]) + m["kv_lora"] + m["kv_lora"] * H * (m["nope"] + m["v"])
        return q + kv + H * m["v"] * d
    H, KV, hd = c["H"], c["KV"], c["hd"]
    n = d * H * hd + 2 * d * KV * hd + H * hd * d  # Wq、Wk、Wv、Wo
    if c.get("qkv_bias"):
        n += H * hd + 2 * KV * hd
    if c.get("gptoss"):  # q/k/v/o 都有 bias，外加每个头一个 attention sink
        n += H * hd + 2 * KV * hd + d + H
    if c.get("qk_norm"):
        n += 2 * hd
    return n


def expert_params(c: dict, width: int) -> int:
    """一个 SwiGLU 专家：gate、up、down 三个矩阵。"""
    n = 3 * c["d"] * width
    if c.get("gptoss"):  # gpt-oss 的专家带 bias：gate_up 2I，down d
        n += 2 * width + c["d"]
    return n


def count(c: dict) -> dict:
    d, L, E, K = c["d"], c["L"], c["E"], c["K"]
    emb = c["V"] * d
    head = c["V"] * d
    attn = attention_params(c) + 2 * d  # 注意力 + 两个 RMSNorm
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
    print("1) 稠密模型：每层的矩阵参数里 FFN 占多少（非 embedding 部分；每 token 的矩阵乘算力 ≈ 2 × 参数）")
    for name, c in DENSE.items():
        b = dense_breakdown(c)
        print(f"  {name:10} 每层注意力 {b['attn'] / 1e6:6.1f}M  FFN {b['ffn'] / 1e6:6.1f}M  "
              f"→ FFN 占 {b['share']:.1%}")

    print("\n2) MoE 模型：按 config.json 数出来的总参数 / 激活参数（单位 B = 十亿），与官方数字对照")
    print(f"  {'模型':17}{'专家数':>6}{'选几个':>6}{'共享':>5}{'专家宽':>7}{'d':>6}"
          f"{'总(算)':>9}{'激活(算)':>9}{'总(官方)':>9}{'激活(官方)':>10}{'路由专家占总':>11}{'激活比':>8}")
    for r in moe_rows():
        print(f"  {r['name']:17}{r['E']:6d}{r['K']:6d}{r['shared']:5d}{r['I']:7d}{r['d']:6d}"
              f"{r['total']:9.1f}{r['active']:9.2f}{r['off_total']:9.1f}{r['off_active']:10.2f}"
              f"{r['routed_share']:11.1%}{r['active'] / r['total']:8.1%}")
    print("\n  口径差异（算出来的数与官方不完全一致的原因）：")
    for r in moe_rows():
        print(f"  - {r['name']}：{r['src']}")
    rows = {r["name"]: r for r in moe_rows()}
    g = rows["GLM-4.5"]
    print(f"  GLM-4.5 按官方口径去掉 embedding 与输出层：{g['total'] - g['emb'] - g['head']:.1f}B / "
          f"{g['active'] - g['emb'] - g['head']:.1f}B（官方另含 1 个 MTP 层，本表未计）")
    for n in ("gpt-oss-120b", "gpt-oss-20b"):
        o = rows[n]
        print(f"  {n} 按官方口径激活数不含 embedding：{o['active'] - o['emb']:.2f}B")
