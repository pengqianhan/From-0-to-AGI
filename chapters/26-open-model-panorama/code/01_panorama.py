"""第 26 章 · 极简代码 1：把最新的开源旗舰和我们的主线模型放进同一张表

读 models.json（每个字段都抄自 config.json / 模型卡，链接在文件里），做三件事：
  1. 逐层"拆解"：每个模型的层由哪几种注意力组成；
  2. 大对比表：参数、层数、注意力、MoE、MTP、上下文、词表、KV cache（每 token / 128K）；
     KV cache 一律用生产级的 zero.tools.kv_cache_calc 计算（DeepSeek-V4 的压缩注意力它还不认识，
     这里用它的 KVLayout / LayerSpec 自己搭一本账）；
  3. 采用矩阵：每项技术在 6 个最新旗舰家族里有几家在用 → 按 GOAL.md 2.1 的"至少 3 家"判定。

运行：uv run python chapters/26-open-model-panorama/code/01_panorama.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))  # 让脚本能 import 仓库根目录下的 zero

from zero.config import load_model_config  # noqa: E402
from zero.model import count_params  # noqa: E402
from zero.tools.kv_cache_calc import (  # noqa: E402
    KVLayout,
    LayerSpec,
    fixed_state_bytes,
    kv_cache_bytes,
    layout_from_config,
)

torch.set_num_threads(1)

GiB, MiB, KiB = 2**30, 2**20, 2**10
SEQ = 131072  # 128K
FLAGSHIPS = [
    "deepseek-v4-pro",
    "qwen3.8-2.4t-a95b",
    "kimi-k3",
    "gpt-oss-120b",
    "glm-5.3",
    "minimax-m3",
]
DISSECT = ["deepseek-v4-pro", "qwen3.8-2.4t-a95b", "kimi-k3", "gpt-oss-120b"]


def load_models() -> dict[str, dict]:
    data = json.loads((HERE / "models.json").read_text(encoding="utf-8"))
    return {m["id"]: m for m in data["models"]}


# ── 1. 每层是什么 ───────────────────────────────────────────────────────────
def layer_kinds(m: dict) -> list[str]:
    """每层注意力的类型（短名）。依据：layer_types / linear_attn_config / compress_ratios / 稀疏配置。"""
    c = m["config"]
    if m.get("zero_toml"):
        return ["GQA"] * load_model_config(REPO / m["zero_toml"]).n_layers
    n = c.get("num_hidden_layers") or c.get("n_layer")
    # DeepSeek-V4：4 = CSA（压缩 + 稀疏），128 = HCA（重压缩），0 = 纯滑动窗口
    if "compress_ratios" in c:
        name = {4: "CSA", 128: "HCA", 0: "SWA"}
        return [name[r] for r in c["compress_ratios"][:n]]
    if "linear_attn_config" in c:  # Kimi：层号从 1 开始
        full = set(c["linear_attn_config"]["full_attn_layers"])
        return ["MLA" if i + 1 in full else "KDA" for i in range(n)]
    if "layer_types" in c:
        short = {"linear_attention": "GDN", "full_attention": "GQA", "sliding_attention": "SWA"}
        return [short[t] for t in c["layer_types"]]
    if c.get("kv_lora_rank"):
        return ["MLA+DSA" if c.get("index_topk") else "MLA"] * n
    sp = c.get("sparse_attention_config", {})
    if sp.get("use_sparse_attention"):
        return ["GQA+稀疏" if f else "GQA" for f in sp["sparse_attention_freq"]]
    return ["MHA" if c.get("n_head") else "GQA"] * n


def composition(kinds: list[str]) -> str:
    out: list[str] = []
    for k in dict.fromkeys(kinds):
        out.append(f"{kinds.count(k)} {k}")
    return " + ".join(out)


# ── 2. KV cache 账本 ─────────────────────────────────────────────────────────
def dsv4_layout(m: dict, with_indexer: bool = False) -> KVLayout:
    """DeepSeek-V4：每层一份共享 K=V 的压缩条目（每 r 个 token 一条、每条 head_dim 个数）
    + 一个 128 窗口的滑动分支（K=V，head_dim 个数）；CSA 层另有索引器的压缩 key（index_head_dim）。"""
    c = m["config"]
    hd, win = c["head_dim"], c["sliding_window"]
    layers: list[LayerSpec] = []
    for r in c["compress_ratios"][: c["num_hidden_layers"]]:
        if r:  # 按"每个原始 token 摊到多少个数"记账：head_dim / r（精确值是 ⌊T/r⌋ 条 × head_dim）
            layers.append(LayerSpec("full", hd // r))
            if with_indexer and r == 4:
                layers.append(LayerSpec("full", c["index_head_dim"] // r))
        layers.append(LayerSpec("sliding", hd, window=win))
    return KVLayout(m["name"], layers)


def gpt2_view(c: dict) -> dict:
    """GPT-2 的字段名和后来的模型不一样，换成 kv_cache_calc 认识的名字（值不变）。"""
    return dict(
        num_hidden_layers=c["n_layer"], num_attention_heads=c["n_head"], hidden_size=c["n_embd"]
    )


def layout(m: dict) -> KVLayout:
    if m.get("zero_toml"):
        return layout_from_config(REPO / m["zero_toml"], name=m["name"])
    c = m["config"]
    if "compress_ratios" in c:
        return dsv4_layout(m)
    return layout_from_config(gpt2_view(c) if "n_layer" in c else c, name=m["name"])


def kv_numbers(m: dict) -> dict:
    """每 token 的增长量取"长上下文下"的斜率：(KV(256K) − KV(128K)) / 128K。
    这样滑动窗口层（早就填满了）不再计入，DeepSeek-V4 的压缩层按摊到每个 token 的量计入。"""
    lay = layout(m)
    c = m["config"]
    ctx = (
        32768 if m.get("zero_toml") else (c.get("max_position_embeddings") or c.get("n_positions"))
    )
    grow = (kv_cache_bytes(lay, 2 * SEQ) - kv_cache_bytes(lay, SEQ)) / SEQ
    return dict(
        per_token=grow,
        kv_32k=kv_cache_bytes(lay, 32768) if ctx >= 32768 else None,
        kv_128k=kv_cache_bytes(lay, SEQ) if ctx >= SEQ else None,
        state=fixed_state_bytes(lay),
        ctx=ctx,
    )


# ── 3. 从 config 读出"用了哪些技术" ──────────────────────────────────────────
# config 里看不出来的，写明证据来源（代码或报告）
QK_NORM_EVIDENCE = {
    "qwen3": "HF Qwen3Attention.q_norm/k_norm",
    "qwen3_5_text": "HF Qwen3_5Attention.q_norm/k_norm",
    "qwen3_5_moe_text": "HF Qwen3_5MoeAttention.q_norm/k_norm",
    "deepseek_v4": "V4 报告 2.3.3：query 各头与压缩 KV 条目都做 RMSNorm",
}


# 这 6 家之外、前面章节已经核实过的采用方（家族名 + 章节），用来补足"至少 3 家"的计数
EARLIER = {
    "YaRN": ["Kimi K2、Qwen3、SmolLM3（第 15 章）"],
    "滑动窗口": ["Gemma 3、OLMo 3（第 22 章）"],
    "混合线性注意力": ["NVIDIA Nemotron 3（第 23 章）"],
    "MLA": ["Mistral Large 3（第 21 章）"],
}


def features(m: dict) -> dict[str, str]:
    """返回 {技术: 证据}；没有这项技术就不出现在字典里。"""
    c, f = m["config"], {}
    mt = c.get("model_type", "")
    if m.get("zero_toml"):
        mc = load_model_config(REPO / m["zero_toml"])
        f["RMSNorm 前置"] = "zero.model.RMSNorm"
        f["RoPE"] = f"rope_theta {mc.rope_theta:g}"
        f["SwiGLU/GLU"] = "zero.model.SwiGLU"
        f["GQA"] = f"{mc.n_heads} Q / {mc.n_kv_heads} KV"
        if mc.qk_norm:
            f["QK-Norm"] = "qk_norm = true"
        if mc.tie_embeddings:
            f["共享 embedding"] = "tie_embeddings = true"
        return f
    if c.get("rms_norm_eps"):
        f["RMSNorm 前置"] = f"rms_norm_eps {c['rms_norm_eps']:g}" + (
            "（Gemma 式 1+w）" if c.get("use_gemma_norm") else ""
        )
    # 位置编码
    if c.get("mla_use_nope"):
        f["NoPE（全注意力层不加位置）"] = "mla_use_nope: true"
    elif c.get("n_positions"):
        f["学习的绝对位置"] = f"n_positions {c['n_positions']}"
    else:
        rp = c.get("rope_parameters", {})
        prf = c.get("partial_rotary_factor") or rp.get("partial_rotary_factor")
        if prf:
            f["RoPE"] = f"partial_rotary_factor {prf}"
        elif c.get("qk_rope_head_dim"):
            f["RoPE"] = f"只转 {c['qk_rope_head_dim']} 维（qk_rope_head_dim）"
        else:
            f["RoPE"] = f"rope_theta {c.get('rope_theta')}"
    rs = c.get("rope_scaling") or {}
    if (rs.get("type") or rs.get("rope_type")) == "yarn":
        f["YaRN"] = f"factor {rs['factor']:g}，原始 {rs['original_max_position_embeddings']}"
    act = c.get("hidden_act") or c.get("activation_function")
    if act in ("silu", "swigluoai", "situ"):
        f["SwiGLU/GLU"] = f"hidden_act {act}"
    # 注意力
    kinds = layer_kinds(m)
    if any(k.startswith("GQA") for k in kinds) and c.get("num_key_value_heads", 0) < c.get(
        "num_attention_heads", 0
    ):
        f["GQA"] = f"{c['num_attention_heads']} Q / {c['num_key_value_heads']} KV"
    if c.get("num_key_value_heads") == 1 and "compress_ratios" in c:
        f["MQA（共享 K=V）"] = f"num_key_value_heads 1，head_dim {c['head_dim']}"
    if c.get("kv_lora_rank"):
        f["MLA"] = f"kv_lora_rank {c['kv_lora_rank']} + qk_rope_head_dim {c['qk_rope_head_dim']}"
    if c.get("use_qk_norm"):
        f["QK-Norm"] = "use_qk_norm: true"
    elif mt in QK_NORM_EVIDENCE:
        f["QK-Norm"] = QK_NORM_EVIDENCE[mt]
    if "SWA" in kinds or ("sliding_window" in c and c["sliding_window"]):
        f["滑动窗口"] = f"sliding_window {c['sliding_window']}"
    if "GDN" in kinds or "KDA" in kinds:
        lin = "GDN" if "GDN" in kinds else "KDA"
        f["混合线性注意力"] = (
            f"{kinds.count(lin)} 层 {lin} : {len(kinds) - kinds.count(lin)} 层全注意力"
        )
    if c.get("index_topk") or c.get("sparse_attention_config", {}).get("use_sparse_attention"):
        f["稀疏注意力"] = (
            f"index_topk {c['index_topk']}"
            if c.get("index_topk")
            else f"块稀疏：块 {c['sparse_attention_config']['sparse_block_size']}，"
            f"top {c['sparse_attention_config']['sparse_topk_blocks']} 块"
        )
    if "compress_ratios" in c:
        f["压缩注意力（CSA/HCA）"] = "compress_ratios 4 / 128"
    if c.get("hc_mult"):
        f["超连接 mHC"] = f"hc_mult {c['hc_mult']}"
    if c.get("attn_res_block_size"):
        f["注意力残差 AttnRes"] = f"attn_res_block_size {c['attn_res_block_size']}"
    # 前馈
    E = c.get("n_routed_experts") or c.get("num_experts") or c.get("num_local_experts")
    if E:
        k = c.get("num_experts_per_tok") or c.get("num_experts_per_token")
        f["MoE"] = f"{E} 选 {k}"
        sh = (
            c.get("n_shared_experts")
            or c.get("num_shared_experts")
            or (1 if c.get("shared_expert_intermediate_size") else 0)
        )
        if sh:
            f["共享专家"] = f"{sh} 个"
        if c.get("topk_method") == "noaux_tc" or c.get("use_routing_bias"):
            f["无辅助损失均衡"] = (
                "topk_method noaux_tc" if c.get("topk_method") else "use_routing_bias: true"
            )
    if c.get("swiglu_limit"):
        f["SwiGLU 截断"] = f"swiglu_limit {c['swiglu_limit']}"
    # 其他
    if c.get("num_nextn_predict_layers") or c.get("mtp_num_hidden_layers"):
        f["MTP"] = (
            f"num_nextn_predict_layers {c['num_nextn_predict_layers']}"
            if c.get("num_nextn_predict_layers")
            else f"mtp_num_hidden_layers {c['mtp_num_hidden_layers']}"
        )
    if c.get("tie_word_embeddings") or mt == "gpt2":
        f["共享 embedding"] = (
            "tie_word_embeddings: true" if mt != "gpt2" else "GPT2LMHeadModel 默认共享"
        )
    return f


def computed_params() -> dict[str, dict]:
    """按 config 数出来的参数（03_meta_params.py，需要 transformers）；拿不到就返回空。"""
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("meta_params", HERE / "03_meta_params.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.run()
    except Exception as e:  # noqa: BLE001 - transformers 缺失或太旧时只用模型卡数字
        print(f"（跳过按 config 数参数：{e!r}）")
        return {}


def params_str(m: dict, cp: dict) -> tuple[str, str]:
    """(总参数, 激活参数)：优先模型卡；稠密模型和没写的，用按 config 数出来的值。"""
    if m.get("zero_toml"):
        n = count_params(load_model_config(REPO / m["zero_toml"]))["total"]
        return f"{n / 1e6:.1f}M", "稠密"
    card = m.get("card", {})
    r = cp.get(m["id"], {})
    tot = card.get("total_params") or (f"{r['total'] / 1e9:.2f}B" if r.get("total") else "—")
    return tot, card.get("active_params", "稠密")


def fmt(n: float | None) -> str:
    if n is None:
        return "—"
    for unit, k in (("GiB", GiB), ("MiB", MiB), ("KiB", KiB)):
        if n >= k:
            return f"{n / k:.2f} {unit}" if unit != "KiB" else f"{n / k:.1f} {unit}"
    return f"{n:.0f} B"


def main() -> None:
    ms = load_models()

    print("=" * 100)
    print("1. 拆解：每个模型的层由什么组成（读自 config.json）")
    print("=" * 100)
    for mid in DISSECT + ["glm-5.3", "minimax-m3", "qwen3.5-0.8b", "main"]:
        m = ms[mid]
        kinds = layer_kinds(m)
        strip = "".join(
            {
                "GQA": "G",
                "MHA": "G",
                "SWA": "s",
                "GDN": "l",
                "KDA": "l",
                "MLA": "M",
                "MLA+DSA": "M",
                "CSA": "c",
                "HCA": "h",
                "GQA+稀疏": "g",
            }[k]
            for k in kinds
        )
        print(f"{m['name']:<22} {len(kinds):>3} 层：{composition(kinds)}")
        print(f"{'':<22}     {strip}")
    print(
        "  图例：G 全注意力(GQA)  s 滑动窗口  l 线性注意力(GDN/KDA)  M MLA  c CSA  h HCA  g GQA+块稀疏"
    )

    print()
    print("=" * 100)
    print(
        "2. 大对比表（KV cache：BF16、batch 1，只算随长度增长的 K/V 或潜向量；线性层固定状态另列）"
    )
    print("=" * 100)
    cp = computed_params()
    print(
        f"{'模型':<19}{'总参数':>8}{'激活':>7}{'层':>4}{'词表':>9}{'上下文':>7}{'共享emb':>6}"
        f"{'每token KV':>11}{'32K KV':>11}{'128K KV':>11}{'线性层状态':>11}"
    )
    order = [
        "gpt2",
        "qwen3-0.6b",
        "main",
        "qwen3.5-0.8b",
        "gpt-oss-120b",
        "deepseek-v3.2",
        "deepseek-v4-pro",
        "qwen3.8-2.4t-a95b",
        "kimi-k3",
        "glm-5.3",
        "minimax-m3",
    ]
    for mid in order:
        m = ms[mid]
        c = m["config"]
        tot, act = params_str(m, cp)
        kv = kv_numbers(m)
        if m.get("zero_toml"):
            mc = load_model_config(REPO / m["zero_toml"])
            vocab, tie, n = mc.vocab_size, mc.tie_embeddings, mc.n_layers
        else:
            vocab, n = c["vocab_size"], c.get("num_hidden_layers") or c.get("n_layer")
            tie = c.get("tie_word_embeddings", c.get("model_type") == "gpt2")
        ctx = f"{kv['ctx'] // 1024}K" if kv["ctx"] >= 1024 else str(kv["ctx"])
        print(
            f"{m['name']:<19}{tot:>8}{act:>7}{n:>4}{vocab:>9,}{ctx:>7}{'是' if tie else '否':>6}"
            f"{fmt(kv['per_token']):>11}{fmt(kv['kv_32k']):>11}{fmt(kv['kv_128k']):>11}{fmt(kv['state'] or None):>11}"
        )
    print(
        "  注：'每 token KV' 是长上下文下每多一个 token 增加的量（滑动窗口层已填满、不再增长）；超出模型上下文的格子记 —。"
    )
    print(
        "      主线模型按长上下文阶段的 32K 记；DeepSeek-V4 的压缩层按摊到每个 token 的量（CSA 512/4、HCA 512/128）。"
    )
    print(
        "      GLM-5.3 的参数取 GLM-5 模型卡（5.3 的 config 形状与 GLM-5 相同）；稠密小模型的总参数按 config 数出（03_meta_params.py）。"
    )

    # DeepSeek-V4 与 V3.2 的对照（报告说 1M 上下文时 KV 只有 V3.2 的 10%）
    v4, v32 = ms["deepseek-v4-pro"], ms["deepseek-v3.2"]
    T = 1 << 20
    v4b = kv_cache_bytes(dsv4_layout(v4, with_indexer=True), T)
    c32 = v32["config"]
    n32 = c32["num_hidden_layers"]
    v32_lay = KVLayout(
        "v3.2",
        [LayerSpec("mla", c32["kv_lora_rank"] + c32["qk_rope_head_dim"])] * n32
        + [LayerSpec("full", c32["index_head_dim"])] * n32,
    )
    v32b = kv_cache_bytes(v32_lay, T)
    print(
        f"\n  对照：1M 上下文、同一精度（BF16）、都算上索引器的 key：V4-Pro {fmt(v4b)}，V3.2 {fmt(v32b)}，"
        f"比值 {v4b / v32b:.1%}（报告：约 10%；报告里 KV 条目除 RoPE 维外用 FP8 存）"
    )
    main_m = ms["main"]
    print(
        f"  主线模型 32K：{fmt(kv_numbers(main_m)['kv_32k'])}；Qwen3.8-2.4T 128K：{fmt(kv_numbers(ms['qwen3.8-2.4t-a95b'])['kv_128k'])}"
    )

    print()
    print("=" * 100)
    print("3. 采用矩阵：6 个最新旗舰家族（证据：config 字段，或注明的代码 / 报告）")
    print("=" * 100)
    feats = {mid: features(ms[mid]) for mid in FLAGSHIPS}
    all_f = list(dict.fromkeys(k for mid in FLAGSHIPS for k in feats[mid]))
    short = {
        "deepseek-v4-pro": "DS-V4",
        "qwen3.8-2.4t-a95b": "Qwen3.8",
        "kimi-k3": "K3",
        "gpt-oss-120b": "gpt-oss",
        "glm-5.3": "GLM-5.3",
        "minimax-m3": "M3",
    }
    print(
        f"{'技术':<20}"
        + "".join(f"{short[m]:>9}" for m in FLAGSHIPS)
        + f"{'家数':>5}  前几章核实过的其他采用方 → 判定"
    )
    for k in all_f:
        n = sum(k in feats[mid] for mid in FLAGSHIPS)
        extra = EARLIER.get(k, [])
        total = n + len(extra)
        v = "≥3 家，共识" if total >= 3 else "不足 3 家 → 前沿观察"
        more = ("＋" + "、".join(extra) + " → ") if extra else ""
        print(
            f"{k:<20}"
            + "".join(f"{'●' if k in feats[mid] else '·':>9}" for mid in FLAGSHIPS)
            + f"{n:>5}  {more}{v}"
        )
    print("\n  证据明细：")
    for mid in FLAGSHIPS:
        print(f"  {ms[mid]['name']}: " + "；".join(f"{k}={v}" for k, v in feats[mid].items()))

    print()
    print("=" * 100)
    print("4. 小模型这一侧：主线模型 vs 两个千问小模型")
    print("=" * 100)
    for mid in ("main", "qwen3-0.6b", "qwen3.5-0.8b"):
        print(
            f"  {ms[mid]['name']}: " + "；".join(f"{k}={v}" for k, v in features(ms[mid]).items())
        )

    print()
    print("=" * 100)
    print(
        "5. 如果主线模型的'下一版'换注意力：32K 上下文的 KV cache（同样 28 层、GQA 16/8、head_dim 128）"
    )
    print("=" * 100)
    for name, lay in next_version_layouts().items():
        kv, st = kv_cache_bytes(lay, 32768), fixed_state_bytes(lay)
        extra = f"，线性层固定状态约 {fmt(st)}" if st else ""
        print(f"  {name:<34} {fmt(kv):>11}{extra}")
    print(
        "  （线性层用 Qwen3.5-0.8B 的线性层头数与维度：16 个 key 头、16 个 value 头、各 128 维；状态按 float32 估算）"
    )


def next_version_layouts() -> dict[str, KVLayout]:
    """主线模型的几种"下一版"注意力方案，交给 zero.tools.kv_cache_calc 记账。"""
    mc = load_model_config(REPO / "configs/main/pretrain.toml")
    base = dict(
        num_hidden_layers=mc.n_layers,
        num_attention_heads=mc.n_heads,
        num_key_value_heads=mc.n_kv_heads,
        head_dim=mc.head_dim,
        hidden_size=mc.dim,
    )
    lin = dict(  # 取自 Qwen3.5-0.8B 的 config（models.json）
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        mamba_ssm_dtype="float32",
    )
    n = mc.n_layers
    return {
        "现在：28 层全注意力 GQA": layout_from_config(base),
        "3:1 局部-全局（窗口 4096）": layout_from_config(
            dict(
                base,
                sliding_window=4096,
                layer_types=[
                    t for _ in range(n // 4) for t in ["sliding_attention"] * 3 + ["full_attention"]
                ],
            )
        ),
        "3:1 混合线性注意力（Qwen3.5 式）": layout_from_config(
            dict(
                base,
                **lin,
                layer_types=[
                    t for _ in range(n // 4) for t in ["linear_attention"] * 3 + ["full_attention"]
                ],
            )
        ),
    }


if __name__ == "__main__":
    main()
