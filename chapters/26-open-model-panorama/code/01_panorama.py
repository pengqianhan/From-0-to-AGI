"""Chapter 26 · minimal code 1: put the newest open flagships and our main-line model in one table.

The script reads models.json. Each field in that file is copied from a config.json or a model card,
and the file contains the links. The script does three things:
  1. Dissect layer by layer: which kinds of attention make up the layers of each model.
  2. One large comparison table: parameters, layers, attention, MoE, MTP, context, vocabulary,
     KV cache (per token / 128K). All KV cache values come from the production code
     zero.tools.kv_cache_calc. That tool does not know the compressed attention of DeepSeek-V4 yet,
     so this script builds that ledger itself with the KVLayout / LayerSpec of the tool.
  3. Adoption matrix: how many of the 6 newest flagship families use each technique.
     The decision uses the "at least 3 families" rule of GOAL.md 2.1.

Run: uv run python chapters/26-open-model-panorama/code/01_panorama.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))  # lets the script import zero from the repository root

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


# ── 1. What each layer is ───────────────────────────────────────────────────
def layer_kinds(m: dict) -> list[str]:
    """Short name of the attention type of each layer.

    Sources: layer_types / linear_attn_config / compress_ratios / the sparse-attention config.
    """
    c = m["config"]
    if m.get("zero_toml"):
        return ["GQA"] * load_model_config(REPO / m["zero_toml"]).n_layers
    n = c.get("num_hidden_layers") or c.get("n_layer")
    # DeepSeek-V4: 4 = CSA (compressed + sparse), 128 = HCA (heavily compressed), 0 = sliding window only
    if "compress_ratios" in c:
        name = {4: "CSA", 128: "HCA", 0: "SWA"}
        return [name[r] for r in c["compress_ratios"][:n]]
    if "linear_attn_config" in c:  # Kimi: layer numbers start at 1
        full = set(c["linear_attn_config"]["full_attn_layers"])
        return ["MLA" if i + 1 in full else "KDA" for i in range(n)]
    if "layer_types" in c:
        short = {"linear_attention": "GDN", "full_attention": "GQA", "sliding_attention": "SWA"}
        return [short[t] for t in c["layer_types"]]
    if c.get("kv_lora_rank"):
        return ["MLA+DSA" if c.get("index_topk") else "MLA"] * n
    sp = c.get("sparse_attention_config", {})
    if sp.get("use_sparse_attention"):
        # The label means "GQA+sparse". It stays in Chinese because the video uses it as a key.
        return ["GQA+稀疏" if f else "GQA" for f in sp["sparse_attention_freq"]]
    return ["MHA" if c.get("n_head") else "GQA"] * n


def composition(kinds: list[str]) -> str:
    out: list[str] = []
    for k in dict.fromkeys(kinds):
        out.append(f"{kinds.count(k)} {k}")
    return " + ".join(out)


# ── 2. KV cache ledger ───────────────────────────────────────────────────────
def dsv4_layout(m: dict, with_indexer: bool = False) -> KVLayout:
    """DeepSeek-V4 ledger.

    Each layer stores compressed entries with shared K=V: one entry for each r tokens,
    head_dim numbers in each entry. Each layer also has a sliding branch with a window of 128
    (K=V, head_dim numbers). CSA layers also store the compressed keys of the indexer
    (index_head_dim).
    """
    c = m["config"]
    hd, win = c["head_dim"], c["sliding_window"]
    layers: list[LayerSpec] = []
    for r in c["compress_ratios"][: c["num_hidden_layers"]]:
        if r:  # count the numbers per original token: head_dim / r (exact value: ⌊T/r⌋ entries × head_dim)
            layers.append(LayerSpec("full", hd // r))
            if with_indexer and r == 4:
                layers.append(LayerSpec("full", c["index_head_dim"] // r))
        layers.append(LayerSpec("sliding", hd, window=win))
    return KVLayout(m["name"], layers)


def gpt2_view(c: dict) -> dict:
    """Rename the GPT-2 fields to the names that kv_cache_calc knows (the values do not change).

    GPT-2 uses different field names from later models.
    """
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
    """KV numbers of one model.

    The growth per token is the slope at long context: (KV(256K) − KV(128K)) / 128K.
    Thus sliding-window layers (full long before) do not count, and the compressed layers of
    DeepSeek-V4 count with their share per token.
    """
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


# ── 3. Read from the config which techniques a model uses ───────────────────
# If the config does not show a technique, give the source of the evidence (code or report)
QK_NORM_EVIDENCE = {
    "qwen3": "HF Qwen3Attention.q_norm/k_norm",
    "qwen3_5_text": "HF Qwen3_5Attention.q_norm/k_norm",
    "qwen3_5_moe_text": "HF Qwen3_5MoeAttention.q_norm/k_norm",
    "deepseek_v4": "V4 report 2.3.3: RMSNorm on each query head and on the compressed KV entries",
}


# Adopters outside these 6 families that earlier chapters verified (family + chapter).
# They complete the count for the "at least 3 families" rule.
# The keys must be the same as the keys of features().
EARLIER = {
    "YaRN": ["Kimi K2, Qwen3, SmolLM3 (Chapter 15)"],
    "滑动窗口": ["Gemma 3, OLMo 3 (Chapter 22)"],
    "混合线性注意力": ["NVIDIA Nemotron 3 (Chapter 23)"],
    "MLA": ["Mistral Large 3 (Chapter 21)"],
    "稀疏注意力": ["Meituan LongCat-2.0 (Chapter 22)"],
}

# Display names for the printed output. Some keys, the layer kind "GQA+稀疏", the name of the
# main-line model in models.json, and the keys of next_version_layouts() stay in Chinese, because
# the Chinese video (video/scenes.py) reads them.
EN = {
    "RMSNorm 前置": "Pre-Norm RMSNorm",
    "滑动窗口": "Sliding window",
    "稀疏注意力": "Sparse attention",
    "共享专家": "Shared experts",
    "无辅助损失均衡": "Aux-free balancing",
    "混合线性注意力": "Hybrid linear attn",
    "GQA+稀疏": "GQA+sparse",
    "主线模型（本课）": "Main-line model",
    "现在：28 层全注意力 GQA": "now: 28 full-attention GQA layers",
    "3:1 局部-全局（窗口 4096）": "3:1 local-global (window 4096)",
    "3:1 混合线性注意力（Qwen3.5 式）": "3:1 hybrid linear (Qwen3.5 style)",
}


def en(s: str) -> str:
    """English display name of a key or a name (unchanged if it has no entry in EN)."""
    return EN.get(s, s)


def features(m: dict) -> dict[str, str]:
    """Return {technique: evidence}. A technique that the model does not use is not in the dict.

    Some keys stay in Chinese because the Chinese video (video/scenes.py) looks them up:
    pre-norm RMSNorm, sliding window, sparse attention, hybrid linear attention,
    shared experts, and auxiliary-loss-free balancing.
    """
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
            f["Tied embedding"] = "tie_embeddings = true"
        return f
    if c.get("rms_norm_eps"):
        f["RMSNorm 前置"] = f"rms_norm_eps {c['rms_norm_eps']:g}" + (
            " (Gemma-style 1+w)" if c.get("use_gemma_norm") else ""
        )
    # Position encoding
    if c.get("mla_use_nope"):
        f["NoPE (full-attn)"] = "mla_use_nope: true"
    elif c.get("n_positions"):
        f["Learned absolute position"] = f"n_positions {c['n_positions']}"
    else:
        rp = c.get("rope_parameters", {})
        prf = c.get("partial_rotary_factor") or rp.get("partial_rotary_factor")
        if prf:
            f["RoPE"] = f"partial_rotary_factor {prf}"
        elif c.get("qk_rope_head_dim"):
            f["RoPE"] = f"rotates only {c['qk_rope_head_dim']} dims (qk_rope_head_dim)"
        else:
            f["RoPE"] = f"rope_theta {c.get('rope_theta')}"
    rs = c.get("rope_scaling") or {}
    if (rs.get("type") or rs.get("rope_type")) == "yarn":
        f["YaRN"] = f"factor {rs['factor']:g}, original {rs['original_max_position_embeddings']}"
    act = c.get("hidden_act") or c.get("activation_function")
    if act in ("silu", "swigluoai", "situ"):
        f["SwiGLU/GLU"] = f"hidden_act {act}"
    # Attention
    kinds = layer_kinds(m)
    if any(k.startswith("GQA") for k in kinds) and c.get("num_key_value_heads", 0) < c.get(
        "num_attention_heads", 0
    ):
        f["GQA"] = f"{c['num_attention_heads']} Q / {c['num_key_value_heads']} KV"
    if c.get("num_key_value_heads") == 1 and "compress_ratios" in c:
        f["MQA (shared K=V)"] = f"num_key_value_heads 1, head_dim {c['head_dim']}"
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
            f"{kinds.count(lin)} {lin} layers : {len(kinds) - kinds.count(lin)} full-attention layers"
        )
    if c.get("index_topk") or c.get("sparse_attention_config", {}).get("use_sparse_attention"):
        f["稀疏注意力"] = (
            f"index_topk {c['index_topk']}"
            if c.get("index_topk")
            else f"block-sparse: block {c['sparse_attention_config']['sparse_block_size']}, "
            f"top {c['sparse_attention_config']['sparse_topk_blocks']} blocks"
        )
    if "compress_ratios" in c:
        f["CSA/HCA compression"] = "compress_ratios 4 / 128"
    if c.get("hc_mult"):
        f["Hyper-connection mHC"] = f"hc_mult {c['hc_mult']}"
    if c.get("attn_res_block_size"):
        f["Attention residuals"] = f"attn_res_block_size {c['attn_res_block_size']}"
    # Feed-forward
    E = c.get("n_routed_experts") or c.get("num_experts") or c.get("num_local_experts")
    if E:
        k = c.get("num_experts_per_tok") or c.get("num_experts_per_token")
        f["MoE"] = f"top-{k} of {E}"
        sh = (
            c.get("n_shared_experts")
            or c.get("num_shared_experts")
            or (1 if c.get("shared_expert_intermediate_size") else 0)
        )
        if sh:
            f["共享专家"] = f"{sh}"
        if c.get("topk_method") == "noaux_tc" or c.get("use_routing_bias"):
            f["无辅助损失均衡"] = (
                "topk_method noaux_tc" if c.get("topk_method") else "use_routing_bias: true"
            )
    if c.get("swiglu_limit"):
        f["SwiGLU clamp"] = f"swiglu_limit {c['swiglu_limit']}"
    # Other
    if c.get("num_nextn_predict_layers") or c.get("mtp_num_hidden_layers"):
        f["MTP"] = (
            f"num_nextn_predict_layers {c['num_nextn_predict_layers']}"
            if c.get("num_nextn_predict_layers")
            else f"mtp_num_hidden_layers {c['mtp_num_hidden_layers']}"
        )
    if c.get("tie_word_embeddings") or mt == "gpt2":
        f["Tied embedding"] = (
            "tie_word_embeddings: true" if mt != "gpt2" else "GPT2LMHeadModel ties them by default"
        )
    return f


def computed_params() -> dict[str, dict]:
    """Parameters counted from the config (03_meta_params.py, needs transformers).

    Return an empty dict if the count is not available.
    """
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("meta_params", HERE / "03_meta_params.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.run()
    except Exception as e:  # noqa: BLE001 - without transformers (or too old), use only the model-card numbers
        print(f"(Skip the parameter count from the config: {e!r})")
        return {}


def params_str(m: dict, cp: dict) -> tuple[str, str]:
    """(total parameters, active parameters).

    Use the model card first. For dense models and for missing values, use the count from the config.
    """
    if m.get("zero_toml"):
        n = count_params(load_model_config(REPO / m["zero_toml"]))["total"]
        return f"{n / 1e6:.1f}M", "dense"
    card = m.get("card", {})
    r = cp.get(m["id"], {})
    tot = card.get("total_params") or (f"{r['total'] / 1e9:.2f}B" if r.get("total") else "—")
    return tot, card.get("active_params", "dense")


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
    print("1. Dissection: what makes up the layers of each model (read from config.json)")
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
        print(f"{en(m['name']):<22} {len(kinds):>3} layers: {composition([en(k) for k in kinds])}")
        print(f"{'':<22}     {strip}")
    print(
        "  Legend: G full attention (GQA)  s sliding window  l linear attention (GDN/KDA)  M MLA  c CSA  h HCA  g GQA+block-sparse"
    )

    print()
    print("=" * 100)
    print(
        "2. Large comparison table (KV cache: BF16, batch 1, only the K/V or latent vectors that grow with length; fixed state of linear layers in its own column)"
    )
    print("=" * 100)
    cp = computed_params()
    print(
        f"{'Model':<19}{'Total':>8}{'Active':>7}{'Lyr':>4}{'Vocab':>9}{'Ctx':>7}{'Tied':>6}"
        f"{'KV/token':>11}{'32K KV':>11}{'128K KV':>11}{'Lin. state':>11}"
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
            f"{en(m['name']):<19}{tot:>8}{act:>7}{n:>4}{vocab:>9,}{ctx:>7}{'yes' if tie else 'no':>6}"
            f"{fmt(kv['per_token']):>11}{fmt(kv['kv_32k']):>11}{fmt(kv['kv_128k']):>11}{fmt(kv['state'] or None):>11}"
        )
    print(
        "  Note: 'KV/token' is the growth for each additional token at long context (sliding-window layers are full and do not grow). A cell beyond the context of the model shows —."
    )
    print(
        "      The main-line model uses the 32K of the long-context stage. The compressed layers of DeepSeek-V4 count their share per token (CSA 512/4, HCA 512/128)."
    )
    print(
        "      The GLM-5.3 parameters come from the GLM-5 model card (the 5.3 config has the same shapes as GLM-5). The total parameters of small dense models are counted from the config (03_meta_params.py)."
    )

    # DeepSeek-V4 vs V3.2 (the report says that at 1M context, the KV cache is only 10% of V3.2)
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
        f"\n  Comparison at 1M context, same precision (BF16), indexer keys included in both: V4-Pro {fmt(v4b)}, V3.2 {fmt(v32b)}, "
        f"ratio {v4b / v32b:.1%} (report: about 10%; the report stores the KV entries in FP8, except the RoPE dimensions)"
    )
    main_m = ms["main"]
    print(
        f"  Main-line model at 32K: {fmt(kv_numbers(main_m)['kv_32k'])}; Qwen3.8-2.4T at 128K: {fmt(kv_numbers(ms['qwen3.8-2.4t-a95b'])['kv_128k'])}"
    )

    print()
    print("=" * 100)
    print("3. Adoption matrix: the 6 newest flagship families (evidence: config fields, or the code / report given)")
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
        f"{'Technique':<20}"
        + "".join(f"{short[m]:>9}" for m in FLAGSHIPS)
        + f"{'Uses':>5}  other adopters verified in earlier chapters → decision"
    )
    for k in all_f:
        n = sum(k in feats[mid] for mid in FLAGSHIPS)
        extra = EARLIER.get(k, [])
        total = n + len(extra)
        v = "≥3 families: consensus" if total >= 3 else "fewer than 3 → frontier note"
        more = ("+ " + ", ".join(extra) + " → ") if extra else ""
        print(
            f"{en(k):<20}"
            + "".join(f"{'●' if k in feats[mid] else '·':>9}" for mid in FLAGSHIPS)
            + f"{n:>5}  {more}{v}"
        )
    print("\n  Evidence details:")
    for mid in FLAGSHIPS:
        print(f"  {ms[mid]['name']}: " + "; ".join(f"{en(k)}={v}" for k, v in feats[mid].items()))

    print()
    print("=" * 100)
    print("4. The small-model side: the main-line model vs two small Qwen models")
    print("=" * 100)
    for mid in ("main", "qwen3-0.6b", "qwen3.5-0.8b"):
        print(
            f"  {en(ms[mid]['name'])}: " + "; ".join(f"{en(k)}={v}" for k, v in features(ms[mid]).items())
        )

    print()
    print("=" * 100)
    print(
        "5. If the 'next version' of the main-line model changes the attention: KV cache at 32K context (same 28 layers, GQA 16/8, head_dim 128)"
    )
    print("=" * 100)
    for name, lay in next_version_layouts().items():
        kv, st = kv_cache_bytes(lay, 32768), fixed_state_bytes(lay)
        extra = f", fixed state of the linear layers about {fmt(st)}" if st else ""
        print(f"  {en(name):<34} {fmt(kv):>11}{extra}")
    print(
        "  (The linear layers use the head counts and sizes of Qwen3.5-0.8B: 16 key heads and 16 value heads, 128 dims each. The state is estimated in float32.)"
    )


def next_version_layouts() -> dict[str, KVLayout]:
    """Some attention options for the "next version" of the main-line model.

    zero.tools.kv_cache_calc keeps the ledger. The dict keys stay in Chinese because the Chinese
    video shows them.
    """
    mc = load_model_config(REPO / "configs/main/pretrain.toml")
    base = dict(
        num_hidden_layers=mc.n_layers,
        num_attention_heads=mc.n_heads,
        num_key_value_heads=mc.n_kv_heads,
        head_dim=mc.head_dim,
        hidden_size=mc.dim,
    )
    lin = dict(  # from the Qwen3.5-0.8B config (models.json)
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
