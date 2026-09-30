"""第 26 章 · 极简代码 3：不下载权重，按 config 数出参数量（总参数 / 激活参数）

做法：把 models.json 里抄下来的 config 交给 Hugging Face transformers 的**官方模型类**，
在 "meta" 设备上建模型——只有形状、不分配内存，2.4T 的模型也是一秒钟建完——然后数参数。
这些模型类就是各家架构的"生产级参考实现"（DeepseekV4ForCausalLM、Qwen3_5MoeForCausalLM……）。

激活参数 = 总参数 − 路由专家参数 × (1 − k/E)，k 是每个 token 选几个专家，E 是专家总数。
各家模型卡的"激活参数"口径不一样（算不算输入 embedding、输出头），下面三种口径都列出来。

运行：uv run python chapters/26-open-model-panorama/code/03_meta_params.py
（需要 transformers ≥ 5.x，它是本仓库 dev 依赖；约 10 秒）
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

torch.set_num_threads(1)
warnings.filterwarnings("ignore")

B = 1e9


def experts_k_E(c: dict) -> tuple[int, int] | None:
    E = c.get("n_routed_experts") or c.get("num_experts") or c.get("num_local_experts")
    k = c.get("num_experts_per_tok") or c.get("num_experts_per_token")
    return (k, E) if E else None


def meta_count(config: dict) -> dict:
    """用 HF 官方模型类在 meta 设备上建模型并数参数。"""
    import transformers
    from transformers import CONFIG_MAPPING, AutoModelForCausalLM

    transformers.logging.set_verbosity_error()
    d = dict(config)
    cfg = CONFIG_MAPPING[d.pop("model_type")](**d)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    total = sum(p.numel() for p in model.parameters())  # 共享的张量只数一次
    experts = sum(p.numel() for n, p in model.named_parameters() if ".experts." in n)
    emb = model.get_input_embeddings().weight.numel()
    tied = model.get_output_embeddings().weight is model.get_input_embeddings().weight
    head = 0 if tied else model.get_output_embeddings().weight.numel()
    ke = experts_k_E(config)
    active = total - (experts * (1 - ke[0] / ke[1]) if ke else 0)
    return dict(
        cls=type(model).__name__,
        total=total,
        experts=experts,
        emb=emb,
        head=head,
        active=active,
        active_no_emb=active - emb,
        active_no_vocab=active - emb - head,
    )


def kimi_k3_latent_check(c: dict) -> dict:
    """HF 的 KimiLinear 类还不支持 K3 的 Latent MoE（路由专家在 3584 维的"潜空间"里算），
    这里只手算路由专家这一大块，看看 2.8T 是怎么来的（估算）。"""
    n_moe = c["num_hidden_layers"] - c["first_k_dense_replace"]
    E, k, f = c["num_experts"], c["num_experts_per_token"], c["moe_intermediate_size"]
    per_expert_latent = (
        3 * c["routed_expert_hidden_size"] * f
    )  # SwiGLU 三个矩阵，输入输出是潜空间维度
    per_expert_full = 3 * c["hidden_size"] * f  # 假如专家直接在 7168 维上算
    return dict(
        latent=n_moe * E * per_expert_latent,
        full=n_moe * E * per_expert_full,
        active_latent=n_moe * k * per_expert_latent,
    )


def main_model_check() -> dict:
    """主线模型：zero 的公式 vs 同形状的 HF Qwen3 类（两者应完全相等）。"""
    from zero.config import load_model_config
    from zero.model import count_params

    mc = load_model_config(REPO / "configs/main/pretrain.toml")
    ours = count_params(mc)["total"]
    hf = meta_count(
        dict(
            model_type="qwen3",
            vocab_size=mc.vocab_size,
            hidden_size=mc.dim,
            num_hidden_layers=mc.n_layers,
            num_attention_heads=mc.n_heads,
            num_key_value_heads=mc.n_kv_heads,
            head_dim=mc.head_dim,
            intermediate_size=mc.ffn_dim,
            tie_word_embeddings=mc.tie_embeddings,
        )
    )
    return dict(zero=ours, hf=hf["total"], cls=hf["cls"])


def run() -> dict[str, dict]:
    """返回 {模型 id: 参数统计}；01_panorama.py 也会调用它。"""
    data = json.loads((HERE / "models.json").read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for m in data["models"]:
        if m["id"] == "main":
            r = main_model_check()
            out["main"] = dict(
                total=r["zero"],
                active=r["zero"],
                active_no_emb=None,
                active_no_vocab=None,
                cls=f"zero.model.count_params（= HF {r['cls']} {r['hf'] / B:.4f}B）",
            )
        elif m["id"] == "kimi-k3":
            r = kimi_k3_latent_check(m["config"])
            out["kimi-k3"] = dict(
                total=None, active=None, latent=r, cls="手算（HF 类不支持 Latent MoE）"
            )
        else:
            out[m["id"]] = meta_count(m["config"])
    return out


def main() -> None:
    try:
        res = run()
    except (ImportError, KeyError) as e:  # transformers 太旧，认不出新架构
        print(f"需要较新的 transformers（≥ 5.x）：{e!r}")
        return
    data = {
        m["id"]: m for m in json.loads((HERE / "models.json").read_text(encoding="utf-8"))["models"]
    }
    print(
        f"{'模型':<20}{'HF 模型类':<28}{'总参数':>10}{'激活(含全部)':>13}{'不含输入emb':>12}{'不含词表':>10}   模型卡"
    )
    for mid, r in res.items():
        m = data[mid]
        card = m.get("card", {})
        cs = f"{card.get('total_params', '—')} / {card.get('active_params', '—')}" if card else "—"
        if r.get("total") is None:
            continue
        if mid == "main":
            print(f"{m['name']:<20}{'zero（见下）':<28}{r['total'] / B:>9.4f}B{'稠密':>13}")
            continue
        dense = experts_k_E(m["config"]) is None
        act = "稠密" if dense else f"{r['active'] / B:.2f}B"
        a2 = "" if dense else f"{r['active_no_emb'] / B:.2f}B"
        a3 = "" if dense else f"{r['active_no_vocab'] / B:.2f}B"
        print(
            f"{m['name']:<20}{r['cls']:<28}{r['total'] / B:>9.3f}B{act:>13}{a2:>12}{a3:>10}   {cs}"
        )
    print(f"\n主线模型：{res['main']['cls']}，zero 的公式 {res['main']['total'] / B:.4f}B")
    k3 = res["kimi-k3"]["latent"]
    print("\nKimi-K3（手算路由专家，估算）：")
    print(
        f"  专家在 3584 维潜空间里算：{k3['latent'] / 1e12:.2f}T（每 token 激活其中 {k3['active_latent'] / B:.1f}B）"
    )
    print(
        f"  假如专家直接在 7168 维上算：{k3['full'] / 1e12:.2f}T —— 模型卡是 2.8T 总 / 104B 激活，"
        "与模型卡写的「Latent MoE Dimension 3584」对得上"
    )
    print(
        "\n说明：DeepSeek-V4、V3.2 的 HF 类不含 MTP 层，所以比模型卡略少；"
        "MiniMax-M3 的模型卡只给约数（~428B / ~23B），这里只建了语言模型部分。"
    )


if __name__ == "__main__":
    main()
