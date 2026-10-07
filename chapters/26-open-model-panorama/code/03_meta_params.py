"""Chapter 26 · minimal code 3: count the parameters from the config without the weights.

The script counts the total parameters and the active parameters.
Method: give each config from models.json to the **official model class** in Hugging Face
transformers. Build the model on the "meta" device: it has only shapes and allocates no memory,
so even a 2.4T model builds in one second. Then count the parameters.
These model classes are the "production reference implementations" of each architecture
(DeepseekV4ForCausalLM, Qwen3_5MoeForCausalLM, ...).

active parameters = total parameters − routed-expert parameters × (1 − k/E).
k is the number of experts that each token selects. E is the total number of experts.
The model cards use different definitions of "active parameters" (with or without the input
embedding and the output head). The script prints all three definitions.

Run: uv run python chapters/26-open-model-panorama/code/03_meta_params.py
(Needs transformers ≥ 5.x, a dev dependency of this repository. About 10 s.)
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
    """Build the model with the official HF model class on the meta device and count the parameters."""
    import transformers
    from transformers import CONFIG_MAPPING, AutoModelForCausalLM

    transformers.logging.set_verbosity_error()
    d = dict(config)
    cfg = CONFIG_MAPPING[d.pop("model_type")](**d)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    total = sum(p.numel() for p in model.parameters())  # a tied tensor counts only once
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
    """Count the routed experts of Kimi-K3 by hand (an estimate).

    The HF KimiLinear class does not support the Latent MoE of K3 yet: the routed experts compute
    in a "latent space" of 3584 dimensions. Here we count only the large block of routed experts,
    to see where the 2.8T comes from.
    """
    n_moe = c["num_hidden_layers"] - c["first_k_dense_replace"]
    E, k, f = c["num_experts"], c["num_experts_per_token"], c["moe_intermediate_size"]
    per_expert_latent = (
        3 * c["routed_expert_hidden_size"] * f
    )  # 3 SwiGLU matrices; input and output have the latent size
    per_expert_full = 3 * c["hidden_size"] * f  # if the experts computed directly in 7168 dimensions
    return dict(
        latent=n_moe * E * per_expert_latent,
        full=n_moe * E * per_expert_full,
        active_latent=n_moe * k * per_expert_latent,
    )


def main_model_check() -> dict:
    """Main-line model: the zero formula vs the HF Qwen3 class with the same shapes (they must be equal)."""
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
    """Return {model id: parameter counts}. 01_panorama.py also calls this function."""
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
                cls=f"zero.model.count_params (= HF {r['cls']} {r['hf'] / B:.4f}B)",
            )
        elif m["id"] == "kimi-k3":
            r = kimi_k3_latent_check(m["config"])
            out["kimi-k3"] = dict(
                total=None, active=None, latent=r, cls="by hand (HF class has no Latent MoE)"
            )
        else:
            out[m["id"]] = meta_count(m["config"])
    return out


def main() -> None:
    try:
        res = run()
    except (ImportError, KeyError) as e:  # transformers is too old to know the new architectures
        print(f"A newer transformers (≥ 5.x) is necessary: {e!r}")
        return
    data = {
        m["id"]: m for m in json.loads((HERE / "models.json").read_text(encoding="utf-8"))["models"]
    }
    print(
        f"{'Model':<20}{'HF model class':<28}{'Total':>10}{'Active (all)':>13}{'No in-emb':>12}{'No vocab':>10}   Model card"
    )
    for mid, r in res.items():
        m = data[mid]
        card = m.get("card", {})
        cs = f"{card.get('total_params', '—')} / {card.get('active_params', '—')}" if card else "—"
        if r.get("total") is None:
            continue
        if mid == "main":
            # models.json keeps the Chinese name (the video shows it); print the English name
            print(f"{'Main-line model':<20}{'zero (see below)':<28}{r['total'] / B:>9.4f}B{'dense':>13}")
            continue
        dense = experts_k_E(m["config"]) is None
        act = "dense" if dense else f"{r['active'] / B:.2f}B"
        a2 = "" if dense else f"{r['active_no_emb'] / B:.2f}B"
        a3 = "" if dense else f"{r['active_no_vocab'] / B:.2f}B"
        print(
            f"{m['name']:<20}{r['cls']:<28}{r['total'] / B:>9.3f}B{act:>13}{a2:>12}{a3:>10}   {cs}"
        )
    print(f"\nMain-line model: {res['main']['cls']}, zero formula {res['main']['total'] / B:.4f}B")
    k3 = res["kimi-k3"]["latent"]
    print("\nKimi-K3 (routed experts counted by hand, an estimate):")
    print(
        f"  Experts compute in the 3584-dim latent space: {k3['latent'] / 1e12:.2f}T ({k3['active_latent'] / B:.1f}B of it active per token)"
    )
    print(
        f"  If the experts computed directly in 7168 dims: {k3['full'] / 1e12:.2f}T. The model card says 2.8T total / 104B active: "
        "only the latent count agrees, as 'Latent MoE Dimension 3584' on the model card says"
    )
    print(
        "\nNote: the HF classes of DeepSeek-V4 and V3.2 do not include the MTP layer, so the counts are a little smaller than the model card. "
        "The MiniMax-M3 model card gives only approximate numbers (~428B / ~23B); here we build only the language-model part."
    )


if __name__ == "__main__":
    main()
