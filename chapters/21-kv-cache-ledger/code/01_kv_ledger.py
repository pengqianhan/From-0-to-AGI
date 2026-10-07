"""Chapter 21 · Minimal code 1: the KV cache ledger (plain Python, no libraries).

The basic formula (Chapter 10): each layer stores one K and one V at each position.
    KV cache bytes = 2 × layers × KV heads × head_dim × sequence length × bytes (× batch)
In real models, the layers are not all the same. Thus we keep the ledger layer by layer:
    Full-attention layer (MHA/GQA/MQA): 2 × KV heads × head_dim numbers per position.
        It grows linearly with the sequence.
    MLA layer: kv_lora_rank + qk_rope_head_dim numbers per position
        (the compressed latent vector + the shared RoPE key).
    Sliding-window layer: the same as full attention, but it keeps at most `window` positions.
    Linear-attention layer: it stores no K/V, only a state of fixed size.
        (Here we count only the part that grows with the sequence.)

All model numbers come from the config.json in the Hugging Face repository of each model
(read in 2026-09). The README has the links.
Run: uv run python chapters/21-kv-cache-ledger/code/01_kv_ledger.py
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BF16 = 2


def layer_list(m: dict) -> list[tuple[str, int, int | None]]:
    """Expand a model description into one (kind, numbers per position, window) per layer."""
    if m.get("kv_lora_rank"):  # MLA
        per = m["kv_lora_rank"] + m["qk_rope_head_dim"]
        kind = "mla"
    else:  # MHA / GQA / MQA
        per = 2 * m["kv_heads"] * m["head_dim"]
        kind = "full"
    out = []
    for t in m.get("types") or [kind] * m["layers"]:
        if t == "linear":
            out.append(("linear", 0, None))
        elif t == "sliding":
            out.append(("sliding", per, m["window"]))
        else:
            out.append((kind, per, None))
    return out


def kv_bytes(m: dict, seq_len: int, batch: int = 1, nbytes: int = BF16) -> int:
    total = 0
    for kind, per, window in layer_list(m):
        kept = min(seq_len, window) if kind == "sliding" else seq_len  # sliding window: keep at most `window` positions
        total += per * kept  # linear layer: per = 0, so it does not grow with the sequence
    return total * batch * nbytes


def per_token(m: dict, nbytes: int = BF16) -> int:
    return sum(per for _, per, _ in layer_list(m)) * nbytes


def fmt(n: float) -> str:
    for unit, k in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if n >= k:
            return f"{n / k:.2f} {unit}"
    return f"{n:.0f} B"


def main_model() -> dict:
    with open(ROOT / "configs" / "main" / "pretrain.toml", "rb") as f:
        c = tomllib.load(f)["model"]
    return dict(layers=c["n_layers"], heads=c["n_heads"], kv_heads=c["n_kv_heads"],
                head_dim=c["head_dim"])


def mixed(n_layers: int, full_every: int) -> list[str]:
    """One full-attention layer in each full_every layers; the others are linear attention.

    Qwen3.5 uses full_attention_interval = 4.
    """
    return ["full" if (i + 1) % full_every == 0 else "linear" for i in range(n_layers)]


# The KV cache fields from the config.json of each model (read in 2026-09).
# The source table in the README has the links.
MODELS = {
    "Qwen3-0.6B": dict(layers=28, heads=16, kv_heads=8, head_dim=128),
    "Qwen3-8B": dict(layers=36, heads=32, kv_heads=8, head_dim=128),
    "Llama-3.1-8B": dict(layers=32, heads=32, kv_heads=8, head_dim=128),
    # gpt-oss: in layer_types, sliding-window (128) and full-attention layers alternate
    "gpt-oss-120b": dict(layers=36, heads=64, kv_heads=8, head_dim=64, window=128,
                         types=["sliding", "full"] * 18),
    # Qwen3.5: 3 Gated DeltaNet layers (linear attention) + 1 full-attention layer, repeated
    "Qwen3.5-0.8B": dict(layers=24, heads=8, kv_heads=2, head_dim=256, types=mixed(24, 4)),
    "Qwen3.5-9B": dict(layers=32, heads=16, kv_heads=4, head_dim=256, types=mixed(32, 4)),
    "Qwen3.5-397B-A17B": dict(layers=60, heads=32, kv_heads=2, head_dim=256, types=mixed(60, 4)),
    # MLA: kv_lora_rank 512 + qk_rope_head_dim 64 = 576
    "DeepSeek-V3/V3.2": dict(layers=61, heads=128, kv_lora_rank=512, qk_rope_head_dim=64),
    "Kimi-K2": dict(layers=61, heads=64, kv_lora_rank=512, qk_rope_head_dim=64),
    "GLM-5": dict(layers=78, heads=64, kv_lora_rank=512, qk_rope_head_dim=64),
    "Mistral-Large-3": dict(layers=61, heads=128, kv_lora_rank=512, qk_rope_head_dim=64),
    # Kimi K3: of 93 layers, 24 are MLA (full_attn_layers) and 69 are KDA linear attention
    "Kimi-K3": dict(layers=93, heads=96, kv_lora_rank=512, qk_rope_head_dim=64,
                    types=["linear" if i not in set(range(4, 93, 4)) | {93} else "mla"
                           for i in range(1, 94)]),
}


if __name__ == "__main__":
    m = main_model()
    L, H, D = m["layers"], m["heads"], m["head_dim"]
    print(f"1. Main-line model configs/main: {L} layers, {H} query heads, {m['kv_heads']} KV heads, "
          f"head_dim {D}, BF16")
    print("   Variant                 Per layer & pos. Per token        32K       128K")
    variants = [
        ("MHA (16 KV heads)", dict(m, kv_heads=H)),
        ("GQA (main-line, 8 KV)", m),
        ("MQA (1 KV head)", dict(m, kv_heads=1)),
        ("Assumed MLA (512+64)", dict(layers=L, kv_lora_rank=512, qk_rope_head_dim=64)),
    ]
    for name, v in variants:
        elems = layer_list(v)[0][1]
        print(f"   {name:22} {elems:6d} numbers   {fmt(per_token(v)):>10} {fmt(kv_bytes(v, 32768)):>10}"
              f" {fmt(kv_bytes(v, 131072)):>10}")

    print("\n2. Public models (BF16, batch 1; only the part that grows with the sequence)")
    print("   Model               Growing/total     Per token        32K       128K   "
          "All-MHA 128K")
    for name, mm in MODELS.items():
        grow = sum(1 for k, _, _ in layer_list(mm) if k != "linear")
        # Reference: the same layers and heads, but MHA in every layer and head_dim 128.
        # MLA models have K heads of width 192; here we use 128 to be conservative.
        mha = dict(layers=mm["layers"], kv_heads=mm["heads"], head_dim=mm.get("head_dim", 128))
        print(f"   {name:19} {grow:4d} / {mm['layers']:<4d}      {fmt(per_token(mm)):>10} "
              f"{fmt(kv_bytes(mm, 32768)):>10} {fmt(kv_bytes(mm, 131072)):>10}   "
              f"{fmt(kv_bytes(mha, 131072)):>10}")

    print("\n3. Parity check: compare with the production code zero/tools/kv_cache_calc.py")
    sys.path.insert(0, str(ROOT))
    from zero.tools.kv_cache_calc import kv_cache_bytes

    ok = kv_cache_bytes(ROOT / "configs/main/pretrain.toml", 32768) == kv_bytes(m, 32768)
    print(f"   Main-line model 32K: minimal {kv_bytes(m, 32768):,} bytes, zero {kv_cache_bytes(ROOT / 'configs/main/pretrain.toml', 32768):,} bytes, same: {ok}")
    ds = {"num_hidden_layers": 61, "num_attention_heads": 128, "kv_lora_rank": 512,
          "qk_rope_head_dim": 64}
    same = kv_cache_bytes(ds, 131072) == kv_bytes(MODELS["DeepSeek-V3/V3.2"], 131072)
    print(f"   DeepSeek-V3 128K: both give the same result: {same}")
