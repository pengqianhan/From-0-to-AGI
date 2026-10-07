"""Chapter 20 · Minimal code 3: how much memory does the main-line model need on a laptop?

    weight memory ≈ number of parameters × bits per parameter / 8

Read configs/main/pretrain.toml (the provisional shape of the main-line model). Count the parameters
of each tensor. Then calculate the file size in several formats:

- fp32 / bf16 (fp16): 32 / 16 bits;
- Q8_0: one block for each 32 numbers, 32 int8 + 1 fp16 scale = 34 bytes → 8.5 bits;
- Q4_K: one super-block for each 256 numbers = 144 bytes → 4.5 bits; Q6_K: 210 bytes for 256 numbers
  → 6.5625 bits (for the block layouts, see ggml/src/ggml-common.h in llama.cpp);
- Q4_K_M: not all tensors use Q4_K. The rule of llama.cpp src/llama-quant.cpp (dense models):
  the output layer (the token embedding when the embedding is shared) uses Q6_K. attn_v and ffn_down
  use Q6_K in the "use more bits" layers: the first 1/8 of the layers, the last 1/8 of the layers,
  and every third layer in the middle. All other matrices use Q4_K. 1D norm weights stay in f32.

Then add the KV cache for inference (Chapter 21): 2 × layers × KV heads × head_dim × sequence length
× 2 bytes (fp16). A GGUF file also contains the tokenizer, metadata, and more. These add a small
amount (a few MB) to the weights. We do not count them here.

Run: uv run python chapters/20-release/code/03_memory_calculator.py [config file]
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CFG = ROOT / "configs" / "main" / "pretrain.toml"

BPW = {"f32": 32.0, "bf16": 16.0, "q8_0": 34 * 8 / 32, "q6_k": 210 * 8 / 256, "q4_k": 144 * 8 / 256}


def tensors(m: dict) -> list[tuple[str, int, int]]:
    """(name, layer index or -1, number of elements). The names are the tensor names in GGUF."""
    d, L, hd = m["dim"], m["n_layers"], m.get("head_dim") or m["dim"] // m["n_heads"]
    q, kv, f = m["n_heads"] * hd, m["n_kv_heads"] * hd, m["ffn_dim"]
    out = [("token_embd", -1, m["vocab_size"] * d), ("output_norm", -1, d)]
    if not m.get("tie_embeddings", True):
        out.append(("output", -1, m["vocab_size"] * d))
    for i in range(L):
        out += [
            ("attn_norm", i, d), ("attn_q", i, d * q), ("attn_k", i, d * kv), ("attn_v", i, d * kv),
            ("attn_output", i, q * d), ("attn_q_norm", i, hd), ("attn_k_norm", i, hd),
            ("ffn_norm", i, d), ("ffn_gate", i, d * f), ("ffn_up", i, d * f), ("ffn_down", i, f * d),
        ]
    return out


def use_more_bits(i: int, n: int) -> bool:
    """The llama.cpp rule: the first 1/8, the last 1/8, and every third layer in the middle."""
    return i < n // 8 or i >= 7 * n // 8 or (i - n // 8) % 3 == 2


def q4_k_m_type(name: str, layer: int, n_layers: int, tied: bool) -> str:
    if name.endswith("norm"):
        return "f32"
    if name == "output" or (name == "token_embd" and tied):
        return "q6_k"
    if name in ("attn_v", "ffn_down") and use_more_bits(layer, n_layers):
        return "q6_k"
    return "q4_k"


def file_bytes(m: dict, fmt: str) -> float:
    total = 0.0
    for name, layer, n in tensors(m):
        if name.endswith("norm"):
            t = "f32"                                   # 1D weights stay in f32 in all formats
        elif fmt == "q4_k_m":
            t = q4_k_m_type(name, layer, m["n_layers"], m.get("tie_embeddings", True))
        else:
            t = fmt
        total += n * BPW[t] / 8
    return total


def kv_cache_bytes(m: dict, seq_len: int, bytes_per: int = 2) -> int:
    hd = m.get("head_dim") or m["dim"] // m["n_heads"]
    return 2 * m["n_layers"] * m["n_kv_heads"] * hd * seq_len * bytes_per


def load_model_section(path: Path) -> dict:
    """Read the [model] section. Support `base = "xxx.toml"` inheritance in configs (the child file overrides the parent)."""
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    m = load_model_section(path.parent / cfg["base"]) if "base" in cfg else {}
    return {**m, **cfg.get("model", {})}


def main() -> None:
    cfg_path = (Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CFG).resolve()
    m = load_model_section(cfg_path)
    n = sum(k for _, _, k in tensors(m))
    emb = m["vocab_size"] * m["dim"]
    print(f"Config {cfg_path.relative_to(ROOT)}: dim={m['dim']}, {m['n_layers']} layers, "
          f"{m['n_heads']} query heads / {m['n_kv_heads']} KV heads, vocabulary {m['vocab_size']}")
    print(f"Parameters {n / 1e6:.1f}M (embedding {emb / 1e6:.1f}M)\n")

    GiB = 2**30
    print(f"{'Format':<10}{'bits/param':>10}{'Weight size':>12}")
    sizes = {}
    for fmt in ("f32", "bf16", "q8_0", "q4_k_m", "q4_k"):
        b = file_bytes(m, fmt)
        sizes[fmt] = b
        print(f"{fmt:<10}{b * 8 / n:>10.2f}{b / GiB:>10.2f} GiB")
    print("(q4_k is the lower bound 'all matrices use Q4_K'. The release uses Q4_K_M.)\n")

    print("KV cache (fp16):")
    for T in (4096, 32768):
        print(f"  sequence length {T:>6}: {kv_cache_bytes(m, T) / GiB:.2f} GiB")
    kv_per_tok = kv_cache_bytes(m, 1)
    print(f"  per token {kv_per_tok / 1024:.0f} KiB\n")

    print("Minimum memory for inference (rough: weights + KV cache for a 4K context; no compute buffers or runtime overhead):")
    for fmt in ("bf16", "q8_0", "q4_k_m"):
        tot = sizes[fmt] + kv_cache_bytes(m, 4096)
        print(f"  {fmt:<7} {tot / GiB:.2f} GiB")


if __name__ == "__main__":
    main()
