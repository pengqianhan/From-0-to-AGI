"""KV cache calculator (Chapter 21, "The KV cache ledger").

    uv run python -m zero.tools.kv_cache_calc configs/main/pretrain.toml --seq 32768
    uv run python -m zero.tools.kv_cache_calc path/to/hf/config.json --seq 131072 --batch 8

The basic formula (each layer stores one K and one V for each position):

    KV cache bytes = 2 × layers × KV heads × head_dim × sequence length × bytes per value × batch

In real models, the layers are not all the same. Thus this tool keeps a ledger layer by layer.
It supports these layer types:

| Layer type | What it stores for each position | How it grows with sequence length |
|---|---|---|
| `full` (MHA / GQA / MQA) | `n_kv_heads × head_dim` values for K, and the same for V | Linear growth |
| `mla` (multi-head latent attention: DeepSeek-V2/V3, Kimi K2, GLM-5, Mistral Large 3) | A compressed latent vector `kv_lora_rank` + a shared RoPE key `qk_rope_head_dim` | Linear growth, but much smaller for each position |
| `sliding` (sliding window) | The same as `full` | Only the last `sliding_window` positions; no growth after the window is full |
| `linear` (linear attention layers such as Gated DeltaNet and KDA) | No K/V; only a state matrix of fixed size | No growth (constant) |

The input can be a `zero.config.ModelConfig`, the path of a TOML config, or a Hugging Face style
`config.json` (a dict or a path). The tool also reads the `text_config` that multimodal models
use, and the field names of the native Mistral `params.json`.

What the numbers include: `kv_cache_bytes` counts only the K/V (or latent vectors) that grow with
the sequence. `fixed_state_bytes` gives the fixed state of the linear attention layers separately.
That value is an estimate, because the dtype and the layout of the convolution cache differ
between implementations.
`tests/test_kv_cache_calc.py` makes sure that, for zero configs, the result is equal to the bytes
that `KVCache.nbytes()` really allocates.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zero.config import ModelConfig


@dataclass
class LayerSpec:
    """One attention layer in the KV cache ledger."""

    kind: str  # "full" | "sliding" | "mla" | "linear"
    per_token: int = 0  # values stored for each position (K and V together; for MLA: latent vector + RoPE key)
    window: int | None = None  # sliding: the maximum number of positions to keep
    state: int = 0  # linear: number of values in the fixed state (independent of sequence length)

    def tokens_kept(self, seq_len: int) -> int:
        if self.kind in ("full", "mla"):
            return seq_len
        if self.kind == "sliding":
            assert self.window is not None
            return min(seq_len, self.window)
        return 0


@dataclass
class KVLayout:
    """The ledger of the full model: one LayerSpec for each layer."""

    name: str
    layers: list[LayerSpec]
    state_bytes: int | None = None  # bytes per value of the linear-layer state; None = the same as KV
    notes: list[str] = field(default_factory=list)

    def count(self, kind: str) -> int:
        return sum(1 for s in self.layers if s.kind == kind)


# ---------------------------------------------------------------------------
# Read the config
# ---------------------------------------------------------------------------


def _get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _load(cfg: Any) -> tuple[str, Any]:
    """Path → (name, ModelConfig or dict)."""
    if isinstance(cfg, (str, Path)):
        path = Path(cfg)
        if path.suffix == ".toml":
            with path.open("rb") as f:
                raw = tomllib.load(f)
            if "model" in raw or "base" in raw:
                from zero.config import load_model_config

                return str(path), load_model_config(path)
            return str(path), raw
        return str(path), json.loads(path.read_text(encoding="utf-8"))
    return getattr(cfg, "name", type(cfg).__name__), cfg


def layout_from_config(cfg: Any, name: str | None = None) -> KVLayout:
    """Parse any supported config into a ledger with one entry for each layer."""
    label, cfg = _load(cfg)
    label = name or label
    if isinstance(cfg, ModelConfig):
        assert cfg.head_dim is not None
        per = 2 * cfg.n_kv_heads * cfg.head_dim
        return KVLayout(label, [LayerSpec("full", per) for _ in range(cfg.n_layers)])
    if not isinstance(cfg, dict):
        raise TypeError(f"Unsupported config type {type(cfg)}")
    d = cfg.get("text_config") or cfg  # multimodal models put the language-model config in text_config
    n_layers = int(_get(d, "num_hidden_layers", "n_layers"))
    n_heads = int(_get(d, "num_attention_heads", "n_heads"))
    n_kv = int(_get(d, "num_key_value_heads", "n_kv_heads", default=n_heads))
    hidden = _get(d, "hidden_size", "dim")
    head_dim = int(_get(d, "head_dim", default=(hidden // n_heads) if hidden else 0))
    notes: list[str] = []

    # How the attention layer stores data: MLA if kv_lora_rank is set, else normal K/V heads
    if _get(d, "kv_lora_rank"):
        rank, rope = int(d["kv_lora_rank"]), int(_get(d, "qk_rope_head_dim", default=0))
        attn = LayerSpec("mla", rank + rope)
        notes.append(f"MLA: each layer stores, for each position, a latent vector of {rank} + a RoPE key of {rope} = {rank + rope} values")
    else:
        attn = LayerSpec("full", 2 * n_kv * head_dim)

    # Fixed state of the linear attention layers (estimate: recurrent state matrix + short convolution cache)
    lin_state = 0
    state_bytes = None
    if "linear_num_value_heads" in d:  # Gated DeltaNet of Qwen3-Next / Qwen3.5
        hv, hk = int(d["linear_num_value_heads"]), int(d["linear_num_key_heads"])
        dk, dv = int(d["linear_key_head_dim"]), int(d["linear_value_head_dim"])
        conv = int(_get(d, "linear_conv_kernel_dim", default=4))
        lin_state = hv * dk * dv + (conv - 1) * (2 * hk * dk + hv * dv)
        if str(d.get("mamba_ssm_dtype", "")) == "float32":
            state_bytes = 4
    lac = d.get("linear_attn_config")
    if lac:  # KDA of Kimi Linear / Kimi K3
        h, hd = int(lac["num_heads"]), int(lac["head_dim"])
        conv = int(lac.get("short_conv_kernel_size", 4))
        lin_state = h * hd * hd + (conv - 1) * 3 * h * hd

    # The type of each layer
    window = _get(d, "sliding_window")
    layer_types = d.get("layer_types")
    if layer_types:
        kinds = [str(t) for t in layer_types][:n_layers]
    elif lac and lac.get("full_attn_layers"):
        full = {int(i) for i in lac["full_attn_layers"]}  # Kimi counts layers from 1
        kinds = ["full_attention" if i + 1 in full else "linear_attention" for i in range(n_layers)]
    elif window and d.get("use_sliding_window", True):
        kinds = ["sliding_attention"] * n_layers  # for example Mistral-7B-v0.1: all layers use a sliding window
    else:
        kinds = ["full_attention"] * n_layers

    layers: list[LayerSpec] = []
    for t in kinds:
        if "linear" in t:
            layers.append(LayerSpec("linear", state=lin_state))
        elif "sliding" in t or "local" in t:
            layers.append(LayerSpec("sliding", attn.per_token, window=int(window)))
        else:
            layers.append(LayerSpec(attn.kind, attn.per_token))
    return KVLayout(label, layers, state_bytes, notes)


# ---------------------------------------------------------------------------
# Count the bytes
# ---------------------------------------------------------------------------


def _as_layout(cfg: Any) -> KVLayout:
    return cfg if isinstance(cfg, KVLayout) else layout_from_config(cfg)


def kv_cache_bytes(cfg: Any, seq_len: int, batch: int = 1, dtype_bytes: int = 2) -> int:
    """Total bytes of the KV cache that grows with the sequence, at length seq_len.

    This includes MLA latent vectors and sliding windows. It does not include the fixed state of
    linear attention layers (see `fixed_state_bytes`). For a zero ModelConfig, the result is equal
    to `KVCache.from_config(cfg, batch, seq_len, dtype=...).nbytes()`.
    """
    lay = _as_layout(cfg)
    elems = sum(s.per_token * s.tokens_kept(seq_len) for s in lay.layers)
    return elems * batch * dtype_bytes


def kv_bytes_per_token(cfg: Any, dtype_bytes: int = 2) -> int:
    """Bytes that the KV cache adds for each new token (before all windows are full)."""
    lay = _as_layout(cfg)
    return (
        sum(s.per_token for s in lay.layers if s.kind in ("full", "mla", "sliding")) * dtype_bytes
    )


def fixed_state_bytes(cfg: Any, batch: int = 1, dtype_bytes: int = 2) -> int:
    """Fixed state of the linear attention layers (estimate). It does not depend on sequence length."""
    lay = _as_layout(cfg)
    b = lay.state_bytes or dtype_bytes
    return sum(s.state for s in lay.layers if s.kind == "linear") * batch * b


def breakdown(cfg: Any, seq_len: int, batch: int = 1, dtype_bytes: int = 2) -> dict[str, Any]:
    lay = _as_layout(cfg)
    kinds = {}
    for k in ("full", "mla", "sliding", "linear"):
        n = lay.count(k)
        if n:
            sub = KVLayout(lay.name, [s for s in lay.layers if s.kind == k], lay.state_bytes)
            kinds[k] = {
                "layers": n,
                "bytes": kv_cache_bytes(sub, seq_len, batch, dtype_bytes),
                "state_bytes": fixed_state_bytes(sub, batch, dtype_bytes),
            }
    return {
        "name": lay.name,
        "seq_len": seq_len,
        "batch": batch,
        "dtype_bytes": dtype_bytes,
        "per_token": kv_bytes_per_token(lay, dtype_bytes),
        "kv_bytes": kv_cache_bytes(lay, seq_len, batch, dtype_bytes),
        "state_bytes": fixed_state_bytes(lay, batch, dtype_bytes),
        "by_kind": kinds,
        "notes": lay.notes,
    }


def fmt_bytes(n: float) -> str:
    for unit, k in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if n >= k:
            return f"{n / k:.2f} {unit}"
    return f"{n:.0f} B"


_KIND_ZH = {"full": "full", "mla": "MLA", "sliding": "window", "linear": "linear"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Calculate the KV cache memory from a config")
    ap.add_argument("configs", nargs="+", help="a zero .toml, or a Hugging Face style config.json")
    ap.add_argument("--seq", type=int, default=32768, help="sequence length (context + generated tokens)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dtype-bytes", type=int, default=2, help="bytes per value: BF16=2, FP8=1")
    args = ap.parse_args(argv)
    for path in args.configs:
        r = breakdown(path, args.seq, args.batch, args.dtype_bytes)
        print(f"{r['name']}  (seq={args.seq:,}, batch={args.batch}, {args.dtype_bytes} bytes/value)")
        print(f"  Per token: {r['per_token']:,} bytes = {fmt_bytes(r['per_token'])}")
        for k, v in r["by_kind"].items():
            extra = f", fixed state about {fmt_bytes(v['state_bytes'])}" if v["state_bytes"] else ""
            print(f"  {_KIND_ZH[k]:<6} {v['layers']:3d} layers: KV {fmt_bytes(v['bytes'])}{extra}")
        print(f"  KV cache total: {r['kv_bytes']:,} bytes = {fmt_bytes(r['kv_bytes'])}")
        if r["state_bytes"]:
            print(f"  Fixed state of linear layers (estimate, does not grow with the sequence): {fmt_bytes(r['state_bytes'])}")
        for note in r["notes"]:
            print(f"  Note: {note}")


if __name__ == "__main__":
    main()
