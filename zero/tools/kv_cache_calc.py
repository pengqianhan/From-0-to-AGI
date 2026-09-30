"""KV cache 计算器（对应第 21 章"KV cache 的账本"）。

    uv run python -m zero.tools.kv_cache_calc configs/main/pretrain.toml --seq 32768
    uv run python -m zero.tools.kv_cache_calc path/to/hf/config.json --seq 131072 --batch 8

基本公式（每层、每个位置存一份 K 和一份 V）：

    KV cache 字节数 = 2 × 层数 × KV 头数 × head_dim × 序列长 × 每个数的字节数 × batch

真实模型不是每层都一样，这里按层逐个记账，支持五种层：

| 层类型 | 每个位置存什么 | 随序列长怎么涨 |
|---|---|---|
| `full`（MHA / GQA / MQA） | K、V 各 `n_kv_heads × head_dim` 个数 | 线性增长 |
| `mla`（多头潜在注意力，DeepSeek-V2/V3、Kimi K2、GLM-5、Mistral Large 3） | 压缩后的潜向量 `kv_lora_rank` + 共享的 RoPE key `qk_rope_head_dim` | 线性增长，但每个位置小得多 |
| `sliding`（滑动窗口） | 同 `full` | 只存最近 `sliding_window` 个位置，到窗口后不再增长 |
| `linear`（Gated DeltaNet、KDA 等线性注意力层） | 不存 K/V，只存一个固定大小的状态矩阵 | 不增长（常数） |

输入可以是 `zero.config.ModelConfig`、TOML 配置路径、Hugging Face 风格的 `config.json`
（字典或路径，支持多模态模型的 `text_config` 嵌套、Mistral 原生 `params.json` 的字段名）。

数字口径：`kv_cache_bytes` 只算随序列增长的 K/V（或潜向量）；线性注意力层的固定状态单独
由 `fixed_state_bytes` 给出（它的 dtype、卷积缓存的写法因实现而异，属于估算）。
`tests/test_kv_cache_calc.py` 保证 zero 配置的结果与 `KVCache.nbytes()` 真实分配的字节数一致。
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
    """一层注意力在 KV cache 账本里的样子。"""

    kind: str  # "full" | "sliding" | "mla" | "linear"
    per_token: int = 0  # 每个位置要存多少个数（K、V 合计；MLA 是潜向量 + RoPE key）
    window: int | None = None  # sliding：最多存多少个位置
    state: int = 0  # linear：固定状态有多少个数（与序列长无关）

    def tokens_kept(self, seq_len: int) -> int:
        if self.kind in ("full", "mla"):
            return seq_len
        if self.kind == "sliding":
            assert self.window is not None
            return min(seq_len, self.window)
        return 0


@dataclass
class KVLayout:
    """整个模型的账本：每层一条 LayerSpec。"""

    name: str
    layers: list[LayerSpec]
    state_bytes: int | None = None  # 线性层状态的每个数占几字节；None = 与 KV 相同
    notes: list[str] = field(default_factory=list)

    def count(self, kind: str) -> int:
        return sum(1 for s in self.layers if s.kind == kind)


# ---------------------------------------------------------------------------
# 读配置
# ---------------------------------------------------------------------------


def _get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _load(cfg: Any) -> tuple[str, Any]:
    """路径 → (名字, ModelConfig 或 dict)。"""
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
    """把任意支持的配置解析成逐层账本。"""
    label, cfg = _load(cfg)
    label = name or label
    if isinstance(cfg, ModelConfig):
        assert cfg.head_dim is not None
        per = 2 * cfg.n_kv_heads * cfg.head_dim
        return KVLayout(label, [LayerSpec("full", per) for _ in range(cfg.n_layers)])
    if not isinstance(cfg, dict):
        raise TypeError(f"不支持的配置类型 {type(cfg)}")
    d = cfg.get("text_config") or cfg  # 多模态模型把语言模型配置放在 text_config 里
    n_layers = int(_get(d, "num_hidden_layers", "n_layers"))
    n_heads = int(_get(d, "num_attention_heads", "n_heads"))
    n_kv = int(_get(d, "num_key_value_heads", "n_kv_heads", default=n_heads))
    hidden = _get(d, "hidden_size", "dim")
    head_dim = int(_get(d, "head_dim", default=(hidden // n_heads) if hidden else 0))
    notes: list[str] = []

    # 注意力层怎么存：MLA 看 kv_lora_rank，否则是普通的 K/V 头
    if _get(d, "kv_lora_rank"):
        rank, rope = int(d["kv_lora_rank"]), int(_get(d, "qk_rope_head_dim", default=0))
        attn = LayerSpec("mla", rank + rope)
        notes.append(f"MLA：每层每位置存潜向量 {rank} + RoPE key {rope} = {rank + rope} 个数")
    else:
        attn = LayerSpec("full", 2 * n_kv * head_dim)

    # 线性注意力层的固定状态（估算：递推状态矩阵 + 短卷积缓存）
    lin_state = 0
    state_bytes = None
    if "linear_num_value_heads" in d:  # Qwen3-Next / Qwen3.5 的 Gated DeltaNet
        hv, hk = int(d["linear_num_value_heads"]), int(d["linear_num_key_heads"])
        dk, dv = int(d["linear_key_head_dim"]), int(d["linear_value_head_dim"])
        conv = int(_get(d, "linear_conv_kernel_dim", default=4))
        lin_state = hv * dk * dv + (conv - 1) * (2 * hk * dk + hv * dv)
        if str(d.get("mamba_ssm_dtype", "")) == "float32":
            state_bytes = 4
    lac = d.get("linear_attn_config")
    if lac:  # Kimi Linear / Kimi K3 的 KDA
        h, hd = int(lac["num_heads"]), int(lac["head_dim"])
        conv = int(lac.get("short_conv_kernel_size", 4))
        lin_state = h * hd * hd + (conv - 1) * 3 * h * hd

    # 每层是什么类型
    window = _get(d, "sliding_window")
    layer_types = d.get("layer_types")
    if layer_types:
        kinds = [str(t) for t in layer_types][:n_layers]
    elif lac and lac.get("full_attn_layers"):
        full = {int(i) for i in lac["full_attn_layers"]}  # Kimi 的层号从 1 开始
        kinds = ["full_attention" if i + 1 in full else "linear_attention" for i in range(n_layers)]
    elif window and d.get("use_sliding_window", True):
        kinds = ["sliding_attention"] * n_layers  # 如 Mistral-7B-v0.1：全部层都是滑动窗口
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
# 记账
# ---------------------------------------------------------------------------


def _as_layout(cfg: Any) -> KVLayout:
    return cfg if isinstance(cfg, KVLayout) else layout_from_config(cfg)


def kv_cache_bytes(cfg: Any, seq_len: int, batch: int = 1, dtype_bytes: int = 2) -> int:
    """序列长 seq_len 时，随序列增长的 KV cache（含 MLA 潜向量、滑动窗口）总字节数。

    不含线性注意力层的固定状态（见 `fixed_state_bytes`）。对 zero 的 ModelConfig，
    结果等于 `KVCache.from_config(cfg, batch, seq_len, dtype=...).nbytes()`。
    """
    lay = _as_layout(cfg)
    elems = sum(s.per_token * s.tokens_kept(seq_len) for s in lay.layers)
    return elems * batch * dtype_bytes


def kv_bytes_per_token(cfg: Any, dtype_bytes: int = 2) -> int:
    """每多一个 token（在所有窗口都填满之前），KV cache 增加多少字节。"""
    lay = _as_layout(cfg)
    return (
        sum(s.per_token for s in lay.layers if s.kind in ("full", "mla", "sliding")) * dtype_bytes
    )


def fixed_state_bytes(cfg: Any, batch: int = 1, dtype_bytes: int = 2) -> int:
    """线性注意力层的固定状态（估算），与序列长无关。"""
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


_KIND_ZH = {"full": "全注意力", "mla": "MLA", "sliding": "滑动窗口", "linear": "线性注意力"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="按配置计算 KV cache 显存")
    ap.add_argument("configs", nargs="+", help="zero 的 .toml 或 Hugging Face 风格的 config.json")
    ap.add_argument("--seq", type=int, default=32768, help="序列长度（上下文 + 生成）")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dtype-bytes", type=int, default=2, help="每个数几字节：BF16=2，FP8=1")
    args = ap.parse_args(argv)
    for path in args.configs:
        r = breakdown(path, args.seq, args.batch, args.dtype_bytes)
        print(f"{r['name']}  (seq={args.seq:,}, batch={args.batch}, {args.dtype_bytes} 字节/数)")
        print(f"  每 token：{r['per_token']:,} 字节 = {fmt_bytes(r['per_token'])}")
        for k, v in r["by_kind"].items():
            extra = f"，固定状态约 {fmt_bytes(v['state_bytes'])}" if v["state_bytes"] else ""
            print(f"  {_KIND_ZH[k]:<6} {v['layers']:3d} 层：KV {fmt_bytes(v['bytes'])}{extra}")
        print(f"  KV cache 合计：{r['kv_bytes']:,} 字节 = {fmt_bytes(r['kv_bytes'])}")
        if r["state_bytes"]:
            print(f"  线性层固定状态（估算，不随序列增长）：{fmt_bytes(r['state_bytes'])}")
        for note in r["notes"]:
            print(f"  注：{note}")


if __name__ == "__main__":
    main()
