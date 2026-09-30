"""第 20 章 · 极简代码 3：主线模型放进笔记本要多少内存？

    权重内存 ≈ 参数量 × 每个参数的 bit 数 / 8

读 configs/main/pretrain.toml（主线模型的暂定形状），逐个张量算参数量，再按几种格式算文件大小：

- fp32 / bf16（fp16）：32 / 16 bit；
- Q8_0：每 32 个数一块，32 个 int8 + 1 个 fp16 scale = 34 字节 → 8.5 bit；
- Q4_K：每 256 个数一个超级块 = 144 字节 → 4.5 bit；Q6_K：256 个数 210 字节 → 6.5625 bit
  （块结构见 llama.cpp 的 ggml/src/ggml-common.h）；
- Q4_K_M：不是所有张量都用 Q4_K。按 llama.cpp src/llama-quant.cpp 的规则（稠密模型）：
  输出层（共享 embedding 时就是 token embedding）用 Q6_K；attn_v 和 ffn_down 在"要多给 bit 的层"
  用 Q6_K——前 1/8 的层、后 1/8 的层、中间每 3 层一个；其余矩阵 Q4_K；一维的 norm 权重保持 f32。

再加上推理时的 KV cache（第 21 章）：2 × 层数 × KV 头数 × head_dim × 序列长 × 2 字节（fp16）。
GGUF 文件里还有分词器、元数据等，比纯权重多一点点（MB 级），这里不计。

运行：uv run python chapters/20-release/code/03_memory_calculator.py [配置文件]
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CFG = ROOT / "configs" / "main" / "pretrain.toml"

BPW = {"f32": 32.0, "bf16": 16.0, "q8_0": 34 * 8 / 32, "q6_k": 210 * 8 / 256, "q4_k": 144 * 8 / 256}


def tensors(m: dict) -> list[tuple[str, int, int]]:
    """(名字, 层号或 -1, 元素个数)。名字用 GGUF 里的叫法。"""
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
    """llama.cpp 的规则：前 1/8、后 1/8，以及中间每 3 层一个。"""
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
            t = "f32"                                   # 一维权重在各种格式里都保持 f32
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
    """读 [model] 小节；支持 configs 里的 `base = "xxx.toml"` 继承（子文件覆盖父文件）。"""
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    m = load_model_section(path.parent / cfg["base"]) if "base" in cfg else {}
    return {**m, **cfg.get("model", {})}


def main() -> None:
    cfg_path = (Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CFG).resolve()
    m = load_model_section(cfg_path)
    n = sum(k for _, _, k in tensors(m))
    emb = m["vocab_size"] * m["dim"]
    print(f"配置 {cfg_path.relative_to(ROOT)}：dim={m['dim']}、{m['n_layers']} 层、"
          f"{m['n_heads']} 个查询头 / {m['n_kv_heads']} 个 KV 头、词表 {m['vocab_size']}")
    print(f"参数量 {n / 1e6:.1f}M（其中 embedding {emb / 1e6:.1f}M）\n")

    GiB = 2**30
    print(f"{'格式':<10}{'bit/参数':>10}{'权重大小':>12}")
    sizes = {}
    for fmt in ("f32", "bf16", "q8_0", "q4_k_m", "q4_k"):
        b = file_bytes(m, fmt)
        sizes[fmt] = b
        print(f"{fmt:<10}{b * 8 / n:>10.2f}{b / GiB:>10.2f} GiB")
    print("（q4_k 是'全部矩阵都用 Q4_K'的下限；实际发布用 Q4_K_M。）\n")

    print("KV cache（fp16）：")
    for T in (4096, 32768):
        print(f"  序列长 {T:>6}：{kv_cache_bytes(m, T) / GiB:.2f} GiB")
    kv_per_tok = kv_cache_bytes(m, 1)
    print(f"  每个 token {kv_per_tok / 1024:.0f} KiB\n")

    print("推理时的最少内存（粗算：权重 + 4K 上下文的 KV cache，不含计算缓冲区和运行时开销）：")
    for fmt in ("bf16", "q8_0", "q4_k_m"):
        tot = sizes[fmt] + kv_cache_bytes(m, 4096)
        print(f"  {fmt:<7} {tot / GiB:.2f} GiB")


if __name__ == "__main__":
    main()
