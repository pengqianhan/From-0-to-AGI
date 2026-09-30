"""第 21 章 · 极简代码 1：KV cache 的账本（纯 Python，不依赖任何库）

基本公式（第 10 章）：每层、每个位置存一份 K 和一份 V
    KV cache 字节数 = 2 × 层数 × KV 头数 × head_dim × 序列长 × 字节数（× batch）
真实模型每层不一样，所以按层记账：
    全注意力层（MHA/GQA/MQA）：每位置 2 × KV 头数 × head_dim 个数，随序列线性增长
    MLA 层：每位置 kv_lora_rank + qk_rope_head_dim 个数（压缩后的潜向量 + 共享 RoPE key）
    滑动窗口层：同全注意力，但最多只存 window 个位置
    线性注意力层：不存 K/V，只有固定大小的状态（这里只算随序列增长的部分）

模型数字全部抄自各模型 Hugging Face 仓库的 config.json（2026-09 读取），链接见 README。
运行：uv run python chapters/21-kv-cache-ledger/code/01_kv_ledger.py
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BF16 = 2


def layer_list(m: dict) -> list[tuple[str, int, int | None]]:
    """把一个模型描述展开成逐层的 (类型, 每位置存几个数, 窗口)。"""
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
        kept = min(seq_len, window) if kind == "sliding" else seq_len  # 滑动窗口：最多存 window 个
        total += per * kept  # 线性层 per = 0：不随序列增长
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
    """每 full_every 层一个全注意力层，其余是线性注意力（Qwen3.5：full_attention_interval = 4）。"""
    return ["full" if (i + 1) % full_every == 0 else "linear" for i in range(n_layers)]


# 各模型 config.json 里与 KV cache 有关的字段（2026-09 读取；链接见 README 的来源表）
MODELS = {
    "Qwen3-0.6B": dict(layers=28, heads=16, kv_heads=8, head_dim=128),
    "Qwen3-8B": dict(layers=36, heads=32, kv_heads=8, head_dim=128),
    "Llama-3.1-8B": dict(layers=32, heads=32, kv_heads=8, head_dim=128),
    # gpt-oss：layer_types 里滑动窗口（128）与全注意力逐层交替
    "gpt-oss-120b": dict(layers=36, heads=64, kv_heads=8, head_dim=64, window=128,
                         types=["sliding", "full"] * 18),
    # Qwen3.5：3 层 Gated DeltaNet（线性注意力）+ 1 层全注意力，循环
    "Qwen3.5-0.8B": dict(layers=24, heads=8, kv_heads=2, head_dim=256, types=mixed(24, 4)),
    "Qwen3.5-9B": dict(layers=32, heads=16, kv_heads=4, head_dim=256, types=mixed(32, 4)),
    "Qwen3.5-397B-A17B": dict(layers=60, heads=32, kv_heads=2, head_dim=256, types=mixed(60, 4)),
    # MLA：kv_lora_rank 512 + qk_rope_head_dim 64 = 576
    "DeepSeek-V3/V3.2": dict(layers=61, heads=128, kv_lora_rank=512, qk_rope_head_dim=64),
    "Kimi-K2": dict(layers=61, heads=64, kv_lora_rank=512, qk_rope_head_dim=64),
    "GLM-5": dict(layers=78, heads=64, kv_lora_rank=512, qk_rope_head_dim=64),
    "Mistral-Large-3": dict(layers=61, heads=128, kv_lora_rank=512, qk_rope_head_dim=64),
    # Kimi K3：93 层里 24 层 MLA（full_attn_layers），69 层 KDA 线性注意力
    "Kimi-K3": dict(layers=93, heads=96, kv_lora_rank=512, qk_rope_head_dim=64,
                    types=["linear" if i not in set(range(4, 93, 4)) | {93} else "mla"
                           for i in range(1, 94)]),
}


if __name__ == "__main__":
    m = main_model()
    L, H, D = m["layers"], m["heads"], m["head_dim"]
    print(f"1. 主线模型 configs/main：{L} 层，{H} 个查询头，{m['kv_heads']} 个 KV 头，"
          f"head_dim {D}，BF16")
    print("   方案                     每层每位置存   每 token      32K        128K")
    variants = [
        ("MHA（16 个 KV 头）", dict(m, kv_heads=H)),
        ("GQA（主线，8 个 KV 头）", m),
        ("MQA（1 个 KV 头）", dict(m, kv_heads=1)),
        ("假设换成 MLA（512+64）", dict(layers=L, kv_lora_rank=512, qk_rope_head_dim=64)),
    ]
    for name, v in variants:
        elems = layer_list(v)[0][1]
        print(f"   {name:22} {elems:6d} 个数   {fmt(per_token(v)):>10} {fmt(kv_bytes(v, 32768)):>10}"
              f" {fmt(kv_bytes(v, 131072)):>10}")

    print("\n2. 公开模型（BF16，batch 1；只算随序列增长的部分）")
    print("   模型                 增长的层 / 总层   每 token     32K         128K      "
          "若全部层都是 MHA 的 128K")
    for name, mm in MODELS.items():
        grow = sum(1 for k, _, _ in layer_list(mm) if k != "linear")
        # 对照：同样层数和头数，每层都是 MHA、head_dim 128（MLA 模型的 K 头宽 192，这里保守按 128 算）
        mha = dict(layers=mm["layers"], kv_heads=mm["heads"], head_dim=mm.get("head_dim", 128))
        print(f"   {name:19} {grow:4d} / {mm['layers']:<4d}      {fmt(per_token(mm)):>10} "
              f"{fmt(kv_bytes(mm, 32768)):>10} {fmt(kv_bytes(mm, 131072)):>10}   "
              f"{fmt(kv_bytes(mha, 131072)):>10}")

    print("\n3. 对拍：和生产级 zero/tools/kv_cache_calc.py 的结果比较")
    sys.path.insert(0, str(ROOT))
    from zero.tools.kv_cache_calc import kv_cache_bytes

    ok = kv_cache_bytes(ROOT / "configs/main/pretrain.toml", 32768) == kv_bytes(m, 32768)
    print(f"   主线模型 32K：极简 {kv_bytes(m, 32768):,} 字节，zero {kv_cache_bytes(ROOT / 'configs/main/pretrain.toml', 32768):,} 字节，一致：{ok}")
    ds = {"num_hidden_layers": 61, "num_attention_heads": 128, "kv_lora_rank": 512,
          "qk_rope_head_dim": 64}
    same = kv_cache_bytes(ds, 131072) == kv_bytes(MODELS["DeepSeek-V3/V3.2"], 131072)
    print(f"   DeepSeek-V3 128K：两边一致：{same}")
