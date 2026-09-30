"""第 21 章 · 极简代码 2：长上下文为什么贵——prefill 算力按 T² 涨，decode 被显存带宽卡住

两笔账，都用主线模型（configs/main）的真实配置：

1. prefill（一次处理整段提示词，T 个 token）的前向浮点运算：
       矩阵乘部分：2 × N × T                         （N = 参与矩阵乘的参数，含 lm_head）
       注意力部分：2 × 层数 × q_dim × T²            （QKᵀ 和 AV 两次矩阵乘、因果掩码只算一半）
   T 一长，T² 那一项就压过了参数那一项。

2. decode（每步生成 1 个 token，batch 里有 B 条对话，每条已有 T 个 token）：
       要读的字节  ≈ 权重（整份读一遍，B 条共享）+ B × KV cache（每条各读自己的）
       要算的次数 ≈ B × (2N + 4 × 层数 × q_dim × T)
   一步的最短时间 = max(算力时间, 读显存时间)。算术强度（次/字节）远低于硬件的"脊点"时，
   GPU 在等数据，这叫带宽受限（memory-bound）。

硬件按 H100 SXM 的公开规格粗算：稠密 BF16 989.5 TFLOPS，HBM3 带宽 3.35 TB/s，显存 80 GB。
这是理论下限，不是实测；真实系统还有 kernel 效率、激活、碎片等开销。
运行：uv run python chapters/21-kv-cache-ledger/code/02_prefill_decode.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from zero.config import load_model_config  # noqa: E402
from zero.model import count_params  # noqa: E402

PEAK_FLOPS = 989.5e12  # H100 SXM 稠密 BF16（zero/tools/estimate_cost.py 同一口径）
HBM_BW = 3.35e12  # 字节/秒
HBM_BYTES = 80e9
BF16 = 2


def model_numbers():
    c = load_model_config(ROOT / "configs" / "main" / "pretrain.toml")
    p = count_params(c)
    n_matmul = c.n_layers * (p["attention_per_layer"] + p["ffn_per_layer"]) + c.dim * c.vocab_size
    kv_per_token = 2 * c.n_layers * c.n_kv_heads * c.head_dim * BF16
    return c, p["total"], n_matmul, kv_per_token


def prefill_flops(c, n_matmul: int, T: int) -> tuple[float, float]:
    linear = 2 * n_matmul * T
    attn = 2 * c.n_layers * c.q_dim * T * T  # 4·q_dim 每对 (i, j)，因果只有 T²/2 对
    return linear, attn


def decode_step(c, n_params: int, n_matmul: int, kv_per_token: int, T: int, B: int) -> dict:
    flops = B * (2 * n_matmul + 4 * c.n_layers * c.q_dim * T)
    w_bytes = n_params * BF16
    kv = B * T * kv_per_token
    t_compute, t_memory = flops / PEAK_FLOPS, (w_bytes + kv) / HBM_BW
    t = max(t_compute, t_memory)
    return dict(flops=flops, weights=w_bytes, kv=kv, intensity=flops / (w_bytes + kv),
                t=t, bound="带宽" if t_memory > t_compute else "算力", tok_s=B / t)


if __name__ == "__main__":
    c, n_params, n_matmul, kv_tok = model_numbers()
    print(f"主线模型：总参数 {n_params / 1e6:.1f}M，参与矩阵乘的参数 N = {n_matmul / 1e6:.1f}M，"
          f"q_dim = {c.q_dim}，每 token KV {kv_tok:,} 字节")

    print("\n1. prefill 的前向运算量（一次喂 T 个 token）")
    print("        T     矩阵乘部分      注意力部分     注意力占比   H100 理论耗时")
    for T in (1024, 4096, 32768, 131072):
        lin, att = prefill_flops(c, n_matmul, T)
        print(f"   {T:7,d}   {lin:12.3e}   {att:12.3e}   {att / (lin + att):8.1%}   "
              f"{(lin + att) / PEAK_FLOPS * 1e3:9.1f} ms")

    ridge = PEAK_FLOPS / HBM_BW
    print(f"\n2. decode 一步（H100 的脊点：{ridge:.0f} 次/字节；算术强度低于它就是带宽受限）")
    print("     T      B   读权重      读 KV cache   算术强度   受限于   一步耗时   吞吐 token/s  80GB 装得下？")
    for T in (4096, 32768):
        for B in (1, 16, 64):
            d = decode_step(c, n_params, n_matmul, kv_tok, T, B)
            print(f"   {T:6,d} {B:4d}   {d['weights'] / 2**30:6.2f} GiB  {d['kv'] / 2**30:8.2f} GiB"
                  f"   {d['intensity']:7.1f}    {d['bound']}    {d['t'] * 1e3:6.2f} ms  "
                  f"{d['tok_s']:10,.0f}   {'是' if d['weights'] + d['kv'] <= HBM_BYTES else '否'}")

    print("\n3. 显存装得下几条对话？（80 GB 减去权重，全部给 KV cache；不计激活与碎片）")
    free = HBM_BYTES - n_params * BF16
    for T in (4096, 32768, 131072):
        for name, per in (("GQA 主线", kv_tok), ("若是 MHA", kv_tok * 2),
                          ("若换 MLA 512+64", c.n_layers * 576 * BF16)):
            print(f"   上下文 {T:7,d}  {name:14} 最多 {int(free // (T * per)):5d} 条")
