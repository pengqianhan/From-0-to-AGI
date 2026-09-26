"""第 22 章 · 极简代码 1：三种掩码、感受野、以及滑动窗口能省多少 KV cache

三件事，都不需要训练：
  1. 画出全因果、滑动窗口（W=4）、局部-全局交替三种注意力掩码，数一数"要算多少对"；
  2. 感受野：只用滑动窗口时，信息每过一层最多往前多传 W-1 个位置；插一层全局就一步到头；
  3. KV cache 账本：用几个公开模型的真实配置（config.json，2026-09 读取），算 128K 上下文下
     全注意力 vs 实际的局部-全局配置各要多少显存。

运行：uv run python chapters/22-local-sparse-attention/code/01_masks_and_ledger.py
"""

from __future__ import annotations

import torch

torch.set_num_threads(1)


# ── 1. 掩码 ────────────────────────────────────────────────────────────────
def causal_mask(T: int) -> torch.Tensor:
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    return j <= i  # 只看自己和过去


def sliding_mask(T: int, W: int) -> torch.Tensor:
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    return (j <= i) & (i - j < W)  # 只看最近 W 个位置（含自己）


def show(mask: torch.Tensor, title: str) -> None:
    print(f"{title}（■ = 能看见，共 {int(mask.sum())} 对）")
    for row in mask.tolist():
        print("  " + " ".join("■" if v else "·" for v in row))


# ── 2. 感受野：L 层之后，最后一个 token 的信息能来自多远 ──────────────────────
def receptive_field(masks: list[torch.Tensor]) -> list[int]:
    """reach[i, j] = 第 j 个位置的信息能否经过前面这些层流到位置 i（布尔矩阵连乘）。"""
    T = masks[0].shape[0]
    reach = torch.eye(T, dtype=torch.bool)
    out = []
    for m in masks:
        reach = (m.float() @ reach.float()) > 0  # 这一层 i 看 k，k 已经汇集了 j
        far = int(torch.nonzero(reach[-1])[0])  # 最后一个 token 能"看到"的最早位置
        out.append(T - 1 - far)  # 最远距离
    return out


# ── 3. KV cache 账本 ────────────────────────────────────────────────────────
def kv_bytes(n_full: int, n_sliding: int, W: int, T: int, kv_heads: int, head_dim: int) -> int:
    """2（K 和 V）× 每层存的位置数 × KV 头 × head_dim × 2 字节（BF16）。"""
    per_pos = 2 * kv_heads * head_dim * 2
    return (n_full * T + n_sliding * min(W, T)) * per_pos


# 数字来自各模型的 config.json（链接见 README "采用方与来源"）
MODELS = [
    # 名字, 层数, 其中全局层, 窗口, KV 头, head_dim
    ("Mistral-7B-v0.1（全部滑动）", 32, 0, 4096, 8, 128),
    ("Gemma-3-27B（5 局部:1 全局）", 62, 10, 1024, 16, 128),
    ("gpt-oss-120b（1:1 交替）", 36, 18, 128, 8, 64),
    ("OLMo-3-7B（3 局部:1 全局）", 32, 8, 4096, 32, 128),
]


def main() -> None:
    T, W = 10, 4
    full = causal_mask(T)
    swa = sliding_mask(T, W)
    show(full, f"全因果注意力 T={T}")
    show(swa, f"滑动窗口 W={W}")
    for T_big in (1024, 32768):
        n_full = T_big * (T_big + 1) // 2
        n_swa = int(sliding_mask(T_big, W).sum()) if T_big <= 4096 else W * T_big - W * (W - 1) // 2
        print(f"T={T_big:>6}: 全因果 {n_full:>12,} 对，W={W} 的滑动窗口 {n_swa:>9,} 对"
              f"（{n_full / n_swa:,.0f} 倍）")

    print("\n感受野：最后一个 token 在第 l 层之后，信息最远能来自多少个位置之前（T=64, W=4）")
    T = 64
    L = 6
    all_swa = [sliding_mask(T, W)] * L
    interleave = [sliding_mask(T, W) if (l + 1) % 3 else causal_mask(T) for l in range(L)]
    print("  层             :", " ".join(f"{l + 1:>3}" for l in range(L)))
    print("  全部滑动窗口    :", " ".join(f"{d:>3}" for d in receptive_field(all_swa)))
    print("  2 局部 + 1 全局 :", " ".join(f"{d:>3}" for d in receptive_field(interleave)))
    print(f"  理论：只用滑动窗口时 l 层的感受野 = l × (W − 1) = l × {W - 1}")

    T = 131072
    print(f"\nKV cache 账本：上下文 {T:,}（128K）token、batch 1、BF16")
    print(f"  {'模型':<28}{'假如每层都是全注意力':>14}{'实际配置':>12}{'节省':>8}")
    for name, n_layers, n_global, win, kvh, hd in MODELS:
        full_b = kv_bytes(n_layers, 0, win, T, kvh, hd)
        real_b = kv_bytes(n_global, n_layers - n_global, win, T, kvh, hd)
        print(f"  {name:<28}{full_b / 2**30:>12.2f} GiB{real_b / 2**30:>10.2f} GiB"
              f"{1 - real_b / full_b:>9.1%}")


if __name__ == "__main__":
    main()
