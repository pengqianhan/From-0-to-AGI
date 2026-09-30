"""第 15 章 · 极简代码 1：RoPE 每个维度对的"波长"——为什么模型读不了比训练时更长的文本

RoPE 把 head_dim 维两两配成 d/2 对，第 i 对在位置 m 旋转 m·ω_i 弧度：
    ω_i = θ^(-2i/d)            （转速，i = 0 .. d/2-1）
    λ_i = 2π / ω_i = 2π·θ^(2i/d) （波长：转满一圈要多少个 token）
训练长度 L 内，第 i 对一共转了 L / λ_i 圈。转不满一圈的维度对，训练时只见过圆周的一部分角度，
读更长的文本时就会遇到"从没见过的角度"。

本脚本用主线模型的 head_dim = 128，对比 θ = 1 万、50 万（Llama 3）、100 万（Qwen3 长上下文阶段）。
运行：uv run python chapters/15-midtraining-long-context/code/01_rope_wavelengths.py
"""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]   # 仓库根目录，用来 import zero

HEAD_DIM = 128          # 主线模型（configs/main/pretrain.toml）
TRAIN_LEN = 4096        # 主线预训练长度
TARGET_LEN = 32768      # 长上下文阶段的目标长度
BASES = [10_000.0, 500_000.0, 1_000_000.0]


def inv_freq(head_dim: int, theta: float) -> np.ndarray:
    """ω_i = θ^(-2i/d)，与 zero.model.compute_rope_inv_freq 的默认分支相同。"""
    i = np.arange(0, head_dim, 2) / head_dim          # 2i/d
    return theta ** (-i)


def wavelengths(head_dim: int, theta: float) -> np.ndarray:
    return 2 * math.pi / inv_freq(head_dim, theta)    # λ_i = 2π/ω_i


def fmt(x: float) -> str:
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    if x >= 1e4:
        return f"{x / 1e3:.0f}K"
    return f"{x:.1f}"


def main() -> None:
    print(f"head_dim = {HEAD_DIM}，共 {HEAD_DIM // 2} 个维度对；训练长度 {TRAIN_LEN}，目标长度 {TARGET_LEN}\n")

    # 1) 选几个维度对，看波长（单位：token）
    picks = [0, 8, 16, 24, 32, 40, 48, 56, 63]
    print("1) 各维度对的波长 λ_i（转一圈需要多少 token）")
    print("   i   " + "".join(f"{'θ=' + fmt(b):>12}" for b in BASES))
    for i in picks:
        row = "".join(f"{fmt(wavelengths(HEAD_DIM, b)[i]):>12}" for b in BASES)
        print(f"  {i:>3} {row}")

    # 2) 训练长度内转不满一圈的维度对有多少个
    print(f"\n2) 在长度 L 内转不满一圈（λ_i > L）的维度对个数（共 {HEAD_DIM // 2} 对）")
    print("   L       " + "".join(f"{'θ=' + fmt(b):>12}" for b in BASES))
    for L in [TRAIN_LEN, TARGET_LEN]:
        row = "".join(f"{int((wavelengths(HEAD_DIM, b) > L).sum()):>12}" for b in BASES)
        print(f"  {L:>6}  {row}")

    # 3) 读到 32K 时，哪些维度对会遇到"训练时没见过的角度"
    print("\n3) 读到 32K 时，会转到训练时（θ=1 万、长度 4K）没见过的角度的维度对个数")
    w_train = inv_freq(HEAD_DIM, 1e4)
    settings = {
        "什么都不改（θ=1 万）": w_train,
        "调大基频到 θ=100 万": inv_freq(HEAD_DIM, 1e6),
        "位置内插 PI（全部 ÷8）": w_train / 8,
    }
    for name, w in settings.items():
        n = unseen_pairs(w_train, w, TRAIN_LEN, TARGET_LEN)
        print(f"   {name:<16} {n:>3} 对；相邻两个 token 在最快那一对上相差 {w[0]:.3f} 弧度")
    print("   注意：调大基频后没有“没见过的角度”了，但同一个角度对应的距离变了（见第 4 部分），")
    print("   所以模型要在新基频下接着训练一段；PI 则把最快那一对也压慢 8 倍，近处的位置变得难分辨。")

    # 4) 调大基频之后：同一个维度对的转速变慢了多少倍
    print("\n4) 把 θ 从 1 万调到 100 万后，同一个维度对的转速变慢了多少倍")
    w_old, w_new = inv_freq(HEAD_DIM, 1e4), inv_freq(HEAD_DIM, 1e6)
    for i in [0, 16, 32, 48, 63]:
        print(f"   i={i:>2}: 慢了 {w_old[i] / w_new[i]:>7.1f} 倍")
    print("   （i = 0 那一对不受影响；越往后的维度对被放慢得越多，最多约 100 倍）")

    # 5) 长上下文的代价：主线模型每个训练 token 的算力（zero 的口径）
    sys.path.insert(0, str(ROOT))
    from zero.config import load_model_config
    from zero.model import estimate_flops_per_token

    cfg = load_model_config(ROOT / "configs/main/pretrain.toml")
    print("\n5) 主线模型每个训练 token 的算力（前向 + 反向，zero.model.estimate_flops_per_token）")
    base = estimate_flops_per_token(cfg, 0)
    for T in [4096, 32768]:
        total = estimate_flops_per_token(cfg, T)
        print(f"   序列长 {T:>6}: {total / 1e9:5.2f} GFLOP/token，其中注意力的 QKᵀ 与 AV 占 {(total - base) / total:.0%}")
    ratio = estimate_flops_per_token(cfg, 32768) / estimate_flops_per_token(cfg, 4096)
    print(f"   同样多的 token，用 32K 序列训练比 4K 贵 {ratio:.2f} 倍")


def unseen_pairs(w_train: np.ndarray, w_new: np.ndarray, train_len: int, new_len: int) -> int:
    """训练时第 i 对见过的角度是 [0, L_train·ω_i]；转满过一圈的维度对见过所有角度，不会"没见过"。
    读到 new_len 时角度最大到 new_len·ω'_i；超出训练时的范围，就是"没见过的角度"。"""
    seen = train_len * w_train
    full_circle = seen >= 2 * math.pi              # 训练时转满过一圈：所有角度都见过
    return int(((~full_circle) & (new_len * w_new > seen + 1e-9)).sum())


if __name__ == "__main__":
    main()
